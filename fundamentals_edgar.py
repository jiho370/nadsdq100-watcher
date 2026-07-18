#!/usr/bin/env python3
"""
fundamentals_edgar.py — SEC EDGAR 에서 '과거 시점' 펀더멘탈을 무료로 수집(백테스트용).

왜 필요한가: 무료 yfinance 는 현재 펀더멘탈만 준다. 밸류·퀄리티 지표를 '과거'에 정직하게
백테스트하려면 그 시점에 '공시돼 있던' 재무가 필요하다. EDGAR companyconcept API 가
각 재무 항목을 제출일(filed)과 함께 제공하므로, filed<=t 조건으로 미래참조를 피할 수 있다.

수집 항목(연간 10-K 기준, 시점정보):
  eps    = 희석 EPS               → 밸류(이익수익률 = EPS/가격 = 1/PER)
  ni     = 순이익                 → 퀄리티(ROE = 순이익/자본)
  equity = 자기자본
  ocf    = 영업활동현금흐름        → 현금흐름(FCF = OCF - CapEx)
  capex  = 유형자산취득(CapEx)

출력: output/fundamentals_cache.json = { "AAPL": {"eps":[{end,filed,val}...], "ni":[...], ...}, ... }
      (한 번 받으면 캐시. 재실행은 캐시 사용 → 빠르고 SEC 레이트리밋 안전)

실행(네트워크 되는 PC): python fundamentals_edgar.py            # 현재 S&P500 전체 수집
                        python fundamentals_edgar.py --tickers AAPL,MSFT   # 일부만
SEC 예절: User-Agent 헤더 필수(SEC_UA 환경변수로 지정 가능), 초당 요청 제한 준수.
"""
from __future__ import annotations
import os, sys, json, time, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SEC_UA = os.environ.get("SEC_UA", "sp500-daily-report research (contact: example@example.com)")
HDRS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
RATE = float(os.environ.get("SEC_RATE", "0.12"))   # 요청 간 최소 간격(초) ≈ 8req/s
CACHE = os.environ.get("FUND_CACHE", "output/fundamentals_cache.json")
WORKERS = int(os.environ.get("SEC_WORKERS", "6"))  # 2026-07-18: 종목 단위 병렬 처리자 수


class _RateLimiter:
    """전체 스레드 합산 요청 간격을 RATE로 강제하는 전역 레이트리미터(2026-07-18,
    지호 님 제안 — "속도 느리면 병렬로"). SEC 가이드라인(초당 ~10회)은 '전체 합산'
    기준이지 종목당이 아니라서, 종목을 병렬로 처리해도 이 리미터를 다 같이 통과시키면
    총 요청속도는 그대로 RATE 이하로 유지되면서 처리량(종목/분)만 워커 수만큼 늘어난다."""
    def __init__(self, rate: float):
        self._rate = rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._rate
        delay = start - now
        if delay > 0:
            time.sleep(delay)


_LIMITER = _RateLimiter(RATE)

# 각 항목마다 회사별로 태그가 다를 수 있어 '후보 태그'를 순서대로 시도(첫 성공 사용).
CONCEPTS = {
    "eps":     [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasic")],
    "ni":      [("us-gaap", "NetIncomeLoss")],
    "equity":  [("us-gaap", "StockholdersEquity"),
                ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "ocf":     [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
                ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "capex":   [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
                ("us-gaap", "PaymentsToAcquireProductiveAssets")],
    "revenue": [("us-gaap", "Revenues"),
                ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "SalesRevenueNet"), ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax")],
    "assets":  [("us-gaap", "Assets")],
    "liab":    [("us-gaap", "Liabilities")],
    "debt":    [("us-gaap", "LongTermDebtNoncurrent"), ("us-gaap", "LongTermDebt")],
    "opinc":   [("us-gaap", "OperatingIncomeLoss")],
    "gross":   [("us-gaap", "GrossProfit")],
    # 2026-07-18(지호 님 지적 — REGN 등 실측): GrossProfit 단일 태그만 쓰면 협업매출·복합
    # 매출구조 기업(GOOGL·LLY·ABBV·MRK·CAT 등 다수, 전체 유니버스의 54.3% 실측)이 전부
    # 누락돼 gp_assets/int_gp_assets가 조용히 z=0(중립)으로 빠짐 — "수익성이 안 좋다"가
    # 아니라 "안 보인다"인데 구분이 안 됐음. 매출원가 태그로 역산하는 폴백 추가
    # (factor_values에서 gross 없으면 revenue-cost_of_revenue로 대체).
    "cost_of_revenue": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold"),
                        ("us-gaap", "CostOfGoodsSold"), ("us-gaap", "CostOfServices")],
    "cash":    [("us-gaap", "CashAndCashEquivalentsAtCarryingValue")],
    "divs":    [("us-gaap", "PaymentsOfDividendsCommonStock"), ("us-gaap", "PaymentsOfDividends")],
    "dep":     [("us-gaap", "DepreciationDepletionAndAmortization"),
                ("us-gaap", "DepreciationAmortizationAndAccretionNet"),
                ("us-gaap", "DepreciationAndAmortization")],
    "buyback": [("us-gaap", "PaymentsForRepurchaseOfCommonStock"),
                ("us-gaap", "PaymentsForRepurchaseOfEquity")],
    "issuance": [("us-gaap", "ProceedsFromIssuanceOfCommonStock")],
    # 운전자본 변동(현금흐름표) — COP(현금기준 영업수익성) 계산용
    "ar_chg":  [("us-gaap", "IncreaseDecreaseInAccountsReceivable")],
    "inv_chg": [("us-gaap", "IncreaseDecreaseInInventories")],
    "ap_chg":  [("us-gaap", "IncreaseDecreaseInAccountsPayable"),
                ("us-gaap", "IncreaseDecreaseInAccountsPayableTrade")],
    # 무형자산 투자 — 무형조정 가치·수익성 팩터용 (Eisfeldt et al.; Berkin et al. 2024 JOI:
    # R&D 전액 + SG&A 30%를 자본화, R&D 상각 15%/년). 캐시에 없으면 증분 수집됨.
    # 2026-07-18: 제약·바이오 다수(ABBV·AMGN 등 실측 확인)가 표준 태그 대신 아래 변형
    # (인수 진행중 R&D 비용 제외분)을 씀 — 첫 태그 없으면 이걸로 폴백.
    "rnd":     [("us-gaap", "ResearchAndDevelopmentExpense"),
               ("us-gaap", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost")],
    "sga":     [("us-gaap", "SellingGeneralAndAdministrativeExpense"),
                ("us-gaap", "GeneralAndAdministrativeExpense")],
}


def ticker_cik_map() -> dict:
    """SEC 공식 ticker→CIK(10자리) 매핑. company_tickers.json은 불완전하다(2026-07-17
    실측 — Comerica·Kellanova·Hologic·TEGNA 등 활성 대형주도 종종 빠져 있음)."""
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HDRS, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


# 2026-07-17: browse-edgar 티커 폴백이 잘못 매칭하는 걸로 실측 확인된 종목(그 티커를 나중에
# 완전히 무관한 회사가 재사용) — 폴백을 절대 안 씀. 지호 님 지적으로 발견("전혀 다른
# 회사가 티커를 재활용"): HCP(옛 헬스케어 REIT)→HashiCorp, LLL(옛 L3)→JX Luxventure,
# MON(옛 몬산토)→Monument Circle Acquisition, PX(옛 프락스에어)→Ridgepost Capital,
# XL(옛 XL그룹)→Spruce Power. 전부 이름이 완전히 다른 무관 기업으로 확인됨.
_TICKER_COLLISION_BLOCKLIST = {"HCP", "LLL", "MON", "PX", "XL"}


def _browse_edgar_cik(ticker: str) -> str | None:
    """company_tickers.json에 없는 티커의 폴백 조회 — SEC browse-edgar의 CIK 파라미터는
    티커 심볼도 직접 받아준다(회사명 검색과 별개 경로, company_tickers.json보다 넓은
    커버리지 실측 확인). 회사명은 반환하지 않으므로 호출부가 별도 검증해야 한다
    (_TICKER_COLLISION_BLOCKLIST 참고 — 티커 재활용으로 완전히 다른 회사가 잡힐 수 있음)."""
    if ticker.upper() in _TICKER_COLLISION_BLOCKLIST:
        return None
    _LIMITER.wait()
    try:
        r = requests.get("https://www.sec.gov/cgi-bin/browse-edgar",
                         params={"action": "getcompany", "CIK": ticker, "type": "10-K",
                                 "dateb": "", "owner": "include", "count": "1", "output": "atom"},
                         headers=HDRS, timeout=15)
        m = re.search(r"<cik>(\d+)</cik>", r.text)
        return m.group(1).zfill(10) if m else None
    except Exception:
        return None


def annual_points(units: list) -> list:
    """companyconcept units → 연간(10-K/FY) 값들을 '최초 공시(filed)' 기준으로 정리.
       각 회계연도(end)마다 가장 먼저 제출된 값 1개만(재작성 이전 = 그 시점 실제 공시치)."""
    best = {}  # end -> (filed, val)
    for it in units or []:
        form = str(it.get("form", ""))
        if not form.startswith("10-K"):
            continue
        end, filed, val = it.get("end"), it.get("filed"), it.get("val")
        if not (end and filed and val is not None):
            continue
        # 연간 흐름값만(기간 ~1년). 잔고항목(equity)은 start 없음 → 그대로 통과.
        start = it.get("start")
        if start:
            days = (_d(end) - _d(start)).days
            if not (330 <= days <= 400):
                continue
        if end not in best or filed < best[end][0]:
            best[end] = (filed, float(val))
    return [{"end": e, "filed": f, "val": v} for e, (f, v) in sorted(best.items())]


def _d(s):
    import datetime as dt
    return dt.date.fromisoformat(s)


_STALE_YEARS = 2   # 최신 태그 판정 기준 — 이보다 오래전에 멈춘 데이터는 "회사가 태그를 갈아탔다"로 간주


def fetch_concept(cik: str, candidates) -> list:
    """후보 (taxonomy, tag) 들을 순서대로 시도하되, 2026-07-18 수정(지호 님 지적으로
    발견한 버그 대응): 첫 성공 태그의 데이터가 **최근(_STALE_YEARS년 이내)까지 있으면**
    거기서 멈추고(기존처럼 빠름), **오래전에 멈춘 stale 데이터면** 다음 후보도 마저 조회해
    합친다. 예전 버전은 무조건 첫 성공에서 멈춰서, 회사가 태그를 갈아탄 이력(가장 흔한
    사례: 2018년 ASC 606 회계기준 전환으로 Revenues→RevenueFromContractWithCustomer...
    로 업계 전체가 이동)이 있을 때 옛 태그에 조금이라도 데이터가 있으면 최신 데이터를
    통째로 놓쳤다(실측: AAPL·MSFT 등 175종목이 매출 데이터가 2010~2020년경에서 멈춰
    있었음). 반대로 첫 시도만 무조건 전부 다 조회하게 하면(중간 수정본) 정상 케이스까지
    매번 4배 가까운 요청을 만들어 SEC 레이트리밋에 걸려 사실상 멈추는 문제가 있었음 —
    이 버전은 그 중간(느린 회사만 추가 조회)."""
    import datetime as _dt
    if candidates and isinstance(candidates[0], str):     # (tax, tag) 단일도 허용
        candidates = [candidates]
    cutoff = (_dt.date.today() - _dt.timedelta(days=365 * _STALE_YEARS)).isoformat()
    merged = []
    for i, (taxonomy, tag) in enumerate(candidates):
        _LIMITER.wait()   # 전체 스레드 합산 속도 제한(병렬 처리 시에도 SEC 레이트리밋 준수)
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
        try:
            r = requests.get(url, headers=HDRS, timeout=30)
            if r.status_code != 200:
                continue
            units = r.json().get("units", {})
            got = None
            for key in ("USD/shares", "USD", *units.keys()):
                if key in units and units[key]:
                    got = units[key]; break
            if not got:
                continue
            merged.extend(got)
            last_end = max((p.get("end") or "") for p in got)
            if last_end >= cutoff:
                break   # 최신까지 있음 — 다음 후보(다른 태그) 조회 불필요
        except Exception:
            continue
    return merged


def _collect_one(t: str, cik: str | None, refresh: bool, existing_rec: dict):
    """종목 하나의 전체 CONCEPTS 수집(스레드에서 실행). cik가 없으면(company_tickers.json
    누락) browse-edgar 폴백을 이 스레드 안에서 시도 — 반환 (ticker, rec, used_fallback)."""
    used_fallback = False
    if not cik:
        cik = _browse_edgar_cik(t.upper())
        used_fallback = bool(cik)
    rec = {} if refresh else dict(existing_rec or {})
    if not cik:
        return t, rec, used_fallback
    for name, cands in CONCEPTS.items():
        if name in rec and not refresh:
            continue
        rec[name] = annual_points(fetch_concept(cik, cands))
    return t, rec, used_fallback


def build(tickers: list, cache_path: str = CACHE, refresh: bool = False, workers: int = WORKERS) -> dict:
    """2026-07-18(지호 님 제안 — "속도 느리면 병렬로"): 종목 단위로 최대 workers개 스레드
    동시 처리. SEC 가이드라인(초당 ~10회)은 전체 합산 기준이라 _LIMITER(전역 레이트리미터)
    가 스레드 수와 무관하게 총 요청속도를 RATE로 묶어주므로, 병렬화해도 SEC 서버 부담은
    순차 실행과 동일하고 처리량(종목/분)만 workers배 가까이 늘어난다."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    cache = {}
    if os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cikmap = ticker_cik_map()
    # 항목별 증분 수집: 캐시에 없는 항목(새로 추가된 지표 등)만 골라 받는다.
    todo = [t for t in tickers if refresh or any(k not in (cache.get(t.upper()) or {}) for k in CONCEPTS)]
    print(f"[EDGAR] 대상 {len(tickers)}종목 · 갱신필요 {len(todo)} · 워커 {workers}개 · "
         f"수집 항목 {list(CONCEPTS)}", file=sys.stderr)
    n_fallback, n_done = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_collect_one, t, cikmap.get(t.upper()), refresh,
                          cache.get(t.upper())): t for t in todo}
        try:
            for fut in as_completed(futs):
                t, rec, used_fb = fut.result()
                cache[t.upper()] = rec
                n_fallback += int(used_fb)
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ...{n_done}/{len(todo)} 수집, 중간 저장", file=sys.stderr)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
        finally:
            # 중단(Ctrl-C 등) 시에도 지금까지 완료분은 저장 — 미완료 future는 취소만 시도
            # (실행 중인 스레드는 강제 종료 못 함, ThreadPoolExecutor의 알려진 한계).
            for f in futs:
                f.cancel()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"[EDGAR] 저장 완료: {cache_path} ({len(cache)}종목, browse-edgar 폴백으로 {n_fallback}종목 추가 복구)", file=sys.stderr)
    return cache


def asof(points: list, date_iso: str):
    """date_iso 시점에 '이미 공시된' 가장 최근 값. 없으면 None."""
    v = None
    for p in points or []:
        if p["filed"] <= date_iso:
            v = p["val"]
        else:
            break
    return v


def asof_pair(points: list, date_iso: str):
    """date_iso 시점 기준 (최근값, 직전연도값). 성장률 계산용."""
    seen = [p["val"] for p in (points or []) if p["filed"] <= date_iso]
    if not seen:
        return None, None
    return seen[-1], (seen[-2] if len(seen) >= 2 else None)


def asof_end(points: list, date_iso: str):
    """date_iso 시점의 최근값 + 그 회계기간 종료일(end). 2026-07-18: gross_profit 폴백
    (revenue-cost_of_revenue)이 회계기간이 다른 두 값을 섞지 않도록 기간매칭용으로 추가
    (REGN처럼 cost_of_revenue 태그가 몇 년째 갱신 안 된 케이스에서 옛 값을 최신인 것처럼
    잘못 쓰는 걸 방지)."""
    v, e = None, None
    for p in points or []:
        if p["filed"] <= date_iso:
            v, e = p["val"], p["end"]
        else:
            break
    return v, e


# 팩터 이름(모두 '높을수록 좋다' 방향으로 정의) — 백테스트·라이브 공용
# (학술 리포트 반영: gp_assets=Novy-Marx 품질, ebitda_ev·fcf_ev=자본구조 반영 가치, asset_growth=과잉투자 배제)
FUND_FACTOR_NAMES = ["value", "sales_yield", "roe", "roa", "net_margin", "op_margin",
                     "gross_margin", "gp_assets", "fcf_yield", "ebitda_ev", "fcf_ev",
                     "leverage", "asset_growth", "rev_growth", "ni_growth", "div_yield",
                     "accruals", "shareholder_yield", "cop",
                     # 2023~2024 문헌 기반 추가(SCORE_MODEL_DESIGN.md 참고):
                     # droe=이익성장(JKP profit-growth 군집/q5 기대성장 대용),
                     # debt_issuance=부채발행 억제(JKP debt-issuance 군집),
                     # rd_mktcap=R&D 집약도, int_gp_assets·int_value=무형조정 수익성·가치
                     "droe", "debt_issuance", "rd_mktcap", "int_gp_assets", "int_value"]


def _intangible_capital(rec: dict, date_iso: str,
                        rd_delta=0.15, sga_frac=0.30, sga_delta=0.20):
    """공시된(filed<=t) 연간 R&D·SG&A 흐름을 영구재고법으로 자본화한 무형자본 K_int.
    R&D 100%(상각 15%/년) + SG&A 30%(상각 20%/년) — Berkin-Dugar-Pozharny(2024) 방식."""
    k = 0.0; seen = False
    pts = [p for p in rec.get("rnd") or [] if p["filed"] <= date_iso]
    for age, p in enumerate(reversed(pts)):
        k += p["val"] * (1 - rd_delta) ** age; seen = True
    pts = [p for p in rec.get("sga") or [] if p["filed"] <= date_iso]
    for age, p in enumerate(reversed(pts)):
        k += sga_frac * p["val"] * (1 - sga_delta) ** age; seen = True
    return k if seen else None


def factor_values(rec: dict, date_iso: str, price: float) -> dict:
    """시점(date_iso)의 펀더멘탈 재무비율 팩터들. 계산 불가한 건 생략(호출부에서 결측 처리)."""
    rec = rec or {}
    g = lambda k: asof(rec.get(k), date_iso)
    eps, ni, eq = g("eps"), g("ni"), g("equity")
    assets, debt, cash = g("assets"), g("debt"), g("cash")
    rev, opinc, gross = g("revenue"), g("opinc"), g("gross")
    if gross is None and rev is not None:
        # 2026-07-18: GrossProfit 미태깅 기업(협업매출·복합매출구조) 폴백 — 매출원가로 역산.
        # 회계기간(end)이 revenue와 정확히 같은 값만 사용 — REGN처럼 cost_of_revenue 태그가
        # 몇 년째 갱신 안 된 경우 옛 값을 최신처럼 잘못 섞어 쓰는 걸 방지(지호 님 지적).
        _, rev_end = asof_end(rec.get("revenue"), date_iso)
        cor, cor_end = asof_end(rec.get("cost_of_revenue"), date_iso)
        if cor is not None and rev_end is not None and cor_end == rev_end:
            gross = rev - cor
    ocf, capex, divs, dep = g("ocf"), g("capex"), g("divs"), g("dep")
    buyback, issuance = g("buyback"), g("issuance")
    ar_chg, inv_chg, ap_chg = g("ar_chg"), g("inv_chg"), g("ap_chg")
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None
    shares = (ni / eps) if (ni is not None and eps not in (None, 0)) else None
    mktcap = (price * shares) if (shares and price and price > 0) else None
    ev = (mktcap + (debt or 0) - (cash or 0)) if mktcap else None    # 기업가치
    ebitda = (opinc + dep) if (opinc is not None and dep is not None) else None
    rev_now, rev_prev = asof_pair(rec.get("revenue"), date_iso)
    ni_now, ni_prev = asof_pair(rec.get("ni"), date_iso)
    asset_now, asset_prev = asof_pair(rec.get("assets"), date_iso)
    out = {}
    if price and price > 0:
        if eps is not None:
            out["value"] = eps / price
        if rev is not None and shares:
            out["sales_yield"] = (rev / shares) / price
        if fcf is not None and shares:
            out["fcf_yield"] = (fcf / shares) / price
        if divs is not None and shares:
            out["div_yield"] = (abs(divs) / shares) / price     # 배당지출은 음수로 기록됨 → abs
    if ni is not None and eq not in (None, 0):
        out["roe"] = ni / eq
    if ni is not None and assets not in (None, 0):
        out["roa"] = ni / assets
    if ni is not None and rev not in (None, 0):
        out["net_margin"] = ni / rev
    if opinc is not None and rev not in (None, 0):
        out["op_margin"] = opinc / rev
    if gross is not None and rev not in (None, 0):
        out["gross_margin"] = gross / rev
    if gross is not None and assets not in (None, 0):
        out["gp_assets"] = gross / assets                       # Novy-Marx 품질(매출총이익/자산)
    if ebitda is not None and ev not in (None, 0):
        out["ebitda_ev"] = ebitda / ev                         # EV/EBITDA 역수(자본구조 반영 가치)
    if fcf is not None and ev not in (None, 0):
        out["fcf_ev"] = fcf / ev                               # FCF/EV
    if debt is not None and eq not in (None, 0):
        out["leverage"] = -(debt / eq)                          # 부채 적을수록 가점
    if asset_now is not None and asset_prev not in (None, 0):
        out["asset_growth"] = -(asset_now / asset_prev - 1)     # 자산성장 낮을수록 가점(과잉투자 배제)
    if rev_now is not None and rev_prev not in (None, 0):
        out["rev_growth"] = rev_now / rev_prev - 1
    if ni_now is not None and ni_prev not in (None, 0):
        out["ni_growth"] = ni_now / ni_prev - 1
    # 발생액 품질(Sloan): 현금이익 비중 높을수록 가점 = -(순이익-영업현금)/자산
    if ni is not None and ocf is not None and assets not in (None, 0):
        out["accruals"] = -((ni - ocf) / assets)
    # COP(현금기준 영업수익성, Ball et al. 2016): (영업이익+감가상각 − 운전자본증가) / 자산
    if opinc is not None and assets not in (None, 0):
        wc = (ar_chg or 0) + (inv_chg or 0) - (ap_chg or 0)   # 운전자본 증가(현금 유출)
        out["cop"] = (opinc + (dep or 0) - wc) / assets
    # 주주환원수익률: (배당 + 자사주매입 − 신주발행) / 시총 (현금흐름표 기준, 분할 무관)
    if mktcap and (divs is not None or buyback is not None or issuance is not None):
        yield_cash = abs(divs or 0) + abs(buyback or 0) - (issuance or 0)
        out["shareholder_yield"] = yield_cash / mktcap
    # --- 2023~2024 문헌 기반 추가 팩터 (SCORE_MODEL_DESIGN.md) ---
    # 이익성장(ΔROE): JKP(2023) profit-growth 군집 / q5 기대성장 팩터의 실측 가능한 대용
    eq_now, eq_prev = asof_pair(rec.get("equity"), date_iso)
    if (ni_now is not None and eq_now not in (None, 0)
            and ni_prev is not None and eq_prev not in (None, 0)):
        out["droe"] = ni_now / eq_now - ni_prev / eq_prev
    # 부채발행 억제: JKP(2023) debt-issuance 군집 — 순부채증가 적을수록 가점
    debt_now, debt_prev = asof_pair(rec.get("debt"), date_iso)
    if debt_now is not None and debt_prev is not None and assets not in (None, 0):
        out["debt_issuance"] = -((debt_now - debt_prev) / assets)
    # R&D 집약도 + 무형조정 수익성·가치 (R&D 자본화 문헌, Berkin et al. 2024)
    rnd = g("rnd")
    if rnd is not None and mktcap:
        out["rd_mktcap"] = rnd / mktcap
    if rnd is not None and gross is not None and assets not in (None, 0):
        out["int_gp_assets"] = (gross + rnd) / assets   # R&D는 비용처리돼 이익을 깎으므로 가산
    k_int = _intangible_capital(rec, date_iso)
    if k_int is not None and eq is not None and mktcap:
        out["int_value"] = (eq + k_int) / mktcap        # 무형조정 장부가/시총
    return out


def main():
    ap = argparse.ArgumentParser(description="SEC EDGAR 과거 펀더멘탈 수집")
    ap.add_argument("--tickers", default=None, help="쉼표구분(미지정 시 현재 S&P500 전체)")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 재수집")
    args = ap.parse_args()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        import sp500_daily_report as R
        tickers, _ = R.get_sp500()
        bad = ("-W", "-WI", "-WS", "-U", "-RT", "-R", ".W", ".U")
        tickers = [s for s in tickers if not any(s.upper().endswith(x) for x in bad)]
    build(tickers, refresh=args.refresh)


if __name__ == "__main__":
    main()
