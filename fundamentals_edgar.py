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
import os, sys, json, time, argparse, re

import requests

SEC_UA = os.environ.get("SEC_UA", "sp500-daily-report research (contact: example@example.com)")
HDRS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
RATE = float(os.environ.get("SEC_RATE", "0.12"))   # 요청 간 최소 간격(초) ≈ 8req/s
CACHE = os.environ.get("FUND_CACHE", "output/fundamentals_cache.json")

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
    "rnd":     [("us-gaap", "ResearchAndDevelopmentExpense")],
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


def fetch_concept(cik: str, candidates) -> list:
    """후보 (taxonomy, tag) 들을 순서대로 시도, 첫 성공 units 반환."""
    if candidates and isinstance(candidates[0], str):     # (tax, tag) 단일도 허용
        candidates = [candidates]
    for taxonomy, tag in candidates:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
        try:
            r = requests.get(url, headers=HDRS, timeout=30)
            if r.status_code != 200:
                continue
            units = r.json().get("units", {})
            for key in ("USD/shares", "USD", *units.keys()):
                if key in units and units[key]:
                    return units[key]
        except Exception:
            continue
    return []


def build(tickers: list, cache_path: str = CACHE, refresh: bool = False) -> dict:
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
    print(f"[EDGAR] 대상 {len(tickers)}종목 · 갱신필요 {len(todo)} · 수집 항목 {list(CONCEPTS)}", file=sys.stderr)
    n_fallback = 0
    for i, t in enumerate(todo, 1):
        cik = cikmap.get(t.upper())
        if not cik:
            cik = _browse_edgar_cik(t.upper())   # company_tickers.json 누락 폴백(2026-07-17)
            if cik:
                n_fallback += 1
            time.sleep(RATE)
        rec = {} if refresh else dict(cache.get(t.upper()) or {})
        if not cik:
            cache[t.upper()] = rec; continue
        for name, cands in CONCEPTS.items():
            if name in rec and not refresh:
                continue                              # 이미 있는 항목은 건너뜀
            rec[name] = annual_points(fetch_concept(cik, cands))
            time.sleep(RATE)
        cache[t.upper()] = rec
        if i % 25 == 0:
            print(f"  ...{i}/{len(todo)} 수집, 중간 저장", file=sys.stderr)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
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
