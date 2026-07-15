#!/usr/bin/env python3
"""
holdings.py — '추천 이력 자동 추적' + 느슨한(장기보유) 매도 시그널.

동작: 매일 '신규 매수'로 뽑힌 종목을 가상 보유목록에 자동 편입하고,
      보유 종목이 아래 '느슨한' 조건에 걸리면 매도 검토로 알린다(그 후 보유목록에서 제거).
  · (2026-07-15 비활성) 200일선 이탈: 종가 < 200일선 × (1 - MA_BUFFER)
  · 트레일링 스톱: 종가 ≤ 보유 중 고점 × (1 - TRAIL)  (기본 비활성)
  · 6개월 정기 재평가: 보유 ≈6개월 경과 후 당일 팩터 후보풀 밖이면 정리(현재 유일하게
    살아있는 매도 트리거 — 21조합 검증 champion과 동일 조건: 가격 개입 없음)
장기보유 지향이라 평소엔 매도 신호가 거의 없고, 6개월 재평가 때만 순환매가 일어난다.

2026-07-15 200일선 백업 비활성 이유: backtest_exec.py 21조합 검증에서 200일선 단독
(exit_ma200only)이 champion(exit_time6m, 가격 무개입) 대비 순수익 -5.93%p로 하위권임이
확인됐고, 진입 쪽엔 애초에 추세 필터가 없어(2026-07 진입게이트 폐기) 이미 200일선 아래인
종목이 편입 직후 이 규칙에 걸려 즉시 매도되는 버그도 발견됨(지호 님 리포트). 대체 방어층
(포지션 사이징·개별종목 파국 이벤트 킬스위치 등, Fable5 자문 진행 중)이 아직 검증 전이라
당분간 champion과 동일하게 가격 개입 없음 상태 — 되살리려면 SELL_MA200_BACKUP=1.

상태파일(output/ai_holdings.json):
  {"holdings": {"NVDA": {"since":"2026-07-01","entry_price":1200.0,"peak":1250.0}, ...},
   "last_run": "2026-07-02"}
"""
from __future__ import annotations
import os, json

STATE = os.environ.get("HOLDINGS_FILE", "output/ai_holdings.json")
# 2026-07 재검증(backtest_exec.py 21조합·PBO 1.6%·DSR 0.97 통과): 트레일링 -20%가 트레이드의
# 88%를 중도 손절시키며 순수익을 절반으로 깎는 것으로 확인(+7.7% vs 고정6개월 +14.9%,
# 200일선only +9.4%) → 기본 비활성(0). 되살리려면 SELL_TRAIL=0.20.
TRAIL = float(os.environ.get("SELL_TRAIL", "0"))
MA_BUFFER = float(os.environ.get("SELL_MA_BUFFER", "0.03"))  # 200일선 -3% 아래로 확실히 이탈
# 2026-07-15: 200일선 백업 자체를 기본 비활성화(위 docstring 근거) — 되살리려면 "1"로.
MA200_BACKUP = os.environ.get("SELL_MA200_BACKUP", "0") == "1"
REEVAL_DAYS = int(os.environ.get("SELL_REEVAL_DAYS", "180"))  # ≈6개월(달력일) — 검증된 보유기간


def load(path=STATE) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"holdings": {}, "last_run": None}


def save(state: dict, path=STATE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _isnan(x):
    return x is None or (isinstance(x, float) and x != x)


def update(state: dict, buy_now_syms: list, ind_map: dict, today: str, pool_syms=None):
    """보유 갱신 + 매도 시그널 산출. 반환: sells[list].  state 는 제자리 수정.
    매도 규칙(2026-07-15 재검증 반영): ①200일선 -3% 이탈은 기본 비활성(SELL_MA200_BACKUP=1
    이어야 켜짐 — 근거는 모듈 docstring) ②보유 ≈6개월 경과 후 현재 후보풀(pool_syms) 밖이면
    정기 재평가 매도(현재 유일한 활성 트리거) ③트레일링은 SELL_TRAIL>0일 때만."""
    import datetime as _dt
    holdings = state.setdefault("holdings", {})
    sells = []
    for sym in list(holdings):
        ind = ind_map.get(sym) or {}
        price, ma200 = ind.get("price"), ind.get("ma200")
        if _isnan(price):
            continue
        h = holdings[sym]
        h["peak"] = max(h.get("peak") or price, price)
        reason = None
        held_days = None
        try:
            held_days = (_dt.date.fromisoformat(today) - _dt.date.fromisoformat(h.get("since"))).days
        except Exception:
            pass
        if MA200_BACKUP and not _isnan(ma200) and price < ma200 * (1 - MA_BUFFER):
            reason = f"200일선 이탈 (종가 {price:,.0f} < 200일선 {ma200:,.0f})"
        elif TRAIL > 0 and h.get("peak") and price <= h["peak"] * (1 - TRAIL):
            drop = (price / h["peak"] - 1) * 100
            reason = f"고점 대비 {drop:.0f}% 하락 (트레일링 -{int(TRAIL*100)}%)"
        elif (pool_syms is not None and held_days is not None and held_days >= REEVAL_DAYS
              and sym not in pool_syms):
            reason = (f"6개월 정기 재평가 — 보유 {held_days}일 경과, 현재 팩터 후보풀 밖 "
                      f"(검증된 보유기간 종료 후 순환매)")
        if reason:
            ret = ((price / h["entry_price"] - 1) * 100) if h.get("entry_price") else None
            sells.append({"symbol": sym, "reason": reason, "since": h.get("since"),
                          "entry": h.get("entry_price"), "price": price, "ret_pct": ret,
                          "peak": h.get("peak")})
            del holdings[sym]
    # 신규 매수 종목 자동 편입(이미 보유면 유지)
    add(state, buy_now_syms, ind_map, today)
    state["last_run"] = today
    return sells


# 동일 회사 복수 클래스(의결권 차이만) — 둘 다 담으면 사실상 같은 종목 2배 보유
_CLASS_ALIAS = {"GOOGL": "GOOG", "FOXA": "FOX", "NWSA": "NWS"}


def _canon(sym: str) -> str:
    return _CLASS_ALIAS.get(sym, sym)


def add(state: dict, buy_syms: list, ind_map: dict, today: str, max_n: int | None = None):
    """신규 매수 편입. 동일 회사 중복(GOOG/GOOGL 등) 배제 + 보유 상한(max_n).
    상한 초과 시 추가하지 않는다 — 기존 보유는 매도 시그널로만 빠진다(팔아야 산다)."""
    holdings = state.setdefault("holdings", {})
    held_canon = {_canon(s) for s in holdings}
    for sym in buy_syms:
        if sym in holdings or _canon(sym) in held_canon:
            continue
        if max_n is not None and len(holdings) >= max_n:
            break
        p = (ind_map.get(sym) or {}).get("price")
        holdings[sym] = {"since": today, "entry_price": p, "peak": p}
        held_canon.add(_canon(sym))


# ------------------------- 라이브 트래킹(보유현황) -------------------------
def live_summary(state: dict, ind_map: dict) -> list:
    """보유 종목별 현재 상태: 매수일·진입가·현재가·수익률·고점대비·보유일수.
    반환은 수익률 내림차순. 종가 조회가 안 되는 종목은 건너뜀."""
    import datetime as _dt
    today = _dt.date.today()
    rows = []
    for sym, h in (state.get("holdings") or {}).items():
        ind = ind_map.get(sym) or {}
        price = ind.get("price")
        entry = h.get("entry_price")
        if _isnan(price) or _isnan(entry) or not entry:
            continue
        held_days = None
        try:
            held_days = (today - _dt.date.fromisoformat(h.get("since"))).days
        except Exception:
            pass
        rows.append({"symbol": sym, "since": h.get("since"), "entry": entry, "price": price,
                     "ret_pct": (price / entry - 1) * 100, "peak": h.get("peak"),
                     "held_days": held_days})
    rows.sort(key=lambda r: r["ret_pct"], reverse=True)
    return rows


def portfolio_series(summary: list, price_map: dict, bench_dates: list, bench_closes: list) -> dict:
    """'각 보유종목을 진입일에 동일 금액씩 샀다' 가정의 포트폴리오 누적수익률(%) 시계열과,
    같은 날짜들에 같은 금액을 지수에 넣었을 때의 시계열(라이브 트래킹 그래프용).
    price_map: {sym: {"dates":[...], "closes":[...]}} — 종목별 일별 종가(오름차순).
    지수 달력(bench_dates)을 마스터로 쓰고, 종목 종가가 빠진 날은 직전가를 유지한다.
    반환: {"dates","portfolio","bench"} 또는 {} (비교 불가)."""
    import bisect
    entries = [r for r in summary if r.get("since") and r.get("entry")]
    if not entries or not bench_dates or len(bench_dates) != len(bench_closes):
        return {}
    start = min(r["since"] for r in entries)
    i0 = bisect.bisect_left(bench_dates, start)
    if i0 >= len(bench_dates):
        return {}
    dates = bench_dates[i0:]
    aligned = {}
    for r in entries:
        pm = price_map.get(r["symbol"]) or {}
        d, c = pm.get("dates") or [], pm.get("closes") or []
        if not d:
            continue
        arr, j, lastv = [], 0, None
        for day in dates:
            while j < len(d) and d[j] <= day:
                lastv = c[j]; j += 1
            arr.append(lastv)
        aligned[r["symbol"]] = arr
    port, bench = [], []
    for k, day in enumerate(dates):
        rs, bs = [], []
        for r in entries:
            if r["since"] > day:
                continue
            arr = aligned.get(r["symbol"])
            if arr and arr[k]:
                rs.append((arr[k] / r["entry"] - 1) * 100)
            bi = bisect.bisect_right(bench_dates, r["since"]) - 1
            if bi >= 0 and bench_closes[bi]:
                bs.append((bench_closes[i0 + k] / bench_closes[bi] - 1) * 100)
        port.append(sum(rs) / len(rs) if rs else None)
        bench.append(sum(bs) / len(bs) if bs else None)
    if not any(v is not None for v in port):
        return {}
    return {"dates": dates, "portfolio": port, "bench": bench}
