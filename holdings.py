#!/usr/bin/env python3
"""
holdings.py — '추천 이력 자동 추적' + 느슨한(장기보유) 매도 시그널.

동작: 매일 '신규 매수'로 뽑힌 종목을 가상 보유목록에 자동 편입하고,
      보유 종목이 아래 '느슨한' 조건에 걸리면 매도 검토로 알린다(그 후 보유목록에서 제거).
  · 200일선 이탈: 종가 < 200일선 × (1 - MA_BUFFER)   (기본 3% 여유)
  · 트레일링 스톱: 종가 ≤ 보유 중 고점 × (1 - TRAIL)  (기본 25%)
장기보유 지향이라 평소엔 매도 신호가 거의 없고, 추세가 확실히 꺾일 때만 뜬다.

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
    매도 규칙(2026-07 재검증 반영): ①200일선 -3% 이탈(폭락 방어 백업) ②보유 ≈6개월 경과 후
    현재 후보풀(pool_syms) 밖이면 정기 재평가 매도 ③트레일링은 SELL_TRAIL>0일 때만."""
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
        if not _isnan(ma200) and price < ma200 * (1 - MA_BUFFER):
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
    for sym in buy_now_syms:
        if sym not in holdings:
            p = (ind_map.get(sym) or {}).get("price")
            holdings[sym] = {"since": today, "entry_price": p, "peak": p}
    state["last_run"] = today
    return sells


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


def benchmark_compare(summary: list, bench_dates: list, bench_closes: list) -> dict:
    """live_summary() 결과의 각 종목 진입일을 기준으로 '동일 기간' 지수 수익률과 비교.
    bench_dates/bench_closes는 오름차순 정렬된 종가 시계열(날짜는 ISO 문자열).
    반환: {} (비교 불가) 또는 {"avg_strategy","avg_bench","rows":[{symbol,strategy_ret,bench_ret}]}"""
    if not summary or not bench_dates or not bench_closes or len(bench_dates) != len(bench_closes):
        return {}
    bench_now = bench_closes[-1]

    def _bench_at(date_str):
        idx = None
        for i, d in enumerate(bench_dates):
            if d <= date_str:
                idx = i
            else:
                break
        return bench_closes[idx] if idx is not None else None

    rows = []
    for r in summary:
        since = r.get("since")
        b0 = _bench_at(since) if since else None
        if b0:
            rows.append({"symbol": r["symbol"], "strategy_ret": r["ret_pct"],
                        "bench_ret": (bench_now / b0 - 1) * 100})
    if not rows:
        return {}
    return {"avg_strategy": sum(x["strategy_ret"] for x in rows) / len(rows),
            "avg_bench": sum(x["bench_ret"] for x in rows) / len(rows),
            "rows": rows}
