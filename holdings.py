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
TRAIL = float(os.environ.get("SELL_TRAIL", "0.20"))       # 고점 대비 -20% (STRATEGY.md §2 — 연구 최적 15~20%)
MA_BUFFER = float(os.environ.get("SELL_MA_BUFFER", "0.03"))  # 200일선 -3% 아래로 확실히 이탈


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


def update(state: dict, buy_now_syms: list, ind_map: dict, today: str):
    """보유 갱신 + 매도 시그널 산출. 반환: sells[list].  state 는 제자리 수정."""
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
        if not _isnan(ma200) and price < ma200 * (1 - MA_BUFFER):
            reason = f"200일선 이탈 (종가 {price:,.0f} < 200일선 {ma200:,.0f})"
        elif h.get("peak") and price <= h["peak"] * (1 - TRAIL):
            drop = (price / h["peak"] - 1) * 100
            reason = f"고점 대비 {drop:.0f}% 하락 (트레일링 -{int(TRAIL*100)}%)"
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
