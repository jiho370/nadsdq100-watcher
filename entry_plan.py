#!/usr/bin/env python3
"""
entry_plan.py — 매수/매도 '실행 계획'을 규칙으로 확정하는 모듈 (AI 개입 없음 = 비용 0).

왜 하드코딩인가:
  분할매수 비율·가격대·손절선·처분 계획은 지표(20/50/200일선, RSI, 52주고점, 보유 고점)의
  '일관된 패턴'이다. 매일 같은 규칙이므로 AI에게 시킬 이유가 없다(비용·환각 모두 제거).
  AI는 이 계획을 '바꿀 수 없고', 뉴스 기반 한 줄 코멘트만 덧붙인다(ai_report.py).

규칙 요약 (STRATEGY.md 매도 규칙과 정합):
  · 매수(과열 아님): 2분할 — 1차 50% 현재가 / 2차 50% 20일선 부근(현재가보다 위면 -3% 지점).
  · 매수(과열 hot): 3분할 — 30% 현재가 / 30% 20일선 부근 / 40% 50일선 부근.
                    과열 판정은 export_data.split_by_entry(RSI≥72 또는 50일선 +15% 이상)가 확정.
  · 하한선: 분할 가격이 200일선 아래로 내려가면 200일선까지로 올림(추세 이탈 구간 매수 금지).
  · 손절선(2026-07-13 정정): 기본은 200일선 × (1-SELL_MA_BUFFER)만 표시 — 실제 매도 규칙이
            "6개월 정기 재평가 + 200일선 백업"으로 바뀐 뒤에도(holdings.py) 여기 SELL_TRAIL
            기본값이 옛 -20%로 남아 있어 손절선이 실제 규칙과 안 맞는 문제가 있었다. 트레일링은
            기본 비활성(SELL_TRAIL=0, holdings.py와 동일 env) — 켜져 있으면 참고선으로 병기.
  · 관찰 → 매수 전환: 20일선 위(-2% 초과 이탈 상태면 회복), RSI<70, 1주 수익률 > -2% 회복.
                      전환 시 계획은 '과열 아님' 2분할과 동일.
  · 매도(2026-07-13 단순화): 확정된 매도 시그널은 **전량 즉시 정리**. backtest_exec.py
    --disposal-sweep 검증 결과 분할+반등대기(기존 방식)와 즉시전량이 net 차이 0.36%p
    이내로 통계적으로 구분 안 됨(PBO 47.7%, 매수 비율 스윕의 6.1%와 달리 신뢰 게이트
    미통과) — 구분이 안 되면 더 단순한 쪽을 쓴다는 판단(지호 님 확인)으로 분할 폐기.

가격 단위: 미국 달러(krw=False)는 소수 2자리, 한국 원화(krw=True)는 정수.
반환 구조는 ai_report.py 렌더러(_plan_table)가 그대로 표로 그린다.
"""
from __future__ import annotations
import os
import numpy as np

try:
    from backtest_exec import support_level_asof  # 트랙 C 산출물 재사용(청산 "규칙"이 아니라 참고 표시용)
except Exception:
    support_level_asof = None

TRAIL = float(os.environ.get("SELL_TRAIL", "0"))              # 기본 비활성(holdings.py와 동일 env)
MA_BUFFER = float(os.environ.get("SELL_MA_BUFFER", "0.03"))  # 200일선 -3% 이탈


def _fmt(p, krw=False):
    """가격 표기: 원화는 '73,000원', 달러는 '$123.45'."""
    if p is None:
        return "-"
    return f"{p:,.0f}원" if krw else f"${p:,.2f}"


def _valid(x):
    return x is not None and isinstance(x, (int, float)) and x == x and x > 0


def buy_plan(c: dict, krw: bool = False) -> dict:
    """매수 후보 c(export_data.build_candidates / kr_stocks 형식)의 분할매수 계획.
    반환: {"tranches":[{"label","price","pct","basis"}...], "stop":{"price","basis"}, "note"}"""
    price, ma20, ma50, ma200 = (c.get("price"), c.get("ma20"), c.get("ma50"), c.get("ma200"))
    if not _valid(price):
        return {}
    hot = bool(c.get("hot"))

    def below(base, fallback_ratio, basis):
        """현재가 아래의 매수 지점: 기준선(base)이 유효하고 현재가 아래면 그 값,
        아니면 현재가 대비 고정 비율 지점. 200일선 아래로는 내리지 않는다."""
        if _valid(base) and base < price:
            p, b = base, basis
        else:
            p, b = price * fallback_ratio, f"현재가 {int((1-fallback_ratio)*100)}% 조정 시"
        if _valid(ma200) and p < ma200:
            p, b = ma200, "200일선(추세 하한)"
        return round(p, 0 if krw else 2), b

    p2, b2 = below(ma20, 0.97, "20일선 부근")
    if hot:
        p3, b3 = below(ma50, 0.92, "50일선 부근")
        tranches = [
            {"label": "1차", "price": round(price, 0 if krw else 2), "pct": 30, "basis": "현재가(소량 시작)"},
            {"label": "2차", "price": p2, "pct": 30, "basis": b2},
            {"label": "3차", "price": p3, "pct": 40, "basis": b3},
        ]
        note = "과열 구간 — 반드시 나눠서, 조정을 기다리며 채운다"
    else:
        tranches = [
            {"label": "1차", "price": round(price, 0 if krw else 2), "pct": 50, "basis": "현재가"},
            {"label": "2차", "price": p2, "pct": 50, "basis": b2},
        ]
        note = "2분할 — 1차 후 조정 오면 2차, 안 오면 1차분만 보유"
    # 손절선(진입 직후 참고): 기본은 200일선 백업만. TRAIL>0(SELL_TRAIL env)일 때만
    # 트레일링 참고선도 후보에 넣는다 — 둘 중 높은(더 가까운) 쪽을 표시.
    candidates = []
    if TRAIL > 0:
        candidates.append((price * (1 - TRAIL), f"1차 매수가 -{int(TRAIL*100)}%"))
    if _valid(ma200):
        candidates.append((ma200 * (1 - MA_BUFFER), f"200일선 -{int(MA_BUFFER*100)}%"))
    if candidates:
        stop_val, stop_basis = max(candidates, key=lambda x: x[0])
        stop = {"price": round(stop_val, 0 if krw else 2), "basis": stop_basis}
    else:
        stop = {"price": None, "basis": "6개월 후 정기 재평가 시 판단"}
    return {"tranches": tranches, "stop": stop, "note": note, "krw": krw}


def watch_trigger(c: dict, krw: bool = False) -> str:
    """관찰 종목의 '매수 전환 조건'을 구체 가격으로 서술(코드 확정 — AI가 못 바꿈)."""
    price, ma20, ma50 = c.get("price"), c.get("ma20"), c.get("ma50")
    rsi = c.get("rsi")
    conds = []
    if _valid(ma20):
        if _valid(price) and price < ma20:
            conds.append(f"20일선({_fmt(ma20, krw)}) 회복 마감")
        else:
            conds.append(f"20일선({_fmt(ma20, krw)}) 위 유지 + 1주 수익률 플러스 전환")
    if rsi is not None and rsi <= 40:
        conds.append(f"또는 RSI {rsi:.0f} 과매도권 — 소량(계획의 30%) 선취매 가능")
    if _valid(ma50) and _valid(price) and price < ma50:
        conds.append(f"50일선({_fmt(ma50, krw)}) 회복 시 나머지 추가")
    return " · ".join(conds) if conds else "20일선 회복 확인 후 2분할 매수"


def _support_note(s: dict, krw: bool = False) -> str:
    """참고용 최근 지지선(backtest_exec.support_level_asof 재사용) — 청산 '기준'이 아니라
    현재가 판단에 참고할 정보만 병기(STRATEGY.md 매도 기준은 6개월 재평가/200일선 그대로)."""
    closes = s.get("closes")
    if not closes or not support_level_asof or len(closes) < 11:
        return ""
    vals = np.asarray(closes, dtype=float)
    lvl = support_level_asof(vals, len(vals) - 1)
    if not lvl:
        return ""
    return f" · 참고 지지선 {_fmt(lvl, krw)} 부근(매도 기준 아님, 참고용)"


def sell_plan(s: dict, krw: bool = False) -> str:
    """매도 후보(holdings.update 반환 형식)의 처분 계획 — 전량 즉시 정리(2026-07-13 단순화,
    분할+반등대기와 통계적으로 구분 안 됨 확인, backtest_exec.py --disposal-sweep 참고)."""
    ret = s.get("ret_pct")
    price = s.get("price")
    note = _support_note(s, krw)
    if ret is not None and ret > 0:
        tail = f" · 매수 후 수익 {ret:+.0f}% 확보 차원"
    elif ret is not None:
        tail = f" · 매수가 대비 {ret:+.0f}%"
    else:
        tail = ""
    return f"현재가({_fmt(price, krw)}) 부근에서 전량 즉시 정리{tail}{note}"


def plan_text(plan: dict) -> str:
    """plan(buy_plan 반환)을 이메일 텍스트 한 줄로 (HTML 표 못 쓰는 곳·AI 컨텍스트용)."""
    if not plan:
        return ""
    krw = plan.get("krw", False)
    parts = [f'{t["label"]} {_fmt(t["price"], krw)} {t["pct"]}%({t["basis"]})'
             for t in plan.get("tranches", [])]
    stop = plan.get("stop") or {}
    if stop:
        parts.append(f'손절 {_fmt(stop.get("price"), krw)}({stop.get("basis")})')
    return " → ".join(parts)


if __name__ == "__main__":   # 스모크 테스트: python entry_plan.py
    hot_c = {"price": 552.05, "ma20": 518.0, "ma50": 483.0, "ma200": 280.0, "hot": True, "rsi": 57}
    cold_c = {"price": 678.24, "ma20": 685.0, "ma50": 676.0, "ma200": 554.0, "hot": False, "rsi": 49}
    watch_c = {"price": 984.75, "ma20": 1042.0, "ma50": 861.0, "rsi": 49}
    kr_c = {"price": 73000, "ma20": 71500.0, "ma50": 69000.0, "ma200": 61000.0, "hot": False}
    sell_win = {"price": 430.0, "ret_pct": 38.2}
    sell_lose = {"price": 90.0, "ret_pct": -18.0}
    import numpy as _np
    _rng = _np.random.default_rng(0)
    sell_with_support = {"price": 430.0, "ret_pct": 12.0,
                          "closes": list(100 + _np.cumsum(_rng.normal(0, 1.5, 260)))}
    print("HOT :", plan_text(buy_plan(hot_c)))
    print("COLD:", plan_text(buy_plan(cold_c)))
    print("WATCH:", watch_trigger(watch_c))
    print("KR  :", plan_text(buy_plan(kr_c, krw=True)))
    print("SELL+:", sell_plan(sell_win))
    print("SELL-:", sell_plan(sell_lose))
    print("SELL(지지선):", sell_plan(sell_with_support))
