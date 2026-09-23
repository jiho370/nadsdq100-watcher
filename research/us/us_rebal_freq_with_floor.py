#!/usr/bin/env python3
"""
us_rebal_freq_with_floor.py — 지호 님 질문(2026-09-23): "바뀐 방식(floor=3.25 반영)으로도
6개월 정기재평가가 최적인지 확인". us_factor_formula_pit_sweep.build_snaps()는 REBAL_DAYS·
TD_DAYS가 모듈 상수라 재사용이 안 돼(테스트된 기존 코드는 안 건드림), 이 파일에 파라미터화된
버전을 새로 둔다.

방법: 재평가 주기 N ∈ {1개월(21일)·3개월(63일)·6개월(126일)·12개월(252일)}마다 그 주기로
스냅샷을 다시 만들고(관찰 간격=보유기간=N — 재평가 주기만큼 들고 있다가 다시 고른다는 라이브
방식 그대로), floor=3.25 적용한 topn8을 뽑아 이벤트 평균 총초과수익·승률을 구한다. 주기가
짧을수록 연간 재평가 횟수가 늘어 회전비용이 커지므로, backtest_costs.CostModel(us)의 왕복
비용을 "연 재평가횟수 × 왕복비용"으로 근사해 총초과수익에서 빼 비용조정 연환산 초과수익을
같이 낸다(선형 근사 — 절대수익 주장이 아니라 주기 간 상대비교용).

실행: python -m research.us.us_rebal_freq_with_floor [--years 10]
결과: output/us_rebal_freq_with_floor.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
from research.us.us_factor_value_vs_rank import _composite

LOOKBACK = 252
TOPN = 8
FLOOR = 3.25
FREQ_GRID = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
OUT_PATH = "output/us_rebal_freq_with_floor.json"


def _log(m): print(f"[US재평가주기]  {m}", file=sys.stderr)


def _build_snaps(panel, spy, funds, pit, rebal_days, td_days):
    """us_factor_formula_pit_sweep.build_snaps()와 동일 로직, rebal_days·td_days만
    파라미터화(재사용 원본은 손 안 댐 — 다른 스크립트들이 그 상수 조합에 의존)."""
    import fundamentals_edgar as F
    n = len(panel)
    ps = list(range(LOOKBACK, n - td_days - 1, rebal_days))
    snaps = []
    for p in ps:
        date_iso = panel.index[p].date().isoformat()
        members = BC.membership_asof(pit, date_iso)
        price = panel.iloc[p]
        valid = [s for s in price.dropna().index
                if s in members and not np.isnan(panel.iloc[p - LOOKBACK][s])]
        if not valid:
            continue
        v = pd.Index(valid)
        rows = {}
        for s in v:
            fd = funds.get(s) or {}
            fv = F.factor_values(fd, date_iso, float(price[s]))
            rows[s] = {f: fv.get(f) for f in ["int_gp_assets", "rd_mktcap", "shareholder_yield"]}
        raw = pd.DataFrame(rows).T.astype(float)
        mom = (panel.iloc[p] / panel.iloc[p - 126] - 1).reindex(v) if p >= 126 else pd.Series(dtype=float)
        raw = raw[mom.notna()] if len(mom) else raw
        if raw.empty:
            continue
        e = p + 1
        if e + td_days >= n:
            continue
        fwd = panel.iloc[e + td_days][raw.index] / panel.iloc[e][raw.index] - 1
        bench = float(spy.iloc[e + td_days] / spy.iloc[e] - 1)
        snaps.append({"date": date_iso, "raw": raw, "fwd": fwd, "bench": bench})
    return snaps


def _one_freq(snaps, td_days) -> dict:
    ex = []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score >= FLOOR]
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        r = fwd.reindex(top).dropna()
        if len(r) == 0:
            continue
        ex.append(float(r.mean()) - bench)
    if len(ex) < 5:
        return {"note": "표본 부족", "n_events": len(ex)}
    a = np.array(ex)
    cost = BC.CostModel(market="us")
    round_trip_bp = (cost.buy + cost.sell) * 1e4
    rebal_per_year = 252.0 / td_days
    annualized_gross_pct = 100 * float(a.mean()) * rebal_per_year
    annualized_cost_drag_pct = round_trip_bp / 1e2 * rebal_per_year
    return {"n_events": len(a), "mean_excess_pct": round(100 * float(a.mean()), 3),
           "win_rate_pct": round(100 * float((a > 0).mean()), 1),
           "excess_sharpe": round(float(a.mean() / a.std()) * np.sqrt(rebal_per_year), 3) if a.std() else None,
           "rebal_per_year_approx": round(rebal_per_year, 2),
           "round_trip_cost_bp": round(round_trip_bp, 2),
           "annualized_gross_excess_pct_approx": round(annualized_gross_pct, 2),
           "annualized_cost_drag_pct_approx": round(annualized_cost_drag_pct, 2),
           "annualized_net_excess_pct_approx": round(annualized_gross_pct - annualized_cost_drag_pct, 2)}


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()

    rows = {}
    for label, days in FREQ_GRID.items():
        snaps = _build_snaps(panel, spy, funds, pit, days, days)
        r = _one_freq(snaps, days)
        rows[label] = r
        _log(f"{label}(재평가{days}일): n={r.get('n_events')} 초과 {r.get('mean_excess_pct')}%p "
            f"승률 {r.get('win_rate_pct')}% → 연환산 순초과(근사) {r.get('annualized_net_excess_pct_approx')}%p")

    best = max((k for k in rows if rows[k].get("annualized_net_excess_pct_approx") is not None),
              key=lambda k: rows[k]["annualized_net_excess_pct_approx"], default=None)
    payload = {"floor": FLOOR, "topn": TOPN, "by_freq": rows, "best_by_annualized_net_excess": best,
              "method": ("재평가 주기=보유기간으로 맞춰(라이브 방식 그대로) 스냅샷을 다시 만들고 "
                        "floor=3.25 topn8 이벤트 평균 초과수익을 구한 뒤, 주기가 짧을수록 늘어나는 "
                        "왕복비용(연 재평가횟수×비용)을 선형근사로 차감한 연환산 순초과수익으로 비교. "
                        "절대수익 주장이 아니라 주기 간 상대순위용 근사치.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    run(years=args.years)


if __name__ == "__main__":
    main()
