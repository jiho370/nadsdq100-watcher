#!/usr/bin/env python3
"""
us_takeprofit_exit_check.py — 지호 님 질문(2026-09-23): "급등 후(10%,15%,20%,25%,30%등)
매도하는 게 좋았는지도 돌려볼래?" 현재 라이브 매도 규칙은 6개월 정기재평가뿐(가격 개입
없음). us_ma100_exit_check.py(100일선 이탈 매도 — 손절 방향)와 정확히 대칭인 익절
방향 검증: 목표 수익률 도달 시 파는 게 그냥 6개월 끝까지 들고 가는 것보다 나은지.

방법: floor+100일선(진입 시점) 통과한 topn10(현행 라이브와 동일)을, 매 종목마다 진입일부터
126거래일(6개월) 동안 일별로 따라가며:
  a) 기준: 무조건 126거래일 보유(현행 라이브와 동일)
  b) 목표수익률 T(10/15/20/25/30%) 도달 즉시 매도: 보유 중 처음으로 종가/진입가-1 >= T인
     날 매도, 그 이후 잔여기간은 현금(0%) 취급 — 같은 126일 창 안에서 공정 비교.
T별로 독립적인 페어드 비교 + 보유 중 그 문턱을 실제로 찍어본 종목 비율도 같이 보고.

실행: python -m research.us.us_takeprofit_exit_check [--years 10]
결과: output/us_takeprofit_exit_check.json
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
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features

TOPN = 10
FLOOR = 3.25
HOLD_DAYS = 126
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
OUT_PATH = "output/us_takeprofit_exit_check.json"


def _log(m): print(f"[US익절매도]  {m}", file=sys.stderr)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    hold_ex = []
    exit_ex = {t: [] for t in THRESHOLDS}
    hit_flags = {t: [] for t in THRESHOLDS}

    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score >= FLOOR]
        if len(score) < TOPN:
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        above = mom["ma100_gap"] > 0 if "ma100_gap" in mom.columns else pd.Series(True, index=score.index)
        score = score[above.reindex(score.index).fillna(False)]
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]

        p = panel.index.get_indexer([pd.Timestamp(snap["date"])], method="pad")[0]
        e = p + 1
        if e + HOLD_DAYS >= len(panel):
            continue

        hold_rets = []
        t_rets = {t: [] for t in THRESHOLDS}
        t_hits = {t: [] for t in THRESHOLDS}
        for sym in top:
            if sym not in panel.columns:
                continue
            path = panel.iloc[e:e + HOLD_DAYS + 1][sym]
            if path.isna().any() or len(path) < HOLD_DAYS + 1:
                continue
            entry = float(path.iloc[0])
            ret_path = path / entry - 1.0
            final_ret = float(ret_path.iloc[-1])
            hold_rets.append(final_ret)
            for t in THRESHOLDS:
                reached = ret_path[ret_path >= t]
                if len(reached):
                    t_rets[t].append(float(reached.iloc[0]))
                    t_hits[t].append(1)
                else:
                    t_rets[t].append(final_ret)
                    t_hits[t].append(0)
        if not hold_rets:
            continue
        hold_ex.append(float(np.mean(hold_rets)) - bench)
        for t in THRESHOLDS:
            exit_ex[t].append(float(np.mean(t_rets[t])) - bench)
            hit_flags[t].append(float(np.mean(t_hits[t])))

    h = np.array(hold_ex)
    rows = []
    for t in THRESHOLDS:
        x = np.array(exit_ex[t])
        diff = x - h
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
        tt = float(diff.mean() / se) if se else None
        row = {"threshold_pct": round(100 * t, 0), "n_events": len(x),
              "exit_mean_excess_pct": round(100 * float(x.mean()), 3),
              "exit_win_rate_pct": round(100 * float((x > 0).mean()), 1),
              "paired_diff_vs_always_hold_pct": round(100 * float(diff.mean()), 3),
              "paired_t_stat": round(tt, 3) if tt is not None else None,
              "pct_positions_that_hit_threshold_during_hold": round(100 * float(np.mean(hit_flags[t])), 1)}
        rows.append(row)
        _log(f"T={row['threshold_pct']:.0f}%: 익절시 {row['exit_mean_excess_pct']:+.2f}%p"
            f"(승률{row['exit_win_rate_pct']}%) vs 무조건보유 diff={row['paired_diff_vs_always_hold_pct']:+.2f}%p "
            f"t={row['paired_t_stat']} 도달비율={row['pct_positions_that_hit_threshold_during_hold']}%")

    payload = {"n_events": len(h), "hold_days": HOLD_DAYS, "topn": TOPN, "floor": FLOOR,
              "always_hold_mean_excess_pct": round(100 * float(h.mean()), 3) if len(h) else None,
              "always_hold_win_rate_pct": round(100 * float((h > 0).mean()), 1) if len(h) else None,
              "thresholds": rows,
              "method": ("floor=3.25+100일선필터 통과 topn10을 126거래일 보유 vs 목표수익률 "
                        "T(10~30%) 도달 즉시 매도 후 잔여기간 현금(0%) 취급 — 같은 126일 창 "
                        "안에서 T별 독립 페어드 비교. us_ma100_exit_check.py(손절 방향)와 대칭.")}
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
