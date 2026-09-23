#!/usr/bin/env python3
"""
us_selected_score_distribution.py — kr_selected_score_distribution.py의 미국판
(2026-09-22, 지호 님 "미국 한국 둘다?"). 실제 topN=8 라이브 선정에서, 순위별(1~8등)로
뽑힌 종목의 종합점수(1:2:2) 분포를 스냅샷 전체에서 집계. 미국은 캡을 걸어도 도움이
안 됐으므로(us_factor_cap_extreme.py) 참고용 cap 기준선은 표시만 하고 배제 논리는 없음.

실행: python -m research.us.us_selected_score_distribution [--years 10]
결과: output/us_selected_score_distribution.json
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

TOPN = 8
REF_CAP = 9.0   # 참고용(미국은 이 근방부터 표본이 급감하던 지점, us_composite_score_vs_index.py)


def _log(m): print(f"[US선정점수분포] {m}", file=sys.stderr)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    by_rank = {r: [] for r in range(1, TOPN + 6)}
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"]
        score = _composite(raw).reindex(fwd.index).dropna().sort_values(ascending=False)
        for r in range(1, min(TOPN + 6, len(score) + 1)):
            by_rank[r].append(float(score.iloc[r - 1]))

    rank_stats = []
    for r in range(1, TOPN + 6):
        vals = by_rank[r]
        if not vals:
            continue
        rank_stats.append({"rank": r, "n": len(vals),
                          "mean_score": round(float(np.mean(vals)), 2),
                          "median_score": round(float(np.median(vals)), 2),
                          "min_score": round(float(np.min(vals)), 2),
                          "max_score": round(float(np.max(vals)), 2),
                          "pct_over_refcap": round(100 * float(np.mean([v > REF_CAP for v in vals])), 1)})
        _log(f"{r}등: 평균 {rank_stats[-1]['mean_score']} (중앙 {rank_stats[-1]['median_score']}, "
            f"범위 {rank_stats[-1]['min_score']}~{rank_stats[-1]['max_score']}) · "
            f"참고캡({REF_CAP}) 초과 비율 {rank_stats[-1]['pct_over_refcap']}%")

    payload = {"n_snaps": len(snaps), "topn": TOPN, "reference_cap": REF_CAP,
              "rank_stats": rank_stats,
              "note": ("미국은 캡을 걸어도 성과가 개선되지 않았음(us_factor_cap_extreme.py) "
                      "— 이 분포는 참고용(한국과 비교 목적)이며 배제 규칙 채택 근거는 아님.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_selected_score_distribution.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_selected_score_distribution.json")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    run(years=args.years)


if __name__ == "__main__":
    main()
