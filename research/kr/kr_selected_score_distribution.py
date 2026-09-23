#!/usr/bin/env python3
"""
kr_selected_score_distribution.py — 지호 님 질문(2026-09-22): "보통 뽑히는 종목들
점수가 어떻게 되나?" 실제 topN=5 라이브 선정에서, 순위별(1~5등)로 뽑힌 종목의 종합점수
분포를 스냅샷 전체에서 집계 — 캡=6.5가 실전에서 얼마나 자주 작동하는지 감을 잡기 위함.

실행: python -m research.kr.kr_selected_score_distribution
결과: output/kr_selected_score_distribution.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

TOPN = 5
HORIZON = "6m"
CAP = 6.5
OUT_PATH = "output/kr_selected_score_distribution.json"


def _log(m): print(f"[KR선정점수분포] {m}", file=sys.stderr)


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개")

    by_rank = {r: [] for r in range(1, TOPN + 6)}   # 1~10등까지(참고용, 캡 이후 대체선 확인)
    n_snaps_where_cap_binds = 0
    n_excluded_from_top5 = []
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna().sort_values(ascending=False)
        for r in range(1, min(TOPN + 6, len(score) + 1)):
            by_rank[r].append(float(score.iloc[r - 1]))
        top5_scores = score.iloc[:TOPN]
        n_over = int((top5_scores > CAP).sum())
        n_excluded_from_top5.append(n_over)
        if n_over > 0:
            n_snaps_where_cap_binds += 1

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
                          "pct_over_cap": round(100 * float(np.mean([v > CAP for v in vals])), 1)})
        _log(f"{r}등: 평균 {rank_stats[-1]['mean_score']} (중앙 {rank_stats[-1]['median_score']}, "
            f"범위 {rank_stats[-1]['min_score']}~{rank_stats[-1]['max_score']}) · "
            f"캡({CAP}) 초과 비율 {rank_stats[-1]['pct_over_cap']}%")

    _log(f"현행 top5 중 캡{CAP} 초과가 하나라도 있던 스냅샷: {n_snaps_where_cap_binds}/{len(snaps)} "
        f"({100*n_snaps_where_cap_binds/len(snaps):.1f}%)")
    _log(f"스냅샷당 top5 중 캡 초과 종목 수 평균: {np.mean(n_excluded_from_top5):.2f}개")

    payload = {"n_snaps": len(snaps), "topn": TOPN, "cap_reference": CAP,
              "rank_stats": rank_stats,
              "cap_binds_pct_of_snaps": round(100 * n_snaps_where_cap_binds / len(snaps), 1),
              "mean_n_excluded_per_snap_from_top5": round(float(np.mean(n_excluded_from_top5)), 2),
              "note": ("6~10등도 참고로 같이 보여줌 — 캡으로 top5에서 빠진 자리를 채울 "
                      "'다음 대기 종목'들의 점수가 어느 수준인지 확인용.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
