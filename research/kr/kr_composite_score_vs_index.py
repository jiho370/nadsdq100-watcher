#!/usr/bin/env python3
"""
kr_composite_score_vs_index.py — us_composite_score_vs_index.py의 한국판(2026-09-22).
라이브 valuediv 종합점수(동일가중)를 실제 값(구간)으로 나눠, 코스피200(B1) 대비 평균
초과수익을 집계한다. backtest_kr.build_kr_snaps()·kr_factor_value_vs_rank._composite()
그대로 재사용.

실행: python -m research.kr.kr_composite_score_vs_index
결과: output/kr_composite_score_vs_index.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

OUT_PATH = "output/kr_composite_score_vs_index.json"
HORIZON = "6m"


def _log(m): print(f"[KR점수vs지수] {m}", file=sys.stderr)


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개")

    rows = []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw)
        df = pd.DataFrame({"score": score, "fwd": fwd}).dropna()
        df["excess"] = df["fwd"] - bnc
        df["date"] = snap["date"]
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    _log(f"풀링 관측치 {len(pooled)}개 · 점수범위 {pooled['score'].min():.2f}~{pooled['score'].max():.2f}")

    bin_width = 1.0
    lo, hi = np.floor(pooled["score"].min()), np.ceil(pooled["score"].max())
    edges = np.arange(lo, hi + bin_width, bin_width)
    pooled["bin"] = pd.cut(pooled["score"], edges, include_lowest=True)
    bin_rows = []
    for interval, g in pooled.groupby("bin", observed=True):
        if len(g) < 15:
            continue
        bin_rows.append({"bin_mid": round(float(interval.mid), 2), "bin_lo": round(float(interval.left), 2),
                         "bin_hi": round(float(interval.right), 2), "n": len(g),
                         "mean_excess_pct": round(100 * float(g["excess"].mean()), 3),
                         "mean_raw_ret_pct": round(100 * float(g["fwd"].mean()), 3)})
    bin_rows.sort(key=lambda r: r["bin_mid"])

    thresholds = np.round(np.arange(lo, hi, 0.5), 2)
    thr_rows = []
    for t in thresholds:
        sub = pooled[pooled["score"] >= t]
        if len(sub) < 15:
            continue
        thr_rows.append({"threshold": float(t), "n": len(sub),
                         "pct_of_universe": round(100 * len(sub) / len(pooled), 1),
                         "mean_excess_pct": round(100 * float(sub["excess"].mean()), 3),
                         "mean_raw_ret_pct": round(100 * float(sub["fwd"].mean()), 3)})

    payload = {"n_snaps": len(snaps), "n_pooled_obs": len(pooled),
              "score_range": [round(float(pooled["score"].min()), 2), round(float(pooled["score"].max()), 2)],
              "bin_width": bin_width,
              "method": ("종합점수(라이브 valuediv 동일가중)를 6개월 forward return - "
                        "코스피200(B1) forward return(초과수익)에 대해 고정폭1.0구간·"
                        "누적임계값(score>=T)별로 집계. 전체 풀링(참고용, 엄밀한 유의성 "
                        "검정 아님)."),
              "by_bin": bin_rows, "by_threshold": thr_rows}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    run()


if __name__ == "__main__":
    main()
