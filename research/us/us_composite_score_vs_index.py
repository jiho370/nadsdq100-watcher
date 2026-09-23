#!/usr/bin/env python3
"""
us_composite_score_vs_index.py — 지호 님 질문(2026-09-22): "몇 점부터 지수보다 얼마나
좋아지는지, 그래프로 보여달라." 라이브 종합점수(1:2:2)를 실제 값(구간)으로 나눠, 그 구간에
속한 종목들이 SPY(벤치마크) 대비 평균 초과수익이 얼마였는지를 집계한다.

재사용: us_factor_formula_pit_sweep.build_snaps()(PIT 스냅샷)와 us_factor_value_vs_rank의
_composite() 그대로. 여기서 새로 하는 건 "점수 구간별 초과수익 집계"뿐(재구현 아님).

두 가지 뷰:
  (1) 고정폭 구간별(bin) 평균 초과수익 — "이 점수대 종목들은 평균적으로 지수 대비 얼마나
      좋았나" (막대/구간별)
  (2) 누적 임계값별(threshold) — "점수 T 이상인 종목만 골랐으면 평균 초과수익이 얼마였나"
      (선그래프, "몇 점부터 뽑아야 하는가"에 대한 직접적인 답)

실행: python -m research.us.us_composite_score_vs_index [--years 10]
결과: output/us_composite_score_vs_index.json
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

OUT_PATH = "output/us_composite_score_vs_index.json"


def _log(m): print(f"[US점수vs지수] {m}", file=sys.stderr)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    rows = []  # 풀링된 (종목,스냅샷) 관측치: score, excess_ret
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw)
        df = pd.DataFrame({"score": score, "fwd": fwd}).dropna()
        df["excess"] = df["fwd"] - bench
        df["date"] = snap["date"]
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    _log(f"풀링 관측치 {len(pooled)}개 (종목x스냅샷) · 점수범위 {pooled['score'].min():.2f}~{pooled['score'].max():.2f}")

    # (1) 고정폭 구간
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

    # (2) 누적 임계값(score >= T)
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
              "method": ("종합점수(라이브 1:2:2, shareholder_yield만 ±5클립)를 6개월 forward "
                        "return - SPY forward return(초과수익)에 대해 (1)고정폭 1.0구간별 "
                        "평균, (2)누적임계값(score>=T)별 평균으로 집계. 모든 스냅샷·종목을 "
                        "풀링(스냅샷 반복관측 문제 있음 — 참고용, 엄밀한 유의성 검정 아님)."),
              "by_bin": bin_rows, "by_threshold": thr_rows}
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
