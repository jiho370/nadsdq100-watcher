#!/usr/bin/env python3
"""
kr_smoothed_floor.py — us_smoothed_floor.py의 한국판(2026-09-22, 지호 님 "침체장 대비
안전판으로 평활화된 하한을 구간별 평균수익률로 계산해달라").

실행: python -m research.kr.kr_smoothed_floor
결과: output/kr_smoothed_floor.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

BIN_WIDTH = 0.5
SMOOTH_WINDOW = 5
TOPN = 5
HORIZON = "6m"
OUT_PATH = "output/kr_smoothed_floor.json"


def _log(m): print(f"[KR평활하한] {m}", file=sys.stderr)


def build_bucket_curve(snaps):
    rows = []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        df = pd.DataFrame({"score": score, "fwd": fwd}).dropna()
        df["excess"] = df["fwd"] - bnc
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    lo = np.floor(pooled["score"].min() / BIN_WIDTH) * BIN_WIDTH
    hi = np.ceil(pooled["score"].max() / BIN_WIDTH) * BIN_WIDTH
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    pooled["bin"] = pd.cut(pooled["score"], edges, include_lowest=True)
    bucket = pooled.groupby("bin", observed=True).agg(
        n=("excess", "size"), mean_excess=("excess", "mean")).reset_index()
    bucket["bin_mid"] = bucket["bin"].apply(lambda iv: round(float(iv.mid), 3))
    bucket = bucket[bucket["n"] >= 15].sort_values("bin_mid").reset_index(drop=True)
    bucket["smoothed_excess"] = bucket["mean_excess"].rolling(SMOOTH_WINDOW, center=True, min_periods=3).mean()
    return bucket.drop(columns=["bin"])


def robust_floor_from_curve(bucket: pd.DataFrame) -> float | None:
    s = bucket["smoothed_excess"].to_numpy()
    mids = bucket["bin_mid"].to_numpy()
    for i in range(len(s)):
        if np.isnan(s[i]):
            continue
        tail = s[i:]
        tail = tail[~np.isnan(tail)]
        if len(tail) >= 3 and np.all(tail > 0):
            return float(mids[i])
    return None


def apply_floor_to_topn(snaps, floor, topn=TOPN):
    ex_list, fill_list = [], []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        pool = score[score >= floor] if floor is not None else score
        n_avail = len(pool)
        if n_avail == 0:
            ex_list.append(-bnc); fill_list.append(0); continue
        top = pool.sort_values(ascending=False).index[:topn]
        r = fwd.reindex(top).dropna()
        port_ret = float(r.sum()) / topn
        ex_list.append(port_ret - bnc)
        fill_list.append(min(n_avail, topn))
    return ex_list, fill_list


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개")

    bucket = build_bucket_curve(snaps)
    for _, row in bucket.iterrows():
        smoothed = row["smoothed_excess"]
        _log(f"구간중앙 {row['bin_mid']:+.2f}: n={int(row['n'])} 평균초과 {row['mean_excess']*100:+.2f}%p "
            f"평활초과 {'' if pd.isna(smoothed) else f'{smoothed*100:+.2f}%p'}")

    floor = robust_floor_from_curve(bucket)
    _log(f"강건한 하한 후보: {floor}")

    ex_base, fill_base = apply_floor_to_topn(snaps, None)
    ex_floor, fill_floor = (apply_floor_to_topn(snaps, floor) if floor is not None else (None, None))

    result_base = {"floor": None, "mean_excess_pct": round(100 * float(np.mean(ex_base)), 3),
                   "win_rate_pct": round(100 * float(np.mean([x > 0 for x in ex_base])), 1),
                   "n_binding": 0}
    result_floor = None
    if floor is not None:
        n_binding = sum(1 for f in fill_floor if f < TOPN)
        result_floor = {"floor": floor, "mean_excess_pct": round(100 * float(np.mean(ex_floor)), 3),
                        "win_rate_pct": round(100 * float(np.mean([x > 0 for x in ex_floor])), 1),
                        "n_binding": n_binding, "n_snaps": len(snaps),
                        "pct_binding": round(100 * n_binding / len(snaps), 1)}
        _log(f"하한 {floor} 적용: 초과 {result_floor['mean_excess_pct']:+.2f}%p "
            f"(발동 {n_binding}/{len(snaps)}회, {result_floor['pct_binding']}%)")

    payload = {"n_snaps": len(snaps), "bin_width": BIN_WIDTH, "smooth_window": SMOOTH_WINDOW,
              "bucket_curve": bucket.to_dict("records"),
              "robust_floor_candidate": floor,
              "baseline_no_floor": result_base, "with_robust_floor": result_floor}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
