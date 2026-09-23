#!/usr/bin/env python3
"""
us_smoothed_floor.py — 지호 님 요청(2026-09-22): "시장이 전반적으로 침체될 때를 대비한
안전판으로 하한선을 두고 싶다. 전체를 다 사는 게 아니라 점수 구간별 평균수익률을 계산해서
안 좋은 구간을 제외하되, 중간의 노이즈성 특이구간 하나 때문에 잘못 판단하지 않도록
평활화해서 봐달라."

방법: 점수를 0.5 단위로 촘촘히 나눠 구간별 평균 초과수익률을 구하고, 인접 구간
이동평균(window=5, 즉 앞뒤 2구간씩)으로 평활화한다. 평활화된 곡선이 "이 지점 이후로는
계속 양(+)을 유지"하는 마지막 구간을 강건한 하한 후보로 제시(단일 구간의 우연한 튐에
흔들리지 않게). 그 하한을 실제 topN 선정에 적용했을 때 (a) 과거에 실제로 몇 번이나
발동했는지(=거의 항상 무의미했다는 것 자체가 "평상시엔 공짜 보험"이라는 뜻) (b) 발동 시
실제로 방어 효과가 있었는지를 함께 보고한다.

실행: python -m research.us.us_smoothed_floor [--years 10]
결과: output/us_smoothed_floor.json
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

BIN_WIDTH = 0.5
SMOOTH_WINDOW = 5   # 홀수, 앞뒤 2구간씩
TOPN = 8
OUT_PATH = "output/us_smoothed_floor.json"


def _log(m): print(f"[US평활하한] {m}", file=sys.stderr)


def build_bucket_curve(snaps):
    rows = []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        df = pd.DataFrame({"score": score, "fwd": fwd}).dropna()
        df["excess"] = df["fwd"] - bench
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    lo, hi = np.floor(pooled["score"].min() / BIN_WIDTH) * BIN_WIDTH, np.ceil(pooled["score"].max() / BIN_WIDTH) * BIN_WIDTH
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    pooled["bin"] = pd.cut(pooled["score"], edges, include_lowest=True)
    bucket = pooled.groupby("bin", observed=True).agg(
        n=("excess", "size"), mean_excess=("excess", "mean")).reset_index()
    bucket["bin_mid"] = bucket["bin"].apply(lambda iv: round(float(iv.mid), 3))
    bucket = bucket[bucket["n"] >= 15].sort_values("bin_mid").reset_index(drop=True)
    bucket["smoothed_excess"] = bucket["mean_excess"].rolling(SMOOTH_WINDOW, center=True, min_periods=3).mean()
    return bucket.drop(columns=["bin"])


def robust_floor_from_curve(bucket: pd.DataFrame) -> float | None:
    """평활곡선이 이 지점부터 끝까지 계속 양(+)인 '가장 낮은' 지점을 강건한 하한으로 제안."""
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
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        pool = score[score >= floor] if floor is not None else score
        n_avail = len(pool)
        if n_avail == 0:
            ex_list.append(-bench); fill_list.append(0); continue
        top = pool.sort_values(ascending=False).index[:topn]
        r = fwd.reindex(top).dropna()
        port_ret = float(r.sum()) / topn
        ex_list.append(port_ret - bench)
        fill_list.append(min(n_avail, topn))
    return ex_list, fill_list


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
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
              "baseline_no_floor": result_base, "with_robust_floor": result_floor,
              "note": ("평활 하한은 역사적으로 거의 발동 안 해도(=공짜 보험) 정상 — 발동률이 "
                      "낮다고 '의미없다'가 아니라, 침체장 등 예외 상황에서만 작동하는 안전판을 "
                      "찾는 게 목적이었음.")}
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
