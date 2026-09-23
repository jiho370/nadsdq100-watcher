#!/usr/bin/env python3
"""
us_factor_floor.py — kr_factor_floor.py의 미국판(2026-09-22, 지호 님 "하한컷 방식도
백테스트"). 미국은 상한 캡이 도움이 안 됐으므로(us_factor_cap_extreme.py) 상한 없이
하한만 스윕.

실행: python -m research.us.us_factor_floor [--years 10]
결과: output/us_factor_floor.json
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

FLOOR_GRID = [None, -2, 0, 2, 3, 4, 5, 6, 7]
TOPN = 8
OUT_PATH = "output/us_factor_floor.json"


def _log(m): print(f"[US하한컷] {m}", file=sys.stderr)


def _pick(snap, floor, topn=TOPN):
    """미체결 슬롯은 현금(수익률 0)으로 희석 — kr_factor_floor.py와 동일 보정."""
    raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool = score[score >= floor] if floor is not None else score
    n_avail = len(pool)
    if n_avail == 0:
        return -bench, 0
    top = pool.sort_values(ascending=False).index[:topn]
    r = fwd.reindex(top).dropna()
    n_filled = len(r)
    if n_filled == 0:
        return -bench, 0
    port_ret = float(r.sum()) / topn
    return port_ret - bench, min(n_avail, topn)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개, topn={TOPN}")

    rows = []
    for floor in FLOOR_GRID:
        ex_list, fill_list = [], []
        for snap in snaps:
            ex, filled = _pick(snap, floor)
            if ex is not None:
                ex_list.append(ex); fill_list.append(filled)
        if not ex_list:
            continue
        rows.append({"floor": floor, "n_events": len(ex_list),
                    "mean_excess_pct": round(100 * float(np.mean(ex_list)), 3),
                    "win_rate_pct": round(100 * float(np.mean([x > 0 for x in ex_list])), 1),
                    "mean_positions_filled": round(float(np.mean(fill_list)), 2),
                    "pct_snaps_underfilled": round(100 * float(np.mean([f < TOPN for f in fill_list])), 1)})
        _log(f"floor={floor}: 초과 {rows[-1]['mean_excess_pct']:+.2f}%p 승률 {rows[-1]['win_rate_pct']}% "
            f"평균채움 {rows[-1]['mean_positions_filled']}/{TOPN}")

    baseline = next(r for r in rows if r["floor"] is None)
    payload = {"n_snaps": len(snaps), "topn": TOPN,
              "method": "하한 F 이상인 종목만 후보 풀에 남기고 그 안에서 top8(모자라면 있는 만큼만) 선정.",
              "baseline_no_floor": baseline, "rows": rows}
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
