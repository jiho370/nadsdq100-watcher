#!/usr/bin/env python3
"""
kr_factor_floor.py — 지호 님 요청(2026-09-22): "하한컷 방식도 백테스트." 지금까지는
상한(너무 높은 점수 제외)만 봤는데, 하한(너무 낮은 점수는 애초에 후보에서 빼고, 모자라면
topN을 못 채워도 그냥 둔다)도 같은 방식으로 검증.

방법: 매 스냅샷마다 (a) 하한 F 이상인 종목만 후보 풀로 남기고 (b) 그 풀에서 상위
topN(5)을 뽑되 풀이 topN보다 작으면 있는 만큼만 채운다(억지로 하위권을 채우지 않음).
상한 캡과 독립적으로(cap=None) 먼저 보고, 이미 찾은 최우수 상한(6.5)과 결합한 밴드
[F, 6.5]도 같이 스윕.

실행: python -m research.kr.kr_factor_floor
결과: output/kr_factor_floor.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

FLOOR_GRID = [None, -1, 0, 1, 2, 3, 3.5, 4, 4.5, 5]
TOPN = 5
HORIZON = "6m"
BEST_CAP = 6.5
OUT_PATH = "output/kr_factor_floor.json"


def _log(m): print(f"[KR하한컷] {m}", file=sys.stderr)


def _pick(snap, floor, cap, topn=TOPN):
    """미체결 슬롯은 현금(수익률 0)으로 둔다 — 슬롯이 모자라면 '뽑힌 종목만의 평균'이
    아니라 포트폴리오 전체(현금 희석 포함) 수익률로 계산해야 캡/하한 간 공정 비교가 된다."""
    raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool = score
    if cap is not None:
        pool = pool[pool <= cap]
    if floor is not None:
        pool = pool[pool >= floor]
    n_avail = len(pool)
    if n_avail == 0:
        return -bnc, 0   # 전액 현금 → 포트폴리오 수익률 0, 초과수익 = -벤치마크
    top = pool.sort_values(ascending=False).index[:topn]
    r = fwd.reindex(top).dropna()
    n_filled = len(r)
    if n_filled == 0:
        return -bnc, 0
    port_ret = float(r.sum()) / topn   # 현금 슬롯은 0% 기여(동일비중 토대에서 희석)
    return port_ret - bnc, min(n_avail, topn)


def _sweep(snaps, cap_label, cap):
    rows = []
    for floor in FLOOR_GRID:
        ex_list, fill_list = [], []
        for snap in snaps:
            ex, filled = _pick(snap, floor, cap)
            if ex is not None:
                ex_list.append(ex); fill_list.append(filled)
        if not ex_list:
            continue
        rows.append({"floor": floor, "cap": cap_label, "n_events": len(ex_list),
                    "mean_excess_pct": round(100 * float(np.mean(ex_list)), 3),
                    "win_rate_pct": round(100 * float(np.mean([x > 0 for x in ex_list])), 1),
                    "mean_positions_filled": round(float(np.mean(fill_list)), 2),
                    "pct_snaps_underfilled": round(100 * float(np.mean([f < TOPN for f in fill_list])), 1)})
        _log(f"[cap={cap_label}] floor={floor}: 초과 {rows[-1]['mean_excess_pct']:+.2f}%p "
            f"승률 {rows[-1]['win_rate_pct']}% 평균채움 {rows[-1]['mean_positions_filled']}/{TOPN} "
            f"미달스냅 {rows[-1]['pct_snaps_underfilled']}%")
    return rows


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개, topn={TOPN}")

    _log("=== 하한만(상한 없음) ===")
    floor_only = _sweep(snaps, "없음", None)
    _log(f"=== 하한 + 기존 최우수 상한({BEST_CAP}) 결합 ===")
    floor_plus_cap = _sweep(snaps, BEST_CAP, BEST_CAP)

    baseline = next(r for r in floor_only if r["floor"] is None)
    best_floor_only = max((r for r in floor_only if r["floor"] is not None), key=lambda r: r["mean_excess_pct"])
    best_combo = max((r for r in floor_plus_cap if r["floor"] is not None), key=lambda r: r["mean_excess_pct"])
    cap_only = next(r for r in floor_plus_cap if r["floor"] is None)

    payload = {"n_snaps": len(snaps), "topn": TOPN, "best_cap_reference": BEST_CAP,
              "method": ("하한 F 이상인 종목만 후보 풀에 남기고 그 안에서 topN(모자라면 "
                        "있는 만큼만) 선정. 상한 없음/상한=6.5(기존 최우수) 두 조건에서 "
                        "각각 스윕."),
              "baseline_no_floor_no_cap": baseline,
              "cap_only_6_5_no_floor": cap_only,
              "floor_sweep_no_cap": floor_only,
              "floor_sweep_with_cap_6_5": floor_plus_cap,
              "best_floor_only": best_floor_only,
              "best_floor_plus_cap": best_combo}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
