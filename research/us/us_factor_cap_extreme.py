#!/usr/bin/env python3
"""
us_factor_cap_extreme.py — 지호 님 질문(2026-09-22): "극단값을 오히려 제외하면 수익률이
올라가나?" us_composite_score_vs_index.py에서 본 점수-초과수익 곡선(미국은 대체로 계속
상승, 한국은 6점 이상에서 역전)을 실제 topN 종목선정에 적용해 검증한다.

방법: 매 스냅샷(us_factor_formula_pit_sweep.build_snaps 재사용, 6개월 forward)마다
종합점수(1:2:2)로 정렬 → (a) 그냥 top8(현행) vs (b) 점수 상한 C 이상인 종목은 제외하고
그 다음 top8을 고른 경우, 를 비교. C를 여러 값으로 스윕. 이벤트(스냅샷) 단위 평균
forward-return·초과수익으로 비교(전체 포트폴리오 NAV 재구현 아님 — backtest_weights.
eval_config와 동일한 이벤트 평균 방식, 가볍고 충분히 빠름).

실행: python -m research.us.us_factor_cap_extreme [--years 10]  (topn 8/9/10 전부 스윕)
결과: output/us_factor_cap_extreme.json
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

CAP_GRID = [None, 15, 13, 12, 11, 10.5, 10, 9.5, 9, 8, 7, 6, 5, 4]   # None=현행(무제한)
TOPN_GRID = [8, 9, 10]   # 지호 님 질문: "잘라도 상위 8~10종목이면?"
OUT_PATH = "output/us_factor_cap_extreme.json"


def _log(m): print(f"[US극단제외] {m}", file=sys.stderr)


def _one_topn(snaps, topn: int, cap_grid=CAP_GRID) -> list[dict]:
    rows = []
    for cap in cap_grid:
        ev, ex, sels = [], [], []
        for snap in snaps:
            raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
            score = _composite(raw).reindex(fwd.index).dropna()
            pool = score[score <= cap] if cap is not None else score
            if len(pool) < topn:
                continue
            top = pool.sort_values(ascending=False).index[:topn]
            r = fwd.reindex(top).dropna()
            if len(r) == 0:
                continue
            ev.append(float(r.mean())); ex.append(float(r.mean()) - bench)
            sels.append(set(top))
        if not ev:
            continue
        ev_a, ex_a = np.array(ev), np.array(ex)
        turns = [1 - len(sels[j] & sels[j - 1]) / max(len(sels[j]), 1) for j in range(1, len(sels))]
        rows.append({"cap": cap, "n_events": len(ev),
                    "mean_fwd_ret_pct": round(100 * float(ev_a.mean()), 3),
                    "mean_excess_pct": round(100 * float(ex_a.mean()), 3),
                    "excess_sharpe": round(float(ex_a.mean() / ex_a.std()) * (252 / 126) ** 0.5, 3) if ex_a.std() else None,
                    "win_rate_pct": round(100 * float((ex_a > 0).mean()), 1),
                    "turnover_pct": round(100 * float(np.mean(turns)), 1) if turns else None})
        _log(f"cap={cap}: 초과 {rows[-1]['mean_excess_pct']:+.2f}%p 승률 {rows[-1]['win_rate_pct']}% "
            f"n={rows[-1]['n_events']}")

    return rows


def run(years: float = 10, topn_grid=TOPN_GRID, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개 · topn 스윕 {topn_grid}")

    by_topn = {}
    for topn in topn_grid:
        _log(f"--- topn={topn} ---")
        rows = _one_topn(snaps, topn)
        baseline = next((r for r in rows if r["cap"] is None), None)
        by_topn[str(topn)] = {"baseline_cap_none": baseline, "rows": rows}

    payload = {"n_snaps": len(snaps), "topn_grid": topn_grid,
              "method": ("매 스냅샷 종합점수(1:2:2) 상위 topn(cap 이하 종목만 후보) 선정 후 "
                        "6개월 forward return 이벤트평균. 전체 NAV 재시뮬레이션 아님(가벼운 "
                        "이벤트 프레임, backtest_weights.eval_config와 동일 방식). "
                        "지호 님 질문: 컷 적용하면서 상위 8~10종목으로 바꿔도 결론이 "
                        "같은지(topn 자체를 늘려 캡으로 빠진 슬롯을 메꾸면 나아지는지)."),
              "by_topn": by_topn}
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
