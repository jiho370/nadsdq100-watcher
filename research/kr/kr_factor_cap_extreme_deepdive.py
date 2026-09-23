#!/usr/bin/env python3
"""
kr_factor_cap_extreme_deepdive.py — kr_factor_cap_extreme.py 결과("점수 6~7 초과 제외 시
초과수익 개선") 정밀 검증 (2026-09-22, 지호 님 "한국 더 자세히").

kr_factor_cap_extreme.py가 확인한 건 점추정치(캡 걸었을 때 평균 초과수익이 더 높다)뿐이고
39개 스냅샷으로는 우연일 위험이 크다. 여기서는:
  1) 캡을 0.5 단위로 더 촘촘히(topn=5 고정, 라이브 값) — 정확한 최적점 위치 확인
  2) 최우수 캡 vs 무캡(현행) 쌍대 블록부트스트랩 — Δ초과수익 90%CI가 0을 배제하는지
     (이 프로젝트 표준 방법론, backtest_regime_assets.paired_block_bootstrap와 동일 정신)
  3) 전반기/후반기 분할 — 특정 시기(예: 2024+ 반도체 쏠림장) 하나가 견인한 결과는 아닌지
  4) 캡을 걸었을 때 실제로 배제되는 종목이 뭔지(빈도 상위) — 메커니즘 확인

실행: python -m research.kr.kr_factor_cap_extreme_deepdive
결과: output/kr_factor_cap_extreme_deepdive.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

FINE_CAP_GRID = [None, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3]
TOPN = 5
HORIZON = "6m"
N_BOOT = 5000
SEED = 11
OUT_PATH = "output/kr_factor_cap_extreme_deepdive.json"


def _log(m): print(f"[KR극단제외정밀] {m}", file=sys.stderr)


def _pick(snap, cap, topn):
    raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool = score[score <= cap] if cap is not None else score
    excluded = score[score > cap].index.tolist() if cap is not None else []
    if len(pool) < topn:
        return None, None, None, None
    top = pool.sort_values(ascending=False).index[:topn]
    r = fwd.reindex(top).dropna()
    if len(r) == 0:
        return None, None, None, None
    return float(r.mean()), float(r.mean()) - bnc, set(top), excluded


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개, topn={TOPN}(라이브 고정)")

    # 1) 0.5 단위 촘촘한 그리드
    fine_rows = []
    per_snap_ex = {}   # cap -> [snap별 초과수익]
    for cap in FINE_CAP_GRID:
        ex_list = []
        for snap in snaps:
            _, ex, _, _ = _pick(snap, cap, TOPN)
            ex_list.append(ex)   # None 포함 가능(표본부족)
        per_snap_ex[cap] = ex_list
        valid = [x for x in ex_list if x is not None]
        if not valid:
            continue
        fine_rows.append({"cap": cap, "n_events": len(valid),
                          "mean_excess_pct": round(100 * float(np.mean(valid)), 3),
                          "win_rate_pct": round(100 * float(np.mean([x > 0 for x in valid])), 1)})
        _log(f"cap={cap}: 초과 {fine_rows[-1]['mean_excess_pct']:+.2f}%p 승률 {fine_rows[-1]['win_rate_pct']}%")

    best = max((r for r in fine_rows if r["cap"] is not None), key=lambda r: r["mean_excess_pct"])
    baseline = next(r for r in fine_rows if r["cap"] is None)
    _log(f"최우수 캡={best['cap']}(초과 {best['mean_excess_pct']:+.2f}%p) vs 무캡(초과 "
        f"{baseline['mean_excess_pct']:+.2f}%p)")

    # 2) 최우수 캡 vs 무캡 쌍대 블록부트스트랩(스냅샷 단위 리샘플 — 이벤트 자체가 이미
    #    6개월 forward라 스냅샷=블록 1개로 취급, 이 프로젝트의 다른 분석들과 동일 관례)
    ex_best = np.array([x for x in per_snap_ex[best["cap"]] if x is not None])
    ex_base = np.array([x for x in per_snap_ex[None] if x is not None])
    n = min(len(ex_best), len(ex_base))
    ex_best, ex_base = ex_best[:n], ex_base[:n]
    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = ex_best[idx].mean() - ex_base[idx].mean()
    ci90 = (round(100 * float(np.percentile(diffs, 5)), 3), round(100 * float(np.percentile(diffs, 95)), 3))
    ci95 = (round(100 * float(np.percentile(diffs, 2.5)), 3), round(100 * float(np.percentile(diffs, 97.5)), 3))
    prob_better = round(100 * float((diffs > 0).mean()), 1)
    bootstrap = {"best_cap": best["cap"], "n_events": n, "delta_excess_ci90": ci90,
                "delta_excess_ci95": ci95, "ci90_excludes_zero": bool(ci90[0] > 0 or ci90[1] < 0),
                "ci95_excludes_zero": bool(ci95[0] > 0 or ci95[1] < 0),
                "prob_cap_beats_baseline_pct": prob_better}
    _log(f"부트스트랩: Δ초과 90%CI {ci90} (0 배제={bootstrap['ci90_excludes_zero']}) · "
        f"95%CI {ci95} · 캡이 이길 확률 {prob_better}%")

    # 3) 전반기/후반기 분할
    dates = [snap["date"] for snap in snaps]
    mid = len(snaps) // 2
    split_date = dates[mid]
    halves = {}
    for label, snap_subset in [("전반기", snaps[:mid]), ("후반기", snaps[mid:])]:
        row = {}
        for cap in [None, best["cap"]]:
            vals = []
            for snap in snap_subset:
                _, ex, _, _ = _pick(snap, cap, TOPN)
                if ex is not None:
                    vals.append(ex)
            row[str(cap)] = {"n": len(vals), "mean_excess_pct": round(100 * float(np.mean(vals)), 3) if vals else None}
        halves[label] = row
    _log(f"기간분할(경계 {split_date}): {json.dumps(halves, ensure_ascii=False)}")

    # 4) 캡=best에서 자주 배제되는 종목
    excl_counter = {}
    for snap in snaps:
        _, _, _, excluded = _pick(snap, best["cap"], TOPN)
        for sym in (excluded or []):
            excl_counter[sym] = excl_counter.get(sym, 0) + 1
    top_excluded = sorted(excl_counter.items(), key=lambda kv: -kv[1])[:15]

    payload = {"n_snaps": len(snaps), "topn": TOPN,
              "fine_grid": fine_rows, "best": best, "baseline": baseline,
              "bootstrap_best_vs_baseline": bootstrap,
              "half_period_split": {"split_date": split_date, "halves": halves},
              "most_frequently_excluded_at_best_cap": [{"sym": s, "n_snaps_excluded": n} for s, n in top_excluded],
              "note": ("39개 스냅샷(6개월 forward, 분기 리밸런싱 시점)으로 이 프로젝트 다른 "
                      "분석과 같은 표본 한계가 그대로 적용됨 — 부트스트랩·기간분할로 정직하게 "
                      "강건성만 확인, 확정적 채택 근거는 아님.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
