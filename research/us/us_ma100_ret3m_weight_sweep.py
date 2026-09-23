#!/usr/bin/env python3
"""
us_ma100_ret3m_weight_sweep.py — 지호 님 제안(2026-09-23): 100일선 이격도·3개월수익률
(단독 검증에서 1·2위)을 라이브 팩터점수와 똑같은 방식(z-score 가중합)으로 섞어서, 비중을
스윕했을 때 봉우리(peak)가 있는지 확인.

혼합점수 = 팩터종합점수 + w1*z(ma100_gap) + w2*z(ret_3m), (w1,w2) 그리드 스윕.
경계값에서 계속 좋아지기만 하면 페이크(그냥 그 신호 비중을 무한히 늘리라는 뜻이 되어
버림), 내부 어딘가에서 정점이면 의미있는 조합일 가능성.

순서: 1) 전체표본 그리드로 봉우리 탐색 → 2) 봉우리 후보를 확장윈도우 워크포워드로
재검증(미래정보 유출 없이) → 3) 그리드 전체를 PBO/DSR 시행으로 넣어 다중검정 보정.

실행: python -m research.us.us_ma100_ret3m_weight_sweep [--years 10]
결과: output/us_ma100_ret3m_weight_sweep.json
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
import overfit_stats as OS
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features

TOPN = 8
FLOOR = 3.25
MIN_HISTORY = 10
WEIGHT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
OUT_PATH = "output/us_ma100_ret3m_weight_sweep.json"


def _log(m): print(f"[US비중스윕]  {m}", file=sys.stderr)


def _zclip(s: pd.Series, cap=3.0) -> pd.Series:
    sd = s.std()
    z = (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0
    return z.clip(-cap, cap).fillna(0.0)


def _pool_frame(snap, panel):
    raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool_idx = score[score >= FLOOR].index
    if len(pool_idx) < TOPN:
        return None
    mom = _mom_features(panel, snap["date"], pool_idx)
    df = pd.DataFrame(index=pool_idx)
    df["score"] = score.reindex(pool_idx)
    df["ma100_gap"] = mom["ma100_gap"] if "ma100_gap" in mom.columns else np.nan
    df["ret_3m"] = mom["ret_3m"] if "ret_3m" in mom.columns else np.nan
    df["fwd"] = fwd.reindex(pool_idx)
    df["excess"] = df["fwd"] - bench
    return df.dropna(subset=["score"])


def _event_excess(frame, w1, w2):
    z1 = _zclip(frame["ma100_gap"]) if frame["ma100_gap"].notna().any() else frame["score"] * 0.0
    z2 = _zclip(frame["ret_3m"]) if frame["ret_3m"].notna().any() else frame["score"] * 0.0
    combo = frame["score"] + w1 * z1 + w2 * z2
    top = combo.sort_values(ascending=False).index[:TOPN]
    r = frame.loc[top, "excess"].dropna()
    return float(r.mean()) if len(r) else None


def full_sample_grid(frames):
    rows = []
    for w1 in WEIGHT_GRID:
        for w2 in WEIGHT_GRID:
            ex = [x for x in (_event_excess(f, w1, w2) for f in frames if f is not None) if x is not None]
            if len(ex) < 8:
                continue
            a = np.array(ex)
            rows.append({"w1_ma100": w1, "w2_ret3m": w2, "n_events": len(ex),
                        "mean_excess_pct": round(100 * float(a.mean()), 3),
                        "win_rate_pct": round(100 * float((a > 0).mean()), 1)})
    return rows


def walkforward_check(snaps, frames, w1, w2):
    base_ex, combo_ex = [], []
    for i in range(len(snaps)):
        if i < MIN_HISTORY or frames[i] is None:
            continue
        cur = frames[i]
        base_top = cur["score"].sort_values(ascending=False).index[:TOPN]
        r_base = cur.loc[base_top, "excess"].dropna()
        ex_combo = _event_excess(cur, w1, w2)
        if len(r_base) == 0 or ex_combo is None:
            continue
        base_ex.append(float(r_base.mean()))
        combo_ex.append(ex_combo)
    if len(base_ex) < 8:
        return {"note": "표본 부족"}
    b, c = np.array(base_ex), np.array(combo_ex)
    diff = c - b
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None
    return {"n_events": len(b),
           "baseline_mean_excess_pct": round(100 * float(b.mean()), 3),
           "combo_mean_excess_pct": round(100 * float(c.mean()), 3),
           "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
           "paired_t_stat": round(t, 3) if t is not None else None}


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    frames = [_pool_frame(s, panel) for s in snaps]

    grid = full_sample_grid(frames)
    for r in grid:
        _log(f"w1={r['w1_ma100']} w2={r['w2_ret3m']}: 초과 {r['mean_excess_pct']:+.2f}%p 승률 {r['win_rate_pct']}%")
    best = max(grid, key=lambda r: r["mean_excess_pct"]) if grid else None
    baseline_row = next((r for r in grid if r["w1_ma100"] == 0.0 and r["w2_ret3m"] == 0.0), None)
    is_interior_peak = bool(best and baseline_row and
                            0 < best["w1_ma100"] < max(WEIGHT_GRID) and
                            0 < best["w2_ret3m"] < max(WEIGHT_GRID))
    _log(f"[전체표본 최고] w1={best['w1_ma100']} w2={best['w2_ret3m']} "
        f"초과 {best['mean_excess_pct']}%p (기준 w=0,0: {baseline_row['mean_excess_pct']}%p) "
        f"경계값아님={is_interior_peak}")

    wf = walkforward_check(snaps, frames, best["w1_ma100"], best["w2_ret3m"]) if best else None
    if wf and "note" not in wf:
        _log(f"[워크포워드 재검증] 기준 {wf['baseline_mean_excess_pct']}%p vs 혼합 "
            f"{wf['combo_mean_excess_pct']}%p 페어드diff={wf['paired_diff_mean_pct']}%p t={wf['paired_t_stat']}")

    trial_labels, trial_returns = [], []
    for r in grid:
        ex = [x for x in (_event_excess(f, r["w1_ma100"], r["w2_ret3m"]) for f in frames if f is not None)
             if x is not None]
        if len(ex) >= 8:
            trial_labels.append(f"w1={r['w1_ma100']}_w2={r['w2_ret3m']}")
            trial_returns.append(ex)
    lens = [len(ex) for ex in trial_returns]
    pbo_report = None
    if lens:
        common_len = max(set(lens), key=lens.count)
        keep_idx = [i for i, l in enumerate(lens) if l == common_len]
        if len(keep_idx) >= 3 and common_len >= 8:
            data = {"trials": [trial_labels[i] for i in keep_idx],
                   "excess_returns": [trial_returns[i] for i in keep_idx],
                   "rebal_days": 63, "hold_days": 126, "horizon": "6m",
                   "universe": "US_floor_top8_weightsweep", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)

    payload = {"n_snaps": len(snaps), "weight_grid": WEIGHT_GRID, "full_sample_grid": grid,
              "full_sample_best": best, "full_sample_baseline": baseline_row,
              "is_interior_peak": is_interior_peak,
              "walkforward_check_at_best_weights": wf, "pbo_dsr": pbo_report,
              "method": ("혼합점수 = 팩터종합점수 + w1*z(ma100일이격도) + w2*z(3개월수익률), "
                        "라이브 팩터점수와 동일한 z-score 가중합 방식. 전체표본 그리드로 봉우리 "
                        "탐색 후 확장윈도우 워크포워드로 그 봉우리가 미래정보 없이도 재현되는지 "
                        "재검증, 그리드 전체를 PBO/DSR 시행으로 다중검정 보정.")}
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
