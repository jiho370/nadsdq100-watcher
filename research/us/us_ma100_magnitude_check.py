#!/usr/bin/env python3
"""
us_ma100_magnitude_check.py — 지호 님 질문(2026-09-23): "100일선은 높을수록(이격도 클수록)
좋고 이런 건 없나". us_momentum_overlay.py의 전체표본 버킷은 사후적합 위험이 있었으므로
(us_grading_bonus_score.py에서 확인된 문제와 동일), 여기서는:
  1) 100일선 위 후보들 안에서만 이격도(ma100_gap)와 개별종목 forward 초과수익의 스냅샷별
     순위상관(IC)+t검정 — "양음"이 아니라 "정도"만 따로 떼어 검정.
  2) 확장윈도우 워크포워드: "100일선 위 topn8" vs "100일선 위 종목 중 이격도 상위 절반만
     topn8" — 이격도로 추가 선별하면 더 좋아지는지, 미래정보 없이 재검증.

실행: python -m research.us.us_ma100_magnitude_check [--years 10]
결과: output/us_ma100_magnitude_check.json
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
from research.us.us_momentum_overlay import _mom_features

TOPN = 8
FLOOR = 3.25
MIN_HISTORY = 10
OUT_PATH = "output/us_ma100_magnitude_check.json"


def _log(m): print(f"[US100일선정도]  {m}", file=sys.stderr)


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
    df["fwd"] = fwd.reindex(pool_idx)
    df["excess"] = df["fwd"] - bench
    return df.dropna(subset=["score", "ma100_gap"])


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    frames = [_pool_frame(s, panel) for s in snaps]

    # 1) 100일선 위 후보 안에서만 IC(스냅샷별 순위상관+t검정)
    per_snap_corr = []
    for f in frames:
        if f is None:
            continue
        above = f[f["ma100_gap"] > 0]
        if len(above) < 6:
            continue
        c = above["ma100_gap"].corr(above["excess"], method="spearman")
        if pd.notna(c):
            per_snap_corr.append(float(c))
    ic = None
    if len(per_snap_corr) >= 5:
        a = np.array(per_snap_corr)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None
        t = float(a.mean() / se) if se else None
        ic = {"mean_spearman": round(float(a.mean()), 4),
             "t_stat": round(t, 3) if t is not None else None, "n_snaps": len(a)}
        _log(f"[IC: 100일선 위 후보 내 이격도 크기] spearman {ic['mean_spearman']:+.4f} "
            f"(t={ic['t_stat']}, n={ic['n_snaps']})")

    # 2) 워크포워드: "100일선 위 topn8" vs "100일선 위 중 이격도 상위 절반만 topn8"
    base_ex, deep_ex = [], []
    for i, f in enumerate(frames):
        if i < MIN_HISTORY or f is None:
            continue
        above = f[f["ma100_gap"] > 0]
        if len(above) < TOPN * 2:
            continue
        base_top = above["score"].sort_values(ascending=False).index[:TOPN]
        r_base = above.loc[base_top, "excess"].dropna()

        deep_half = above[above["ma100_gap"] >= above["ma100_gap"].median()]
        if len(deep_half) < TOPN:
            continue
        deep_top = deep_half["score"].sort_values(ascending=False).index[:TOPN]
        r_deep = deep_half.loc[deep_top, "excess"].dropna()

        if len(r_base) == 0 or len(r_deep) == 0:
            continue
        base_ex.append(float(r_base.mean()))
        deep_ex.append(float(r_deep.mean()))

    wf = None
    if len(base_ex) >= 8:
        b, d = np.array(base_ex), np.array(deep_ex)
        diff = d - b
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
        t = float(diff.mean() / se) if se else None
        wf = {"n_events": len(b),
             "above_ma100_topn_mean_excess_pct": round(100 * float(b.mean()), 3),
             "above_ma100_topn_win_rate_pct": round(100 * float((b > 0).mean()), 1),
             "deep_above_only_mean_excess_pct": round(100 * float(d.mean()), 3),
             "deep_above_only_win_rate_pct": round(100 * float((d > 0).mean()), 1),
             "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
             "paired_t_stat": round(t, 3) if t is not None else None}
        _log(f"[워크포워드] 100일선위 {wf['above_ma100_topn_mean_excess_pct']}%p"
            f"(승률{wf['above_ma100_topn_win_rate_pct']}%) vs 이격도상위절반만 "
            f"{wf['deep_above_only_mean_excess_pct']}%p(승률{wf['deep_above_only_win_rate_pct']}%) "
            f"페어드diff={wf['paired_diff_mean_pct']}%p t={wf['paired_t_stat']}")

    payload = {"n_snaps": len(snaps), "ic_gap_magnitude_within_above_ma100": ic,
              "walkforward_deep_above_vs_above": wf,
              "method": ("100일선 위 후보군에서만(양음 효과 제거) 이격도 크기와 forward 초과수익의 "
                        "순위상관 — '정도'만 따로 검정. 워크포워드는 확장윈도우(미래정보 유출 없음)로 "
                        "이격도 상위 절반만 추가로 걸렀을 때 실제 개선되는지 재확인.")}
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
