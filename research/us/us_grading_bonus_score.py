#!/usr/bin/env python3
"""
us_grading_bonus_score.py — 지호 님 제안(2026-09-23): us_momentum_overlay.py 버킷곡선
(150/200일선 이격도·3개월수익률)을 보고 정한 가점표로 추가점수를 만들어 종합점수에
더했을 때 효과가 있는지 확인.

⚠ 순환논리 위험: 그 버킷곡선 자체가 전체 34개 스냅샷을 다 본 뒤 나온 모양이라, 같은
데이터로 다시 테스트하면 당연히 좋게 나온다(사후적합). 이를 피하려고 분위구간 경계를
"그 시점까지의 과거 데이터"로만 계산하는 확장윈도우 워크포워드로 검증한다 — 가점표
숫자 자체(지호 님이 지정)는 고정이지만, 그 숫자를 어느 분위구간에 적용할지는 매 시점
과거 데이터만으로 다시 정해서 미래정보 유출을 막는다.

가점표(지호 님 지정, 분위 1~5):
  ma150_gap: 0, 0.2, 0.4, 0.6, 0.5
  ma200_gap: 0, 0.1, 0.1, 0.5, 0.4
  ret_3m:    0, 1,   1,   1,   1
combined_bonus = 세 가점의 합, composite z-score에 그대로 더해 topn8 재선정.

실행: python -m research.us.us_grading_bonus_score [--years 10]
결과: output/us_grading_bonus_score.json
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
MIN_HISTORY = 10   # 확장윈도우 최소 과거 스냅샷 수(이보다 적으면 워크포워드 평가 제외)
BONUS_TABLE = {
    "ma150_gap": [0.0, 0.2, 0.4, 0.6, 0.5],
    "ma200_gap": [0.0, 0.1, 0.1, 0.5, 0.4],
    "ret_3m":    [0.0, 1.0, 1.0, 1.0, 1.0],
}
OUT_PATH = "output/us_grading_bonus_score.json"


def _log(m): print(f"[US가점제]  {m}", file=sys.stderr)


def _pool_frame(snap, panel):
    raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool_idx = score[score >= FLOOR].index
    if len(pool_idx) < TOPN:
        return None
    mom = _mom_features(panel, snap["date"], pool_idx)
    df = pd.DataFrame(index=pool_idx)
    df["score"] = score.reindex(pool_idx)
    for c in BONUS_TABLE:
        df[c] = mom[c] if c in mom.columns else np.nan
    df["fwd"] = fwd.reindex(pool_idx)
    df["excess"] = df["fwd"] - bench
    return df.dropna(subset=["score"])


def _quintile_bonus(train_vals: pd.Series, apply_vals: pd.Series, weights: list) -> pd.Series:
    """train_vals(과거 데이터 풀링)로 분위 경계 계산 → apply_vals(현재 스냅샷)에 적용."""
    train = train_vals.dropna()
    if len(train) < 20:
        return pd.Series(0.0, index=apply_vals.index)
    try:
        edges = np.unique(np.quantile(train, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    except Exception:
        return pd.Series(0.0, index=apply_vals.index)
    if len(edges) < 3:
        return pd.Series(0.0, index=apply_vals.index)
    bins = pd.cut(apply_vals, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    n_bins = len(edges) - 1
    w = weights[:n_bins] if n_bins <= len(weights) else weights + [weights[-1]] * (n_bins - len(weights))
    return bins.map(lambda b: w[int(b)] if pd.notna(b) else 0.0)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    frames = []
    for snap in snaps:
        f = _pool_frame(snap, panel)
        frames.append(f)

    base_ex, bonus_ex = [], []
    for i, snap in enumerate(snaps):
        if i < MIN_HISTORY or frames[i] is None:
            continue
        hist = pd.concat([frames[j] for j in range(i) if frames[j] is not None], axis=0)
        if len(hist) < 30:
            continue
        cur = frames[i]
        combined_bonus = pd.Series(0.0, index=cur.index)
        for col, weights in BONUS_TABLE.items():
            combined_bonus = combined_bonus.add(
                _quintile_bonus(hist[col], cur[col], weights), fill_value=0.0)

        base_top = cur["score"].sort_values(ascending=False).index[:TOPN]
        bonus_score = cur["score"] + combined_bonus
        bonus_top = bonus_score.sort_values(ascending=False).index[:TOPN]

        r_base = cur.loc[base_top, "excess"].dropna()
        r_bonus = cur.loc[bonus_top, "excess"].dropna()
        if len(r_base) == 0 or len(r_bonus) == 0:
            continue
        base_ex.append(float(r_base.mean()))
        bonus_ex.append(float(r_bonus.mean()))
        _log(f"{snap['date']}: 기준 {100*float(r_base.mean()):+.2f}%p vs 가점 {100*float(r_bonus.mean()):+.2f}%p")

    b, c = np.array(base_ex), np.array(bonus_ex)
    diff = c - b
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None

    payload = {"n_events": len(b), "min_history": MIN_HISTORY, "bonus_table": BONUS_TABLE,
              "method": ("확장윈도우 워크포워드 — 각 시점의 분위구간 경계는 그 이전 스냅샷들의 "
                        "풀링 데이터로만 계산(미래정보 유출 없음). 가점표 숫자 자체는 지호 님이 "
                        "전체표본 버킷곡선을 보고 지정한 값이라 그 선택 자체엔 사전 정보가 "
                        "섞여있음 — 완전한 사전등록은 아니지만, 분위 경계 계산은 매 시점 정직하게 "
                        "그 시점까지의 데이터만 사용."),
              "baseline_mean_excess_pct": round(100 * float(b.mean()), 3),
              "baseline_win_rate_pct": round(100 * float((b > 0).mean()), 1),
              "bonus_mean_excess_pct": round(100 * float(c.mean()), 3),
              "bonus_win_rate_pct": round(100 * float((c > 0).mean()), 1),
              "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
              "paired_t_stat": round(t, 3) if t is not None else None,
              "n_wins_bonus_over_baseline": int((diff > 0).sum())}
    _log(f"[결과] n={len(b)} 기준 {payload['baseline_mean_excess_pct']}%p(승률{payload['baseline_win_rate_pct']}%) "
        f"vs 가점 {payload['bonus_mean_excess_pct']}%p(승률{payload['bonus_win_rate_pct']}%) "
        f"페어드diff={payload['paired_diff_mean_pct']}%p t={payload['paired_t_stat']}")

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
