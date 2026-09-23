#!/usr/bin/env python3
"""
us_grading_system.py — kr_grading_system.py의 미국판(2026-09-23, 지호 님 요청): "등급제도"
검증. 대대적 백테스트로 유의한 결과를 얻되, 같은 점수를 반으로 가르는 순환논리 없이 —
이미 각각 독립적으로 유의성이 검증된 조건들(추세·수익률·점수위치)을 조합해 등급을 매기고,
그 조합들을 PBO/DSR의 진짜 시행으로 넣어 다중검정을 정직하게 보정한다.

조건 10개(floor=3.25 이상 topn8 후보군에서 계산):
  점수위치: 종합점수 후보군 중앙값 이상
  추세(세분화): 20/50/100/150/200일선 이격도 양수 각각 (5개)
  수익률: 1/3/6개월 후행수익률 양수 각각(3개) + 1개월수익률 중앙값이하(역추세, 1개)

실행: python -m research.us.us_grading_system [--years 10]
결과: output/us_grading_system.json
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
OUT_PATH = "output/us_grading_system.json"


def _log(m): print(f"[US등급제]  {m}", file=sys.stderr)


def _pool_signals(snap, panel):
    raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool_idx = score[score >= FLOOR].index
    if len(pool_idx) < TOPN * 2:
        return None
    mom = _mom_features(panel, snap["date"], pool_idx)
    sig = pd.DataFrame(index=pool_idx)
    sig["score"] = score.reindex(pool_idx)
    for c in mom.columns:
        sig[c] = mom[c]
    sig["fwd"] = fwd.reindex(pool_idx)
    sig["excess"] = sig["fwd"] - bench
    return sig.dropna(subset=["score"])


def _cond_flags(sig: pd.DataFrame) -> pd.DataFrame:
    med = sig.median(numeric_only=True)
    c = pd.DataFrame(index=sig.index)
    c["score_top"] = sig["score"] >= med["score"]
    for n in (20, 50, 100, 150, 200):
        col = f"ma{n}_gap"
        c[f"ma{n}"] = sig[col] > 0 if col in sig.columns else False
    for label in ("1m", "3m", "6m"):
        col = f"ret_{label}"
        c[f"ret_{label}_pos"] = sig[col] > 0 if col in sig.columns else False
    c["rev1m"] = sig["ret_1m"] <= med.get("ret_1m", np.nan) if "ret_1m" in sig.columns else False
    return c.fillna(False)


COND_COLS = ["score_top", "ma20", "ma50", "ma100", "ma150", "ma200",
            "ret_1m_pos", "ret_3m_pos", "ret_6m_pos", "rev1m"]
TREND_GROUP = ["ma20", "ma50", "ma100", "ma150", "ma200"]
RET_GROUP = ["ret_1m_pos", "ret_3m_pos", "ret_6m_pos"]


def _make_variants():
    def base_sel(sig, cond):
        return sig["score"].sort_values(ascending=False).index[:TOPN]

    def make_single(col):
        def sel(sig, cond):
            pool = sig.index[cond[col]]
            if len(pool) < TOPN:
                return base_sel(sig, cond)
            return sig.loc[pool, "score"].sort_values(ascending=False).index[:TOPN]
        return sel

    def make_group(cols, frac):
        def sel(sig, cond):
            grade = cond[cols].sum(axis=1)
            pool = sig.index[grade >= max(1, int(len(cols) * frac))]
            if len(pool) < TOPN:
                return base_sel(sig, cond)
            return sig.loc[pool, "score"].sort_values(ascending=False).index[:TOPN]
        return sel

    variants = {"baseline_floor_topn": base_sel}
    for col in COND_COLS:
        variants[f"cond_{col}"] = make_single(col)
    variants["grade_trend_half"] = make_group(TREND_GROUP, 0.5)
    variants["grade_trend_high"] = make_group(TREND_GROUP, 0.75)
    variants["grade_ret_half"] = make_group(RET_GROUP, 0.5)
    variants["grade_ret_all"] = make_group(RET_GROUP, 0.99)
    variants["grade_all_half"] = make_group(COND_COLS, 0.5)
    return variants


def _event_stats(sel_fn, snaps, panel) -> list:
    ex = []
    for snap in snaps:
        sig = _pool_signals(snap, panel)
        if sig is None or len(sig) < TOPN:
            ex.append(None)
            continue
        cond = _cond_flags(sig)
        sel = sel_fn(sig, cond)
        r = sig.loc[sig.index.intersection(sel), "excess"].dropna()
        ex.append(float(r.mean()) if len(r) else None)
    return ex


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    variants = _make_variants()
    trial_labels, trial_returns, summary = [], [], []
    baseline_ex = None
    for label, fn in variants.items():
        ex_raw = _event_stats(fn, snaps, panel)
        ex = [x for x in ex_raw if x is not None]
        if len(ex) < 8:
            _log(f"{label}: 이벤트 부족({len(ex)}) — 스킵")
            continue
        ex_a = np.array(ex)
        if label == "baseline_floor_topn":
            baseline_ex = ex_a
        row = {"label": label, "n_events": len(ex),
              "mean_excess_pct": round(100 * float(ex_a.mean()), 3),
              "win_rate_pct": round(100 * float((ex_a > 0).mean()), 1)}
        summary.append(row)
        trial_labels.append(label)
        trial_returns.append(ex)
        _log(f"{label}: 초과 {row['mean_excess_pct']:+.2f}%p 승률 {row['win_rate_pct']}% n={row['n_events']}")

    per_snap_corr = {c: [] for c in COND_COLS}
    for snap in snaps:
        sig = _pool_signals(snap, panel)
        if sig is None or len(sig) < 10:
            continue
        cond = _cond_flags(sig).astype(int)
        for col in COND_COLS:
            if cond[col].nunique() < 2:
                continue
            corr = cond[col].corr(sig["excess"], method="spearman")
            if pd.notna(corr):
                per_snap_corr[col].append(float(corr))
    ic_table = {}
    for col, vals in per_snap_corr.items():
        if len(vals) < 5:
            continue
        a = np.array(vals)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None
        t = float(a.mean() / se) if se else None
        ic_table[col] = {"mean_spearman": round(float(a.mean()), 4),
                         "t_stat": round(t, 3) if t is not None else None, "n_snaps": len(a)}
        _log(f"IC[{col}]: spearman {ic_table[col]['mean_spearman']:+.4f} "
            f"(t={ic_table[col]['t_stat']}, n={ic_table[col]['n_snaps']})")

    paired = []
    if baseline_ex is not None:
        for label, ex in zip(trial_labels, trial_returns):
            if label == "baseline_floor_topn" or len(ex) != len(baseline_ex):
                continue
            diff = np.array(ex) - baseline_ex
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
            t = float(diff.mean() / se) if se else None
            paired.append({"label": label, "mean_diff_pct": round(100 * float(diff.mean()), 3),
                          "t_stat": round(t, 3) if t is not None else None, "n": len(diff)})

    lens = [len(ex) for ex in trial_returns]
    pbo_report = None
    mc = None
    if lens:
        common_len = max(set(lens), key=lens.count)
        keep_idx = [i for i, l in enumerate(lens) if l == common_len]
        if len(keep_idx) >= 3 and common_len >= 8:
            data = {"trials": [trial_labels[i] for i in keep_idx],
                   "excess_returns": [trial_returns[i] for i in keep_idx],
                   "rebal_days": 63, "hold_days": 126, "horizon": "6m",
                   "universe": "US_floor_top8", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)
    if baseline_ex is not None:
        mc = OS.monte_carlo_test(baseline_ex.tolist(), n=1000, seed=0)

    payload = {"n_snaps": len(snaps), "topn": TOPN, "floor": FLOOR, "n_trials": len(trial_labels),
              "method": ("floor=3.25 이상 후보군에서 10개 독립조건(추세5+수익률4+점수위치1)을 "
                        "단독/그룹으로 필터해 종합점수 topN을 재선정 — 같은 점수 반쪼개기가 아니라 "
                        "실제로 다르게 설계될 수 있었던 후보 전략들이라 PBO/DSR 다중검정 보정이 유효함."),
              "summary": summary, "paired_vs_baseline": paired, "ic": ic_table,
              "pbo_dsr": pbo_report, "monte_carlo_baseline": mc}
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
