#!/usr/bin/env python3
"""
kr_grading_system.py — 지호 님 요청(2026-09-23): "등급제도" 설계 검증. 대대적 백테스트로
유의한 결과를 얻되, 같은 점수를 반으로 가르는 순환논리(그룹 나누기) 없이 — 이미 각각
독립적으로 유의성이 검증된 조건들(밴드 내 상위·모멘텀 세부지표·밸류 지표)을 조합해
"몇 개 조건을 만족하는가"로 등급을 매기고, 그 조합들을 PBO/DSR의 진짜 시행(trial)으로
넣어 다중검정을 정직하게 보정한다(지호 님: "데이터고문말고 진짜로").

조건 12개(cap=6.0 밴드 내에서 계산):
  밴드위치: 밴드 내 종합점수 중앙값 이상
  밸류: value(1/PER)·pbr_inv(1/PBR)·div_yield 각각 밴드 중앙값 이상 (3개)
  모멘텀(세분화): mom6 양수·mom6 중앙값이상·mom12_1 양수·mom12_1 중앙값이상·
                 ma100/150/200일선 이격도 양수(3개)·1개월수익률 중앙값 이하(역추세) (8개)

시행 구성(순환논리 회피 — 같은 점수 반쪼개기 아님):
  1) 기준: 밴드+종합점수 topN(현행)
  2) 조건 단독 12개: 밴드 후보를 그 조건으로 필터 후 종합점수 topN
     (us_momentum_overlay.py에서 검증된 "필터 후 원점수 랭킹" 패턴과 동일)
  3) 그룹등급 5개: 밸류/모멘텀/전체 그룹별로 "만족 조건 수 ≥ 그룹크기 절반"을 필터로

실행: python -m research.kr.kr_grading_system
결과: output/kr_grading_system.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

import overfit_stats as OS
from research.kr.kr_factor_value_vs_rank import _composite
from research.kr.kr_momentum_overlay import _extra_mom_features

TOPN = 5
CAP = 6.0
HORIZON = "6m"
OUT_PATH = "output/kr_grading_system.json"


def _log(m): print(f"[KR등급제]  {m}", file=sys.stderr)


def _band_signals(snap, panel):
    """cap 밴드 내 후보의 종합점수+밸류3+모멘텀8 신호를 한 DataFrame으로."""
    raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
    score = _composite(raw).reindex(fwd.index).dropna()
    band_idx = score[score <= CAP].index
    if len(band_idx) < TOPN * 2:
        return None
    extra = _extra_mom_features(panel, snap["date"], band_idx)
    sig = pd.DataFrame(index=band_idx)
    sig["score"] = score.reindex(band_idx)
    for c in ("value", "pbr_inv", "div_yield", "mom6", "mom12_1"):
        sig[c] = raw.reindex(band_idx)[c]
    for c in ("ma100_gap", "ma150_gap", "ma200_gap", "ret_1m"):
        sig[c] = extra[c] if c in extra.columns else np.nan
    sig["fwd"] = fwd.reindex(band_idx)
    sig["excess"] = sig["fwd"] - bnc
    return sig.dropna(subset=["score"])


def _cond_flags(sig: pd.DataFrame) -> pd.DataFrame:
    med = sig.median(numeric_only=True)
    c = pd.DataFrame(index=sig.index)
    c["band_top"] = sig["score"] >= med["score"]
    c["value_med"] = sig["value"] >= med["value"]
    c["pbrinv_med"] = sig["pbr_inv"] >= med["pbr_inv"]
    c["div_med"] = sig["div_yield"] >= med["div_yield"]
    c["mom6_pos"] = sig["mom6"] > 0
    c["mom6_med"] = sig["mom6"] >= med["mom6"]
    c["mom121_pos"] = sig["mom12_1"] > 0
    c["mom121_med"] = sig["mom12_1"] >= med["mom12_1"]
    c["ma100"] = sig["ma100_gap"] > 0
    c["ma150"] = sig["ma150_gap"] > 0
    c["ma200"] = sig["ma200_gap"] > 0
    c["rev1m"] = sig["ret_1m"] <= med["ret_1m"]
    return c.fillna(False)


COND_COLS = ["band_top", "value_med", "pbrinv_med", "div_med", "mom6_pos", "mom6_med",
            "mom121_pos", "mom121_med", "ma100", "ma150", "ma200", "rev1m"]
VALUE_GROUP = ["value_med", "pbrinv_med", "div_med"]
MOM_GROUP = ["mom6_pos", "mom6_med", "mom121_pos", "mom121_med", "ma100", "ma150", "ma200", "rev1m"]


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

    variants = {"baseline_band_topn": base_sel}
    for col in COND_COLS:
        variants[f"cond_{col}"] = make_single(col)
    variants["grade_value_half"] = make_group(VALUE_GROUP, 0.5)
    variants["grade_value_all"] = make_group(VALUE_GROUP, 0.99)
    variants["grade_mom_half"] = make_group(MOM_GROUP, 0.5)
    variants["grade_mom_high"] = make_group(MOM_GROUP, 0.75)
    variants["grade_all_half"] = make_group(COND_COLS, 0.5)
    return variants


def _event_stats(sel_fn, snaps, panel) -> list:
    ex = []
    for snap in snaps:
        sig = _band_signals(snap, panel)
        if sig is None or len(sig) < TOPN:
            ex.append(None)
            continue
        cond = _cond_flags(sig)
        sel = sel_fn(sig, cond)
        r = sig.loc[sig.index.intersection(sel), "excess"].dropna()
        ex.append(float(r.mean()) if len(r) else None)
    return ex


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
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
        if label == "baseline_band_topn":
            baseline_ex = ex_a
        row = {"label": label, "n_events": len(ex),
              "mean_excess_pct": round(100 * float(ex_a.mean()), 3),
              "win_rate_pct": round(100 * float((ex_a > 0).mean()), 1)}
        summary.append(row)
        trial_labels.append(label)
        trial_returns.append(ex)
        _log(f"{label}: 초과 {row['mean_excess_pct']:+.2f}%p 승률 {row['win_rate_pct']}% n={row['n_events']}")

    # IC: 조건별 개별종목 초과수익과의 순위상관(그룹 안 나누고 연속형 검정)
    per_snap_corr = {c: [] for c in COND_COLS}
    for snap in snaps:
        sig = _band_signals(snap, panel)
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
            if label == "baseline_band_topn" or len(ex) != len(baseline_ex):
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
                   "rebal_days": 63, "hold_days": 126, "horizon": HORIZON,
                   "universe": "KR_band_top5", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)
    if baseline_ex is not None:
        mc = OS.monte_carlo_test(baseline_ex.tolist(), n=1000, seed=0)

    payload = {"n_snaps": len(snaps), "topn": TOPN, "cap": CAP, "n_trials": len(trial_labels),
              "method": ("cap=6 밴드 내에서 12개 독립조건(밸류3+모멘텀8+밴드위치1)을 단독/그룹으로 "
                        "필터해 종합점수 topN을 재선정 — 같은 점수 반쪼개기가 아니라 실제로 다르게 "
                        "설계될 수 있었던 후보 전략들이라 PBO/DSR 다중검정 보정이 유효함."),
              "summary": summary, "paired_vs_baseline": paired, "ic": ic_table,
              "pbo_dsr": pbo_report, "monte_carlo_baseline": mc}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
