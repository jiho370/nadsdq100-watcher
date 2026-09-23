#!/usr/bin/env python3
"""
kr_topn_by_mktcap.py — 지호 님 질문(2026-09-23): "최종 선별된건 팩터랭킹이 아니라
시총 순위로 나열하는 거 어때? 수익률에 영향 없나? 국장도."

방법: 현행 라이브와 동일한 후보풀(cap=6.0 밴드, kr_grading_system._band_signals 재사용)에서
"최종 topN(=5, KR_MAX_HOLD와 동일)"을 뽑는 두 가지 방식을 페어드 비교한다:
  a) 기준(현행): 종합점수(value+pbr_inv+div_yield 가중합) 내림차순 topN
  b) 시총 내림차순 topN — 같은 밴드 안에서 순서만 다르게
시총은 pykrx 실측(backtest_kr.fetch_mktcap, PIT) — 미국과 달리 역산 근사가 아니라
그 날짜 실제 시가총액.

실행: python -m research.kr.kr_topn_by_mktcap
결과: output/kr_topn_by_mktcap.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_grading_system import _band_signals

TOPN = 5
OUT_PATH = "output/kr_topn_by_mktcap.json"


def _log(m): print(f"[KR시총정렬]  {m}", file=sys.stderr)


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개")

    ex_score, ex_mktcap, overlap, ic_vals = [], [], [], []
    n_skip_thin = 0
    for snap in snaps:
        sig = _band_signals(snap, panel)
        if sig is None or len(sig) < TOPN:
            continue
        d8 = snap["date"].replace("-", "")
        mc_by_t = mktcaps.get(d8) or {}
        mktcap = sig.index.to_series().map(mc_by_t).dropna()
        if len(mktcap) < TOPN:
            n_skip_thin += 1
            continue

        sel_score = sig["score"].sort_values(ascending=False).index[:TOPN]
        sel_mktcap = mktcap.sort_values(ascending=False).index[:TOPN]

        r1 = sig.loc[sig.index.intersection(sel_score), "excess"].dropna()
        r2 = sig.loc[sig.index.intersection(sel_mktcap), "excess"].dropna()
        if len(r1) == 0 or len(r2) == 0:
            continue
        ex_score.append(float(r1.mean()))
        ex_mktcap.append(float(r2.mean()))
        overlap.append(len(set(sel_score) & set(sel_mktcap)) / TOPN)

        common = mktcap.index
        if len(common) >= 10:
            corr = mktcap.reindex(common).rank().corr(sig.loc[common, "excess"], method="spearman")
            if pd.notna(corr):
                ic_vals.append(float(corr))

    a = np.array(ex_score)
    b = np.array(ex_mktcap)
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None

    ic_arr = np.array(ic_vals)
    ic_se = ic_arr.std(ddof=1) / np.sqrt(len(ic_arr)) if len(ic_arr) > 1 else None
    ic_t = float(ic_arr.mean() / ic_se) if ic_se else None

    payload = {
        "n_events": len(a), "n_skipped_thin_mktcap": n_skip_thin, "topn": TOPN, "cap": 6.0,
        "baseline_score_mean_excess_pct": round(100 * float(a.mean()), 3) if len(a) else None,
        "baseline_score_win_rate_pct": round(100 * float((a > 0).mean()), 1) if len(a) else None,
        "mktcap_order_mean_excess_pct": round(100 * float(b.mean()), 3) if len(b) else None,
        "mktcap_order_win_rate_pct": round(100 * float((b > 0).mean()), 1) if len(b) else None,
        "paired_diff_mean_pct": round(100 * float(diff.mean()), 3) if len(diff) else None,
        "paired_t_stat": round(t, 3) if t is not None else None,
        "avg_overlap_frac": round(float(np.mean(overlap)), 3) if overlap else None,
        "mktcap_rank_ic_spearman_mean": round(float(ic_arr.mean()), 4) if len(ic_arr) else None,
        "mktcap_rank_ic_t_stat": round(ic_t, 3) if ic_t is not None else None,
        "mktcap_rank_ic_n_snaps": len(ic_arr),
        "method": ("cap=6.0 밴드 통과 풀에서 topN(=5)을 종합점수순 vs 시총(pykrx 실측)순으로 뽑아 "
                  "페어드 비교. mktcap_rank_ic = 밴드 내 시총순위와 개별종목 향후초과수익의 "
                  "스피어만 상관.")}
    _log(f"[결과] n={payload['n_events']} 기준(점수순) {payload['baseline_score_mean_excess_pct']}%p"
        f"(승률{payload['baseline_score_win_rate_pct']}%) vs 시총순 "
        f"{payload['mktcap_order_mean_excess_pct']}%p(승률{payload['mktcap_order_win_rate_pct']}%) "
        f"페어드diff={payload['paired_diff_mean_pct']}%p t={payload['paired_t_stat']} "
        f"겹침률={payload['avg_overlap_frac']} IC={payload['mktcap_rank_ic_spearman_mean']}"
        f"(t={payload['mktcap_rank_ic_t_stat']})")

    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    run()


if __name__ == "__main__":
    main()
