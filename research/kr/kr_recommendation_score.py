#!/usr/bin/env python3
"""
kr_recommendation_score.py — us_recommendation_score.py의 한국판(2026-09-22, 지호 님
"추천 점수 랭킹 시스템도 만들어보자").

한국은 6.5점 근처부터 역전(밸류트랩)이 실측 확인됐으므로, 이 비단조성이 자동 반영되는
"평활화된 구간별 기대초과수익 -> 0~10점" 방식이 특히 의미가 크다 — 단순 백분위 방식은
"점수가 제일 높은 종목 = 10점"을 주지만, 실제로는 그게 밸류트랩 구간이라 낮은 점수를
받아야 정직하다.

실행: python -m research.kr.kr_recommendation_score
결과: output/kr_recommendation_score.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite
from research.kr.kr_smoothed_floor import build_bucket_curve

HORIZON = "6m"
OUT_PATH = "output/kr_recommendation_score.json"


def _log(m): print(f"[KR추천점수] {m}", file=sys.stderr)


def build_score_mapper(bucket: pd.DataFrame):
    b = bucket.dropna(subset=["smoothed_excess"]).sort_values("bin_mid")
    mids = b["bin_mid"].to_numpy()
    smoothed = b["smoothed_excess"].to_numpy()
    all_smoothed_sorted = np.sort(smoothed)

    def expected_excess(composite_score: float) -> float:
        return float(np.interp(composite_score, mids, smoothed, left=smoothed[0], right=smoothed[-1]))

    def to_0_10(exp_excess: float) -> float:
        pct = float(np.searchsorted(all_smoothed_sorted, exp_excess) / len(all_smoothed_sorted))
        return round(min(max(pct * 10, 0), 10), 1)

    return expected_excess, to_0_10


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개")

    bucket = build_bucket_curve(snaps)
    expected_excess, to_0_10 = build_score_mapper(bucket)

    latest = snaps[-1]
    score = _composite(latest["raw"]).dropna().sort_values(ascending=False)
    _log(f"최신 스냅샷({latest['date']}) 후보 {len(score)}종목")

    rows = []
    for sym, s in score.head(20).items():
        ee = expected_excess(float(s))
        rec = to_0_10(ee)
        rows.append({"symbol": sym, "composite_score": round(float(s), 2),
                    "raw_score_rank": len(rows) + 1,
                    "expected_excess_pct": round(100 * ee, 2), "recommendation_0_10": rec})
    for r in rows[:15]:
        flag = " <- 원점수 1등인데 밸류트랩 구간이라 추천점수 낮음" if r["raw_score_rank"] <= 3 and r["recommendation_0_10"] < 5 else ""
        _log(f"{r['symbol']:8s} 원점수순위{r['raw_score_rank']:2d} 종합점수 {r['composite_score']:6.2f} -> "
            f"기대초과 {r['expected_excess_pct']:+6.2f}%p -> 추천점수 {r['recommendation_0_10']:4.1f}/10{flag}")

    max_score = score.max(); min_score = score.min()
    old_style = [round(10 * (s - min_score) / (max_score - min_score), 1) for s in score.head(20)]
    n_diverge = sum(1 for r, o in zip([r["recommendation_0_10"] for r in rows], old_style) if abs(r - o) >= 2.0)

    payload = {"as_of": latest["date"], "n_snaps_used_for_curve": len(snaps),
              "method": ("kr_smoothed_floor.build_bucket_curve 재사용 — 점수구간별 평활초과"
                        "수익을 선형보간해 0~10점 환산. 전체표본 기반(워크포워드 아님, "
                        "프로토타입). 6.5점 근처 역전(밸류트랩)이 자동 반영됨."),
              "top20_recommendations": rows,
              "n_top20_diverge_ge_2pt_from_naive_percentile": n_diverge}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
