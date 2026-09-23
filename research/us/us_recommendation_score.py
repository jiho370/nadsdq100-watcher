#!/usr/bin/env python3
"""
us_recommendation_score.py — 지호 님 요청(2026-09-22): "추천 점수 랭킹 시스템도 만들어보자."

기존 score_calibration.py(v2)는 "군집점수의 백분위 × 10"으로 0~10점을 매긴다 — 이는
점수가 높을수록 무조건 좋다(단조)는 전제다. 그런데 이번 세션에서 확인한 건:
  · 미국은 대체로 단조(점수 높을수록 좋음, us_factor_cap_extreme.py)
  · 한국은 6.5점 근처부터 역전(밸류트랩, kr_factor_cap_extreme_deepdive.py)
즉 "백분위"가 아니라 "그 점수대 종목들이 실제로 평균 얼마나 벌었는가"(평활화된 구간별
기대초과수익, us_smoothed_floor.py의 bucket_curve)를 직접 0~10점으로 환산하는 게 더
정직하다 — 논모노토닉(비단조) 관계도 자동으로 반영된다.

방법:
  1) us_smoothed_floor.build_bucket_curve() 재사용 — 점수구간별 평활초과수익 곡선.
  2) 임의의 종합점수 s에 대해 그 곡선을 선형보간해 "기대초과수익 E[excess|score=s]"을 구함.
  3) 기대초과수익의 히스토리 분포(백분위)로 0~10점 환산 — 기대초과수익 자체가 이미
     비단조성을 반영하므로, 점수가 아무리 높아도 그 구간 기대수익이 낮으면 추천점수도 낮다.
  4) 최신 스냅샷(현재 후보군)에 적용해 실제 추천표 예시 생성.

한계(정직하게 명시): 곡선 자체가 전체 역사표본으로 만들어져 있어(워크포워드 아님) 사후
적합 위험이 있다 — us_factor_cap_extreme.py처럼 나중에 워크포워드로 재검증 가능.
지금은 "구조 프로토타입"이다.

실행: python -m research.us.us_recommendation_score [--years 10]
결과: output/us_recommendation_score.json
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
from research.us.us_smoothed_floor import build_bucket_curve

OUT_PATH = "output/us_recommendation_score.json"


def _log(m): print(f"[US추천점수] {m}", file=sys.stderr)


def build_score_mapper(bucket: pd.DataFrame):
    """구간중앙값 -> 평활초과수익 선형보간 함수, 그리고 그 평활초과수익들의 백분위 환산기."""
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


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
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
                    "expected_excess_pct": round(100 * ee, 2), "recommendation_0_10": rec})
    for r in rows[:15]:
        _log(f"{r['symbol']:6s} 종합점수 {r['composite_score']:6.2f} -> 기대초과 "
            f"{r['expected_excess_pct']:+6.2f}%p -> 추천점수 {r['recommendation_0_10']:4.1f}/10")

    # 검증: 신형(기대초과수익 기반) vs 구형(단순 백분위) 점수의 순서가 어디서 갈리는지 확인
    max_score, min_score = score.max(), score.min()
    old_style = [round(10 * (s - min_score) / (max_score - min_score), 1) for s in score.head(20)]
    n_diverge = sum(1 for r, o in zip([r["recommendation_0_10"] for r in rows], old_style) if abs(r - o) >= 2.0)

    payload = {"as_of": latest["date"], "n_snaps_used_for_curve": len(snaps),
              "method": ("점수구간별 평활초과수익 곡선(us_smoothed_floor.build_bucket_curve)을 "
                        "선형보간해 종합점수 -> 기대초과수익 -> (역사적 기대초과수익 분포 내 "
                        "백분위)*10 으로 환산. 전체표본 기반(워크포워드 아님, 프로토타입)."),
              "top20_recommendations": rows,
              "n_top20_diverge_ge_2pt_from_naive_percentile": n_diverge,
              "note": "naive_percentile은 '점수 단순 백분위*10'(기존 score_calibration.py 방식)과 비교용."}
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
