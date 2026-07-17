#!/usr/bin/env python3
"""
us_factor_bootstrap_stability.py — 지호 님 요청(2026-07-17): "1등 조합이 스윕마다 계속
바뀌는게 이 표본크기에서 원래 그런건지 확인." 블록부트스트랩으로 5개 대표 조합(현행+
두 그리드 1등+지호 님이 매력적으로 본 1:3:3+C3)의 '승률'을 직접 재본다 — 34개 이벤트를
반복 리샘플해서 매 회차 어느 조합이 최고 샤프인지 집계. 한 조합이 압도적으로 자주
이기면 "진짜 신호", 여러 조합이 비슷한 비율로 나눠 이기면 "이 표본에선 원래 구분 불가"
(=오늘 그리드마다 1등이 바뀐 게 특별히 이상한 게 아니라 34개 표본의 근본적 한계).

실행: python us_factor_bootstrap_stability.py
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

import backtest_costs as BC
import us_factor_formula_pit_sweep as PS
import us_factor_formula_pit_grid_top8 as G8

CANDIDATES = [
    {"label": "현행(1:2:2 raw)",      "weights": {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}, "rd_mode": "raw"},
    {"label": "147그리드 1등(rd1sy3 qgate)", "weights": {"rd_mktcap": 1, "shareholder_yield": 3}, "rd_mode": "qgate"},
    {"label": "345그리드 1등(gp5rd3sy5 raw)", "weights": {"int_gp_assets": 5, "rd_mktcap": 3, "shareholder_yield": 5}, "rd_mode": "raw"},
    {"label": "지호님 관심(1:3:3 raw)", "weights": {"int_gp_assets": 1, "rd_mktcap": 3, "shareholder_yield": 3}, "rd_mode": "raw"},
    {"label": "C3(1:1:2 raw)",        "weights": {"int_gp_assets": 1, "rd_mktcap": 1, "shareholder_yield": 2}, "rd_mode": "raw"},
]


def _log(m): print(f"[안정성검증] {m}", file=sys.stderr)


def run(n_boot=5000, block=6, seed=42):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(10, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = PS.build_snaps(panel, spy, funds, pit)
    sector_map = PS.sector_of_map()

    series = []   # 각 후보의 34개 이벤트별 초과수익 배열
    for c in CANDIDATES:
        excess, _ = G8.eval_trial_nocap(snaps, c["weights"], c["rd_mode"])  # topn8, 무캡(그리드와 동일 조건)
        series.append(np.array([v if v == v else 0.0 for v in excess]))
        _log(f"{c['label']:28s} 평균초과6M {100*np.mean(excess):+.2f}%p")

    n = len(series[0])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    wins = np.zeros(len(CANDIDATES))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sharpes = []
        for s in series:
            sample = s[idx]
            sd = sample.std()
            sharpes.append(float(sample.mean() / sd) if sd > 0 else 0.0)
        wins[int(np.argmax(sharpes))] += 1

    win_pct = 100 * wins / n_boot
    _log(f"=== 블록부트스트랩 {n_boot}회 승률(각 회차 최고샤프 후보) ===")
    for c, w in zip(CANDIDATES, win_pct):
        _log(f"  {c['label']:28s} 승률 {w:5.1f}%")
    _log("(5개 후보가 똑같이 20%씩이면 완전 무작위와 다를 바 없음 — 그것과 비교해서 판단)")

    payload = {"n_boot": n_boot, "block": block, "n_events": n,
              "candidates": [c["label"] for c in CANDIDATES],
              "win_pct": [round(float(w), 1) for w in win_pct],
              "note": "각 후보 승률이 균등분포(1/5=20%)에 가까울수록 '이 표본 크기에서는 "
                      "원래 구분 불가'라는 뜻 — 오늘 그리드마다 1등이 바뀐 게 특이한 게 "
                      "아니라 34개 표본의 근본적 한계임을 시사."}
    os.makedirs("output", exist_ok=True)
    with open("output/us_factor_bootstrap_stability.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_factor_bootstrap_stability.json")
    return payload


if __name__ == "__main__":
    run()
