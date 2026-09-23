#!/usr/bin/env python3
"""
kr_cap_walkforward.py — 지호 님 요청(2026-09-22): "일정 이상 커트라인 형성하는 방법,
점수화도 시도." kr_factor_cap_extreme_deepdive.py가 찾은 "6.5점 초과 제외"는 전체 표본을
다 보고 찾은 값이라(사후적) look-ahead 편향 위험이 있다 — score_calibration.py가 이미
채택한 원칙("가중치는 워크포워드, t 이전 스냅샷 정보만 사용")을 그대로 적용해 두 가지를
시도한다:

(A) 워크포워드 적응형 컷라인 — 각 시점 t마다, t 이전 스냅샷들'만'으로 캡 그리드를 다시
    탐색해 그 시점까지 최선이었던 캡을 선택 → t에 적용(미래 참조 없음). 고정 6.5와 실제
    성과를 비교(고정값은 사후적이라 성과가 부풀려질 수 있음 — 워크포워드가 정직한 검정).
(B) 점수화(하드컷 대신 완만한 페널티) — 상한을 넘는 만큼 점수를 깎는 연속함수
    adj_score = score - λ·max(0, score-θ)^2 로 재점수화 후 재정렬. 배제(0/1)가 아니라
    "얼마나 넘었는지"에 비례해 벌점을 줘서, 커트라인 바로 위 종목이 완전히 탈락하는 대신
    순위만 밀리게 한다(경계값 근처의 불연속성 완화).

실행: python -m research.kr.kr_cap_walkforward
결과: output/kr_cap_walkforward.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

TOPN = 5
HORIZON = "6m"
CAP_SEARCH_GRID = [None, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4]
MIN_HISTORY = 12   # 워크포워드 캡 탐색에 필요한 최소 과거 스냅샷 수
PENALTY_THETA_GRID = [7, 6.5, 6, 5.5]
PENALTY_LAMBDA_GRID = [0.3, 0.6, 1.0, 2.0]
OUT_PATH = "output/kr_cap_walkforward.json"


def _log(m): print(f"[KR워크포워드컷] {m}", file=sys.stderr)


def _excess_for_cap(snap, cap, topn=TOPN):
    raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
    score = _composite(raw).reindex(fwd.index).dropna()
    pool = score[score <= cap] if cap is not None else score
    if len(pool) < topn:
        return None
    top = pool.sort_values(ascending=False).index[:topn]
    r = fwd.reindex(top).dropna()
    if len(r) == 0:
        return None
    return float(r.mean()) - bnc


def _excess_for_penalty(snap, theta, lam, topn=TOPN):
    raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
    score = _composite(raw).reindex(fwd.index).dropna()
    penalty = lam * np.maximum(0.0, score - theta) ** 2
    adj = score - penalty
    if len(adj) < topn:
        return None
    top = adj.sort_values(ascending=False).index[:topn]
    r = fwd.reindex(top).dropna()
    if len(r) == 0:
        return None
    return float(r.mean()) - bnc


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    n = len(snaps)
    _log(f"스냅샷 {n}개, 워크포워드 최소이력 {MIN_HISTORY}개 → OOS 구간 {n - MIN_HISTORY}개")

    # ---- (A) 워크포워드 적응형 컷라인 ----
    # 사전계산: 모든 (스냅샷, 캡) 조합의 초과수익
    excess_by_cap = {cap: [_excess_for_cap(s, cap) for s in snaps] for cap in CAP_SEARCH_GRID}

    wf_excess, wf_chosen_cap, baseline_oos = [], [], []
    for t in range(MIN_HISTORY, n):
        best_cap, best_mean = None, -np.inf
        for cap in CAP_SEARCH_GRID:
            hist = [excess_by_cap[cap][i] for i in range(t) if excess_by_cap[cap][i] is not None]
            if len(hist) < MIN_HISTORY // 2:
                continue
            m = float(np.mean(hist))
            if m > best_mean:
                best_mean, best_cap = m, cap
        ex_t = excess_by_cap[best_cap][t]
        if ex_t is not None:
            wf_excess.append(ex_t); wf_chosen_cap.append(best_cap)
        base_t = excess_by_cap[None][t]
        if base_t is not None:
            baseline_oos.append(base_t)

    fixed65_oos = [excess_by_cap[6.5][t] for t in range(MIN_HISTORY, n) if excess_by_cap[6.5][t] is not None]

    wf_result = {
        "n_oos_events": len(wf_excess),
        "mean_excess_pct": round(100 * float(np.mean(wf_excess)), 3) if wf_excess else None,
        "win_rate_pct": round(100 * float(np.mean([x > 0 for x in wf_excess])), 1) if wf_excess else None,
        "chosen_cap_distribution": {str(c): wf_chosen_cap.count(c) for c in set(wf_chosen_cap)},
    }
    baseline_result = {"n_oos_events": len(baseline_oos),
                       "mean_excess_pct": round(100 * float(np.mean(baseline_oos)), 3) if baseline_oos else None}
    fixed65_result = {"n_oos_events": len(fixed65_oos),
                      "mean_excess_pct": round(100 * float(np.mean(fixed65_oos)), 3) if fixed65_oos else None,
                      "note": "고정 6.5는 전체표본으로 사후에 찾은 값 — look-ahead 있음, 참고용"}
    _log(f"워크포워드 적응형: OOS 초과 {wf_result['mean_excess_pct']}%p (n={wf_result['n_oos_events']}) · "
        f"선택된 캡 분포 {wf_result['chosen_cap_distribution']}")
    _log(f"같은 OOS구간 무캡(기준): {baseline_result['mean_excess_pct']}%p")
    _log(f"같은 OOS구간 고정6.5(사후적, 참고): {fixed65_result['mean_excess_pct']}%p")

    # ---- (B) 점수화(완만한 페널티) ----
    penalty_rows = []
    for theta in PENALTY_THETA_GRID:
        for lam in PENALTY_LAMBDA_GRID:
            vals = [_excess_for_penalty(s, theta, lam) for s in snaps]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            penalty_rows.append({"theta": theta, "lambda": lam, "n_events": len(vals),
                                 "mean_excess_pct": round(100 * float(np.mean(vals)), 3),
                                 "win_rate_pct": round(100 * float(np.mean([v > 0 for v in vals])), 1)})
    penalty_rows.sort(key=lambda r: -r["mean_excess_pct"])
    for r in penalty_rows[:8]:
        _log(f"페널티(θ={r['theta']}, λ={r['lambda']}): 초과 {r['mean_excess_pct']:+.2f}%p 승률 {r['win_rate_pct']}%")
    best_penalty = penalty_rows[0] if penalty_rows else None
    full_sample_65 = [x for x in excess_by_cap[6.5] if x is not None]
    hardcap_65 = {"cap": 6.5, "mean_excess_pct": round(100 * float(np.mean(full_sample_65)), 3)}

    payload = {"n_snaps": n, "min_history": MIN_HISTORY,
              "walkforward_adaptive_cap": wf_result,
              "same_window_baseline_no_cap": baseline_result,
              "same_window_fixed_cap_6_5_lookahead_reference": fixed65_result,
              "penalty_scoring_all_rows": penalty_rows,
              "penalty_scoring_best": best_penalty,
              "hardcap_6_5_full_sample_reference": hardcap_65,
              "note": ("(A)는 미래 참조 없이 매 시점 과거 데이터만으로 캡을 다시 고른 정직한 "
                      "검정 — 고정 6.5(전체표본 사후 최적값)보다 낮게 나오는 게 정상(그게 "
                      "look-ahead의 크기). (B)는 하드컷 대신 초과분에 비례한 연속 벌점을 준 "
                      "재점수화 — θ·λ 그리드 중 최우수를 보여줌.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
