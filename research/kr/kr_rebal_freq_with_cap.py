#!/usr/bin/env python3
"""
kr_rebal_freq_with_cap.py — us_rebal_freq_with_floor.py의 한국판(2026-09-23, 지호 님
질문: "바뀐 방식(cap=6 반영)으로도 6개월 정기재평가가 최적인지 확인").

방법: 재평가 주기 N ∈ {1개월(21일)·3개월(63일)·6개월(126일)·12개월(252일)}마다
backtest_kr.build_kr_snaps(rebal_days=N)로 그 주기로 스냅샷을 다시 만들고(보유기간도
같은 N — fwd[해당 horizon] 사용, 재평가 주기만큼 들고 있다가 다시 고른다는 라이브 방식
그대로), cap=6.0 적용한 밴드 내 topn5를 뽑아 이벤트 평균 초과수익·승률을 구한다. 주기가
짧을수록 늘어나는 회전비용을 "연 재평가횟수 × 코스피 매도세(20bp)"로 근사해 연환산
순초과수익도 같이 낸다(선형 근사 — 상대비교용).

실행: python -m research.kr.kr_rebal_freq_with_cap
결과: output/kr_rebal_freq_with_cap.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from research.kr.kr_factor_value_vs_rank import _composite

TOPN = 5
CAP = 6.0
FREQ_GRID = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
KR_ROUND_TRIP_BP = 20.0   # backtest_costs.KR_SELL_TAX(코스피 0.20%) — 매수측은 세금 없음, 왕복 근사
OUT_PATH = "output/kr_rebal_freq_with_cap.json"


def _log(m): print(f"[KR재평가주기] {m}", file=sys.stderr)


def _one_freq(snaps, horizon, days) -> dict:
    ex = []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][horizon], snap["bench"][horizon]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score <= CAP]
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        r = fwd.reindex(top).dropna()
        if len(r) == 0:
            continue
        ex.append(float(r.mean()) - bnc)
    if len(ex) < 5:
        return {"note": "표본 부족", "n_events": len(ex)}
    a = np.array(ex)
    rebal_per_year = 252.0 / days
    annualized_gross_pct = 100 * float(a.mean()) * rebal_per_year
    annualized_cost_drag_pct = KR_ROUND_TRIP_BP / 1e2 * rebal_per_year
    return {"n_events": len(a), "mean_excess_pct": round(100 * float(a.mean()), 3),
           "win_rate_pct": round(100 * float((a > 0).mean()), 1),
           "excess_sharpe": round(float(a.mean() / a.std()) * np.sqrt(rebal_per_year), 3) if a.std() else None,
           "rebal_per_year_approx": round(rebal_per_year, 2),
           "annualized_gross_excess_pct_approx": round(annualized_gross_pct, 2),
           "annualized_cost_drag_pct_approx": round(annualized_cost_drag_pct, 2),
           "annualized_net_excess_pct_approx": round(annualized_gross_pct - annualized_cost_drag_pct, 2)}


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()

    rows = {}
    for label, days in FREQ_GRID.items():
        snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                        rebal_days=days, flows=flows, mktcaps=mktcaps)
        r = _one_freq(snaps, label, days)
        rows[label] = r
        _log(f"{label}(재평가{days}일): n={r.get('n_events')} 초과 {r.get('mean_excess_pct')}%p "
            f"승률 {r.get('win_rate_pct')}% → 연환산 순초과(근사) {r.get('annualized_net_excess_pct_approx')}%p")

    best = max((k for k in rows if rows[k].get("annualized_net_excess_pct_approx") is not None),
              key=lambda k: rows[k]["annualized_net_excess_pct_approx"], default=None)
    payload = {"cap": CAP, "topn": TOPN, "by_freq": rows, "best_by_annualized_net_excess": best,
              "method": ("재평가 주기=보유기간으로 맞춰(라이브 방식 그대로) 스냅샷을 다시 만들고 "
                        "cap=6.0 밴드 내 topn5 이벤트 평균 초과수익을 구한 뒤, 주기가 짧을수록 "
                        "늘어나는 회전비용(연 재평가횟수×코스피 매도세 20bp)을 선형근사로 차감한 "
                        "연환산 순초과수익으로 비교. 절대수익 주장이 아니라 주기 간 상대순위용 근사치.")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
