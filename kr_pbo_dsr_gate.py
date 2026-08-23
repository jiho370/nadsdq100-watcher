#!/usr/bin/env python3
"""
kr_pbo_dsr_gate.py — 한국 팩터 가중치 재탐색(§9-K-10~11, keep=5·levels0-3·961조합)의
정식 PBO/DSR 게이트 (2026-08-23, 지호 님 요청: "한국 후보 PBO/DSR 정식 게이트 마저").

미국(backtest_costs.py `compare()`의 PBO 입력 절)과 동일한 절차 — topn=30 풀에서
조합별 6개월 순초과수익 이벤트행렬을 만들어 overfit_stats.analyze()에 그대로 투입.
overfit_stats.analyze(save=True)는 output/pbo_report.json(미국용)을 덮어쓰므로 여기선
save=False로 받아 output/kr_pbo_report.json에 별도 저장.

⚠ 한계(정직하게 명시): 이 게이트는 topn=30 넓은 풀 기준이다. §9-K-10~11에서 이미
topn=30 순위가 topn=5(실제 라이브) 성과를 완벽히 예측하지 못함을 확인했다 — 즉 이 게이트
통과 여부는 "탐색 자체가 노이즈인지"를 보는 표준 절차이지, "후보#8이 topn=5에서 진짜
나은지"에 대한 직접 증명은 아니다(그건 §9-K-11의 topn=5 짝지은 부트스트랩이 담당).

실행: python kr_pbo_dsr_gate.py
결과: output/kr_trial_returns.json · output/kr_pbo_report.json
"""
from __future__ import annotations
import os, sys, json

import backtest_costs as BC
import backtest_weights as BW
import backtest_kr as BK
import overfit_stats as OS

KEEP, LEVELS = 5, (0, 1, 2, 3)
POOL_KR = 30
REBAL_DAYS = 63


def _log(m): print(f"[KR PBO/DSR]{m}", file=sys.stderr)


def main():
    from benchmarks_kr import load_research_data
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, n_pit_ok, n_total = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                                 rebal_days=REBAL_DAYS, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개(PIT 확인 {n_pit_ok}/{n_total})")
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    allidx = list(range(len(snaps)))

    ic_sorted = BC._agg_ic(snaps, allidx)
    selected = BC._pick(ic_sorted, KEEP)
    _log(f"채택 팩터(keep={KEEP}): {selected}")

    trials, matrix, dates0 = [], [], None
    for w in BW._weight_grid(selected, LEVELS):
        row, ev6 = BC.eval_config(w, snaps, allidx, cost, POOL_KR, collect_6m=True)
        trials.append(BW._wstr(w))
        matrix.append(ev6)
    n_ev = min(len(m) for m in matrix) if matrix else 0
    _log(f"조합 {len(trials)}개 × 이벤트 {n_ev}회 행렬 구축 완료")

    trial_data = {"horizon": "6m", "universe": "kr_kospi200_pit",
                 "cost": cost.describe(), "rebal_days": REBAL_DAYS,
                 "hold_days": BW.TD["6m"],
                 "dates": [s["date"] for s in snaps[:n_ev]],
                 "trials": trials, "excess_returns": [m[:n_ev] for m in matrix]}
    os.makedirs("output", exist_ok=True)
    with open("output/kr_trial_returns.json", "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    _log("저장: output/kr_trial_returns.json")

    report = OS.analyze(trial_data, save=False)
    with open("output/kr_pbo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: output/kr_pbo_report.json (passed={report['passed']})")

    # 후보#8·라이브 1:1:1이 이 961조합 중 몇 위인지, DSR 최고점 조합과 같은지 확인
    cand8_str = BW._wstr({"mom12_1": 1, "pbr_inv": 1, "div_yield": 3, "low_vol": 1})
    live_str = BW._wstr({"value": 1, "pbr_inv": 1, "div_yield": 1})
    _log(f"후보#8 문자열={cand8_str} · 라이브 문자열={live_str}")
    _log(f"DSR 최고샤프 조합={report['dsr']['best_trial']}")
    _log(f"후보#8이 DSR 최고점과 동일한가: {cand8_str == report['dsr']['best_trial']}")


if __name__ == "__main__":
    main()
