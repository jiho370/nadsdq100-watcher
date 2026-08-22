#!/usr/bin/env python3
"""
fx_cost_breakeven_check.py — KRW=X 레짐필터(Stage1 최우수 파라미터) vs 매수후보유
CAGR 우위가 정확히 몇 bp 비용에서 역전되는지 미세 격자로 재확인 (2026-08-22).

배경: STRATEGY.md §6-R-2는 "0bp 1.4% → 10bp 1.04% → 20bp 0.69%(B&H 0.81%보다 열위 전환)
→ 50bp -0.37%(완전 역전)"이라고 기록했다(당시 B&H CAGR 0.81%). 2026-08-22 재검증 세션에서
데이터를 2026-07-29→2026-08-22로 갱신하자 B&H CAGR 자체가 0.62%로 낮아져(최근 원화 강세
구간이 새로 추가됨) 전략(같은 파라미터, 절대 CAGR은 거의 그대로)의 상대 우위가 유지되는
비용 구간이 넓어졌을 가능성이 있다 — backtest_regime_fx.py의 COST_BPS_SWEEP은
[0,5,10,20,50]로 성겨서 정확한 교차점을 못 짚는다. 이 스크립트는 재구현 없이
backtest_regime_assets.simulate()만 재사용해 1bp 간격으로 교차점을 이분탐색한다.

실행: python fx_cost_breakeven_check.py
결과: output/fx_cost_breakeven.json + stderr 요약
"""
from __future__ import annotations
import os, sys, json
import numpy as np

import backtest_regime_assets as RA

WINNER = {"trend_ma": 100, "band": 0.0, "confirm": 5}   # Stage1/wide 공통 최우수(§6-R-1/2)


def _log(m): print(f"[비용교차점] {m}", file=sys.stderr)


def main():
    price = RA.fetch("KRW=X", "output/regime_price_cache_fx.pkl")
    closes = price.to_numpy()
    exp = RA.regime_series(closes, WINNER["trend_ma"], WINNER["band"], WINNER["confirm"])

    bh_cagr = RA.simulate(closes, exp, 0)["bh_cagr"]  # cost_bps는 bh_cagr에 영향 없음(항상보유는 무회전)

    fine_bps = list(range(0, 61, 2))
    rows = []
    for bps in fine_bps:
        m = RA.simulate(closes, exp, bps)
        rows.append({"cost_bps": bps, "cagr": round(m["cagr"], 3), "bh_cagr": round(bh_cagr, 3),
                    "excess_cagr": round(m["cagr"] - bh_cagr, 3)})

    # 1bp 이분탐색으로 정확한 교차점(초과CAGR=0) 좁히기
    lo, hi = 0, 60
    lo_m = RA.simulate(closes, exp, lo)["cagr"] - bh_cagr
    hi_m = RA.simulate(closes, exp, hi)["cagr"] - bh_cagr
    crossover_bps = None
    if lo_m > 0 and hi_m < 0:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_m = RA.simulate(closes, exp, mid)["cagr"] - bh_cagr
            if mid_m > 0:
                lo = mid
            else:
                hi = mid
        crossover_bps = hi   # 초과CAGR이 처음 음수로 바뀌는 bp

    payload = {"winner_params": WINNER, "bh_cagr": round(bh_cagr, 3),
              "date_range": [str(price.index.min().date()), str(price.index.max().date())],
              "fine_sweep": rows, "crossover_bps_approx": crossover_bps,
              "note": "excess_cagr>0 구간이 전략 우위. crossover_bps_approx는 이분탐색으로 "
                      "초과CAGR이 처음 <=0이 되는 1bp 단위 지점(근사, 비용은 선형이 아니라 "
                      "회전빈도에 따라 계단식으로 영향을 줌)."}
    os.makedirs("output", exist_ok=True)
    with open("output/fx_cost_breakeven.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log(f"B&H CAGR(현재 데이터): {bh_cagr:.3f}%")
    for r in rows:
        flag = "우위" if r["excess_cagr"] > 0 else "열위"
        _log(f"  {r['cost_bps']:>3}bp: 전략 {r['cagr']:>6.3f}% vs B&H {r['bh_cagr']:.3f}% "
            f"(초과 {r['excess_cagr']:+.3f}%p, {flag})")
    _log(f"교차점(초과CAGR=0) 근사: {crossover_bps}bp 부근")
    _log(f"저장: output/fx_cost_breakeven.json")


if __name__ == "__main__":
    main()
