#!/usr/bin/env python3
"""
topn_sweep_candidates.py — 라이브 1:2:2 + topn=8 재검증 1·2위 대안 후보를 topn=4~10
전 구간에서 비교 (2026-08-23, 지호 님 요청: "4~10개로 다시").

효율화: 종목 랭킹(decisions)은 가중치당 딱 1번만 계산 — BP.simulate()가 랭킹 리스트의
상위 topn만 골라 쓰므로, topn만 바꿔 재시뮬레이션하는 건 저렴하다(재다운로드·재랭킹 없음).

실행: python topn_sweep_candidates.py
결과: output/topn_sweep_candidates.json
"""
from __future__ import annotations
import os, sys, json

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP
import tech_factors as T
import core_satellite_kr as CS

YEARS = 10
TOPN_RANGE = [4, 5, 6, 7, 8, 9, 10]
CANDIDATES = {
    "라이브1:2:2": {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2},
    "대안#1(topn8 1위)": {"int_gp_assets": 2, "shareholder_yield": 3, "fcf_ev": 3},
    "대안#2(topn8 2위)": {"int_gp_assets": 2, "shareholder_yield": 3, "fcf_ev": 2},
}


def _log(m): print(f"[topN스윕]{m}", file=sys.stderr)


def main():
    pit = BC.load_pit()
    panel, spy, opens = BC.build_panel_pit(YEARS, pit)
    funds = BW.load_funds()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = SP._sector_of_factory()
    cross = T.build_panels(panel)

    results = {}
    for name, w in CANDIDATES.items():
        decisions = []
        for p in range(BW.LOOKBACK, len(panel) - 1, BP.MONTH):
            ranked = SP._select_basket_live_clip(panel, p, funds, cross, pit, w, BP.POOL_SIZE)
            if ranked:
                decisions.append((p, ranked))
        _log(f"[{name}] 가중치={BW._wstr(w)} 결정시점 {len(decisions)}개")
        rows = {}
        for topn in TOPN_RANGE:
            nav = BP.simulate(panel, ma200, decisions, topn, cost, ma200_backup=False,
                              sector_of=sector_of, sector_cap=2)
            if nav is None:
                _log(f"    topn={topn}: NAV 실패")
                continue
            idx = nav.index.intersection(spy.reindex(nav.index).ffill().dropna().index)
            nav_a = nav.reindex(idx); nav_a = nav_a / nav_a.iloc[0]
            s = CS.stats(nav_a)
            rows[str(topn)] = s
            _log(f"    topn={topn}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}%")
        results[name] = {"weights": w, "by_topn": rows}

    spy_a = spy.reindex(idx).ffill(); spy_a = spy_a / spy_a.iloc[0]
    s_spy = CS.stats(spy_a)
    _log(f"[SPY 동일구간] CAGR {s_spy['cagr_pct']}% 샤프 {s_spy['sharpe']} MDD {s_spy['mdd_pct']}%")

    payload = {"years": YEARS, "topn_range": TOPN_RANGE, "spy": s_spy, "candidates": results}
    os.makedirs("output", exist_ok=True)
    with open("output/topn_sweep_candidates.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/topn_sweep_candidates.json")


if __name__ == "__main__":
    main()
