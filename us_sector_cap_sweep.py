#!/usr/bin/env python3
"""
us_sector_cap_sweep.py — 지호 님 질문(2026-07-17): "바이오/제약이 유독 많이 걸리는데
구조적인 이유인가, 부정적 영향은 없나?" 실측 확인 결과 오늘 상위 8종목 중 4개(MRNA·BMY·
INCY·REGN)가 Health Care — 현재 라이브 가중치(rd_mktcap 2·shareholder_yield 2·
gp_assets 1) 중 rd_mktcap(R&D지출/시가총액)이 구조적으로 바이오/제약을 편애함(REGN
0.539로 전체 218종목 중 압도적 1위, 2위의 2배). 확인 결과 미국 선정 로직(export_data.
select_by_weights)엔 섹터 상한이 전혀 없음(pick_with_sector_cap/RECO_SECTOR_MAX는 안 쓰는
구식 하이브리드 방식 전용). 한국은 이미 Stage 4(STRATEGY.md)에서 섹터캡을 검증했으나
(캡을 걸어도 성과에 거의 영향 없음 확인) 미국은 한 번도 테스트한 적 없음 — 이 스크립트가
그 공백을 메운다.

방법: backtest_portfolio.simulate()의 기존 sector_of/sector_cap 파라미터(KR Stage 4용으로
이미 추가돼 있음) 재사용. topn=8(현재 champion) 고정, sector_cap ∈ {None(무제한), 3, 2}
스윕. sector_of는 위키피디아 GICS Sector(현재 시점 분류, PIT 아님 — KR Stage 4와 동일한
근사 한계) 사용.

실행: python us_sector_cap_sweep.py
결과: output/us_sector_cap_sweep.json · output/pbo_report_us_sector_cap.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP
import backtest_weights as BW

TOPN = 8
CAP_LIST = [None, 3, 2]


def _log(m): print(f"[US섹터캡] {m}", file=sys.stderr)


def _sector_of_factory():
    import sp500_daily_report as R
    sector_map = R.fetch_wikipedia_sectors()
    _log(f"위키 섹터맵 {len(sector_map)}종목 확보")

    def sector_of(date_s: str, sym: str):
        return sector_map.get(sym)
    return sector_of


def run(years=10, save=True):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    ma200 = panel.rolling(200, min_periods=200).mean()
    decisions = BP.us_decisions(panel, funds, pit)
    sector_of = _sector_of_factory()

    rows = []
    navs = {}
    for cap in CAP_LIST:
        trade_log = []
        nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False,
                          sector_of=sector_of, sector_cap=cap, trade_log=trade_log)
        if nav is None:
            _log(f"cap={cap}: NAV 산출 실패"); continue
        m = BP.metrics(nav, spy.reindex(panel.index).ffill())
        relaxed = sum(1 for t in trade_log if t.get("action") == "sector_cap_relaxed")
        rows.append({"sector_cap": cap if cap is not None else "무제한", **m, "cap_relaxed_events": relaxed})
        navs[str(cap)] = nav
        _log(f"cap={cap}: CAGR {m['cagr_pct']:6.2f}% (초과 {m['excess_cagr_pct']:+.2f}%p) · "
             f"샤프 {m['sharpe']:5.2f} · MDD {m['mdd_pct']:6.1f}% · 캡완화 {relaxed}회")

    if len(rows) < 2:
        raise RuntimeError("결과 부족")

    dates0, matrix = None, []
    for cap in CAP_LIST:
        nav = navs.get(str(cap))
        if nav is None:
            continue
        d, r = BP.monthly_excess(nav, spy.reindex(nav.index).ffill())
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
    n_ev = min(len(r) for r in matrix)
    trial_data = {"horizon": "us_sector_cap", "universe": "sp500_pit", "cost": cost.describe(),
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev], "trials": [f"cap{r['sector_cap']}" for r in rows],
                 "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN,
              "judgment": "SPY 매수후보유 대비 월간초과수익 PBO/DSR (섹터캡 무제한/3/2 비교)",
              "rows": rows,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "note": "sector_of는 위키피디아 GICS Sector 현재 시점 분류(point-in-time 아님) "
                      "— KR Stage 4와 동일한 근사 한계"}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_sector_cap_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open("output/pbo_report_us_sector_cap.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/us_sector_cap_sweep.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


if __name__ == "__main__":
    run()
