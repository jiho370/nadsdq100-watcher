#!/usr/bin/env python3
"""
verify_roa_candidate_live_config.py — §9-K-4~6에서 찾은 샤프1등 후보(int_gp_assets1·
shareholder_yield2·roa1)를 실제 라이브 조건(topn=8·섹터캡2·ma200_backup=False)의 NAV로
직접 검증 (2026-08-23, 지호 님 요청: "채택 전에 추가 검증 먼저").

배경: 지금까지의 비교(§9-K-4~6)는 전부 topn=30 넓은 팩터평가 풀 기준이었다 — 실제 라이브가
쓰는 topn=8 집중 포트폴리오·섹터캡2에서도 같은 우위가 유지되는지는 검증한 적이 없다.
`output/best_weights.json`은 건드리지 않고, `us_spmo_blend_prereg._select_basket_live_clip`
(가중치를 인자로 받음)을 재사용해 두 가중치를 나란히 시뮬레이션한다.

실행: python verify_roa_candidate_live_config.py
결과: output/roa_candidate_live_config.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP
import tech_factors as T
import core_satellite_kr as CS

YEARS = 15
LIVE_WEIGHTS = {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}
ROA_CANDIDATE = {"int_gp_assets": 1, "shareholder_yield": 2, "roa": 1}


def _log(m): print(f"[roa후보라이브검증] {m}", file=sys.stderr)


def decisions_with_weights(panel, funds, pit, weights, step=BP.MONTH):
    cross = T.build_panels(panel)
    out = []
    for p in range(BW.LOOKBACK, len(panel) - 1, step):
        ranked = SP._select_basket_live_clip(panel, p, funds, cross, pit, weights, BP.POOL_SIZE)
        if ranked:
            out.append((p, ranked))
    _log(f"결정 시점 {len(out)}개(가중치={BW._wstr(weights)})")
    return out


def build_nav(panel, ma200, funds, pit, weights, sector_of, cost):
    decisions = decisions_with_weights(panel, funds, pit, weights)
    nav = BP.simulate(panel, ma200, decisions, SP.TOPN, cost, ma200_backup=False,
                      sector_of=sector_of, sector_cap=2)
    if nav is None:
        raise RuntimeError(f"NAV 산출 실패(가중치={weights})")
    return nav / nav.iloc[0]


def winrate_by_horizon(nav_a: pd.Series, nav_b: pd.Series, label: str, min_windows=60) -> dict:
    idx = nav_a.index.intersection(nav_b.index)
    a, b = nav_a.reindex(idx).to_numpy(), nav_b.reindex(idx).to_numpy()
    n = len(a)
    h_max = n - min_windows
    out = {}
    for probe in (21, 63, 126, 252, 504, 756, 1260):
        if probe <= h_max:
            ra = a[probe:] / a[:-probe] - 1
            rb = b[probe:] / b[:-probe] - 1
            excess = ra - rb
            out[str(probe)] = {"win_rate_pct": round(float((excess > 0).mean()) * 100, 1),
                               "n_windows": int(len(excess)),
                               "mean_excess_pct": round(float(excess.mean()) * 100, 2)}
    _log(f"[{label}] 승률-보유기간: " + " | ".join(f"{k}d={v['win_rate_pct']}%" for k, v in out.items()))
    return out


def era_split(nav_a: pd.Series, nav_b: pd.Series, spy: pd.Series) -> dict:
    idx = nav_a.index.intersection(nav_b.index).intersection(spy.reindex(nav_a.index).ffill().dropna().index)
    mid = idx[len(idx) // 2]
    out = {}
    for tag, sl in (("전반부", idx[:len(idx)//2]), ("후반부", idx[len(idx)//2:])):
        a, b, s = nav_a.reindex(sl), nav_b.reindex(sl), spy.reindex(sl).ffill()
        a, b, s = a/a.iloc[0], b/b.iloc[0], s/s.iloc[0]
        sa, sb, ss = CS.stats(a), CS.stats(b), CS.stats(s)
        out[tag] = {"기간": f"{sl[0].date()}~{sl[-1].date()}",
                   "라이브1:2:2": sa, "roa후보": sb, "SPY": ss}
        _log(f"[{tag} {sl[0].date()}~{sl[-1].date()}] 라이브 CAGR {sa['cagr_pct']}% · "
            f"roa후보 CAGR {sb['cagr_pct']}% · SPY {ss['cagr_pct']}%")
    return out


def main():
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(YEARS, pit)
    funds = BW.load_funds()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = SP._sector_of_factory()

    nav_live = build_nav(panel, ma200, funds, pit, LIVE_WEIGHTS, sector_of, cost)
    nav_roa = build_nav(panel, ma200, funds, pit, ROA_CANDIDATE, sector_of, cost)

    idx = nav_live.index.intersection(nav_roa.index).intersection(
        spy.reindex(nav_live.index).ffill().dropna().index)
    nav_live_a, nav_roa_a = nav_live.reindex(idx), nav_roa.reindex(idx)
    nav_live_a, nav_roa_a = nav_live_a/nav_live_a.iloc[0], nav_roa_a/nav_roa_a.iloc[0]
    spy_a = spy.reindex(idx).ffill(); spy_a = spy_a/spy_a.iloc[0]

    s_live, s_roa, s_spy = CS.stats(nav_live_a), CS.stats(nav_roa_a), CS.stats(spy_a)
    _log(f"전체({idx[0].date()}~{idx[-1].date()}, {len(idx)}일): "
        f"라이브 CAGR {s_live['cagr_pct']}%/샤프{s_live['sharpe']}/MDD{s_live['mdd_pct']}% · "
        f"roa후보 CAGR {s_roa['cagr_pct']}%/샤프{s_roa['sharpe']}/MDD{s_roa['mdd_pct']}% · "
        f"SPY CAGR {s_spy['cagr_pct']}%/샤프{s_spy['sharpe']}/MDD{s_spy['mdd_pct']}%")

    wr_roa_vs_live = winrate_by_horizon(nav_roa_a, nav_live_a, "roa후보 vs 라이브1:2:2")
    wr_live_vs_spy = winrate_by_horizon(nav_live_a, spy_a, "라이브1:2:2 vs SPY")
    wr_roa_vs_spy = winrate_by_horizon(nav_roa_a, spy_a, "roa후보 vs SPY")
    eras = era_split(nav_live_a, nav_roa_a, spy_a)

    payload = {"years_requested": YEARS, "data_start": idx[0].date().isoformat(),
              "data_end": idx[-1].date().isoformat(), "n_days": len(idx),
              "live_1_2_2": s_live, "roa_candidate": s_roa, "spy": s_spy,
              "winrate_roa_vs_live": wr_roa_vs_live,
              "winrate_live_vs_spy": wr_live_vs_spy,
              "winrate_roa_vs_spy": wr_roa_vs_spy,
              "era_split": eras}
    os.makedirs("output", exist_ok=True)
    with open("output/roa_candidate_live_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/roa_candidate_live_config.json")


if __name__ == "__main__":
    main()
