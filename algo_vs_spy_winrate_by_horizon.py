#!/usr/bin/env python3
"""
algo_vs_spy_winrate_by_horizon.py — 미국 알고리즘(topn8 라이브 그대로) vs SPY,
보유기간별 승리 확률 (2026-07-26, 지호 님 요청)

배경: algo_vs_index_lenient.py는 월간(21거래일) 승률 하나만 봤다. 여기서는 "몇 개월/몇 년을
들고 있으면 SPY를 이길 확률이 얼마나 되나"를 여러 보유기간(1개월~5년)으로 넓혀 계산한다.
2026-07-26 지호 님 요청으로 이산적인(1개월·3개월·…) 표가 아니라 **1거래일 단위로 하루부터
쭉 이어지는 연속 곡선**으로 확장.

데이터: us_spmo_blend_prereg._us_decisions_live_clip()(라이브와 동일 z-score, shareholder_yield
클립만 ±5)로 만든 topn8 NAV(best_weights.json 1:2:2, sector_cap=2, ma200_backup=False)를
SPY와 직접 비교 — SPMO는 쓰지 않는다(SPMO는 2015-10 상장이라 _load()로 SPMO와 교집합을
취하면 2012~2015 구간이 통째로 잘려나감, 최초 시도에서 확인). "계산 가능한 시점부터"는 이
저장소에서 이미 검증된 최대 구간인 15년(us_algo_15y_champion_check.json 참고 — 2012년~,
EDGAR 펀더멘털 커버리지가 100%가 아니라 결측 종목은 해당 팩터 z=0 중립 처리되는 한계가
있음을 그대로 물려받음).

방법: 보유기간 h=1,2,3,…거래일 전부에 대해 롤링(겹치는) 윈도우로 t, t+h 구간의 알고리즘
수익률과 SPY 수익률을 비교 — algo_ret > spy_ret인 윈도우 비율 = 승률. 윈도우가 겹치므로
(하루씩 이동) 관측치 간 자기상관이 크다는 점은 한계로 명시(그래도 "임의 시점에 진입해서
h기간 들고 있으면 이길 확률"이라는 질문엔 겹치는 롤링이 가장 직접적인 답).

NAV 캐시: Yahoo Finance 대량조회 속도제한(초기 시도에서 2회 연속 실행 시 776종목이
rate-limit로 실패)을 피하기 위해, 한 번 만든 algo_nav/spy_nav를
output/_algo_vs_spy_nav_cache.csv에 저장해두고 재실행 시 재사용한다.

실행: python algo_vs_spy_winrate_by_horizon.py
결과: output/algo_vs_spy_winrate_by_horizon.json (연속 곡선 데이터)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP

YEARS = 15   # 이 저장소에서 검증된 최대 구간(us_algo_15y_champion_check.json)
NAV_CACHE = "output/_algo_vs_spy_nav_cache.csv"
MIN_WINDOWS = 60   # 표본 60개 미만인 초장기 구간은 잘라냄


def _log(m): print(f"[승률계산] {m}", file=sys.stderr)


def _build_algo_vs_spy(years=YEARS):
    """SPMO 없이 algo_nav(topn8 라이브클립) + spy_nav만 새로 구축 — us_spmo_blend_prereg._load()가
    SPMO(2015-10 상장)와 교집합을 취해 앞쪽 구간을 잘라내는 것을 피하기 위함."""
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    decisions = SP._us_decisions_live_clip(panel, funds, pit)
    sector_of = SP._sector_of_factory()
    algo_nav = BP.simulate(panel, ma200, decisions, SP.TOPN, cost, ma200_backup=False,
                           sector_of=sector_of, sector_cap=2)
    if algo_nav is None:
        raise RuntimeError("topn=8 알고리즘 NAV 산출 실패")
    idx = algo_nav.index.intersection(spy.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spy_nav = spy.reindex(idx).ffill(); spy_nav = spy_nav / spy_nav.iloc[0]
    return algo_nav, spy_nav


def _load_algo_vs_spy(years=YEARS, use_cache=True):
    if use_cache and os.path.exists(NAV_CACHE):
        _log(f"NAV 캐시 재사용: {NAV_CACHE}")
        df = pd.read_csv(NAV_CACHE, parse_dates=["date"]).set_index("date")
        return df["algo_nav"], df["spy_nav"]
    algo_nav, spy_nav = _build_algo_vs_spy(years)
    os.makedirs("output", exist_ok=True)
    pd.DataFrame({"algo_nav": algo_nav, "spy_nav": spy_nav}).rename_axis("date").to_csv(NAV_CACHE)
    _log(f"NAV 캐시 저장: {NAV_CACHE}")
    return algo_nav, spy_nav


def run(save=True, use_cache=True) -> dict:
    algo_nav, spy_nav = _load_algo_vs_spy(years=YEARS, use_cache=use_cache)
    n = len(algo_nav)
    _log(f"공통 구간: {algo_nav.index[0].date()} ~ {algo_nav.index[-1].date()} "
         f"({n}거래일, {YEARS}년 요청)")

    algo_arr = algo_nav.to_numpy()
    spy_arr = spy_nav.to_numpy()
    h_max = n - MIN_WINDOWS
    horizon_days, win_rate_pct, n_windows, mean_excess_pct = [], [], [], []
    for h in range(1, h_max + 1):
        algo_ret = algo_arr[h:] / algo_arr[:-h] - 1
        spy_ret = spy_arr[h:] / spy_arr[:-h] - 1
        excess = algo_ret - spy_ret
        horizon_days.append(h)
        win_rate_pct.append(round(float((excess > 0).mean()) * 100, 2))
        n_windows.append(int(len(excess)))
        mean_excess_pct.append(round(float(excess.mean()) * 100, 2))
    _log(f"1~{h_max}거래일 전 구간 계산 완료(1일 단위 {len(horizon_days)}개 지점)")
    for probe_days in (21, 63, 126, 252, 504, 756, 1260, 2520):
        if probe_days <= h_max:
            i = probe_days - 1
            _log(f"  {probe_days}거래일: 승률 {win_rate_pct[i]}% (n={n_windows[i]})")

    payload = {
        "as_of": algo_nav.index[-1].date().isoformat(),
        "data_start": algo_nav.index[0].date().isoformat(),
        "years_requested": YEARS,
        "n_days": n,
        "config": "topn8 라이브 그대로(best_weights.json 1:2:2, sector_cap=2, "
                  "ma200_backup=False, shareholder_yield 클립 ±5)",
        "method": "1거래일 단위 겹치는(rolling) 윈도우 — 각 거래일을 시작점으로 h거래일 뒤 "
                  "수익률을 알고리즘·SPY 각각 계산해 알고리즘이 이긴 비율. 윈도우가 매일 겹쳐 "
                  "자기상관이 큼(독립시행 아님) — '아무 날짜에나 들어가서 h기간 보유하면 "
                  "이길 확률'이라는 질문에 대한 직접적인 경험적 추정치로 읽을 것, "
                  "PBO/DSR류 다중검정 게이트 대상 통계는 아님.",
        "caveat": "EDGAR 펀더멘털 커버리지가 15년 전 구간에서 100%가 아니라 결측 종목은 "
                  "해당 팩터 z=0(중립) 처리됨 — 완전한 PIT 표본은 아님(us_algo_15y_"
                  "champion_check.json과 동일 한계).",
        "horizon_days": horizon_days,
        "win_rate_pct": win_rate_pct,
        "n_windows": n_windows,
        "mean_excess_pct": mean_excess_pct,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/algo_vs_spy_winrate_by_horizon.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    run()
