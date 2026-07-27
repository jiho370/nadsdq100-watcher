#!/usr/bin/env python3
"""
kr_algo_vs_kospi_winrate_by_horizon.py — 한국 알고리즘(valuediv topn5 라이브 그대로)
vs 코스피200(B1), 보유기간별 승리 확률 — 1거래일 단위 연속 (2026-07-26, 지호 님 요청,
algo_vs_spy_winrate_by_horizon.py의 미국판과 동일한 방식을 한국에 적용)

데이터: benchmarks_kr.load_research_data()는 output/kr_panel_cache.pkl이 있으면 그 캐시를
그대로 쓴다 — 이 저장소에 이미 §6-E에서 만든 12년 캐시(2014-07-16~2026-07-16, KRX가
코스피200 멤버십을 2014-05-01 이전은 제공하지 않아 이게 이 프로젝트의 "계산 가능한
시점")가 있어 그걸 재사용한다(추가 pykrx 재수집 없음). 알고리즘 NAV는
algo_vs_index_lenient.run_kr()과 동일 파이프라인(valuediv 랭킹, topn=5,
ma200_backup=False — §3 Stage 6 결론) · 벤치마크는 benchmarks_kr.build_benchmarks()의
B1_kospi200(코스피200 시가총액가중, 리밸런싱 없는 매수후보유)을 그 12년 패널로 직접
재계산(캐시된 output/benchmarks_kr.json은 8년치라 미사용).

방법·NAV 캐시: algo_vs_spy_winrate_by_horizon.py와 동일 — h=1,2,3,…거래일 전부에 대해
겹치는 롤링 윈도우로 승률 계산, NAV는 output/_kr_algo_vs_kospi_nav_cache.csv에 캐시.

실행: python kr_algo_vs_kospi_winrate_by_horizon.py
결과: output/kr_algo_vs_kospi_winrate_by_horizon.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP

YEARS_NOTE = "output/kr_panel_cache.pkl 캐시 그대로(12년, 2014-07~2026-07 — KRX 코스피200 멤버십 하한)"
NAV_CACHE = "output/_kr_algo_vs_kospi_nav_cache.csv"
MIN_WINDOWS = 60
TOPN = 5


def _log(m): print(f"[한국승률계산] {m}", file=sys.stderr)


def _build_algo_vs_kospi():
    from benchmarks_kr import load_research_data, build_benchmarks
    import backtest_kr as BK
    import backtest_kr_strategies as KS

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    algo_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False)
    if algo_nav is None:
        raise RuntimeError("한국 topn=5 valuediv 알고리즘 NAV 산출 실패")

    navs = build_benchmarks(panel, membership, mktcaps, bench)
    b1 = navs["B1_kospi200"].dropna()

    idx = algo_nav.index.intersection(b1.index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    b1 = b1.reindex(idx); b1 = b1 / b1.iloc[0]
    return algo_nav, b1


def _load_algo_vs_kospi(use_cache=True):
    if use_cache and os.path.exists(NAV_CACHE):
        _log(f"NAV 캐시 재사용: {NAV_CACHE}")
        df = pd.read_csv(NAV_CACHE, parse_dates=["date"]).set_index("date")
        return df["algo_nav"], df["kospi200_nav"]
    algo_nav, b1 = _build_algo_vs_kospi()
    os.makedirs("output", exist_ok=True)
    pd.DataFrame({"algo_nav": algo_nav, "kospi200_nav": b1}).rename_axis("date").to_csv(NAV_CACHE)
    _log(f"NAV 캐시 저장: {NAV_CACHE}")
    return algo_nav, b1


def run(save=True, use_cache=True) -> dict:
    algo_nav, bench_nav = _load_algo_vs_kospi(use_cache=use_cache)
    n = len(algo_nav)
    _log(f"공통 구간: {algo_nav.index[0].date()} ~ {algo_nav.index[-1].date()} ({n}거래일)")

    algo_arr = algo_nav.to_numpy()
    bench_arr = bench_nav.to_numpy()
    h_max = n - MIN_WINDOWS
    horizon_days, win_rate_pct, n_windows, mean_excess_pct = [], [], [], []
    for h in range(1, h_max + 1):
        algo_ret = algo_arr[h:] / algo_arr[:-h] - 1
        bench_ret = bench_arr[h:] / bench_arr[:-h] - 1
        excess = algo_ret - bench_ret
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
        "n_days": n,
        "config": "한국 valuediv topn5 라이브 그대로(ma200_backup=False, §3 Stage 6) vs "
                  "코스피200 시가총액가중 매수후보유(B1)",
        "years_note": YEARS_NOTE,
        "method": "1거래일 단위 겹치는(rolling) 윈도우 — 미국판(algo_vs_spy_winrate_by_"
                  "horizon.py)과 동일 방법론. 윈도우가 매일 겹쳐 자기상관이 큼(독립시행 아님) "
                  "— PBO/DSR류 다중검정 게이트 대상 통계는 아님.",
        "caveat": "valuediv 전략 자체가 STRATEGY.md §3에서 단일 트라이얼 우승자 게이트를 "
                  "통과 못함(PBO 77.8%·DSR 0.64) — 코어-새틀라이트 구조로만 조건부 채택된 "
                  "전략이라, 여기서 보는 '새틀라이트 단독' vs 코스피200 비교는 실제 라이브 "
                  "배분(코어65:새틀35 권고)과 다른 조건(새틀라이트 100%)임에 유의.",
        "horizon_days": horizon_days,
        "win_rate_pct": win_rate_pct,
        "n_windows": n_windows,
        "mean_excess_pct": mean_excess_pct,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/kr_algo_vs_kospi_winrate_by_horizon.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    run()
