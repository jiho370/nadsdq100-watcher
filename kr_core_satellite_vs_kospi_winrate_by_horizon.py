#!/usr/bin/env python3
"""
kr_core_satellite_vs_kospi_winrate_by_horizon.py — 코어65:새틀35(valuediv topn5, 라이브 그대로)
혼합 vs 코스피200(B1), 보유기간별 승리 확률 (2026-08-22, 지호 님 요청 — 재검증 세션).

kr_algo_vs_kospi_winrate_by_horizon.py는 새틀라이트(valuediv topn5) **단독** vs B1만 비교한다.
그 스크립트 자신의 caveat이 명시하듯, 이는 실제 라이브 배분(코어65:새틀35 권고, STRATEGY.md
§3 "포트폴리오 구성 권고")과 다른 조건이다(새틀라이트는 단일 트라이얼 우승자 게이트를 통과
못함 — PBO 77.8%·DSR 0.64). 이 스크립트는 core_satellite_kr.py의
regime_series/timed_nav/mix_nav/stats 헬퍼를 그대로 재사용해 실제 권고 구성(코어=B1 200일선
레짐타이밍 65% + 새틀라이트=valuediv topn5 35%, 월간 리밸)의 NAV를 만들고,
kr_algo_vs_kospi_winrate_by_horizon.py와 동일한 롤링윈도우 승률 방법론을 적용한다.

데이터: benchmarks_kr.load_research_data() — output/kr_panel_cache.pkl 캐시(있으면 재사용,
없으면 신규 pykrx+yfinance 수집). 새틀라이트 NAV는 kr_algo_vs_kospi_winrate_by_horizon.py와
동일 파이프라인으로 직접 재계산(valuediv 랭킹, topn=5, ma200_backup=False — §3 Stage 6 결론)
— output/kr_strategy_navs.json(valuediv_flow 변형, 2026-07-14 스냅샷·날짜 미갱신)은 쓰지 않는다.
코어는 B1(코스피200 시가총액가중)의 200일선 ±1% 히스테리시스·3일 확인 레짐(STRATEGY.md §1).

주의(버그 회피): regime_series/timed_nav는 반드시 "전체 이력" B1로 먼저 계산해야 200일
워밍업이 제대로 반영된다 — 새틀라이트 시작일(결정 격자 첫 지점, LOOKBACK=260거래일 이후)로
먼저 잘라낸 B1을 넣으면 워밍업이 새틀라이트 구간 안에서 다시 시작돼 그만큼 "기본 ON" 구간이
밀려들어온다. core_satellite_kr.run()과 동일한 순서(전체 계산 → mix_nav가 내부에서 교집합
정렬)를 따른다.

실행: python kr_core_satellite_vs_kospi_winrate_by_horizon.py
결과: output/kr_core_satellite_vs_kospi_winrate_by_horizon.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import core_satellite_kr as CSK

NAV_CACHE = "output/_kr_core_satellite_vs_kospi_nav_cache.csv"
MIN_WINDOWS = 60
TOPN = 5
PROBE_DAYS = (21, 63, 126, 252, 504, 756, 1260, 2520)


def _log(m): print(f"[코어새틀승률계산] {m}", file=sys.stderr)


def _build_navs():
    from benchmarks_kr import load_research_data, build_benchmarks
    import backtest_kr as BK
    import backtest_kr_strategies as KS

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    sat_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False)
    if sat_nav is None:
        raise RuntimeError("한국 topn=5 valuediv 새틀라이트 NAV 산출 실패")

    navs = build_benchmarks(panel, membership, mktcaps, bench)
    b1_full = navs["B1_kospi200"].dropna()

    regime = CSK.regime_series(b1_full)
    core_full = CSK.timed_nav(b1_full, regime)
    blend_nav = CSK.mix_nav(core_full, sat_nav, CSK.CORE_W, rebal=CSK.MONTH)

    idx = blend_nav.index
    b1 = b1_full.reindex(idx); b1 = b1 / b1.iloc[0]
    sat_cmp = sat_nav.reindex(idx); sat_cmp = sat_cmp / sat_cmp.iloc[0]
    core_cmp = core_full.reindex(idx); core_cmp = core_cmp / core_cmp.iloc[0]
    blend_nav = blend_nav / blend_nav.iloc[0]
    return blend_nav, b1, sat_cmp, core_cmp


def _load_navs(use_cache=True):
    if use_cache and os.path.exists(NAV_CACHE):
        _log(f"NAV 캐시 재사용: {NAV_CACHE}")
        df = pd.read_csv(NAV_CACHE, parse_dates=["date"]).set_index("date")
        return df["blend_nav"], df["kospi200_nav"], df["satellite_nav"], df["core_nav"]
    blend_nav, b1, sat_nav, core_nav = _build_navs()
    os.makedirs("output", exist_ok=True)
    pd.DataFrame({"blend_nav": blend_nav, "kospi200_nav": b1,
                  "satellite_nav": sat_nav, "core_nav": core_nav}
                 ).rename_axis("date").to_csv(NAV_CACHE)
    _log(f"NAV 캐시 저장: {NAV_CACHE}")
    return blend_nav, b1, sat_nav, core_nav


def _winrate_by_horizon(a: np.ndarray, b: np.ndarray, min_windows=MIN_WINDOWS):
    n = len(a)
    h_max = n - min_windows
    horizon_days, win_rate_pct, n_windows, mean_excess_pct = [], [], [], []
    for h in range(1, h_max + 1):
        a_ret = a[h:] / a[:-h] - 1
        b_ret = b[h:] / b[:-h] - 1
        excess = a_ret - b_ret
        horizon_days.append(h)
        win_rate_pct.append(round(float((excess > 0).mean()) * 100, 2))
        n_windows.append(int(len(excess)))
        mean_excess_pct.append(round(float(excess.mean()) * 100, 2))
    return horizon_days, win_rate_pct, n_windows, mean_excess_pct


def run(save=True, use_cache=True) -> dict:
    blend_nav, bench_nav, sat_nav, core_nav = _load_navs(use_cache=use_cache)
    n = len(blend_nav)
    _log(f"공통 구간: {blend_nav.index[0]} ~ {blend_nav.index[-1]} ({n}거래일)")

    horizon_days, win_rate_pct, n_windows, mean_excess_pct = _winrate_by_horizon(
        blend_nav.to_numpy(), bench_nav.to_numpy())
    h_max = len(horizon_days)
    _log(f"1~{h_max}거래일 전 구간 계산 완료(1일 단위 {len(horizon_days)}개 지점)")
    for probe_days in PROBE_DAYS:
        if probe_days <= h_max:
            i = probe_days - 1
            _log(f"  {probe_days}거래일: 승률 {win_rate_pct[i]}% (n={n_windows[i]})")

    # 참고용: B1·새틀라이트 단독·코어 단독·혼합의 CAGR/샤프/MDD를 동일 서브기간 태그로
    # (§3 Stage 2가 인용한 8년 기준 수치와 비교하기 위해 core_satellite_kr.SUBS 그대로 재사용)
    stat_rows = {name: {tag: CSK.stats(nav, a, b) for tag, a, b in CSK.SUBS}
                 for name, nav in (("B1_kospi200", bench_nav),
                                   ("satellite_valuediv_topn5", sat_nav),
                                   ("core_timed", core_nav),
                                   ("core65_sat35_blend", blend_nav))}
    for tag, a, b in CSK.SUBS:
        f = stat_rows["core65_sat35_blend"][tag]
        fb = stat_rows["B1_kospi200"][tag]
        if f and fb:
            _log(f"  [{tag}] 혼합 CAGR {f['cagr_pct']}%/샤프 {f['sharpe']}/MDD {f['mdd_pct']}% "
                 f"vs B1 CAGR {fb['cagr_pct']}%/샤프 {fb['sharpe']}/MDD {fb['mdd_pct']}%")

    payload = {
        "as_of": blend_nav.index[-1].date().isoformat(),
        "data_start": blend_nav.index[0].date().isoformat(),
        "n_days": n,
        "config": f"코어(B1 200일선 레짐타이밍, OFF시 현금 0%) {CSK.CORE_W:.0%} + 새틀라이트"
                  f"(valuediv topn5 라이브 그대로, ma200_backup=False) {1 - CSK.CORE_W:.0%}, "
                  "월간(21거래일) 목표비중 리밸 vs 코스피200 시가총액가중 매수후보유(B1)",
        "method": "1거래일 단위 겹치는(rolling) 윈도우 — kr_algo_vs_kospi_winrate_by_horizon.py와 "
                  "동일 방법론(algo_vs_spy_winrate_by_horizon.py 미국판의 한국 이식). 윈도우가 "
                  "매일 겹쳐 자기상관이 큼(독립시행 아님) — PBO/DSR류 다중검정 게이트 대상 "
                  "통계는 아님.",
        "caveat": "이건 STRATEGY.md §3 '포트폴리오 구성 권고'의 실제 라이브 배분(코어65:새틀35)에 "
                  "대한 win-rate다. 새틀라이트 '단독' win-rate는 "
                  "kr_algo_vs_kospi_winrate_by_horizon.py 참고 — 새틀라이트는 그 자체로는 단일 "
                  "트라이얼 우승자 게이트를 통과 못함(PBO 77.8%·DSR 0.64, STRATEGY.md §3). "
                  "둘은 다른 질문에 답하며 서로 대체하지 않는다.",
        "horizon_days": horizon_days,
        "win_rate_pct": win_rate_pct,
        "n_windows": n_windows,
        "mean_excess_pct": mean_excess_pct,
        "stats_by_subperiod": stat_rows,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/kr_core_satellite_vs_kospi_winrate_by_horizon.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    run()
