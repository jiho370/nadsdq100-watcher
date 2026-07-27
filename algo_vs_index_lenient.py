#!/usr/bin/env python3
"""
algo_vs_index_lenient.py — 미장·국장 현재 알고리즘 vs 지수, 관대한 기준 검증 (2026-07-23)

배경: 이 프로젝트의 표준 게이트(PBO/DSR)는 "여러 후보 중 이게 진짜 최고냐"를 묻는
엄격한 다중검정 보정이라, 지금처럼 이미 확정된 단일 설정(라이브 그대로)을 "그냥 지수보다
낫냐"고 묻는 데는 과도하게 엄격하다(bond_trend_filter_grid.py에서 지호 님이 지적한 것과
동일한 논리). 여기서는 그 대신 다음을 본다:
  ① 페어드 t검정(월간 초과수익 vs 0) — 단일 가설이라 다중검정 벌점 없음
  ② 짝지은 블록부트스트랩 95%CI(CAGR·샤프 차이)
  ③ 월간 승률(비중첩 21거래일 구간 중 지수를 이긴 비율) — "그냥 대체로 이기냐"는
     직관적 질문에 가장 가까운 관대한 지표
  ④ 서브기간 방향 일관성

대상: 미장(topn8, 라이브 가중치 1:2:2·섹터캡2·200일선백업없음) vs SPY,
     국장(valuediv topn5, 라이브 그대로) vs 코스피200(B1).

실행: python algo_vs_index_lenient.py
결과: output/algo_vs_index_us.json · output/algo_vs_index_kr.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import sp500_daily_report as R

MONTH = 21
BLOCK = 6
N_BOOT = 5000
SEED = 42


def _log(m): print(f"[알고vs지수] {m}", file=sys.stderr)


def _monthly_returns(nav: pd.Series) -> np.ndarray:
    return np.array([nav.iloc[t + MONTH] / nav.iloc[t] - 1
                     for t in range(0, len(nav) - MONTH, MONTH)])


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    n = len(d)
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = float(d.mean()) / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def _cagr_from_monthly(sample: np.ndarray) -> float:
    yrs = len(sample) / 12
    return float(np.prod(1 + sample) ** (1 / yrs) - 1) * 100


def _sharpe_from_monthly(sample: np.ndarray) -> float:
    return float(sample.mean() / sample.std() * np.sqrt(12)) if sample.std() else 0.0


def _analyze(algo_nav: pd.Series, bench_nav: pd.Series, subs: list, label: str) -> dict:
    idx = algo_nav.index.intersection(bench_nav.index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    bench_nav = bench_nav.reindex(idx).ffill(); bench_nav = bench_nav / bench_nav.iloc[0]

    def stats(nav, a=None, b=None):
        w = nav.loc[a:b] if (a or b) else nav
        if len(w) < 60:
            return None
        r = w.pct_change().dropna()
        yrs = len(w) / 252
        return {"cagr_pct": round(100 * float((w.iloc[-1] / w.iloc[0]) ** (1 / yrs) - 1), 2),
               "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() else 0.0,
               "mdd_pct": round(100 * float((w / w.cummax() - 1).min()), 1)}

    full_algo, full_bench = stats(algo_nav), stats(bench_nav)
    _log(f"[{label}] 알고리즘: CAGR {full_algo['cagr_pct']}% 샤프 {full_algo['sharpe']} MDD {full_algo['mdd_pct']}%")
    _log(f"[{label}] 지수: CAGR {full_bench['cagr_pct']}% 샤프 {full_bench['sharpe']} MDD {full_bench['mdd_pct']}%")

    r_algo = _monthly_returns(algo_nav)
    r_bench = _monthly_returns(bench_nav)
    n = min(len(r_algo), len(r_bench))
    r_algo, r_bench = r_algo[:n], r_bench[:n]

    tstat, pval = _paired_ttest(r_algo, r_bench)
    win_rate = float((r_algo > r_bench).mean()) * 100
    _log(f"[{label}] 페어드 t검정(월간, n={n}): t={tstat:+.2f} p={pval:.3f} · 월간승률 {win_rate:.1f}%")

    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BLOCK))
    cagr_diffs = np.empty(N_BOOT)
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = _cagr_from_monthly(r_algo[bidx]) - _cagr_from_monthly(r_bench[bidx])
        sharpe_diffs[i] = _sharpe_from_monthly(r_algo[bidx]) - _sharpe_from_monthly(r_bench[bidx])
    cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    cagr_mean = float(cagr_diffs.mean())
    cagr_pos = float((cagr_diffs > 0).mean()) * 100
    sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    _log(f"[{label}] CAGR차이(알고-지수) 95%CI [{cagr_lo:+.2f}, {cagr_hi:+.2f}] "
         f"(평균{cagr_mean:+.2f}, {cagr_pos:.1f}%양수) · 샤프차이 95%CI [{sharpe_lo:+.3f}, {sharpe_hi:+.3f}]")

    sub_rows, sub_wins = [], []
    for name, a, b in subs:
        sa, sb = stats(algo_nav, a, b), stats(bench_nav, a, b)
        if sa is None or sb is None:
            continue
        beat = sa["cagr_pct"] > sb["cagr_pct"]
        sub_wins.append(beat)
        sub_rows.append({"period": name, "algo_cagr": sa["cagr_pct"], "bench_cagr": sb["cagr_pct"],
                         "algo_beats": beat})
        _log(f"[{label}] {name}: 알고 {sa['cagr_pct']}% vs 지수 {sb['cagr_pct']}% · "
             f"{'승' if beat else '패'}")

    verdict_lenient = "지수 우위 확인(관대한 기준)" if (cagr_lo > 0 or (tstat >= 1.96 and win_rate >= 55)) else \
                      ("판정 보류" if cagr_hi > 0 else "지수가 유의하게 우위")

    return {"label": label, "n_months": n, "full_period": {"algo": full_algo, "bench": full_bench},
           "paired_ttest_monthly": {"t": round(tstat, 3), "p": round(pval, 4), "n": n},
           "monthly_win_rate_pct": round(win_rate, 1),
           "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                   "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1)},
           "sharpe_diff_bootstrap": {"ci95_lo": round(sharpe_lo, 3), "ci95_hi": round(sharpe_hi, 3)},
           "subperiods": sub_rows, "subperiod_win_count": f"{sum(sub_wins)}/{len(sub_wins)}",
           "verdict_lenient": verdict_lenient,
           "note": "관대한 기준: PBO/DSR(다중검정) 대신 페어드 t검정·부트스트랩CI·월간승률·서브기간"
                   "승패로 판단 — '이미 확정된 라이브 설정 하나가 지수보다 나은가'라는 단일 질문에"
                   " 맞는 통계(다중검정 벌점 없음)."}


def run_us(save=True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(10, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    ma200 = panel.rolling(200, min_periods=200).mean()
    decisions = BP.us_decisions(panel, funds, pit)   # best_weights.json(1:2:2) 라이브 그대로
    sector_map = R.fetch_wikipedia_sectors()
    sector_of = lambda date_s, sym: sector_map.get(sym)
    algo_nav = BP.simulate(panel, ma200, decisions, 8, cost, ma200_backup=False,
                           sector_of=sector_of, sector_cap=2)
    if algo_nav is None:
        raise RuntimeError("US algo NAV 실패")
    subs = [("2018-2021", None, "2021-12-31"), ("2022-2023", "2022-01-01", "2023-12-31"),
           ("2024+", "2024-01-01", None)]
    result = _analyze(algo_nav, spy, subs, "미장(topn8 vs SPY)")
    if save:
        with open("output/algo_vs_index_us.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_kr(save=True) -> dict:
    from benchmarks_kr import load_research_data, load_benchmarks
    import backtest_kr as BK
    import backtest_kr_strategies as KS
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    navs_bm = load_benchmarks()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    algo_nav = BP.simulate(panel, ma200, decisions, 5, cost, ma200_backup=False)   # 라이브: MA200백업 비활성
    if algo_nav is None:
        raise RuntimeError("KR algo NAV 실패")
    b1 = navs_bm["B1_kospi200"].dropna()
    yrs_available = (panel.index[-1] - panel.index[0]).days / 365
    subs = [("전반", None, str(panel.index[len(panel)//2].date())),
           ("후반", str(panel.index[len(panel)//2].date()), None)]
    result = _analyze(algo_nav, b1, subs, "국장(valuediv topn5 vs 코스피200)")
    if save:
        with open("output/algo_vs_index_kr.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    run_us()
    run_kr()
