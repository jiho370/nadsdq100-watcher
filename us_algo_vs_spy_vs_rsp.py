#!/usr/bin/env python3
"""
us_algo_vs_spy_vs_rsp.py — 알고리즘(topn8 라이브) vs S&P500 시총가중(SPY) vs S&P500
동일가중(RSP) 3방식 비교 (지호 님 요청, 2026-07-29).

배경: SPMO 블렌드는 이미 사전등록 검증에서 기각됐고(§6-H), 현재 라이브 알고리즘은 그냥
S&P500(SPY, 시총가중) 대비로만 비교돼왔다. "동일가중 S&P500과 비교하면 어떤지"도 보자는
요청 — 시총가중 지수는 대형주(특히 메가캡) 쏠림이 커서, 우리 알고리즘(섹터캡2·8종목
분산)과 비교할 때 동일가중이 더 "공정한 분산도 비교"가 될 수 있다는 문제의식.

방법: topn8 라이브 챔피언(가중치 1:2:2, 섹터캡2, ma200_backup=False — us_spmo_blend_prereg의
레시피 그대로 재사용) NAV를 9년 구축하고, SPY·RSP(Invesco S&P500 Equal Weight ETF, 2003년
상장이라 9년 구간 문제없음) 원시가격(buy&hold) NAV와 나란히 비교. 페어드 비교(월간 비중첩
수익률, 6개월 블록부트스트랩 5000회 — 이 세션의 다른 사전등록 스크립트와 동일 방법론)로
알고리즘-SPY·알고리즘-RSP 차이의 유의성까지 확인.

실행: python us_algo_vs_spy_vs_rsp.py [--years 9]
결과: output/us_algo_vs_spy_vs_rsp.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP

YEARS = 9
TOPN = 8
BLOCK = SP.BLOCK
N_BOOT = SP.N_BOOT
SEED = SP.SEED
SUBS = SP.SUBS


def _log(m): print(f"[ALGO-vs-SPY-vs-RSP] {m}", file=sys.stderr)


def _algo_nav(years=YEARS):
    pit = BC.load_pit()
    panel, spy, _opens = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    decisions = SP._us_decisions_live_clip(panel, funds, pit)
    sector_of = SP._sector_of_factory()
    ma200 = panel.rolling(200, min_periods=200).mean()
    nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False,
                      sector_of=sector_of, sector_cap=2)
    if nav is None:
        raise RuntimeError("알고리즘 NAV 산출 실패")
    return nav, spy


def _etf_nav(ticker: str, index_like: pd.Series) -> pd.Series | None:
    """buy&hold NAV(1.0 시작), 알고리즘 NAV와 같은 날짜 인덱스로 정규화."""
    hist = SP.R.download_histories([ticker], period="max").get(ticker)
    if hist is None or hist.empty:
        return None
    s = hist.reindex(index_like.index).ffill().dropna()
    if s.empty:
        return None
    return s / s.iloc[0]


def run(years=YEARS, save=True):
    algo_nav, _spy_from_panel = _algo_nav(years)
    spy_nav = _etf_nav("SPY", algo_nav)
    rsp_nav = _etf_nav("RSP", algo_nav)
    if spy_nav is None or rsp_nav is None:
        raise RuntimeError("SPY/RSP 시세 조회 실패")

    idx = algo_nav.index.intersection(spy_nav.index).intersection(rsp_nav.index)
    if len(idx) < 60:
        raise RuntimeError(f"공통구간 부족(n={len(idx)})")
    algo_nav, spy_nav, rsp_nav = algo_nav.reindex(idx), spy_nav.reindex(idx), rsp_nav.reindex(idx)
    spy_nav, rsp_nav = spy_nav / spy_nav.iloc[0], rsp_nav / rsp_nav.iloc[0]

    algo_full = SP.CS.stats(algo_nav)
    spy_full = SP.CS.stats(spy_nav)
    rsp_full = SP.CS.stats(rsp_nav)
    _log(f"알고리즘(topn8): CAGR {algo_full['cagr_pct']}% 샤프 {algo_full['sharpe']} MDD {algo_full['mdd_pct']}%")
    _log(f"S&P500 시총가중(SPY): CAGR {spy_full['cagr_pct']}% 샤프 {spy_full['sharpe']} MDD {spy_full['mdd_pct']}%")
    _log(f"S&P500 동일가중(RSP): CAGR {rsp_full['cagr_pct']}% 샤프 {rsp_full['sharpe']} MDD {rsp_full['mdd_pct']}%")

    def _paired(nav_a, nav_b, label):
        r_a, r_b = SP._monthly_returns(nav_a), SP._monthly_returns(nav_b)
        n = min(len(r_a), len(r_b))
        r_a, r_b = r_a[:n], r_b[:n]
        tstat, pval = SP._paired_ttest(r_a, r_b)
        rng = np.random.default_rng(SEED)
        n_blocks_needed = int(np.ceil(n / BLOCK))
        cagr_diffs = np.empty(N_BOOT)
        sharpe_diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
            bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
            cagr_diffs[i] = SP._cagr_from_monthly(r_a[bidx]) - SP._cagr_from_monthly(r_b[bidx])
            sharpe_diffs[i] = SP._sharpe_from_monthly(r_a[bidx]) - SP._sharpe_from_monthly(r_b[bidx])
        cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
        cagr_mean, cagr_pos = float(cagr_diffs.mean()), float((cagr_diffs > 0).mean()) * 100
        sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
        _log(f"[{label}] 페어드 t={tstat:+.2f} p={pval:.3f} · CAGR차이 95%CI [{cagr_lo:+.2f}%p,"
             f"{cagr_hi:+.2f}%p] 평균{cagr_mean:+.2f}%p({cagr_pos:.1f}% 양수) · 샤프차이 95%CI "
             f"[{sharpe_lo:+.3f},{sharpe_hi:+.3f}]")
        return {"n": n, "paired_ttest": {"t": round(float(tstat), 3), "p": round(float(pval), 4)},
                "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                        "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1)},
                "sharpe_diff_bootstrap": {"mean": round(float(sharpe_diffs.mean()), 3),
                                         "ci95_lo": round(sharpe_lo, 3), "ci95_hi": round(sharpe_hi, 3)}}

    vs_spy = _paired(algo_nav, spy_nav, "algo-vs-SPY")
    vs_rsp = _paired(algo_nav, rsp_nav, "algo-vs-RSP")

    sub_rows = []
    for label, a, b in SUBS:
        sa, ss, sr = SP.CS.stats(algo_nav, a, b), SP.CS.stats(spy_nav, a, b), SP.CS.stats(rsp_nav, a, b)
        if not (sa and ss and sr):
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        row = {"period": label, "algo_cagr": sa["cagr_pct"], "spy_cagr": ss["cagr_pct"],
              "rsp_cagr": sr["cagr_pct"], "algo_sharpe": sa["sharpe"], "spy_sharpe": ss["sharpe"],
              "rsp_sharpe": sr["sharpe"]}
        sub_rows.append(row)
        _log(f"{label}: 알고리즘 {sa['cagr_pct']}%/{sa['sharpe']} · SPY {ss['cagr_pct']}%/{ss['sharpe']} "
             f"· RSP {sr['cagr_pct']}%/{sr['sharpe']}")

    payload = {"as_of": algo_nav.index[-1].date().isoformat(), "years": years,
              "full_period": {"algo": algo_full, "spy": spy_full, "rsp": rsp_full},
              "algo_vs_spy": vs_spy, "algo_vs_rsp": vs_rsp, "subperiods": sub_rows,
              "note": "SPY=시총가중 S&P500(buy&hold, 레짐타이밍 없음), RSP=Invesco S&P500 "
                      "동일가중 ETF(buy&hold). 알고리즘=topn8 라이브(1:2:2·섹터캡2·"
                      "ma200_backup=False)."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_algo_vs_spy_vs_rsp.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_algo_vs_spy_vs_rsp.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=YEARS)
    a = ap.parse_args()
    run(a.years)
