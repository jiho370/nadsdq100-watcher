#!/usr/bin/env python3
"""
us_topn_1to10_sweep.py — 지호 님 질문(2026-09-24): "top 1~9 다 의미 없었다는 거지? 캡 적용해도?"

과거 topN 비교(HISTORY §2·§9-K-9)는 결함 데이터(분할 미보정·PIT 종료종목 삭제) 위였고, 결함 수정 후엔
top6/8/10 이벤트 단위만 있었다. 여기서 결함 수정 패널·라이브 선정(1:2:2·live_z·floor·100일선)으로
top 1~10 전부를 계좌 NAV로 비교한다. "캡"은 두 의미 모두:

  A. 라이브 엔진(backtest_portfolio.simulate — 보유상한 N·180일+후보풀60 재평가·빈 슬롯 충원, 10거래일 판정)
     × 섹터캡 없음 / 섹터캡 2(현재 GICS — PIT 아님)
  B. 분기 리밸런싱 3슬리브(매번 상위 N 재구성) × 동일비중 / 시가총액 비중(선정 종목 안에서, 상한 없음)
     필터 통과가 N보다 적으면 빈 자리는 현금(두 방식 동일).

지표: CAGR·샤프·MDD(현금=BIL, 비용 반영), SPY 대비 월수익 짝지은 블록부트스트랩, 구간(~2022·2023~),
40개 설정 전체의 월 초과수익(vs SPY)으로 PBO/DSR.

실행: python -m research.us.us_topn_1to10_sweep
결과: output/us_topn_1to10_sweep.json
"""
from __future__ import annotations
import json
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import research.us.us_capweight_momentum_validation as V

TOPNS = list(range(1, 11))
SLEEVES = 3


def _log(m): print(f"[topN1~10]  {m}", file=sys.stderr)


def run(years: float = 10) -> dict:
    import sp500_daily_report as R
    import tech_factors as T
    import backtest_exec as BE
    from research.us import backtest_portfolio as BP
    from research.us.us_full_stack_exec_validation import _live_ranked

    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "QQQ", "BIL"], period=f"{int(years)}y",
                                            drop_stale=False)).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    try:
        sector_map = R.fetch_wikipedia_sectors()
    except Exception:
        sector_map = {}
    idx = panel.index

    cache = {}

    def ranked(p):
        if p not in cache:
            r = _live_ranked(panel, p, funds, cross, pit, weights)
            cache[p] = None if r is None else [s for s in r if s not in dupes]
        return cache[p]

    # ---------- A. 라이브 엔진 ----------
    dec = [(p, ranked(p)[:60]) for p in range(BW.LOOKBACK, len(idx) - 1, 10) if ranked(p)]
    ma200 = panel.rolling(200, min_periods=200).mean()
    sector_of = lambda d, s: sector_map.get(s, "NA")
    navs = {}
    for n in TOPNS:
        navs[f"A_live_top{n}"] = BP.simulate(panel, ma200, dec, n, V.COST, ma200_backup=False)
        navs[f"A_live_top{n}_sectorcap2"] = BP.simulate(panel, ma200, dec, n, V.COST, ma200_backup=False,
                                                        sector_of=sector_of, sector_cap=2)
    _log(f"A 엔진 완료(판정 {len(dec)}회)")

    # ---------- B. 분기 3슬리브 ----------
    months = V.month_starts(idx, BW.LOOKBACK)
    mcap = {}

    def caps(p, syms):
        d = idx[p].date().isoformat()
        out = {}
        for s in syms:
            if (p, s) not in mcap:
                sh = V.shares_asof(funds.get(s) or {}, d)
                mcap[(p, s)] = sh * float(panel.iloc[p][s]) if sh else np.nan
            out[s] = mcap[(p, s)]
        return pd.Series(out, dtype=float)

    def target(p, n, kind):
        r = ranked(p) or []
        top = r[:n]
        if not top:
            return {}
        invested = len(top) / n                     # 필터 통과가 n보다 적으면 나머지는 현금
        if kind == "eq":
            return {s: invested / len(top) for s in top}
        m = caps(p, top).dropna()
        m = m[m > 0]
        if m.empty:
            return {s: invested / len(top) for s in top}
        return {s: invested * float(v / m.sum()) for s, v in m.items()}

    for n in TOPNS:
        for kind in ("eq", "cap"):
            sl = []
            for k in range(SLEEVES):
                sched = [(p + 1, target(p, n, kind)) for p in months[k::SLEEVES]]
                sl.append(V.sleeve_nav(panel, rf, sched))
            t0 = max(s.index[0] for s in sl)
            navs[f"B_q_top{n}_{kind}"] = sum(s.loc[t0:] / s.loc[t0] for s in sl) / SLEEVES
    _log("B 엔진 완료")

    navs = {k: v for k, v in navs.items() if v is not None}
    t0 = max(v.index[0] for v in navs.values())
    for k in ("SPY", "QQQ"):
        navs[k] = etf[k]
    navs = {k: v.loc[t0:] / v.loc[t0] for k, v in navs.items()}
    windows = {"full": (None, None), "to_2022": (None, "2022-12-31"), "2023_on": ("2023-01-01", None)}
    tbl = {w: {k: V.metrics(v.loc[a:b] / v.loc[a:b].iloc[0], rf) for k, v in navs.items()}
           for w, (a, b) in windows.items()}
    m = {k: V.monthly(v) for k, v in navs.items()}
    rf_m = (1 + rf).resample("ME").prod().sub(1).reindex(m["SPY"].index)
    trials = [k for k in navs if k not in ("SPY", "QQQ")]
    boot = {k: V.paired_bootstrap(m[k], m["SPY"], rf_m, n=4000) for k in trials}
    ex = pd.DataFrame({k: m[k] - m["SPY"] for k in trials}).dropna()
    rep = OS.analyze({"horizon": "topn_1to10", "universe": "pit", "cost": V.COST.describe(),
                      "rebal_days": 21, "hold_days": 21, "trials": list(ex.columns),
                      "excess_returns": ex.T.values.tolist()}, save=False)
    for k in trials + ["SPY", "QQQ"]:
        f, a, b = tbl["full"][k], tbl["to_2022"][k], tbl["2023_on"][k]
        extra = (f" | P(샤프>SPY) {boot[k]['p_sharpe_better']:.2f} P(CAGR>SPY) {boot[k]['p_cagr_better']:.2f}"
                 if k in boot else "")
        _log(f"{k:26s} {f['cagr_pct']:6.2f}%/{f['sharpe']:.2f}/{f['mdd_pct']:6.1f} | ~2022 {a['cagr_pct']:6.2f}/{a['sharpe']:.2f}"
             f" | 2023~ {b['cagr_pct']:6.2f}/{b['sharpe']:.2f}{extra}")
    _log(f"PBO {rep['pbo']['pbo']} · DSR {rep['dsr'].get('dsr')} · 최고 {rep['dsr']['best_trial']} · 통과 {rep['passed']}")
    out = {"as_of": idx[-1].date().isoformat(), "window_start": t0.date().isoformat(),
           "engines": {"A": "live sticky (cap N, 180d+pool60 reeval, 10d decisions), equal slot weight",
                       "B": "quarterly full reconstitution, 3 staggered sleeves"},
           "sector_map": "current GICS (not PIT)", "windows": tbl, "bootstrap_vs_spy": boot,
           "pbo_dsr_vs_spy": {"n_trials": len(trials), "pbo": rep["pbo"]["pbo"], "dsr": rep["dsr"].get("dsr"),
                              "best": rep["dsr"]["best_trial"], "passed": rep["passed"]}}
    with open("output/us_topn_1to10_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_topn_1to10_sweep.json")
    return out


if __name__ == "__main__":
    run()
