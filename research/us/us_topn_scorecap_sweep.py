#!/usr/bin/env python3
"""
us_topn_scorecap_sweep.py — 지호 님 질문(2026-09-24): "섹터캡 말고 상한캡" — 종합점수가 너무 높은(극단) 종목을
제외하는 점수 상한(한국 라이브 kr_stocks.SCORE_CAP과 같은 개념)을 top 1~10과 교차 검증.

과거 us_factor_cap_extreme.py는 이벤트 평균·결함 데이터(분할 미보정·PIT 종료종목 삭제)였다. 여기선 결함 수정
패널, 라이브 선정(1:2:2·live_z·floor 3.25·100일선) 점수에서 상한 C 초과 종목을 뺀 순위로, 라이브 엔진
(backtest_portfolio.simulate — 보유상한 N·180일+후보풀60 재평가·빈 슬롯 충원, 10거래일 판정)을 돌린다.
상한은 신규 편입 후보와 재평가 후보풀 모두에 적용(라이브에 넣는다면 그렇게 동작하므로).

실행: python -m research.us.us_topn_scorecap_sweep
결과: output/us_topn_scorecap_sweep.json
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
CAPS = [None, 15, 12, 10, 9, 8, 7, 6, 5]


def _log(m): print(f"[점수상한]  {m}", file=sys.stderr)


def live_scores(panel, p, funds, cross, pit, weights):
    """us_full_stack_exec_validation._live_ranked와 같은 규칙, 점수까지 반환(필터 통과만, 내림차순)."""
    import export_data as E
    raw = BW._raw_frame(panel, p, funds, bool(funds), cross)
    if raw is None or raw.empty:
        return None
    idx = raw.index.intersection(BC.membership_asof(pit, panel.index[p].date().isoformat()))
    if not len(idx):
        return None
    raw = raw.loc[idx]
    w = {k: v for k, v in weights.items() if k in raw.columns}
    score = sum(float(v) * E.live_z(raw[k], k) for k, v in w.items())
    score = score[score >= E.SCORE_FLOOR]
    if "ma100_gap" in raw.columns:
        score = score[raw["ma100_gap"].reindex(score.index) > 0]
    return score.sort_values(ascending=False)


def run(years: float = 10) -> dict:
    import sp500_daily_report as R
    import tech_factors as T
    import backtest_exec as BE
    from research.us import backtest_portfolio as BP

    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "QQQ", "BIL"], period=f"{int(years)}y",
                                            drop_stale=False)).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    ma200 = panel.rolling(200, min_periods=200).mean()

    scores = {}
    for p in range(BW.LOOKBACK, len(panel) - 1, 10):
        s = live_scores(panel, p, funds, cross, pit, weights)
        if s is not None:
            scores[p] = s[[x for x in s.index if x not in dupes]]
    allsc = pd.concat(scores.values())
    _log(f"판정 {len(scores)}회 · 필터 통과 점수 분포 p50 {allsc.median():.2f} p90 {allsc.quantile(.9):.2f} "
         f"p99 {allsc.quantile(.99):.2f} max {allsc.max():.2f}")

    navs, binding = {}, {}
    for cap in CAPS:
        dec = []
        for p, s in scores.items():
            r = list((s if cap is None else s[s <= cap]).index)
            if r:
                dec.append((p, r[:60]))
        for n in TOPNS:
            key = f"top{n}_cap{'none' if cap is None else cap}"
            navs[key] = BP.simulate(panel, ma200, dec, n, V.COST, ma200_backup=False)
            if cap is not None:   # 상한이 상위 n 구성을 바꾼 판정 비율
                binding[key] = round(100 * float(np.mean([list(s.index[:n]) != list(s[s <= cap].index[:n])
                                                          for s in scores.values()])), 1)
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
    boot = {k: V.paired_bootstrap(m[k], m["SPY"], rf_m, n=3000) for k in trials}
    ex = pd.DataFrame({k: m[k] - m["SPY"] for k in trials}).dropna()
    rep = OS.analyze({"horizon": "topn_scorecap", "universe": "pit", "cost": V.COST.describe(),
                      "rebal_days": 21, "hold_days": 21, "trials": list(ex.columns),
                      "excess_returns": ex.T.values.tolist()}, save=False)
    for cap in CAPS:
        c = "none" if cap is None else cap
        cells = []
        for n in TOPNS:
            f = tbl["full"].get(f"top{n}_cap{c}")
            cells.append(f"{f['cagr_pct']:5.1f}/{f['sharpe']:.2f}" if f else "  -  ")
        _log(f"cap {str(c):>4s} | " + " ".join(cells))
    _log(f"PBO {rep['pbo']['pbo']} · DSR {rep['dsr'].get('dsr')} · 최고 {rep['dsr']['best_trial']} · 통과 {rep['passed']}")
    out = {"as_of": panel.index[-1].date().isoformat(), "window_start": t0.date().isoformat(),
           "score_dist": {"p50": round(float(allsc.median()), 2), "p90": round(float(allsc.quantile(.9)), 2),
                          "p99": round(float(allsc.quantile(.99)), 2), "max": round(float(allsc.max()), 2)},
           "windows": tbl, "binding_pct": binding, "bootstrap_vs_spy": boot,
           "pbo_dsr_vs_spy": {"n_trials": len(trials), "pbo": rep["pbo"]["pbo"], "dsr": rep["dsr"].get("dsr"),
                              "best": rep["dsr"]["best_trial"], "passed": rep["passed"]}}
    with open("output/us_topn_scorecap_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_topn_scorecap_sweep.json")
    return out


if __name__ == "__main__":
    run()
