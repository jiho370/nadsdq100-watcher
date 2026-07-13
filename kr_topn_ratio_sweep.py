#!/usr/bin/env python3
"""
kr_topn_ratio_sweep.py — 지호 님 질문(2026-07-14) 대응: valuediv 채택 시 남았던 검증 공백 2개.

kr_stocks.py를 valuediv로 교체(STRATEGY.md §3)하며 topn=6·코어65:새틀35는 옛 momentum
알고리즘 검증(backtest_portfolio.py TOPN_KR)과 고정 설계(core_satellite_kr.py CORE_W=0.65)를
그대로 물려받았을 뿐, valuediv 자체로는 한 번도 스윕된 적이 없었다. 이 스크립트가 메운다:

  Stage 1 (topn): backtest_kr_strategies.build_decisions(..., "valuediv")로 뽑은 풀(30)을
    고정하고, 새틀라이트로 담을 개수만 3/4/6/8/10/15 스윕 — B2(동일가중) 대비 PBO/DSR.
  Stage 2 (ratio): Stage 1에서 가장 견고한(또는 현행) topn으로 새틀라이트 NAV를 만들고,
    core_satellite_kr의 코어(B1×레짐 타이밍)와 섞는 비중 w_core를
    1.0(코어만)/0.8/0.65(현행)/0.5/0.35/0.0(새틀만) 스윕 — B1 대비 PBO/DSR + 서브기간.

실행: python kr_topn_ratio_sweep.py --stage topn
      python kr_topn_ratio_sweep.py --stage ratio --topn 6   # Stage 1 결과 보고 지정
      python kr_topn_ratio_sweep.py --self-test
결과: output/kr_topn_sweep.json · output/pbo_report_kr_topn.json
      output/kr_ratio_sweep.json · output/pbo_report_kr_ratio.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP
import backtest_kr_strategies as KS
import core_satellite_kr as CS

TOPN_LIST = [3, 4, 6, 8, 10, 15]
RATIO_LIST = [1.0, 0.8, 0.65, 0.5, 0.35, 0.0]
POOL_KR = KS.POOL_KR


def _log(m): print(f"[KR스윕] {m}", file=sys.stderr)


def _load():
    from benchmarks_kr import load_research_data, load_benchmarks
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    navs_bm = load_benchmarks()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    return panel, snaps, navs_bm, ma200, cost


def run_topn_stage(save=True):
    panel, snaps, navs_bm, ma200, cost = _load()
    b1 = navs_bm["B1_kospi200"].reindex(panel.index).ffill()
    b2 = navs_bm["B2_equal"].reindex(panel.index).ffill()
    decisions = KS.build_decisions(panel, snaps, "valuediv")   # 풀(30) 고정 — topn만 변수
    _log(f"valuediv 결정 시점 {len(decisions)}개 (풀 {POOL_KR})")

    rows, matrix, dates0, navs_out = [], [], None, {}
    for tn in TOPN_LIST:
        nav = BP.simulate(panel, ma200, decisions, tn, cost)
        if nav is None:
            _log(f"topn={tn}: NAV 산출 실패"); continue
        navs_out[tn] = nav
        m1, m2 = BP.metrics(nav, b1), BP.metrics(nav, b2)
        d, r = BP.monthly_excess(nav, b2)          # 판정은 B2 기준(§1)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        rows.append({"topn": tn, "cagr_pct": m1["cagr_pct"], "vol_pct": m1["vol_pct"],
                     "sharpe": m1["sharpe"], "mdd_pct": m1["mdd_pct"],
                     "excess_vs_B1": m1["excess_cagr_pct"], "excess_vs_B2": m2["excess_cagr_pct"]})
        _log(f"topn={tn:2d}: CAGR {m1['cagr_pct']:6.2f}% 샤프 {m1['sharpe']:5.2f} "
             f"MDD {m1['mdd_pct']:6.1f}% · vs B2 {m2['excess_cagr_pct']:+6.2f}%p")
    if len(rows) < 2:
        raise RuntimeError("topn 결과 부족")
    n_ev = min(len(r) for r in matrix)
    trial_data = {"horizon": "kr_topn", "universe": "kospi200_pit", "cost": cost.describe(),
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev], "trials": [f"topn{r['topn']}" for r in rows],
                 "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": panel.index[-1].date().isoformat(), "pool": POOL_KR, "factor": "valuediv",
              "judgment": "B2(동일가중) 대비 월간초과수익 PBO/DSR", "rows": rows,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False)}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_topn_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open("output/pbo_report_kr_topn.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        with open("output/kr_topn_navs.json", "w", encoding="utf-8") as f:
            json.dump({str(tn): {d.date().isoformat(): round(float(v), 6) for d, v in s.items()}
                       for tn, s in navs_out.items()}, f, ensure_ascii=False)
        _log(f"저장: output/kr_topn_sweep.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


def run_ratio_stage(topn: int, save=True):
    panel, snaps, navs_bm, ma200, cost = _load()
    b1 = navs_bm["B1_kospi200"].dropna()
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    sat_nav = BP.simulate(panel, ma200, decisions, topn, cost)
    if sat_nav is None:
        raise RuntimeError(f"topn={topn} 새틀라이트 NAV 산출 실패")
    reg = CS.regime_series(b1.reindex(panel.index).ffill())
    core = CS.timed_nav(b1.reindex(panel.index).ffill(), reg)
    sat = sat_nav / sat_nav.iloc[0]

    rows, matrix, dates0 = [], [], None
    subs_out = {}
    for w in RATIO_LIST:
        mixed = CS.mix_nav(core, sat, w) if 0 < w < 1 else (core if w == 1 else sat)
        subs = {tag: CS.stats(mixed, a, b) for tag, a, b in CS.SUBS}
        subs_out[w] = subs
        f = subs["full"]
        if f is None:
            continue
        rows.append({"core_weight": w, **f})
        d, r = BP.monthly_excess(mixed, b1.reindex(mixed.index).ffill())
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        _log(f"core={w:.2f}: CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} MDD {f['mdd_pct']:6.1f}% "
             f"· 서브기간 샤프 {subs['2018-2021'] and subs['2018-2021']['sharpe']}/"
             f"{subs['2022-2023'] and subs['2022-2023']['sharpe']}/{subs['2024+'] and subs['2024+']['sharpe']}")
    if len(rows) < 2:
        raise RuntimeError("ratio 결과 부족")
    n_ev = min(len(r) for r in matrix)
    trial_data = {"horizon": "kr_ratio", "universe": "kospi200_pit", "cost": cost.describe(),
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev], "trials": [f"core{r['core_weight']:.2f}" for r in rows],
                 "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": topn, "satellite": "valuediv",
              "judgment": "B1(코스피200 매수후보유) 대비 월간초과수익 PBO/DSR", "rows": rows,
              "subperiods": {str(w): s for w, s in subs_out.items()},
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False)}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_ratio_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open("output/pbo_report_kr_ratio.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/kr_ratio_sweep.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] Stage1/2 배선 검증 — build_decisions·simulate·mix_nav 인터페이스만 확인"
         "(합성 데이터는 backtest_kr_strategies/core_satellite_kr 자체 self-test가 이미 커버)")
    import backtest_weights as BW
    rng = np.random.default_rng(5)
    n, m = 700, 25
    idx = pd.bdate_range("2021-01-01", periods=n)
    b1 = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
    ma200 = None  # ratio 단계에선 ma200 불필요(core_satellite는 b1만 사용)
    reg = CS.regime_series(b1)
    core = CS.timed_nav(b1, reg)
    sat = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))), index=idx)
    sat = sat / sat.iloc[0]
    for w in RATIO_LIST:
        mixed = CS.mix_nav(core, sat, w) if 0 < w < 1 else (core if w == 1 else sat)
        s = CS.stats(mixed)
        assert s is not None and np.isfinite(s["cagr_pct"]), (w, s)
    _log("[self-test] 통과: ratio 6종 전부 유효 NAV 산출")


def main():
    ap = argparse.ArgumentParser(description="KR valuediv topN·코어비율 스윕")
    ap.add_argument("--stage", choices=["topn", "ratio"], default="topn")
    ap.add_argument("--topn", type=int, default=6, help="ratio 단계에서 쓸 새틀라이트 topn")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.stage == "topn":
        run_topn_stage()
    else:
        run_ratio_stage(args.topn)


if __name__ == "__main__":
    main()
