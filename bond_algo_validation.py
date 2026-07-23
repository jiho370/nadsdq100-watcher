#!/usr/bin/env python3
"""
bond_algo_validation.py — "채권 알고리즘" 다각도 검증 (2026-07-23, 지호 님 요청)

배경: STRATEGY.md §4/§6-G 확인 결과, 현재 "채권"은 사실상 알고리즘이 없다 — weekly_report.py가
IEF(미국채 7-10년)를 안정형 40%/공격형 15% **고정비중**으로 넣고 Daryanani(2008) 밴드(±20%)
리밸런싱만 할 뿐, 주식·코인·금과 달리 추세 신호(market_signals.py)가 전혀 안 걸려 있다.
§6-G 열린 실 ③ "채권/현금성 코어 — 아직 한 번도 검증 안 함"에 대한 최초 검증.

4개 스테이지:
  1) 채권 ETF 5종(SHY/IEF/TLT/AGG/BND) 단독 성과 + SPY와의 상관관계(전체·SPY 하락일 조건부)
  2) 각 ETF × 비중(0~100%) 블렌드 스윕 — SPY와 섞었을 때 CAGR/샤프/MDD, PBO/DSR(다중검정)
  3) 위기구간(2008 GFC·2020 코로나·2022 금리인상) 조건부 드릴다운 — 현행(IEF)이 실제로
     "위기 시 완충"이라는 존재 이유를 다하는지
  4) 채권에도 주식과 동일한 추세필터(200일선 히스테리시스)를 걸면 나아지는가 — 단일
     사전등록 비교(페어드 블록부트스트랩)

데이터: SHY/IEF/TLT/AGG/BND/SPY 공통구간(BND 상장 2007-04 이후, 2008/2020/2022 위기 다 포함).

실행: python bond_algo_validation.py
결과: output/bond_algo_*.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import overfit_stats as OS
import core_satellite_kr as CS
import sp500_daily_report as R

BONDS = ["SHY", "IEF", "TLT", "AGG", "BND"]
RATIO_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 비중=채권 비중
MONTH = 21
BLOCK = 6
N_BOOT = 5000
SEED = 42
CRISES = [("2008 GFC", "2007-10-01", "2009-03-31"),
          ("2020 코로나", "2020-02-19", "2020-03-23"),
          ("2022 금리인상", "2022-01-01", "2022-10-14")]


def _log(m): print(f"[채권검증] {m}", file=sys.stderr)


def _load():
    hist = R.download_histories(BONDS + ["SPY"], period="max")

    def series_of(sym):
        s = hist.get(sym)
        if s is None or s.empty:
            return None
        return s.dropna()

    series = {sym: series_of(sym) for sym in BONDS + ["SPY"]}
    common_start = max(s.index[0] for s in series.values())
    navs = {}
    for sym, s in series.items():
        s = s.loc[common_start:]
        navs[sym] = s / s.iloc[0]
    _log(f"공통구간: {common_start.date()} ~ {navs['SPY'].index[-1].date()} "
         f"({len(navs['SPY'])}거래일)")
    return navs


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


# ------------------------- Stage 1: 단독 성과 + 상관관계 -------------------------
def stage1(navs: dict, save=True) -> dict:
    spy = navs["SPY"]
    r_spy = spy.pct_change().dropna()
    rows = []
    for sym in BONDS + ["SPY"]:
        nav = navs[sym]
        s = CS.stats(nav)
        r = nav.pct_change().dropna()
        idx = r.index.intersection(r_spy.index)
        corr_full = float(r.reindex(idx).corr(r_spy.reindex(idx)))
        down = r_spy.reindex(idx) < 0
        corr_down = float(r.reindex(idx)[down].corr(r_spy.reindex(idx)[down])) if down.sum() > 20 else None
        rows.append({"sym": sym, **s, "corr_vs_spy": round(corr_full, 3),
                    "corr_vs_spy_down_days": round(corr_down, 3) if corr_down is not None else None})
        _log(f"{sym}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}% · "
             f"SPY상관 {corr_full:.3f}(하락일 {corr_down})")
    payload = {"rows": rows, "note": "corr_vs_spy_down_days<0이면 '위기 시 완충' 역할 — 주식이 "
                                     "빠질 때 반대로 움직인다는 뜻"}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/bond_algo_stage1_standalone.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


# ------------------------- Stage 2: ETF × 비중 블렌드 스윕 -------------------------
def stage2(navs: dict, save=True) -> dict:
    spy = navs["SPY"]
    results = {}
    for bond_sym in BONDS:
        bond = navs[bond_sym]
        idx = spy.index.intersection(bond.index)
        spy_a, bond_a = spy.reindex(idx), bond.reindex(idx)
        spy_a, bond_a = spy_a / spy_a.iloc[0], bond_a / bond_a.iloc[0]
        rows, matrix, dates0 = [], [], None
        for w in RATIO_LIST:   # w = 채권비중
            mixed = CS.mix_nav(spy_a, bond_a, 1 - w) if 0 < w < 1 else (bond_a if w == 1 else spy_a)
            s = CS.stats(mixed)
            rows.append({"bond_pct": int(round(w * 100)), "stock_pct": int(round((1 - w) * 100)), **s})
            d, r = _excess_vs_spy(mixed, spy_a)
            if dates0 is None:
                dates0 = d
            matrix.append(r[:len(dates0)])
        n_ev = min(len(r) for r in matrix)
        matrix = [r[:n_ev] for r in matrix]
        trial_data = {"horizon": f"bond_ratio_{bond_sym}", "universe": "spy_bond_blend",
                      "cost": "0bp(무비용 가정)", "rebal_days": MONTH, "hold_days": MONTH,
                      "dates": dates0[:n_ev], "trials": [f"bond{r['bond_pct']}" for r in rows],
                      "excess_returns": matrix}
        rpt = OS.analyze(trial_data, save=False)
        best = max(rows, key=lambda r: r["sharpe"])
        _log(f"{bond_sym}: 샤프 최고점 채권{best['bond_pct']}%(샤프 {best['sharpe']}) · "
             f"PBO {rpt.get('pbo', {}).get('pbo')} · DSR {rpt.get('dsr', {}).get('dsr')}")
        results[bond_sym] = {"rows": rows, "best_by_sharpe": best,
                             "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
                             "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
                             "passed": rpt.get("passed", False)}
    if save:
        with open("output/bond_algo_stage2_ratio_sweep.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _excess_vs_spy(nav: pd.Series, spy: pd.Series) -> tuple[list, list]:
    b = spy.reindex(nav.index).ffill()
    out_d, out_r = [], []
    for t in range(0, len(nav) - MONTH, MONTH):
        r = float(nav.iloc[t + MONTH] / nav.iloc[t] - 1)
        rb = float(b.iloc[t + MONTH] / b.iloc[t] - 1)
        out_d.append(nav.index[t + MONTH].date().isoformat())
        out_r.append(round(r - rb, 6))
    return out_d, out_r


# ------------------------- Stage 3: 위기구간 드릴다운 -------------------------
def stage3(navs: dict, save=True) -> dict:
    spy = navs["SPY"]
    out = {}
    for label, a, b in CRISES:
        row = {}
        for sym in BONDS + ["SPY"]:
            nav = navs[sym]
            w = nav.loc[a:b]
            if len(w) < 3:
                row[sym] = None
                continue
            dd = float((w / w.cummax() - 1).min()) * 100
            ret = float(w.iloc[-1] / w.iloc[0] - 1) * 100
            row[sym] = {"mdd_pct": round(dd, 1), "period_return_pct": round(ret, 1)}
        out[label] = row
        _log(f"{label}: " + " · ".join(f"{s}={row[s]}" for s in row))
    if save:
        with open("output/bond_algo_stage3_crisis.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ------------------------- Stage 4: 채권에 추세필터(200일선) 적용 여부 -------------------------
def stage4(navs: dict, bond_sym="IEF", save=True) -> dict:
    """현행(고정보유) vs 채권도 200일선 히스테리시스 레짐타이밍 — 40:60(채권:SPY, 안정형과
    유사 비중)로 블렌드해서 비교. 단일 사전등록 비교(그리드 아님)."""
    spy = navs["SPY"]
    bond = navs[bond_sym]
    idx = spy.index.intersection(bond.index)
    spy_a, bond_a = spy.reindex(idx), bond.reindex(idx)
    spy_a, bond_a = spy_a / spy_a.iloc[0], bond_a / bond_a.iloc[0]

    reg = CS.regime_series(bond_a)   # 채권 자체에 200일선 히스테리시스
    bond_timed = CS.timed_nav(bond_a, reg)

    w_bond = 0.40   # 안정형 비중과 동일
    static_mix = CS.mix_nav(spy_a, bond_a, 1 - w_bond)
    timed_mix = CS.mix_nav(spy_a, bond_timed, 1 - w_bond)

    r_static = _monthly_returns(static_mix)
    r_timed = _monthly_returns(timed_mix)
    n = min(len(r_static), len(r_timed))
    r_static, r_timed = r_static[:n], r_timed[:n]
    tstat, pval = _paired_ttest(r_timed, r_static)

    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n / BLOCK))
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_t = float(np.prod(1 + r_timed[bidx]) ** (12 / n) - 1) * 100
        cagr_s = float(np.prod(1 + r_static[bidx]) ** (12 / n) - 1) * 100
        diffs[i] = cagr_t - cagr_s
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = float(diffs.mean())
    pct_pos = float((diffs > 0).mean()) * 100

    s_static, s_timed = CS.stats(static_mix), CS.stats(timed_mix)
    _log(f"고정보유(현행): CAGR {s_static['cagr_pct']}% 샤프 {s_static['sharpe']} MDD {s_static['mdd_pct']}%")
    _log(f"추세필터채권: CAGR {s_timed['cagr_pct']}% 샤프 {s_timed['sharpe']} MDD {s_timed['mdd_pct']}%")
    _log(f"CAGR차이(추세필터-고정) 95%CI [{lo:+.2f}, {hi:+.2f}] (평균{mean_diff:+.2f}, "
         f"{pct_pos:.1f}% 양수) · 페어드 t={tstat:+.2f}")

    payload = {"bond_sym": bond_sym, "w_bond": w_bond,
              "static_hold": s_static, "trend_timed": s_timed,
              "paired_ttest": {"t": round(tstat, 3), "p": round(pval, 4), "n": n},
              "cagr_diff_bootstrap": {"mean": round(mean_diff, 2), "ci95_lo": round(float(lo), 2),
                                      "ci95_hi": round(float(hi), 2), "pct_positive": round(pct_pos, 1)},
              "note": "가설: 채권에도 추세필터 적용이 유의하게 낫다. 방향 기준 게이트: "
                      "CI 하한>0 AND t>=+1.96 이어야 채택."}
    if save:
        with open("output/bond_algo_stage4_trend_filter.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def run_all():
    navs = _load()
    r1 = stage1(navs)
    r2 = stage2(navs)
    r3 = stage3(navs)
    r4 = stage4(navs, bond_sym="IEF")
    _log("=== 전체 완료 ===")
    return {"stage1": r1, "stage2": r2, "stage3": r3, "stage4": r4}


if __name__ == "__main__":
    run_all()
