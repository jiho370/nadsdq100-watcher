#!/usr/bin/env python3
"""
kr_candidate8_full_verification.py — KR topn=30 상위10 중 #8(라이브를 이겼던 후보)에 대해
미국 roa후보(§9-K-4~7)와 동일한 3단계 정밀검증 (2026-08-23):
  1) 짝지은(paired) 블록부트스트랩 — 일별수익 기준 CAGR·샤프 차이 유의성
  2) 이상치(소수 대박구간) 의존도 진단 — 63일 구간수익 분포
  3) topn=3~10 스윕 — 우위가 특정 topn에 국한된 게 아닌지 확인
(4번 "장기표본 재현"은 이미 12년 캐시로 검증됐음이 확인돼 별도 재실행 불필요.)

실행: python kr_candidate8_full_verification.py
결과: output/kr_candidate8_full_verification.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_kr as BK
import backtest_portfolio as BP
import core_satellite_kr as CS

TOPN_RANGE = [3, 4, 5, 6, 7, 8, 9, 10]
LIVE_WEIGHTS = {"value": 1, "pbr_inv": 1, "div_yield": 1}
CANDIDATE8 = {"mom12_1": 1, "pbr_inv": 1, "div_yield": 3, "low_vol": 1}


def _log(m): print(f"[KR후보8검증]{m}", file=sys.stderr)


def decisions_for_weights(panel, snaps, weights, pool=30):
    pos_by_date = {d.date().isoformat(): i for i, d in enumerate(panel.index)}
    out = []
    for s in snaps:
        p = pos_by_date.get(s["date"])
        if p is None:
            continue
        roe = s["raw"]["roe"]
        pool_idx = roe[roe > 0].index
        if len(pool_idx) < 7:
            continue
        z = s["z"].loc[pool_idx]
        score = sum(z[f] * wt for f, wt in weights.items() if f in z.columns)
        out.append((p, list(score.sort_values(ascending=False).index[:pool])))
    return out


def main(exclude_top=0):
    from benchmarks_kr import load_research_data, build_benchmarks
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    _log(f"패널 {panel.index[0].date()}~{panel.index[-1].date()} ({len(panel)}일) "
        f"exclude_top={exclude_top}")
    snaps, n_pit_ok, n_total = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                                 rebal_days=63, flows=flows, mktcaps=mktcaps,
                                                 exclude_top=exclude_top)
    navs_bm = build_benchmarks(panel, membership, mktcaps, bench)
    b1 = navs_bm["B1_kospi200"].dropna()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    dec_live = decisions_for_weights(panel, snaps, LIVE_WEIGHTS)
    dec_c8 = decisions_for_weights(panel, snaps, CANDIDATE8)

    # ------------------------- 1) topn=5 NAV로 짝지은 부트스트랩 -------------------------
    nav_live = BP.simulate(panel, ma200, dec_live, 5, cost, ma200_backup=False)
    nav_c8 = BP.simulate(panel, ma200, dec_c8, 5, cost, ma200_backup=False)
    idx = nav_live.index.intersection(nav_c8.index).intersection(b1.index)
    nav_live_a = (nav_live.reindex(idx)); nav_live_a /= nav_live_a.iloc[0]
    nav_c8_a = (nav_c8.reindex(idx)); nav_c8_a /= nav_c8_a.iloc[0]
    b1_a = (b1.reindex(idx)); b1_a /= b1_a.iloc[0]
    s_live, s_c8, s_b1 = CS.stats(nav_live_a), CS.stats(nav_c8_a), CS.stats(b1_a)
    _log(f"topn=5 전체구간({idx[0].date()}~{idx[-1].date()}): 라이브 CAGR {s_live['cagr_pct']}%/"
        f"샤프{s_live['sharpe']}/MDD{s_live['mdd_pct']}% · 후보8 CAGR {s_c8['cagr_pct']}%/"
        f"샤프{s_c8['sharpe']}/MDD{s_c8['mdd_pct']}% · B1 CAGR {s_b1['cagr_pct']}%")

    r_live = nav_live_a.pct_change().dropna().to_numpy()
    r_c8 = nav_c8_a.pct_change().dropna().to_numpy()
    n = min(len(r_live), len(r_c8)); r_live, r_c8 = r_live[:n], r_c8[:n]
    rng = np.random.default_rng(7)
    block, n_boot = 60, 5000
    n_blocks = n // block

    def _cagr(nav_ret, n_days):
        nav = np.cumprod(1 + nav_ret)
        yrs = n_days / 252
        return float((nav[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 and nav[-1] > 0 else float("nan")

    d_cagr = np.empty(n_boot); d_sharpe = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(s * block, s * block + block) for s in starts])[:n]
        c8r, lvr = r_c8[sel], r_live[sel]
        d_cagr[i] = _cagr(c8r, n) - _cagr(lvr, n)
        sd_c8, sd_lv = c8r.std(ddof=1), lvr.std(ddof=1)
        sh_c8 = c8r.mean() / sd_c8 * np.sqrt(252) if sd_c8 > 0 else 0.0
        sh_lv = lvr.mean() / sd_lv * np.sqrt(252) if sd_lv > 0 else 0.0
        d_sharpe[i] = sh_c8 - sh_lv
    ci_cagr = (round(float(np.percentile(d_cagr, 5)), 3), round(float(np.percentile(d_cagr, 95)), 3))
    ci_sharpe = (round(float(np.percentile(d_sharpe, 5)), 4), round(float(np.percentile(d_sharpe, 95)), 4))
    pct_c8_wins_cagr = round(float((d_cagr > 0).mean()) * 100, 1)
    pct_c8_wins_sharpe = round(float((d_sharpe > 0).mean()) * 100, 1)
    _log(f"[짝지은 부트스트랩, n={n}일·block={block}·5000회] "
        f"CAGR차이(후보8-라이브) 90%CI {ci_cagr} · 후보8이 높을 확률 {pct_c8_wins_cagr}%")
    _log(f"[짝지은 부트스트랩] 샤프차이(후보8-라이브) 90%CI {ci_sharpe} · "
        f"후보8이 높을 확률 {pct_c8_wins_sharpe}%")

    # ------------------------- 2) 이상치 진단(63일 구간수익) -------------------------
    def period_returns(nav, step=63):
        vals = nav.to_numpy()
        return np.array([vals[i + step] / vals[i] - 1 for i in range(0, len(vals) - step, step)])

    pr_live, pr_c8 = period_returns(nav_live_a), period_returns(nav_c8_a)

    def outlier_diag(name, arr):
        s = np.sort(arr)[::-1]
        mean, med = float(arr.mean()), float(np.median(arr))
        top_share = float(s[:3].sum() / arr.sum()) * 100 if arr.sum() != 0 else float("nan")
        trimmed1 = float(np.delete(arr, arr.argmax()).mean())
        skew = float(((arr - mean) ** 3).mean() / (arr.std(ddof=0) ** 3)) if arr.std(ddof=0) > 0 else 0.0
        _log(f"[이상치:{name}] n={len(arr)} 평균={mean*100:.2f}% 중앙값={med*100:.2f}% "
            f"최고1개={s[0]*100:.2f}% 왜도={skew:.2f} 상위3개비중={top_share:.1f}% "
            f"최고1개제외평균={trimmed1*100:.2f}%")
        return {"n": len(arr), "mean_pct": round(mean*100,2), "median_pct": round(med*100,2),
                "max_pct": round(s[0]*100,2), "skew": round(skew,2),
                "top3_share_pct": round(top_share,1), "mean_excl_top1_pct": round(trimmed1*100,2)}

    diag_live = outlier_diag("라이브", pr_live)
    diag_c8 = outlier_diag("후보8", pr_c8)

    # ------------------------- 3) topn=3~10 스윕 -------------------------
    sweep = {}
    for label, decisions in (("라이브", dec_live), ("후보8", dec_c8)):
        rows = {}
        for topn in TOPN_RANGE:
            nav = BP.simulate(panel, ma200, decisions, topn, cost, ma200_backup=False)
            if nav is None:
                continue
            ii = nav.index.intersection(b1.index)
            nav_a = nav.reindex(ii); nav_a = nav_a / nav_a.iloc[0]
            s = CS.stats(nav_a)
            rows[str(topn)] = s
            _log(f"  [{label}] topn={topn}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}%")
        sweep[label] = rows

    payload = {"exclude_top": exclude_top,
              "data_range": [idx[0].date().isoformat(), idx[-1].date().isoformat()],
              "topn5_live_config": {"라이브": s_live, "후보8": s_c8, "B1": s_b1},
              "paired_bootstrap": {"cagr_diff_ci90_c8_minus_live": ci_cagr,
                                   "pct_c8_higher_cagr": pct_c8_wins_cagr,
                                   "sharpe_diff_ci90_c8_minus_live": ci_sharpe,
                                   "pct_c8_higher_sharpe": pct_c8_wins_sharpe},
              "outlier_diagnostics": {"라이브": diag_live, "후보8": diag_c8},
              "topn_sweep": sweep}
    os.makedirs("output", exist_ok=True)
    suffix = f"_ex{exclude_top}" if exclude_top else ""
    path = f"output/kr_candidate8_full_verification{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log(f"저장: {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-top", type=int, default=0,
                    help="시총 상위 N종목 제외(2=삼성전자·SK하이닉스 배제 진단용)")
    args = ap.parse_args()
    main(exclude_top=args.exclude_top)
