#!/usr/bin/env python3
"""
btc_eth_vs_random_baseline.py — 비트코인·이더리움 추세전략들을 "무작위 매수/매도"와
직접 대조 (2026-08-24, 지호 님 요청: "1등 전략이 아닌, 광범위하게 — 무작위 매수보다
얼마나 유의한지, 낙폭을 줄이고 샤프지수를 올리는 방향으로").

방법론: fx_hedge_validation.gate1_matched_random과 동일한 "시장체류시간 매칭 무작위
대조군" — 실제 전략의 노출비율(p=평균 시장 참여 비율)과 평균 ON/OFF 지속기간을 그대로
갖는 마르코프 무작위 이진신호를 n_rep회 생성해, 실제 전략의 샤프·MDD가 그 무작위 분포
안에서 몇 퍼센타일인지를 계산한다("무작위로 시장에 그 정도 시간 들어갔다 나왔다 했어도
이 정도 성과가 나왔을까"를 직접 검정 — 단순 매수후보유가 아니라 "무작위 매매"가
기준선이라는 점이 §9-C의 bootstrap_vs_bh와 다르다).

**넓게(1등 전략 뽑기 아님)**: 이동평균 돌파 16구간 + 골든/데드크로스 + 기존 레짐 그리드
(추세선×밴드×확인일수 60~75조합) 전부에 대해 계산하고, "몇 %가 무작위를 유의하게
이겼는가"를 보고한다.

실행: python btc_eth_vs_random_baseline.py
결과: output/btc_eth_vs_random_baseline.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import (
    fetch, regime_series, simulate, BTC_GRID, BTC_CURRENT, momentum_ok,
)
from ma_trend_strategies import MA_GRID, golden_cross_series

COST_BPS = 30.0
N_REP = 2000
SEED = 7


def _log(m): print(f"[무작위대조]{m}", file=sys.stderr)


def _episode_stats(exposure: np.ndarray) -> dict | None:
    """유효구간(NaN 제외)에서 평균 노출비율 p·평균 ON/OFF 지속일수 → 마르코프 전이확률."""
    valid = exposure[~np.isnan(exposure)]
    if len(valid) < 30:
        return None
    p = float(valid.mean())
    runs, cur, cur_len = [], valid[0], 1
    for v in valid[1:]:
        if v == cur:
            cur_len += 1
        else:
            runs.append((cur, cur_len)); cur, cur_len = v, 1
    runs.append((cur, cur_len))
    on_runs = [l for v, l in runs if v == 1.0]
    off_runs = [l for v, l in runs if v == 0.0]
    l_on = float(np.mean(on_runs)) if on_runs else 1.0
    l_off = float(np.mean(off_runs)) if off_runs else 1.0
    return {"p": p, "l_on": l_on, "l_off": l_off,
            "q_on_to_off": min(max(1.0 / l_on, 1e-4), 1.0),
            "q_off_to_on": min(max(1.0 / l_off, 1e-4), 1.0),
            "n_on_runs": len(on_runs), "n_off_runs": len(off_runs)}


def _markov_random_paths(stat: dict, T: int, n_rep: int, seed: int) -> np.ndarray:
    """stat(p·전이확률)과 동일한 지속성 통계를 갖는 무작위 이진경로 n_rep개. shape (n_rep, T)."""
    rng = np.random.default_rng(seed)
    q_off_to_on, q_on_to_off = stat["q_off_to_on"], stat["q_on_to_off"]
    p0 = q_off_to_on / (q_off_to_on + q_on_to_off)
    state = rng.random(n_rep) < p0
    out = np.empty((n_rep, T))
    out[:, 0] = state.astype(float)
    for t in range(1, T):
        r = rng.random(n_rep)
        flip_off = state & (r < q_on_to_off)
        flip_on = (~state) & (r < q_off_to_on)
        state = np.where(flip_off, False, state)
        state = np.where(flip_on, True, state)
        out[:, t] = state.astype(float)
    return out


def vs_random(closes: np.ndarray, exposure: np.ndarray, cost_bps: float,
             n_rep: int = N_REP, seed: int = SEED) -> dict | None:
    """실제 전략 vs 동일 노출특성 무작위 전략 n_rep회 — 샤프·MDD·Ulcer 퍼센타일."""
    stat = _episode_stats(exposure)
    if stat is None or stat["n_on_runs"] < 3 or stat["n_off_runs"] < 3:
        return None
    m_real = simulate(closes, exposure, cost_bps)
    real_ret = m_real["strat_ret"]
    T = len(real_ret)

    rand_exp = _markov_random_paths(stat, len(closes), n_rep, seed)   # (n_rep, n_days)
    bh_ret = np.diff(closes) / closes[:-1]
    sharpes, mdds, ulcers = np.empty(n_rep), np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        exp_lag = rand_exp[i, :-1]
        r = exp_lag * bh_ret
        flips = np.diff(rand_exp[i]) != 0
        cost = np.where(flips[:len(r)], cost_bps / 10000.0, 0.0)
        r = r - cost[:len(r)]
        nav = np.cumprod(1 + r)
        sd = r.std(ddof=1)
        sharpes[i] = (r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
        cm = np.maximum.accumulate(nav)
        mdds[i] = float(((nav / cm - 1).min()) * 100)
        dd = (nav / cm - 1) * 100
        ulcers[i] = float(np.sqrt(np.mean(dd ** 2)))

    sd_real = real_ret.std(ddof=1)
    sharpe_real = (real_ret.mean() / sd_real * np.sqrt(252)) if sd_real > 0 else 0.0
    cm_real = np.maximum.accumulate(m_real["nav"])
    mdd_real = float(((m_real["nav"] / cm_real - 1).min()) * 100)
    ulcer_real = m_real["ulcer"]

    # MDD·샤프·Ulcer 전부 "무작위 표본 중 실제보다 나쁜 쪽의 비율" = 실제가 무작위를 이기는
    # 퍼센타일로 통일. MDD·CAGR은 음수(-60이 -30보다 나쁨)라 "무작위 < 실제"가 곧
    # "무작위가 더 나쁨"이다(부호 반전 불필요 — 직접 검증: real=-30, rand=-60이면
    # rand<real=True → "무작위가 더 나쁨"에 정확히 대응).
    pct_sharpe = float((sharpes < sharpe_real).mean()) * 100       # 무작위 샤프가 실제보다 낮은 비율
    pct_mdd_better = float((mdds < mdd_real).mean()) * 100         # 무작위 MDD가 실제보다 나쁜(더 음수) 비율
    pct_ulcer_better = float((ulcers > ulcer_real).mean()) * 100   # 무작위 Ulcer가 실제보다 높은(나쁜) 비율

    return {"p": round(stat["p"], 3), "l_on": round(stat["l_on"], 1), "l_off": round(stat["l_off"], 1),
            "real_sharpe": round(float(sharpe_real), 3), "real_mdd": round(mdd_real, 1),
            "real_ulcer": round(ulcer_real, 2),
            "rand_sharpe_mean": round(float(sharpes.mean()), 3), "rand_mdd_mean": round(float(mdds.mean()), 1),
            "rand_ulcer_mean": round(float(ulcers.mean()), 2),
            "pctile_sharpe": round(pct_sharpe, 1),          # ≥95면 샤프가 무작위 대비 유의하게 높음
            "pctile_mdd_better": round(pct_mdd_better, 1),  # ≥95면 MDD가 무작위 대비 유의하게 얕음
            "pctile_ulcer_better": round(pct_ulcer_better, 1)}


def run_asset(name: str, ticker: str, cache: str) -> dict:
    closes = fetch(ticker, cache).to_numpy()
    _log(f"[{name}] {len(closes)}일")
    results = {"ma_breakout": [], "golden_cross": None, "regime_grid": [], "live_current": None}

    for n in MA_GRID:
        exp = regime_series(closes, n, 0.0, 1)
        r = vs_random(closes, exp, COST_BPS)
        if r:
            results["ma_breakout"].append({"ma": n, **r})

    gc = golden_cross_series(closes, 50, 200)
    results["golden_cross"] = vs_random(closes, gc, COST_BPS)

    for tm in BTC_GRID["trend_ma"]:
        for band in BTC_GRID["band"]:
            for cf in BTC_GRID["confirm"]:
                exp = regime_series(closes, tm, band, cf)
                r = vs_random(closes, exp, COST_BPS)
                if r:
                    results["regime_grid"].append({"trend_ma": tm, "band": band, "confirm": cf, **r})

    live_exp = regime_series(closes, BTC_CURRENT["trend_ma"], BTC_CURRENT["band"], BTC_CURRENT["confirm"])
    mok = momentum_ok(closes, "3m")
    live_full = np.where((live_exp == 1.0) & (mok == 1.0), 1.0,
                         np.where(np.isnan(live_exp) | np.isnan(mok), np.nan, 0.0))
    results["live_current_trend_only"] = vs_random(closes, live_exp, COST_BPS)
    results["live_current_with_momentum"] = vs_random(closes, live_full, COST_BPS)

    # ------------------------- 요약: "1등이 아니라 전체가 무작위를 얼마나 이기나" -------------------------
    all_rows = list(results["ma_breakout"]) + list(results["regime_grid"])
    if results["golden_cross"]:
        all_rows.append(results["golden_cross"])
    n_total = len(all_rows)
    n_sharpe_sig = sum(1 for r in all_rows if r["pctile_sharpe"] >= 95)
    n_mdd_sig = sum(1 for r in all_rows if r["pctile_mdd_better"] >= 95)
    n_both_sig = sum(1 for r in all_rows if r["pctile_sharpe"] >= 95 and r["pctile_mdd_better"] >= 95)
    summary = {"n_strategies_tested": n_total,
              "pct_sharpe_beats_random_95": round(100 * n_sharpe_sig / n_total, 1) if n_total else None,
              "pct_mdd_beats_random_95": round(100 * n_mdd_sig / n_total, 1) if n_total else None,
              "pct_both_beat_random_95": round(100 * n_both_sig / n_total, 1) if n_total else None,
              "median_pctile_sharpe": round(float(np.median([r["pctile_sharpe"] for r in all_rows])), 1) if n_total else None,
              "median_pctile_mdd_better": round(float(np.median([r["pctile_mdd_better"] for r in all_rows])), 1) if n_total else None}
    results["summary"] = summary
    _log(f"[{name}] 전체 {n_total}개 전략 중 샤프 95%ile 이상={n_sharpe_sig}개({summary['pct_sharpe_beats_random_95']}%) · "
        f"MDD 95%ile 이상={n_mdd_sig}개({summary['pct_mdd_beats_random_95']}%) · 둘다={n_both_sig}개")
    _log(f"[{name}] 라이브(추세만) 샤프%ile={results['live_current_trend_only']['pctile_sharpe'] if results['live_current_trend_only'] else None} "
        f"MDD%ile={results['live_current_trend_only']['pctile_mdd_better'] if results['live_current_trend_only'] else None}")
    _log(f"[{name}] 라이브(추세+모멘텀) 샤프%ile={results['live_current_with_momentum']['pctile_sharpe'] if results['live_current_with_momentum'] else None} "
        f"MDD%ile={results['live_current_with_momentum']['pctile_mdd_better'] if results['live_current_with_momentum'] else None}")
    return results


def main():
    out = {}
    out["btc"] = run_asset("btc", "BTC-USD", "output/regime_price_cache_btc.pkl")
    out["eth"] = run_asset("eth", "ETH-USD", "output/regime_price_cache_eth.pkl")
    os.makedirs("output", exist_ok=True)
    with open("output/btc_eth_vs_random_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/btc_eth_vs_random_baseline.json")


if __name__ == "__main__":
    main()
