#!/usr/bin/env python3
"""
eth_ma30_verification.py — 이더리움 MA30일선 후보 후속 정밀검증 (2026-08-24, 지호 님 요청:
"이더리움 추가 검증 ㄱㄱ"). §11-1에서 MA30이 라이브(비트코인 120일선 그대로 이식)보다
무작위 대비 유의성이 훨씬 컸던 걸 확인 — 미국/한국 가중치 후보와 동일 절차(짝지은
부트스트랩·이상치진단·정식 PBO/DSR·시대분리)로 후속검증.

실행: python eth_ma30_verification.py
결과: output/eth_ma30_verification.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import fetch, regime_series, simulate, momentum_ok, pbo_gate, _cagr

COST_BPS = 30.0
MA_CANDIDATE = 30
LIVE_PARAMS = {"trend_ma": 120, "band": 0.03, "confirm": 3}
SHORT_MA_GRID = {"trend_ma": [10, 20, 30, 40, 50, 60, 75, 90], "band": [0.0], "confirm": [1]}


def _log(m): print(f"[ETH MA30검증]{m}", file=sys.stderr)


def main():
    closes = fetch("ETH-USD", "output/regime_price_cache_eth.pkl").to_numpy()
    n = len(closes)
    _log(f"ETH {n}일")

    exp_ma30 = regime_series(closes, MA_CANDIDATE, 0.0, 1)
    live_trend = regime_series(closes, **LIVE_PARAMS)
    mok = momentum_ok(closes, "3m")
    exp_live = np.where((live_trend == 1.0) & (mok == 1.0), 1.0,
                        np.where(np.isnan(live_trend) | np.isnan(mok), np.nan, 0.0))

    m_ma30 = simulate(closes, exp_ma30, COST_BPS)
    m_live = simulate(closes, exp_live, COST_BPS)
    _log(f"MA30: CAGR {m_ma30['cagr']:.2f}% 샤프계산은 별도 · 라이브: CAGR {m_live['cagr']:.2f}%")

    # ------------------------- 1) 짝지은 블록부트스트랩(일별수익) -------------------------
    ra, rb = m_ma30["strat_ret"], m_live["strat_ret"]
    n_ret = min(len(ra), len(rb)); ra, rb = ra[:n_ret], rb[:n_ret]
    rng = np.random.default_rng(7)
    block, n_boot = 60, 5000
    n_blocks = n_ret // block
    d_cagr, d_sharpe = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(s * block, s * block + block) for s in starts])[:n_ret]
        a_sel, b_sel = ra[sel], rb[sel]
        nav_a, nav_b = np.cumprod(1 + a_sel), np.cumprod(1 + b_sel)
        d_cagr[i] = _cagr(nav_a, n_ret) - _cagr(nav_b, n_ret)
        sd_a, sd_b = a_sel.std(ddof=1), b_sel.std(ddof=1)
        sh_a = a_sel.mean() / sd_a * np.sqrt(252) if sd_a > 0 else 0.0
        sh_b = b_sel.mean() / sd_b * np.sqrt(252) if sd_b > 0 else 0.0
        d_sharpe[i] = sh_a - sh_b
    ci_cagr = (round(float(np.percentile(d_cagr, 5)), 2), round(float(np.percentile(d_cagr, 95)), 2))
    ci_sharpe = (round(float(np.percentile(d_sharpe, 5)), 3), round(float(np.percentile(d_sharpe, 95)), 3))
    pct_cagr = round(float((d_cagr > 0).mean()) * 100, 1)
    pct_sharpe = round(float((d_sharpe > 0).mean()) * 100, 1)
    _log(f"[짝지은부트스트랩] CAGR차이(MA30-라이브) 90%CI {ci_cagr} · MA30이 높을 확률 {pct_cagr}%")
    _log(f"[짝지은부트스트랩] 샤프차이(MA30-라이브) 90%CI {ci_sharpe} · MA30이 높을 확률 {pct_sharpe}%")

    # ------------------------- 2) 이상치 진단(63일 구간수익) -------------------------
    def period_returns(nav, step=63):
        return np.array([nav[i + step] / nav[i] - 1 for i in range(0, len(nav) - step, step)])

    pr_ma30, pr_live = period_returns(m_ma30["nav"]), period_returns(m_live["nav"])

    def outlier_diag(name, arr):
        s = np.sort(arr)[::-1]
        mean = float(arr.mean())
        top3 = float(s[:3].sum() / arr.sum()) * 100 if arr.sum() != 0 else float("nan")
        trimmed1 = float(np.delete(arr, arr.argmax()).mean())
        skew = float(((arr - mean) ** 3).mean() / (arr.std(ddof=0) ** 3)) if arr.std(ddof=0) > 0 else 0.0
        _log(f"[이상치:{name}] n={len(arr)} 평균={mean*100:.2f}% 최고1개={s[0]*100:.2f}% "
            f"왜도={skew:.2f} 상위3개비중={top3:.1f}% 최고1개제외평균={trimmed1*100:.2f}%")
        return {"n": len(arr), "mean_pct": round(mean*100, 2), "max_pct": round(s[0]*100, 2),
                "skew": round(skew, 2), "top3_share_pct": round(top3, 1),
                "mean_excl_top1_pct": round(trimmed1*100, 2)}

    diag_ma30 = outlier_diag("MA30", pr_ma30)
    diag_live = outlier_diag("라이브", pr_live)

    # ------------------------- 3) 정식 PBO/DSR(짧은 MA 그리드, band=0·confirm=1) -------------------------
    try:
        gate = pbo_gate(closes, SHORT_MA_GRID, COST_BPS)
        _log(f"[PBO/DSR] PBO={gate.get('pbo',{}).get('pbo')} DSR={gate.get('dsr',{}).get('dsr')} "
            f"passed={gate.get('passed')}")
    except Exception as e:
        _log(f"[PBO/DSR] 실패({type(e).__name__}: {e})")
        gate = None

    # ------------------------- 4) 시대분리(전반부/후반부) -------------------------
    mid = n // 2
    eras = {}
    for tag, sl in (("전반부", slice(0, mid)), ("후반부", slice(mid, n))):
        c = closes[sl]
        bh_cagr = _cagr(c / c[0], len(c))
        e30 = regime_series(c, MA_CANDIDATE, 0.0, 1)
        m30 = simulate(c, e30, COST_BPS)
        lt = regime_series(c, **LIVE_PARAMS)
        lm = momentum_ok(c, "3m")
        el = np.where((lt == 1.0) & (lm == 1.0), 1.0, np.where(np.isnan(lt) | np.isnan(lm), np.nan, 0.0))
        mlive = simulate(c, el, COST_BPS)
        eras[tag] = {"n_days": len(c), "buy_hold_cagr": round(bh_cagr, 2),
                    "ma30_cagr": round(m30["cagr"], 2), "live_cagr": round(mlive["cagr"], 2)}
        _log(f"[{tag}, {len(c)}일] 매수후보유={bh_cagr:.2f}% MA30={m30['cagr']:.2f}% 라이브={mlive['cagr']:.2f}%")

    payload = {
        "full_period": {"ma30_cagr": round(m_ma30["cagr"], 2), "ma30_mdd": round(m_ma30["mdd"], 1),
                        "ma30_ulcer": round(m_ma30["ulcer"], 2),
                        "live_cagr": round(m_live["cagr"], 2), "live_mdd": round(m_live["mdd"], 1),
                        "live_ulcer": round(m_live["ulcer"], 2),
                        "bh_cagr": round(m_ma30["bh_cagr"], 2)},
        "paired_bootstrap": {"cagr_diff_ci90_ma30_minus_live": ci_cagr, "pct_ma30_higher_cagr": pct_cagr,
                             "sharpe_diff_ci90_ma30_minus_live": ci_sharpe, "pct_ma30_higher_sharpe": pct_sharpe},
        "outlier_diagnostics": {"ma30": diag_ma30, "live": diag_live},
        "pbo_dsr_short_ma_grid": gate,
        "era_split": eras,
    }
    os.makedirs("output", exist_ok=True)
    with open("output/eth_ma30_verification.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/eth_ma30_verification.json")


if __name__ == "__main__":
    main()
