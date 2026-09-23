#!/usr/bin/env python3
"""
macd_bollinger_validation.py — MACD·볼린저밴드를 현재 라이브 레짐 시스템·매수후보유
대비 사전등록 검증 (2026-09-13, 지호 님 요청 — "MACD랑 볼린저밴드 구현해서 검증해줘",
gs-quant 조사 후속 — 그 패키지엔 안 없는 지표라 직접 구현해 이 저장소 표준 잣대로 검증).

사전등록(그리드서치 금지, §0 원칙 — 여러 변형을 스윕하면 그중 하나는 우연히 좋아 보이므로
"표준값 그대로 하나만" 미리 정해두고 검증한다):
  MACD(12,26,9) — 업계 표준값 그대로. MACD선(EMA12-EMA26) > 시그널선(MACD의 EMA9)이면
  매수(ON), 아니면 현금(OFF). 크로스 자체가 상태전이라 별도 밴드·확인일수 없음.
  볼린저(20일선·2표준편차) — 종가가 상단밴드(20일선+2표준편차) 위로 올라가면 매수 진입,
  중심선(20일 SMA) 아래로 내려오면 청산. 이 저장소 레짐 시스템과 같은 "진입 문턱≠청산
  문턱" 히스테리시스 사상.

대상: market_signals.CORE_ASSETS 8개 그대로(라이브에서 실제 추적 중인 자산) — 각 자산의
실제 라이브 PARAMS(레짐 파라미터)와 직접 비교한다(매수후보유뿐 아니라 "지금 쓰는 것보다
나은가"까지 같이 봄).

판정: 자산별 페어드 블록부트스트랩(MACD/볼린저 vs 매수후보유, vs 라이브) +
8자산 풀링 PBO/DSR(월간 비중첩 초과수익, backtest_regime_assets.pbo_gate와 동일 방법론
— overfit_stats.analyze 재사용).

실행: python -m research.regime.macd_bollinger_validation
결과: output/macd_bollinger_validation.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

from research.regime.backtest_regime_assets import fetch, regime_series, simulate, _cagr, _ulcer
import overfit_stats as OS
import market_signals as MS

# 자산군별 비용 가정(이 저장소 기존 관례 재사용 — ma_trend_strategies.py 예시와 동일)
COST_BPS = {"equity": 5.0, "bond": 5.0, "crypto": 30.0, "crypto_eth": 30.0}
N_BOOT = 2000
BLOCK = 60
SEED = 7
MONTH = 21


def _log(m): print(f"[MACD/볼린저] {m}", file=sys.stderr)


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return round(float(r.mean() / sd * np.sqrt(252)), 3) if sd > 0 else 0.0


def macd_exposure(closes: np.ndarray, fast=12, slow=26, signal=9) -> np.ndarray:
    """MACD선(EMA_fast-EMA_slow) > 시그널선(MACD의 EMA_signal)이면 ON. 표준(12,26,9) 고정."""
    s = pd.Series(closes, dtype=float)
    macd_line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    exp = np.where(macd_line.to_numpy() > signal_line.to_numpy(), 1.0, 0.0)
    warmup = slow + signal   # EMA가 안정화되기 전 구간은 미확정 처리
    exp = exp.astype(float)
    exp[:warmup] = np.nan
    return exp


def bollinger_exposure(closes: np.ndarray, window=20, k=2.0) -> np.ndarray:
    """종가가 상단밴드(중심선+k*표준편차) 위로 돌파하면 진입, 중심선 아래로 내려오면 청산."""
    s = pd.Series(closes, dtype=float)
    mid = s.rolling(window).mean().to_numpy()
    std = s.rolling(window).std(ddof=0).to_numpy()
    upper = mid + k * std
    c = closes
    n = len(c)
    out = np.full(n, np.nan)
    on = False
    for i in range(n):
        if np.isnan(mid[i]):
            continue
        if not on and c[i] > upper[i]:
            on = True
        elif on and c[i] < mid[i]:
            on = False
        out[i] = 1.0 if on else 0.0
    return out


def paired_bootstrap(closes: np.ndarray, exp_a: np.ndarray, exp_b: np.ndarray, cost_bps: float,
                     block=BLOCK, n_boot=N_BOOT, seed=SEED) -> dict:
    """exp_a(후보) vs exp_b(비교 대상) 페어드 블록부트스트랩 — Δ(CAGR)의 90%CI +
    "후보가 이길 확률"(prob_a_beats_b_pct)."""
    ma_ = simulate(closes, exp_a, cost_bps)
    mb_ = simulate(closes, exp_b, cost_bps)
    ra, rb = ma_["strat_ret"], mb_["strat_ret"]
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    n_blocks = n // block
    if n_blocks < 8:
        return {"error": f"표본 부족(n_blocks={n_blocks})"}
    rng = np.random.default_rng(seed)
    d_cagr = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(j * block, (j + 1) * block) for j in idx])
        nav_a = np.cumprod(1 + ra[sel]); nav_b = np.cumprod(1 + rb[sel])
        d_cagr[i] = _cagr(nav_a, n) - _cagr(nav_b, n)
    ci = (round(float(np.percentile(d_cagr, 5)), 3), round(float(np.percentile(d_cagr, 95)), 3))
    return {"a_cagr": round(ma_["cagr"], 2), "b_cagr": round(mb_["cagr"], 2),
            "a_sharpe": _sharpe(ra), "b_sharpe": _sharpe(rb),
            "a_mdd": round(ma_["mdd"], 1), "b_mdd": round(mb_["mdd"], 1),
            "delta_cagr_ci90": ci, "prob_a_beats_b_pct": round(float((d_cagr > 0).mean()) * 100, 1)}


def _monthly_excess(closes: np.ndarray, exposure: np.ndarray, bh_ret: np.ndarray, cost_bps: float,
                    month=MONTH) -> list:
    m = simulate(closes, exposure, cost_bps)
    r = m["strat_ret"]
    n = min(len(r), len(bh_ret))
    excess = r[:n] - bh_ret[:n]
    return [round(float(excess[t:t + month].sum()), 6) for t in range(0, n - month, month)]


def run(save=True) -> dict:
    assets_out = {}
    pooled = {"live": [], "macd": [], "bollinger": []}

    for key, name, ticker, kind, _when in MS.CORE_ASSETS:
        cost_bps = COST_BPS.get(kind, 10.0)
        live_p = MS.PARAMS[kind]
        cache = f"output/regime_price_cache_{key.lower()}.pkl"
        s = fetch(ticker, cache)
        closes = s.to_numpy()
        _log(f"[{key}] {s.index.min().date()}~{s.index.max().date()} ({len(s)}일, 비용{cost_bps}bp)")

        bh_ret = np.diff(closes) / closes[:-1]
        exp_live = regime_series(closes, live_p["trend_ma"], live_p["band"], live_p["confirm"])
        exp_macd = macd_exposure(closes)
        exp_boll = bollinger_exposure(closes)

        m_bh = simulate(closes, np.ones(len(closes)), cost_bps)
        m_live = simulate(closes, exp_live, cost_bps)
        m_macd = simulate(closes, exp_macd, cost_bps)
        m_boll = simulate(closes, exp_boll, cost_bps)

        assets_out[key] = {
            "ticker": ticker, "n_days": len(closes), "cost_bps": cost_bps,
            "live_params": live_p,
            "buy_hold":  {"cagr": round(m_bh["cagr"], 2), "sharpe": _sharpe(bh_ret), "mdd": round(m_bh["mdd"], 1)},
            "live":      {"cagr": round(m_live["cagr"], 2), "sharpe": _sharpe(m_live["strat_ret"]), "mdd": round(m_live["mdd"], 1)},
            "macd":      {"cagr": round(m_macd["cagr"], 2), "sharpe": _sharpe(m_macd["strat_ret"]), "mdd": round(m_macd["mdd"], 1)},
            "bollinger": {"cagr": round(m_boll["cagr"], 2), "sharpe": _sharpe(m_boll["strat_ret"]), "mdd": round(m_boll["mdd"], 1)},
            "macd_vs_buyhold":  paired_bootstrap(closes, exp_macd, np.ones(len(closes)), cost_bps),
            "macd_vs_live":     paired_bootstrap(closes, exp_macd, exp_live, cost_bps),
            "bollinger_vs_buyhold": paired_bootstrap(closes, exp_boll, np.ones(len(closes)), cost_bps),
            "bollinger_vs_live":    paired_bootstrap(closes, exp_boll, exp_live, cost_bps),
        }

        pooled["live"].extend(_monthly_excess(closes, exp_live, bh_ret, cost_bps))
        pooled["macd"].extend(_monthly_excess(closes, exp_macd, bh_ret, cost_bps))
        pooled["bollinger"].extend(_monthly_excess(closes, exp_boll, bh_ret, cost_bps))

    n_ev = min(len(v) for v in pooled.values())
    trial_data = {"horizon": "macd_bollinger_vs_live_pooled8assets", "universe": "core_assets_8",
                 "cost": "asset-specific", "rebal_days": MONTH, "hold_days": MONTH,
                 "dates": [str(i) for i in range(n_ev)],
                 "trials": ["live(라이브 레짐)", "macd(12,26,9)", "bollinger(20,2.0)"],
                 "excess_returns": [pooled["live"][:n_ev], pooled["macd"][:n_ev], pooled["bollinger"][:n_ev]]}
    pbo_rpt = OS.analyze(trial_data, save=False)

    payload = {"assets": assets_out,
              "pooled_pbo_dsr": {"pbo": pbo_rpt.get("pbo", {}).get("pbo"),
                                 "dsr": pbo_rpt.get("dsr", {}).get("dsr"),
                                 "passed": pbo_rpt.get("passed", False), "n_periods_per_trial": n_ev}}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/macd_bollinger_validation.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/macd_bollinger_validation.json")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 배선 확인")
    rng = np.random.default_rng(3)
    up = 100 * np.exp(np.cumsum(np.full(400, 0.0012)))
    chop = up[-1] * np.exp(np.cumsum(rng.normal(0, 0.012, 400)))
    closes = np.concatenate([up, chop])

    macd = macd_exposure(closes)
    assert set(np.unique(macd[~np.isnan(macd)])) <= {0.0, 1.0}
    assert np.isnan(macd[:34]).all(), "MACD 워밍업(26+9=35봉) 전엔 NaN이어야 함"
    _log("[self-test] 통과: macd_exposure 배선 정상")

    boll = bollinger_exposure(closes)
    assert set(np.unique(boll[~np.isnan(boll)])) <= {0.0, 1.0}
    assert np.isnan(boll[:19]).all(), "볼린저 워밍업(20봉) 전엔 NaN이어야 함"
    _log("[self-test] 통과: bollinger_exposure 배선 정상")

    boot = paired_bootstrap(closes, macd, np.ones(len(closes)), 5.0, n_boot=200)
    assert "prob_a_beats_b_pct" in boot and 0.0 <= boot["prob_a_beats_b_pct"] <= 100.0
    _log(f"[self-test] 통과: paired_bootstrap 배선 정상(P(MACD이김)={boot['prob_a_beats_b_pct']}%)")
    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        run()
