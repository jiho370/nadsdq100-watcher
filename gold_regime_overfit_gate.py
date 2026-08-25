#!/usr/bin/env python3
"""
gold_regime_overfit_gate.py — 금 배분타이밍 레짐신호 4단계: walk-forward + DSR/PBO.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §4단계 참고.

실행: python gold_regime_overfit_gate.py            # output/gold_regime_overfit_gate.json
      python gold_regime_overfit_gate.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import overfit_stats as OS
from gold_regime_data import load_or_build, fetch_all
from gold_regime_signal import (
    build_regime_series, simulate_exposure, fixed_weight_benchmark,
    CORR_THRESHOLDS, REBAL_FREQS, WEIGHT_SCALES,
    DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE, COST_BPS,
)

OUTPUT_PATH = "output/gold_regime_overfit_gate.json"
N_PARAM_COMBOS = len(CORR_THRESHOLDS) * len(REBAL_FREQS) * len(WEIGHT_SCALES)


def _log(m): print(f"[금레짐과최적화]{m}", file=sys.stderr)


def grid_trials(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS) -> dict:
    trials, matrix, dates0 = [], [], None
    block_weeks = 4  # 월간 근사(주간 인덱스 4개 = 1블록)
    for ct in CORR_THRESHOLDS:
        for freq in REBAL_FREQS:
            for scale in WEIGHT_SCALES:
                regime = build_regime_series(features, ct, freq, scale)
                sim = simulate_exposure(gold_daily, regime, cost_bps)
                fw_level = fixed_weight_benchmark(regime)
                fw_regime = regime.copy(); fw_regime["exposure"] = fw_level
                sim_fw = simulate_exposure(gold_daily, fw_regime, cost_bps)
                r, b = sim["strat_ret"], sim_fw["strat_ret"]
                n = min(len(r), len(b))
                excess = r[:n] - b[:n]
                step = block_weeks * 5  # ~21거래일
                d, ex = [], []
                for t in range(0, n - step, step):
                    d.append(str(t)); ex.append(round(float(excess[t:t + step].sum()), 6))
                if dates0 is None or len(d) < len(dates0):
                    dates0 = d
                matrix.append(ex)
                trials.append(f"ct{ct}_{freq}_{scale}")
    matrix = [row[:len(dates0)] for row in matrix]
    return {"horizon": "monthly_approx", "universe": "gold_regime", "cost": f"{cost_bps}bp",
            "rebal_days": 21, "hold_days": 21, "dates": dates0,
            "trials": trials, "excess_returns": matrix}


def run_dsr_pbo(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS) -> dict:
    trial_data = grid_trials(features, gold_daily, cost_bps)
    return OS.analyze(trial_data, n_blocks=12, save=False)


def self_test():
    idx = pd.date_range("2003-01-03", periods=400, freq="W-FRI")  # ~7.7년
    rng = np.random.default_rng(11)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 400), "gold_mom_6m": rng.normal(0, 0.05, 400),
        "gold_mom_12m": rng.normal(0, 0.05, 400),
        "dxy_mom_3m": rng.normal(0, 0.02, 400), "dxy_mom_6m": rng.normal(0, 0.02, 400),
        "dxy_mom_12m": rng.normal(0, 0.02, 400),
        "real_rate_mom_3m": rng.normal(0, 0.3, 400), "real_rate_mom_6m": rng.normal(0, 0.3, 400),
        "real_rate_mom_12m": rng.normal(0, 0.3, 400),
        "ief_mom_3m": rng.normal(0, 0.02, 400), "ief_mom_6m": rng.normal(0, 0.02, 400),
        "ief_mom_12m": rng.normal(0, 0.02, 400),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.1, 400),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.1, 400),
    }, index=idx)
    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=7))
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays)))), index=bdays)

    trial_data = grid_trials(feat, gold_daily)
    assert len(trial_data["trials"]) == N_PARAM_COMBOS, trial_data["trials"]
    assert len(trial_data["excess_returns"]) == N_PARAM_COMBOS

    report = OS.analyze(trial_data, n_blocks=8, save=False)
    assert "dsr" in report and "pbo" in report
    _log("통과: grid_trials/run_dsr_pbo 배선 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
