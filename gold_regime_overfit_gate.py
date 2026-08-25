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


def walk_forward(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS,
                 train_years: int = 5, test_years: int = 1) -> dict:
    train_weeks, test_weeks = train_years * 52, test_years * 52
    n = len(features)
    folds, start = [], 0
    while start + train_weeks + test_weeks <= n:
        train = features.iloc[start:start + train_weeks]
        test = features.iloc[start + train_weeks:start + train_weeks + test_weeks]
        train_gold = gold_daily[(gold_daily.index >= train.index[0]) & (gold_daily.index <= train.index[-1])]
        best = None
        for ct in CORR_THRESHOLDS:
            for freq in REBAL_FREQS:
                for scale in WEIGHT_SCALES:
                    regime = build_regime_series(train, ct, freq, scale)
                    sim = simulate_exposure(train_gold, regime, cost_bps)
                    fw_level = fixed_weight_benchmark(regime)
                    fw_regime = regime.copy(); fw_regime["exposure"] = fw_level
                    sim_fw = simulate_exposure(train_gold, fw_regime, cost_bps)
                    score = ((sim_fw["ulcer"] - sim["ulcer"]) / sim_fw["ulcer"]
                            if sim_fw["ulcer"] > 0 else float("-inf"))
                    if best is None or score > best["score"]:
                        best = {"ct": ct, "freq": freq, "scale": scale, "score": score}
        test_regime = build_regime_series(test, best["ct"], best["freq"], best["scale"])
        test_gold = gold_daily[(gold_daily.index >= test.index[0]) & (gold_daily.index <= test.index[-1])]
        sim_test = simulate_exposure(test_gold, test_regime, cost_bps)
        folds.append({"train_end": str(train.index[-1].date()), "test_end": str(test.index[-1].date()),
                      "chosen_params": {k: best[k] for k in ("ct", "freq", "scale")},
                      "oos_cagr": round(sim_test["cagr"], 2), "oos_ulcer": round(sim_test["ulcer"], 2)})
        start += test_weeks
    return {"n_folds": len(folds), "folds": folds,
            "oos_cagr_mean": round(float(np.mean([f["oos_cagr"] for f in folds])), 2) if folds else None,
            "n_param_combos_tried": N_PARAM_COMBOS}


def cost_sensitivity(features: pd.DataFrame, gold_daily: pd.Series, params: dict,
                     cost_levels: tuple = (5.0, 15.0)) -> dict:
    out = {}
    for cb in cost_levels:
        regime = build_regime_series(features, params["ct"], params["freq"], params["scale"])
        sim = simulate_exposure(gold_daily, regime, cb)
        out[f"{cb}bp"] = {"cagr": round(sim["cagr"], 2), "ulcer": round(sim["ulcer"], 2)}
    return out


def run(save: bool = True) -> dict:
    features = load_or_build()
    gold_daily = fetch_all()["gold"]
    dsr_pbo = run_dsr_pbo(features, gold_daily)
    wf = walk_forward(features, gold_daily)
    default_params = {"ct": DEFAULT_CORR_THRESHOLD, "freq": DEFAULT_REBAL_FREQ, "scale": DEFAULT_WEIGHT_SCALE}
    cost_sens = cost_sensitivity(features, gold_daily, default_params)
    result = {"dsr_pbo": dsr_pbo, "walk_forward": wf, "cost_sensitivity": cost_sens,
             "n_param_combos": N_PARAM_COMBOS}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUTPUT_PATH}")
    return result


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
    self_test_walk_forward()
    self_test_walk_forward_no_leakage()


def self_test_walk_forward():
    idx = pd.date_range("2003-01-03", periods=600, freq="W-FRI")  # ~11.5년(5+1년 fold 최소 2개)
    rng = np.random.default_rng(13)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 600), "gold_mom_6m": rng.normal(0, 0.05, 600),
        "gold_mom_12m": rng.normal(0, 0.05, 600),
        "dxy_mom_3m": rng.normal(0, 0.02, 600), "dxy_mom_6m": rng.normal(0, 0.02, 600),
        "dxy_mom_12m": rng.normal(0, 0.02, 600),
        "real_rate_mom_3m": rng.normal(0, 0.3, 600), "real_rate_mom_6m": rng.normal(0, 0.3, 600),
        "real_rate_mom_12m": rng.normal(0, 0.3, 600),
        "ief_mom_3m": rng.normal(0, 0.02, 600), "ief_mom_6m": rng.normal(0, 0.02, 600),
        "ief_mom_12m": rng.normal(0, 0.02, 600),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.1, 600),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.1, 600),
    }, index=idx)
    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=7))
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays)))), index=bdays)

    wf = walk_forward(feat, gold_daily, train_years=5, test_years=1)
    assert wf["n_folds"] >= 2, wf["n_folds"]
    assert wf["n_param_combos_tried"] == N_PARAM_COMBOS
    for f in wf["folds"]:
        assert set(f["chosen_params"]) == {"ct", "freq", "scale"}

    default_params = {"ct": DEFAULT_CORR_THRESHOLD, "freq": DEFAULT_REBAL_FREQ, "scale": DEFAULT_WEIGHT_SCALE}
    cs = cost_sensitivity(feat, gold_daily, default_params)
    assert set(cs) == {"5.0bp", "15.0bp"}
    assert cs["15.0bp"]["cagr"] <= cs["5.0bp"]["cagr"] + 1e-9, "비용이 높으면 CAGR이 같거나 낮아야 함"

    _log("통과: walk_forward/cost_sensitivity 배선 정상")


def self_test_walk_forward_no_leakage():
    # train_years=1/test_years=1, 156주(3년) -> 정확히 2개 fold:
    #   fold0: train=주[0:52), test=주[52:104)
    #   fold1: train=주[52:104), test=주[104:156)
    # fold1의 test 구간(주[104:156)) 안에만 뚜렷한 가격 충격을 넣은 변형(shock)과
    # 넣지 않은 변형(flat) 두 gold_daily를 만든다. 두 변형은 그 구간 이전(모든 학습
    # 구간 포함)에는 완전히 동일하다. 만약 파라미터 탐색이 미래(test) 구간 데이터를
    # 누출해서 본다면 fold1의 chosen_params가 두 변형 사이에서 달라질 것이고,
    # 누출이 없다면 학습 구간이 동일하므로 chosen_params가 반드시 같아야 한다.
    idx = pd.date_range("2003-01-03", periods=156, freq="W-FRI")  # 3년
    rng = np.random.default_rng(19)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 156), "gold_mom_6m": rng.normal(0, 0.05, 156),
        "gold_mom_12m": rng.normal(0, 0.05, 156),
        "dxy_mom_3m": rng.normal(0, 0.02, 156), "dxy_mom_6m": rng.normal(0, 0.02, 156),
        "dxy_mom_12m": rng.normal(0, 0.02, 156),
        "real_rate_mom_3m": rng.normal(0, 0.3, 156), "real_rate_mom_6m": rng.normal(0, 0.3, 156),
        "real_rate_mom_12m": rng.normal(0, 0.3, 156),
        "ief_mom_3m": rng.normal(0, 0.02, 156), "ief_mom_6m": rng.normal(0, 0.02, 156),
        "ief_mom_12m": rng.normal(0, 0.02, 156),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.1, 156),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.1, 156),
    }, index=idx)

    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=7))
    rng2 = np.random.default_rng(23)
    base_rets = rng2.normal(0.0002, 0.003, len(bdays))  # 잔잔한 변동성

    fold1_test_start = idx[104]  # fold1 test 구간 시작(fold1 train 구간 이후)
    shock_end = fold1_test_start + pd.Timedelta(days=10)
    shock_mask = (bdays >= fold1_test_start) & (bdays <= shock_end)
    assert shock_mask.sum() > 0, "충격 구간이 비어있음 — 테스트 설계 오류"

    shock_rets = base_rets.copy()
    shock_rets[shock_mask] = -0.05  # fold1 test 구간 안에서만 뚜렷한 급락

    gold_daily_flat = pd.Series(100 * np.exp(np.cumsum(base_rets)), index=bdays)
    gold_daily_shock = pd.Series(100 * np.exp(np.cumsum(shock_rets)), index=bdays)

    wf_flat = walk_forward(feat, gold_daily_flat, train_years=1, test_years=1)
    wf_shock = walk_forward(feat, gold_daily_shock, train_years=1, test_years=1)
    assert wf_flat["n_folds"] == 2 and wf_shock["n_folds"] == 2, \
        (wf_flat["n_folds"], wf_shock["n_folds"])

    fold0_flat, fold0_shock = wf_flat["folds"][0], wf_shock["folds"][0]
    assert fold0_flat == fold0_shock, \
        f"fold0은 충격 구간 이전이라 완전히 동일해야 함: {fold0_flat} vs {fold0_shock}"

    fold1_flat, fold1_shock = wf_flat["folds"][1], wf_shock["folds"][1]
    assert fold1_flat["test_end"] == fold1_shock["test_end"]
    assert fold1_flat["chosen_params"] == fold1_shock["chosen_params"], \
        (f"fold1 test 구간에만 넣은 충격이 파라미터 선택을 바꿈 -> lookahead 누출 의심: "
         f"{fold1_flat['chosen_params']} vs {fold1_shock['chosen_params']}")
    assert (fold1_flat["oos_cagr"] != fold1_shock["oos_cagr"]
            or fold1_flat["oos_ulcer"] != fold1_shock["oos_ulcer"]), \
        "충격이 OOS 결과에 아무 영향도 주지 않음 — 테스트가 실제로 검증하는지 확인 필요"

    _log("통과: walk_forward 파라미터 선택에 test 구간 미래 데이터 누출 없음")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
