#!/usr/bin/env python3
"""gold_filter_vs_buyhold_bootstrap.py — STRATEGY.md §6-F 재현(신선 GLD 표본, 2026-08-22).

배경: §6-F는 "금 트렌드필터(현행 200일선·1%밴드·확인3일) vs 무필터(매수후보유)"를 LBMA
58년 확장표본(1968~)으로 쌍대 블록부트스트랩 검정해 Δ(Ulcer)·Δ(CAGR) 90%CI가 둘 다 0을
포함(유의하지 않음)함을 보였다. 그 LBMA CSV(C:\\Users\\JH\\Downloads\\archive\\LBMA-GOLD.csv)는
이 저장소/환경에 없어(재확인 완료) 여기서는 GLD 단독표본(2004-11~, ~22년)만으로 같은
질문을 같은 방법론(블록부트스트랩 block=60·n_boot=2000, backtest_regime_assets.py와 동일)
으로 재현한다 — §1/§6-F가 이미 명시한 한계(LBMA 미포함)를 그대로 유지한 채 짧은 표본에서도
결론이 같은지만 확인하는 목적.

ma_trend_strategies.bootstrap_vs_bh()는 이 프로젝트의 기존 "전략 vs 매수후보유" 블록부트
스트랩 함수이나 CAGR 축만 계산한다(그 스크립트는 수익 프레임이 목적이라). §6-F는 Ulcer
개선분도 함께 검정했으므로, 여기서는 bootstrap_vs_bh를 그대로 호출해 CAGR 쪽 수치(및
prob_beats_buyhold_pct)를 얻고, 동일한 리샘플 방식(block=60·n_boot=2000·seed=7 동일)으로
Δ(Ulcer) 90%CI도 추가 계산한다 — backtest_regime_assets.paired_block_bootstrap과 같은
로직이되, 비교 대상(params_b)을 "다른 파라미터 조합"이 아니라 "무필터(상시 보유)"로 교체.

실행: python gold_filter_vs_buyhold_bootstrap.py
결과: output/gold_filter_vs_buyhold_bootstrap.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import fetch, regime_series, simulate, _cagr, _ulcer, GOLD_CURRENT, COST_BPS
from ma_trend_strategies import bootstrap_vs_bh

BLOCK = 60
N_BOOT = 2000
SEED = 7


def _log(m): print(f"[filter_vs_bh] {m}", file=sys.stderr)


def delta_ulcer_vs_buyhold(closes: np.ndarray, exposure: np.ndarray, cost_bps: float,
                           block=BLOCK, n_boot=N_BOOT, seed=SEED) -> dict:
    """paired_block_bootstrap과 동일한 리샘플 로직으로 Δ(Ulcer) 90%CI만 계산(필터 vs
    무필터). bootstrap_vs_bh와 정확히 같은 block/n_boot/seed를 써서 같은 리샘플 인덱스가
    나오게 함(재현성·비교가능성)."""
    m_f = simulate(closes, exposure, cost_bps)
    exp_bh = np.ones(len(closes))
    m_b = simulate(closes, exp_bh, cost_bps)
    rf, rb = m_f["strat_ret"], m_b["strat_ret"]
    n = min(len(rf), len(rb))
    rf, rb = rf[:n], rb[:n]
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    d_ulcer = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(j * block, (j + 1) * block) for j in idx])
        nav_f = np.cumprod(1 + rf[sel]); nav_b = np.cumprod(1 + rb[sel])
        d_ulcer[i] = _ulcer(nav_b) - _ulcer(nav_f)   # 양수 = 필터가 Ulcer 더 낮음(개선)
    ci = (round(float(np.percentile(d_ulcer, 5)), 3), round(float(np.percentile(d_ulcer, 95)), 3))
    return {"delta_ulcer_ci90": ci, "delta_ulcer_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}


def run(name: str, ticker: str, filter_params: dict, cost_bps: float) -> dict:
    s = fetch(ticker, f"output/regime_price_cache_{name}.pkl")
    closes = s.to_numpy()
    exposure = regime_series(closes, filter_params["trend_ma"], filter_params["band"], filter_params["confirm"])

    cagr_side = bootstrap_vs_bh(closes, exposure, cost_bps, block=BLOCK, n_boot=N_BOOT, seed=SEED)
    ulcer_side = delta_ulcer_vs_buyhold(closes, exposure, cost_bps, block=BLOCK, n_boot=N_BOOT, seed=SEED)

    result = {
        "asset": name, "ticker": ticker, "filter_params": filter_params, "cost_bps": cost_bps,
        "date_range": [s.index.min().date().isoformat(), s.index.max().date().isoformat()],
        "n_days": len(closes),
        "filter_cagr": cagr_side["cagr"], "bh_cagr": cagr_side["bh_cagr"],
        "filter_ulcer": cagr_side["ulcer"], "bh_ulcer": cagr_side["bh_ulcer"],
        "filter_mdd": cagr_side["mdd"], "bh_mdd": cagr_side["bh_mdd"],
        "delta_cagr_ci90": cagr_side["delta_cagr_ci90"],
        "delta_cagr_excludes_zero": bool(cagr_side["delta_cagr_ci90"][0] > 0 or cagr_side["delta_cagr_ci90"][1] < 0),
        "prob_beats_buyhold_pct": cagr_side["prob_beats_buyhold_pct"],
        "delta_ulcer_ci90": ulcer_side["delta_ulcer_ci90"],
        "delta_ulcer_excludes_zero": ulcer_side["delta_ulcer_excludes_zero"],
        "n_boot": N_BOOT, "block": BLOCK,
        "note": ("GLD 단독표본(LBMA 장기 스플라이스 미포함, 파일 부재 재확인) — §6-F는 "
                "1968~58년 표본 기준. 여기는 §6-F와 같은 방법론을 짧은 표본(2004-11~)에 "
                "적용한 재현 시도일 뿐, §6-F 대체 아님."),
    }
    _log(f"[{name}] 필터 CAGR {result['filter_cagr']}% vs B&H {result['bh_cagr']}% | "
        f"dCAGR 90%CI {result['delta_cagr_ci90']}(0포함={not result['delta_cagr_excludes_zero']}) | "
        f"dUlcer 90%CI {result['delta_ulcer_ci90']}(0포함={not result['delta_ulcer_excludes_zero']}) | "
        f"P(필터가 CAGR로 이김)={result['prob_beats_buyhold_pct']}%")
    return result


def main():
    os.makedirs("output", exist_ok=True)
    result = run("gold", "GLD", GOLD_CURRENT, COST_BPS["gold"])
    path = "output/gold_filter_vs_buyhold_bootstrap.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"저장: {path}")


if __name__ == "__main__":
    main()
