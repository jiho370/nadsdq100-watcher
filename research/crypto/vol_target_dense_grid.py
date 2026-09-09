#!/usr/bin/env python3
"""
vol_target_dense_grid.py — vol_target_validation.py 후속(2026-09-06, 지호 님 재질문):
  ①목표변동성을 바꾸면 실제 리밸런싱(매매)이 얼마나 자주·얼마나 크게 발생하는가
  ②5%~100% 전 구간 촘촘한 그리드에서 자산별 샤프지수가 어떻게 변하는가
를 표로 뽑는다. regime_series·vol_target_weight·simulate_weighted 등은
vol_target_validation.py 그대로 재사용(같은 21거래일 월간 리밸런싱 가정).

실행: python vol_target_dense_grid.py
결과: output/vol_target_dense_grid.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from research.regime.backtest_regime_assets import fetch, regime_series, COST_BPS
from research.crypto.vol_target_validation import (vol_target_weight, simulate_weighted, realized_vol_series,
                                   EQUITY_COST_BPS, REBAL_DAYS)

TRADING_DAYS = 252


def _log(m): print(f"[변동성그리드상세] {m}", file=sys.stderr)


def rebalance_stats(closes: np.ndarray, target: float, rebal_days: int = REBAL_DAYS) -> dict:
    """target=None(무타깃팅)이면 리밸런싱 자체가 없음. 아니면 21거래일마다 재계산되는
    가중치가 직전 대비 얼마나 바뀌는지(리밸런싱 '크기') 집계 — 매매 빈도는 목표값과
    무관하게 캘린더로 고정(연 ~12회)이므로, 여기선 '얼마나 큰 매매인가'가 핵심."""
    if target is None:
        return {"rebal_per_year": 0.0, "mean_abs_change_pp": 0.0, "median_abs_change_pp": 0.0,
                "pct_change_gt_10pp": 0.0, "pct_change_gt_20pp": 0.0, "n_rebal_events": 0}
    n = len(closes)
    rv = realized_vol_series(closes)
    weights_at_rebal = []
    for i in range(n):
        if np.isnan(rv[i]) or i % rebal_days != 0:
            continue
        w = min(1.0, target / rv[i]) if rv[i] > 0 else 1.0
        weights_at_rebal.append(w)
    weights_at_rebal = np.array(weights_at_rebal)
    if len(weights_at_rebal) < 2:
        return {"rebal_per_year": 0.0, "mean_abs_change_pp": 0.0, "median_abs_change_pp": 0.0,
                "pct_change_gt_10pp": 0.0, "pct_change_gt_20pp": 0.0, "n_rebal_events": 0}
    changes = np.abs(np.diff(weights_at_rebal)) * 100   # 퍼센트포인트
    years = n / TRADING_DAYS
    return {"rebal_per_year": round(len(weights_at_rebal) / years, 1),
            "mean_abs_change_pp": round(float(np.mean(changes)), 1),
            "median_abs_change_pp": round(float(np.median(changes)), 1),
            "pct_change_gt_10pp": round(float(np.mean(changes > 10) * 100), 1),
            "pct_change_gt_20pp": round(float(np.mean(changes > 20) * 100), 1),
            "n_rebal_events": int(len(weights_at_rebal))}


def regime_flip_rate(closes: np.ndarray, regime_params: dict) -> dict:
    """비교 기준선: 지금도 이미 일어나고 있는 '레짐 전환' 매매 빈도(타깃값과 무관,
    추세신호 자체가 바뀔 때만 발생) — 변동성 리밸런싱과 섞갈리지 않게 별도로 표기."""
    regime = regime_series(closes, **regime_params)
    valid = regime[~np.isnan(regime)]
    flips = int(np.sum(np.abs(np.diff(valid)) > 0))
    years = len(closes) / TRADING_DAYS
    return {"flips_per_year": round(flips / years, 2), "n_flips": flips}


def run_asset(name, ticker, regime_params, grid, current_target, cost_bps):
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    regime = regime_series(closes, **regime_params)
    rows = []
    for target in grid:
        w = vol_target_weight(closes, target)
        exp = regime * w if target is not None else regime
        m = simulate_weighted(closes, exp, cost_bps)
        rb = rebalance_stats(closes, target)
        # 21거래일(=1개월) 동안 누적되는 회전율(레짐전환+변동성리밸 전부 포함한 블렌드값,
        # avg_turnover는 하루 평균이므로 *21로 월 단위 환산 — 2026-09-06 지호 님 요청).
        turnover_month_pct = round(m["avg_turnover"] * 21 * 100, 1)
        rows.append({"target": target, "sharpe": m["sharpe"], "cagr": round(m["cagr"], 2),
                    "ulcer": round(m["ulcer"], 2), "mdd": round(m["mdd"], 1),
                    "turnover_month_pct": turnover_month_pct, **rb})
    flip = regime_flip_rate(closes, regime_params)
    _log(f"[{name}] 완료 · 레짐전환(타깃값 무관, 기준선) 연 {flip['flips_per_year']}회")
    return {"asset": name, "ticker": ticker, "n_days": len(closes), "current_target": current_target,
            "regime_flip_baseline": flip, "rows": rows}


def _print_table(name, current_target, flip, rows):
    print(f"\n### {name.upper()} (레짐전환 매매는 타깃값과 무관하게 이미 연 {flip['flips_per_year']}회 발생 — 별개)")
    print(f"{'목표변동성':>8} | {'샤프':>6} | {'CAGR%':>7} | {'Ulcer':>6} | {'MDD%':>6} | {'회전율/월':>9}")
    print("-" * 60)
    for r in rows:
        tgt = "무타깃팅" if r["target"] is None else f"{r['target']*100:.0f}%"
        mark = " ←현재" if r["target"] == current_target else ""
        print(f"{tgt:>8} | {r['sharpe']:>6.3f} | {r['cagr']:>7.2f} | {r['ulcer']:>6.2f} | {r['mdd']:>6.1f} | "
            f"{r['turnover_month_pct']:>8.1f}%{mark}")


def main():
    os.makedirs("output", exist_ok=True)
    dense_grid = [None] + [round(x * 0.05, 2) for x in range(1, 21)]   # 5%~100%, 5%p 단위

    assets = [
        ("btc", "BTC-USD", {"trend_ma": 120, "band": 0.03, "confirm": 3}, dense_grid, 0.40, COST_BPS["btc"]),
        ("eth", "ETH-USD", {"trend_ma": 30, "band": 0.0, "confirm": 1}, dense_grid, 0.40, COST_BPS["btc"]),
        ("spx", "^GSPC", {"trend_ma": 200, "band": 0.01, "confirm": 3}, dense_grid, 0.15, EQUITY_COST_BPS),
    ]
    out = {}
    for name, ticker, params, grid, cur_target, cost in assets:
        _log(f"=== {name} 시작 ===")
        out[name] = run_asset(name, ticker, params, grid, cur_target, cost)

    with open("output/vol_target_dense_grid.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/vol_target_dense_grid.json")

    for name, a in out.items():
        _print_table(name, a["current_target"], a["regime_flip_baseline"], a["rows"])


if __name__ == "__main__":
    main()
