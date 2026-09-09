#!/usr/bin/env python3
"""
vol_target_fast_response.py — "60일 평균은 급락에 느리게 반응한다"는 문제를 확인하고,
짧은 창(5~20일)·일간 재계산으로 급변동에 더 빨리 반응하는 버전을 백테스트 (2026-09-06,
지호 님 요청 — 업비트 API 자동매매 전 사전검증 단계).

배경: vol_target_validation.py/vol_target_dense_grid.py는 60일 실현변동성을 월간(21거래일)
마다만 재계산해서, 폭락 당일에는 거의 반응하지 못한다(60일 평균이 하루 데이터로 크게
안 움직이므로). "급변동시 바로 일부 현금화"가 목표라면 ①창을 짧게(5~20일) ②재계산을
매일로 바꿔야 한다 — 이 스크립트가 그 대안을 같은 방법론(비용 포함 시뮬레이션)으로 검증한다.

실행: python vol_target_fast_response.py
결과: output/vol_target_fast_response.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

from research.regime.backtest_regime_assets import fetch, regime_series, COST_BPS
from research.crypto.vol_target_validation import vol_target_weight, simulate_weighted, realized_vol_series, composite_score

TRADING_DAYS = 252


def _log(m): print(f"[급변동대응검증] {m}", file=sys.stderr)


# ------------------------- 창×재계산주기 비교 -------------------------
def sweep_window_cadence(closes, regime, target, cost_bps, configs):
    rows = []
    for label, window, rebal_days in configs:
        w = vol_target_weight(closes, target, w=window, rebal_days=rebal_days)
        exp = regime * w
        m = simulate_weighted(closes, exp, cost_bps)
        turnover_month = round(m["avg_turnover"] * 21 * 100, 1)
        rows.append({"label": label, "window": window, "rebal_days": rebal_days,
                    "sharpe": m["sharpe"], "cagr": round(m["cagr"], 2), "ulcer": round(m["ulcer"], 2),
                    "mdd": round(m["mdd"], 1), "turnover_month_pct": turnover_month,
                    "score": composite_score(m)})
    return rows


# ------------------------- 비용 민감도(회전율이 높아지므로 별도 확인) -------------------------
def cost_sensitivity(closes, regime, target, window, rebal_days, cost_grid):
    rows = []
    w = vol_target_weight(closes, target, w=window, rebal_days=rebal_days)
    exp = regime * w
    for cost_bps in cost_grid:
        m = simulate_weighted(closes, exp, cost_bps)
        rows.append({"cost_bps": cost_bps, "sharpe": m["sharpe"], "cagr": round(m["cagr"], 2),
                    "ulcer": round(m["ulcer"], 2), "mdd": round(m["mdd"], 1)})
    return rows


# ------------------------- 실제 폭락일 반응속도 케이스스터디 -------------------------
def crash_case_study(closes, dates, regime, target, crash_date: str, variants: list, days_before=5, days_after=10):
    idx = dates.get_indexer([pd.Timestamp(crash_date)])[0]
    if idx < 0:
        return None
    lo, hi = max(0, idx - days_before), min(len(closes), idx + days_after + 1)
    out = {"dates": [d.date().isoformat() for d in dates[lo:hi]],
          "price": [round(float(c), 1) for c in closes[lo:hi]]}
    for label, window, rebal_days in variants:
        w = vol_target_weight(closes, target, w=window, rebal_days=rebal_days)
        exp = (regime * w)[lo:hi]
        out[label] = [None if np.isnan(x) else round(float(x) * 100, 1) for x in exp]
    return out


def run_asset(name, ticker, regime_params, target_default, cost_bps):
    s = fetch(ticker, f"output/regime_price_cache_{name}.pkl")
    closes = s.to_numpy()
    dates = s.index
    regime = regime_series(closes, **regime_params)

    configs = [
        ("60일/월간(기존)", 60, 21),
        ("20일/일간", 20, 1),
        ("10일/일간", 10, 1),
        ("5일/일간", 5, 1),
    ]
    sweep = sweep_window_cadence(closes, regime, target_default, cost_bps, configs)

    cost_sens = cost_sensitivity(closes, regime, target_default, 10, 1, [10, 30, 50, 80])

    # 2020-03-12(코로나 폭락)은 레짐이 이미 그 며칠 전에 OFF로 꺼져있어 변동성타깃팅 자체의
    # 반응속도를 보여주기 어렵다 — 레짐이 ON으로 유지된 채 급락한 사례로 선택
    # (2026-09-06 케이스스터디 재선정): BTC 2017-09-14(중국 ICO금지, -18.7%),
    # ETH 2021-05-19(중국 채굴금지, -13.8%, 직후 레짐 OFF 전환).
    crashes = {
        "btc": "2017-09-14", "eth": "2021-05-19",
    }
    case = crash_case_study(closes, dates, regime, target_default, crashes[name],
                            [("60일/월간", 60, 21), ("10일/일간", 10, 1), ("5일/일간", 5, 1)])

    for r in sweep:
        _log(f"[{name}] {r['label']:>12} 샤프={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% "
            f"Ulcer={r['ulcer']:.1f} MDD={r['mdd']:.1f}% 회전율/월={r['turnover_month_pct']:.1f}%")

    return {"asset": name, "ticker": ticker, "target": target_default, "cost_bps": cost_bps,
            "window_cadence_sweep": sweep, "cost_sensitivity": cost_sens, "crash_case_study": case}


def main():
    os.makedirs("output", exist_ok=True)
    out = {
        "btc": run_asset("btc", "BTC-USD", {"trend_ma": 120, "band": 0.03, "confirm": 3}, 0.40, COST_BPS["btc"]),
        "eth": run_asset("eth", "ETH-USD", {"trend_ma": 30, "band": 0.0, "confirm": 1}, 0.40, COST_BPS["btc"]),
    }
    with open("output/vol_target_fast_response.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/vol_target_fast_response.json")


if __name__ == "__main__":
    main()
