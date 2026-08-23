#!/usr/bin/env python3
"""
eth_ma30_cost_and_wide_grid.py — ETH MA30 후보 후속검증 2탄 (2026-08-24, 지호 님 요청:
"이더리움 검증 만들고 백그라운드에서"). §11-2에서 남은 두 갈래:
  1) 비용 민감도 — 짧은 이평선(30일)은 회전율이 높아(지호 님이 BTC MA40에도 같은 우려
     제기) 슬리피지 가정이 커지면 우위가 사라지는지 확인(30/50/100bp).
  2) 정식 PBO/DSR을 8종(10~90일) 아닌 전체 MA_GRID(16구간, ma_trend_strategies와 동일
     그리드)로 넓혀 재확인 — 그리드를 넓히면 보통 DSR이 더 나빠지는 게 이 프로젝트의
     일관된 패턴(§6-P-3)인지, 이더리움도 같은지.

실행: python eth_ma30_cost_and_wide_grid.py
결과: output/eth_ma30_cost_and_wide_grid.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import fetch, regime_series, simulate, momentum_ok, pbo_gate
from ma_trend_strategies import MA_GRID

MA_CANDIDATE = 30
LIVE_PARAMS = {"trend_ma": 120, "band": 0.03, "confirm": 3}
COST_LEVELS = [30.0, 50.0, 100.0]


def _log(m): print(f"[ETH비용/광역그리드]{m}", file=sys.stderr)


def main():
    closes = fetch("ETH-USD", "output/regime_price_cache_eth.pkl").to_numpy()
    exp_ma30 = regime_series(closes, MA_CANDIDATE, 0.0, 1)
    live_trend = regime_series(closes, **LIVE_PARAMS)
    mok = momentum_ok(closes, "3m")
    exp_live = np.where((live_trend == 1.0) & (mok == 1.0), 1.0,
                        np.where(np.isnan(live_trend) | np.isnan(mok), np.nan, 0.0))
    bh_cagr = simulate(closes, np.ones(len(closes)), 0.0)["cagr"]

    cost_rows = []
    for cb in COST_LEVELS:
        m30 = simulate(closes, exp_ma30, cb)
        mlive = simulate(closes, exp_live, cb)
        row = {"cost_bps": cb, "ma30_cagr": round(m30["cagr"], 2), "live_cagr": round(mlive["cagr"], 2),
              "bh_cagr": round(bh_cagr, 2), "ma30_excess_over_live": round(m30["cagr"] - mlive["cagr"], 2)}
        cost_rows.append(row)
        _log(f"cost={cb}bp: MA30={row['ma30_cagr']}% 라이브={row['live_cagr']}% "
            f"매수후보유={row['bh_cagr']}% (MA30-라이브={row['ma30_excess_over_live']}%p)")

    try:
        wide_gate = pbo_gate(closes, {"trend_ma": MA_GRID, "band": [0.0], "confirm": [1]}, 30.0)
        _log(f"[광역그리드 PBO/DSR, {len(MA_GRID)}구간] PBO={wide_gate.get('pbo',{}).get('pbo')} "
            f"DSR={wide_gate.get('dsr',{}).get('dsr')} 최고점조합={wide_gate.get('dsr',{}).get('best_trial')} "
            f"passed={wide_gate.get('passed')}")
    except Exception as e:
        _log(f"[광역그리드 PBO/DSR] 실패({type(e).__name__}: {e})")
        wide_gate = None

    payload = {"cost_sensitivity": cost_rows, "pbo_dsr_full_ma_grid": wide_gate,
              "ma_grid_used": MA_GRID}
    os.makedirs("output", exist_ok=True)
    with open("output/eth_ma30_cost_and_wide_grid.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/eth_ma30_cost_and_wide_grid.json")


if __name__ == "__main__":
    main()
