#!/usr/bin/env python3
"""
bond_trend_filter_grid.py — 채권(IEF/TLT/SHY) 추세필터 다각도 그리드 검증 (2026-07-23, 지호 님 요청)

배경: bond_algo_validation.py Stage 4에서 "200일선·±1%·확인3일"(주식과 동일 파라미터를
그대로 물려받은 것) 딱 1개 조합만 테스트해 기각했다. 지호 님 요청: "200일선, 250일선,
+1%, +2% 지지 몇일 확인, 모멘텀 등등 다각도로 많은 옵션을 백테스트" — 즉 1개 조합이 아니라
제대로 된 그리드로 재검증.

방법론: backtest_regime_assets.py(금·비트코인 레짐 파라미터 검증에 쓰인 Fable 5 설계)를
그대로 재사용 — regime_series/simulate/composite_score/pbo_gate 함수 재사용, 채권 전용으로
그리드·베이스라인만 재설정. 핵심 차이: 금·비트코인은 "이미 있는 필터 파라미터"를 교체하는
문제였지만, 채권은 "필터 자체가 아예 없는" 상태(현재 고정보유)라 베이스라인 = 무필터
(always_on, buy&hold)다 — composite_score의 CAGR 손실예산도 이미 buy&hold 기준이라 그대로
맞는다.

그리드: 추세선[100,150,200,250,300일] × 밴드[0%,±0.5%,±1%,±2%,±3%] × 확인일수[1,3,5,10]
       = 100조합, 모멘텀 필터[1m,3m,6m,9m,12m,12_1] 6종(1단계 최우수 위에 조건부 AND)
대상: IEF(현행) 주 + TLT·SHY(듀레이션별 필터 효과 차이 확인용) 부.

실행: python bond_trend_filter_grid.py
결과: output/bond_trend_grid_{ief,tlt,shy}.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import (
    fetch, regime_series, momentum_ok, simulate, composite_score,
    run_stage1, run_stage2, paired_block_bootstrap, pbo_gate, _ulcer, _cagr, _mdd,
)

COST_BPS = 5   # IEF/TLT/SHY 유동성 우량 ETF, GLD와 동일 가정
MIN_OFF_EPISODES = 8

BOND_GRID = {"trend_ma": [100, 150, 200, 250, 300],
            "band": [0.0, 0.005, 0.01, 0.02, 0.03],
            "confirm": [1, 3, 5, 10]}
BOND_MOM_GRID = ["1m", "3m", "6m", "9m", "12m", "12_1"]
# 참고 비교점(주식과 동일 파라미터를 그냥 물려받았을 경우) — "현행"이 아니라 "만약 있었다면"
REFERENCE_PARAMS = {"trend_ma": 200, "band": 0.01, "confirm": 3}


def _log(m): print(f"[채권그리드] {m}", file=sys.stderr)


def run_bond(name: str, ticker: str) -> dict:
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    stage1 = run_stage1(closes, BOND_GRID, REFERENCE_PARAMS, COST_BPS, name)
    payload = {"asset": name, "ticker": ticker, "n_days": len(closes), "stage1": stage1}

    always_on = simulate(closes, np.ones(len(closes)), COST_BPS)
    payload["baseline_no_filter"] = {"cagr": round(always_on["cagr"], 2),
                                     "ulcer": round(always_on["ulcer"], 2),
                                     "mdd": round(always_on["mdd"], 1)}

    if stage1["best"]:
        base = {"trend_ma": stage1["best"]["trend_ma"], "band": stage1["best"]["band"],
               "confirm": stage1["best"]["confirm"]}
        payload["stage2_momentum"] = run_stage2(closes, base, BOND_MOM_GRID, "없음(참고용)",
                                                COST_BPS, name)
        # 사전등록 판정: 최우수 후보 vs 무필터(진짜 현행) — paired_block_bootstrap은
        # params_b를 regime_series에 넣으므로 "무필터"를 표현하려면 band=-1(항상 ON)로 트릭.
        exp_best = regime_series(closes, base["trend_ma"], base["band"], base["confirm"])
        m_best = simulate(closes, exp_best, COST_BPS)
        rng = np.random.default_rng(7)
        block, n_boot = 60, 2000
        ra, rb = m_best["strat_ret"], np.diff(closes) / closes[:-1]
        n = min(len(ra), len(rb))
        ra, rb = ra[:n], rb[:n]
        n_blocks = n // block
        d_ulcer, d_cagr = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_blocks, n_blocks)
            sel = np.concatenate([np.arange(i * block, (i + 1) * block) for i in idx])
            nav_a = np.cumprod(1 + ra[sel]); nav_b = np.cumprod(1 + rb[sel])
            d_ulcer.append(_ulcer(nav_b) - _ulcer(nav_a))
            d_cagr.append(_cagr(nav_a, n) - _cagr(nav_b, n))
        d_ulcer, d_cagr = np.array(d_ulcer), np.array(d_cagr)
        ci = lambda x: (round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3))
        payload["bootstrap_best_vs_no_filter"] = {
            "delta_ulcer_ci90": ci(d_ulcer), "delta_cagr_ci90": ci(d_cagr),
            "delta_ulcer_excludes_zero": bool(ci(d_ulcer)[0] > 0 or ci(d_ulcer)[1] < 0),
            "delta_cagr_excludes_zero": bool(ci(d_cagr)[0] > 0 or ci(d_cagr)[1] < 0)}
        _log(f"[{name}] 최우수 vs 무필터: dUlcer90%CI={ci(d_ulcer)} dCAGR90%CI={ci(d_cagr)}")

    try:
        payload["pbo_gate"] = pbo_gate(closes, BOND_GRID, COST_BPS)
    except Exception as e:
        _log(f"[{name}] PBO 게이트 실패({type(e).__name__}: {e})")
        payload["pbo_gate"] = None

    os.makedirs("output", exist_ok=True)
    with open(f"output/bond_trend_grid_{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"[{name}] 저장: output/bond_trend_grid_{name}.json")
    return payload


def main():
    for name, ticker in [("ief", "IEF"), ("tlt", "TLT"), ("shy", "SHY")]:
        run_bond(name, ticker)


if __name__ == "__main__":
    main()
