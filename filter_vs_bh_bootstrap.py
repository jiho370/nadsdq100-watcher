#!/usr/bin/env python3
"""
filter_vs_bh_bootstrap.py — "필터 자체(있음 vs 없음)"의 효과를 쌍대 블록부트스트랩으로
검정 (2026-08-22).

배경: STRATEGY.md §1은 "무필터(매수후보유) 대비 비트코인: MDD -83.4%→-60.0%, Ulcer
42.51→27.1(CAGR도 33.5%→45.0%로 오히려 개선)"이라 적어 두었지만, 이건 **점추정치**일 뿐
— 금(GLD)은 §6-F에서 58년 확장표본 블록부트스트랩으로 재검정해 "필터가 매수후보유보다
확실히 낫다는 통계적 증명은 실패"로 정정된 전례가 있다. 비트코인은 그 정정이 아직 없었다
(bootstrap_best_vs_current는 "그리드 최우수 후보 vs 현행 파라미터" 비교이지 "현행 파라미터
vs 무필터"가 아니다 — 다른 질문). 이 스크립트가 그 빠진 검정을 채운다.

방법: backtest_regime_assets.paired_block_bootstrap()과 동일한 블록부트스트랩(block=60,
n_boot=2000, seed=7)이지만, 비교 대상이 "그리드 후보 vs 현행"이 아니라 "현행 파라미터
(필터 있음) vs 무필터(exposure=1 고정, 즉 매수후보유)"라 trend_ma/band/confirm 파라미터쌍으로
표현이 안 됨 — 이미 계산된 두 수익률 시계열을 직접 받는 일반화 버전을 새로 작성했다
(재구현이 아니라 확장: _ulcer/_cagr/simulate/regime_series는 backtest_regime_assets에서
그대로 가져다 쓴다). ETH도 동일 정의(BTC_CURRENT 그대로 적용)로 병행 — "필터가 ETH에도
BTC처럼(약하게라도) 도움이 되는가"에 답하기 위함.

실행: python filter_vs_bh_bootstrap.py
결과: output/filter_vs_bh_bootstrap.json (콘솔에도 요약 출력)
"""
from __future__ import annotations
import json
import numpy as np

from backtest_regime_assets import (
    fetch, regime_series, simulate, _ulcer, _cagr, BTC_CURRENT, COST_BPS, _log
)

BLOCK, N_BOOT, SEED = 60, 2000, 7


def paired_block_bootstrap_returns(ret_a: np.ndarray, ret_b: np.ndarray,
                                    block=BLOCK, n_boot=N_BOOT, seed=SEED) -> dict:
    """backtest_regime_assets.paired_block_bootstrap과 동일한 통계(Δulcer, Δcagr 90%CI)를,
    파라미터 조합이 아니라 이미 계산된 두 수익률 시계열(a=후보/필터, b=기준/무필터)에 대해
    직접 계산. a가 b보다 Ulcer 낮으면(방어 개선) delta_ulcer>0, CAGR 높으면 delta_cagr>0."""
    n = min(len(ret_a), len(ret_b))
    ra, rb = ret_a[:n], ret_b[:n]
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    d_ulcer, d_cagr = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(j * block, (j + 1) * block) for j in idx])
        nav_a = np.cumprod(1 + ra[sel]); nav_b = np.cumprod(1 + rb[sel])
        d_ulcer[i] = _ulcer(nav_b) - _ulcer(nav_a)
        d_cagr[i] = _cagr(nav_a, n) - _cagr(nav_b, n)
    ci = lambda x: (round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3))
    ci_u, ci_c = ci(d_ulcer), ci(d_cagr)
    return {"delta_ulcer_ci90": ci_u, "delta_cagr_ci90": ci_c,
            "delta_ulcer_excludes_zero": bool(ci_u[0] > 0 or ci_u[1] < 0),
            "delta_cagr_excludes_zero": bool(ci_c[0] > 0 or ci_c[1] < 0),
            "prob_ulcer_improves_pct": round(float((d_ulcer > 0).mean()) * 100, 1),
            "prob_cagr_beats_pct": round(float((d_cagr > 0).mean()) * 100, 1),
            "n_boot": n_boot, "block": block}


def filter_vs_nofilter(closes: np.ndarray, params: dict, cost_bps: float) -> dict:
    exp_filter = regime_series(closes, **params)
    m_filter = simulate(closes, exp_filter, cost_bps)
    m_bh = simulate(closes, np.ones(len(closes)), cost_bps)     # exposure=1 고정 = 무필터 매수후보유
    boot = paired_block_bootstrap_returns(m_filter["strat_ret"], m_bh["strat_ret"])
    return {"point_estimates": {
                "filter_cagr": round(m_filter["cagr"], 2), "bh_cagr": round(m_bh["cagr"], 2),
                "filter_ulcer": round(m_filter["ulcer"], 2), "bh_ulcer": round(m_bh["ulcer"], 2),
                "filter_mdd": round(m_filter["mdd"], 1), "bh_mdd": round(m_bh["mdd"], 1)},
            "bootstrap": boot}


def main():
    btc_closes = fetch("BTC-USD", "output/regime_price_cache_btc.pkl").to_numpy()
    eth_closes = fetch("ETH-USD", "output/regime_price_cache_eth.pkl").to_numpy()

    result = {
        "params_used": BTC_CURRENT,
        "note": "ETH는 독립 라이브 파라미터가 없어 BTC_CURRENT를 그대로 적용(run_eth_grid.py의 "
                "그리드와 별개 — 여기는 '현재 라이브 규칙이 필터 없음보다 나은가'만 검정)",
        "btc": filter_vs_nofilter(btc_closes, BTC_CURRENT, COST_BPS["btc"]),
        "eth": filter_vs_nofilter(eth_closes, BTC_CURRENT, COST_BPS["btc"]),
    }
    with open("output/filter_vs_bh_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log("저장: output/filter_vs_bh_bootstrap.json")
    for asset in ("btc", "eth"):
        r = result[asset]
        _log(f"  [{asset}] point={r['point_estimates']} boot={r['bootstrap']}")


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 paired_block_bootstrap_returns 배선 확인")
    rng = np.random.default_rng(1)
    n = 3000
    ra = rng.normal(0.001, 0.02, n)    # 후보: 평균수익 높고 변동성 낮음(우월해야 함)
    rb = rng.normal(0.0002, 0.03, n)   # 기준: 무필터
    boot = paired_block_bootstrap_returns(ra, rb, n_boot=300)
    assert "delta_ulcer_ci90" in boot and "delta_cagr_ci90" in boot
    assert boot["prob_cagr_beats_pct"] > 50, "명백히 우월한 합성표본에서 확률이 50% 아래면 배선 문제"
    _log(f"[self-test] 통과: {boot}")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
