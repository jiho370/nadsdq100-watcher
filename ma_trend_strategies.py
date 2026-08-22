#!/usr/bin/env python3
"""
ma_trend_strategies.py — 단일 이동평균선 돌파/이탈 매매 + 골든크로스/데드크로스 매매,
자산군 공통 검증 엔진 (2026-08-22, 지호 님 요청: "120일선 등 이동평균선 매매법으로 했을 때
각 자산군별로 무지성 보유보다 수익률이 어떠했는지, 몇일선이 가장 좋았는지, 골든크로스
매수·데드크로스 매도도 시도").

배경: backtest_regime_assets.py의 regime_series/simulate는 히스테리시스밴드+확인일수까지
갖춘 정교한 레짐필터이자, band=0·confirm=1로 두면 정확히 "종가가 N일선 위/아래로 넘어가면
바로 매수/매도"인 순수 MA 돌파 규칙과 수학적으로 동일하다 — 재구현하지 않고 그대로
재사용한다(파라미터만 다르게 호출). 골든/데드크로스(단기선 vs 장기선)는 이 저장소에
없어 신규 구현.

기존 레짐엔진(§1, composite_score)과 이 스크립트의 목적함수 차이:
  · 기존: "낙폭 방어"가 목적이라 Ulcer 개선을 CAGR 손실예산 안에서만 점수화(위험관리 프레임).
  · 여기: 지호 님이 명시적으로 "수익률이 어떠했는지"·"몇일선이 가장 좋았는지"를 물었으므로
    주 지표를 초과CAGR(전략-매수후보유)로 둔다(수익 프레임). 샤프·MDD·Ulcer는 계속 병기.
  · 유의성 프레이밍도 다르게: "여러 후보 중 1등일 확률이 아니라 지수(매수후보유) 대비 나을
    확률"을 지호 님이 명시 요청 — 블록부트스트랩으로 "전략 CAGR > 매수후보유 CAGR"인
    리샘플 비율을 직접 계산해 prob_beats_buyhold_pct로 보고한다. 다중검정 관점(PBO/DSR)은
    그리드 스윕 자체가 노이즈가 아님을 보이는 새니티 게이트로 참고 병기(주된 판단 기준 아님).

실행: python ma_trend_strategies.py --ticker GLD --name gold --cost-bps 5
      python ma_trend_strategies.py --ticker BTC-USD --name btc --cost-bps 30
      python ma_trend_strategies.py --self-test
결과: output/ma_trend_{name}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

from backtest_regime_assets import fetch, regime_series, simulate, pbo_gate, _cagr, _mdd, _ulcer

MA_GRID = [10, 20, 30, 40, 50, 60, 75, 90, 100, 120, 150, 180, 200, 220, 250, 300]
GOLDEN_PAIRS = [(50, 200)]     # 표준 골든/데드크로스 정의
MIN_OFF_EPISODES = 8           # backtest_regime_assets.py와 동일 기준(순위 매길 최소 표본)
N_BOOT = 2000
BLOCK = 60
SEED = 7


def _log(m): print(f"[MA추세전략] {m}", file=sys.stderr)


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return round(float(r.mean() / sd * np.sqrt(252)), 3) if sd > 0 else 0.0


def golden_cross_series(closes: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """단기선>장기선 → ON(매수 보유), 아니면 OFF(현금). 크로스 자체가 상태전이라
    별도 히스테리시스·확인일수는 두지 않음(표준 골든/데드크로스 정의 그대로, 1봉 지연
    실행은 simulate()가 처리)."""
    ma_f = pd.Series(closes).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(closes).rolling(slow).mean().to_numpy()
    return np.where(np.isnan(ma_f) | np.isnan(ma_s), np.nan, np.where(ma_f > ma_s, 1.0, 0.0))


def bootstrap_vs_bh(closes: np.ndarray, exposure: np.ndarray, cost_bps: float,
                    block=BLOCK, n_boot=N_BOOT, seed=SEED) -> dict:
    """전략 vs 매수후보유 쌍대 블록부트스트랩 — "이 전략이 매수후보유보다 나을 확률"을
    직접 계산(PBO/DSR의 "다중검정 중 1등일 확률"과는 다른 질문, 지호 님이 명시 요청한 정의)."""
    m = simulate(closes, exposure, cost_bps)
    bh_ret = np.diff(closes) / closes[:-1]
    ra, rb = m["strat_ret"], bh_ret
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
    return {"cagr": round(m["cagr"], 2), "bh_cagr": round(m["bh_cagr"], 2),
            "sharpe": _sharpe(ra), "bh_sharpe": _sharpe(rb),
            "mdd": round(m["mdd"], 1), "bh_mdd": round(m["bh_mdd"], 1),
            "ulcer": round(m["ulcer"], 2), "bh_ulcer": round(m["bh_ulcer"], 2),
            "off_episodes": m["off_episodes"],
            "delta_cagr_ci90": ci, "prob_beats_buyhold_pct": round(float((d_cagr > 0).mean()) * 100, 1),
            "n_boot": n_boot, "block": block}


def ma_breakout_sweep(closes: np.ndarray, cost_bps: float, grid=MA_GRID) -> dict:
    """N일선 돌파(위)=매수·이탈(아래)=매도, band=0·confirm=1(지연 없는 순수 돌파)."""
    rows = []
    for n in grid:
        exp = regime_series(closes, n, band=0.0, confirm=1)
        m = simulate(closes, exp, cost_bps)
        rows.append({"ma": n, "cagr": round(m["cagr"], 2), "bh_cagr": round(m["bh_cagr"], 2),
                    "excess_cagr": round(m["cagr"] - m["bh_cagr"], 2),
                    "sharpe": _sharpe(m["strat_ret"]), "mdd": round(m["mdd"], 1),
                    "ulcer": round(m["ulcer"], 2), "off_episodes": m["off_episodes"],
                    "rankable": m["off_episodes"] >= MIN_OFF_EPISODES})
    rankable = [r for r in rows if r["rankable"]]
    best_by_return = max(rankable, key=lambda r: r["excess_cagr"]) if rankable else None
    best_by_risk_adj = max(rankable, key=lambda r: r["sharpe"]) if rankable else None
    return {"rows": rows, "best_by_return": best_by_return, "best_by_risk_adj": best_by_risk_adj}


def golden_cross_stats(closes: np.ndarray, cost_bps: float, pairs=GOLDEN_PAIRS) -> list:
    rows = []
    for fast, slow in pairs:
        exp = golden_cross_series(closes, fast, slow)
        boot = bootstrap_vs_bh(closes, exp, cost_bps)
        rows.append({"fast": fast, "slow": slow, **boot})
    return rows


def run(name: str, ticker: str, cost_bps: float, cache_path: str | None = None, save=True) -> dict:
    cache_path = cache_path or f"output/regime_price_cache_{name}.pkl"
    s = fetch(ticker, cache_path)
    closes = s.to_numpy()
    _log(f"[{name}] 데이터 {s.index.min().date()}~{s.index.max().date()} ({len(s)}일)")

    always_on = simulate(closes, np.ones(len(closes)), cost_bps)
    sweep = ma_breakout_sweep(closes, cost_bps)
    try:
        pbo = pbo_gate(closes, {"trend_ma": MA_GRID, "band": [0.0], "confirm": [1]}, cost_bps)
    except Exception as e:
        _log(f"[{name}] PBO 게이트 실패({type(e).__name__}: {e})")
        pbo = None

    best = sweep["best_by_return"]
    best_boot = None
    if best:
        exp_best = regime_series(closes, best["ma"], 0.0, 1)
        best_boot = bootstrap_vs_bh(closes, exp_best, cost_bps)

    gold_rows = golden_cross_stats(closes, cost_bps)

    payload = {
        "asset": name, "ticker": ticker, "n_days": len(closes),
        "date_range": [s.index.min().date().isoformat(), s.index.max().date().isoformat()],
        "cost_bps": cost_bps,
        "buy_hold": {"cagr": round(always_on["cagr"], 2),
                    "sharpe": _sharpe(np.diff(closes) / closes[:-1]),
                    "mdd": round(always_on["mdd"], 1), "ulcer": round(always_on["ulcer"], 2)},
        "ma_sweep": sweep,
        "ma_sweep_pbo_dsr": pbo,
        "ma_best_vs_buyhold_bootstrap": best_boot,
        "golden_dead_cross": gold_rows,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/ma_trend_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    bcagr = best['cagr'] if best else float('nan')
    bexc = best['excess_cagr'] if best else float('nan')
    bprob = best_boot.get('prob_beats_buyhold_pct') if best_boot else None
    _log(f"[{name}] 매수후보유 CAGR {payload['buy_hold']['cagr']}% vs 최우수 "
        f"MA{best['ma'] if best else '?'} CAGR {bcagr}%(초과 {bexc}%p) — P(이김)={bprob}%")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 배선 확인")
    rng = np.random.default_rng(3)
    up = 100 * np.exp(np.cumsum(np.full(500, 0.0012)))
    crash = up[-1] * np.exp(np.cumsum(np.full(120, -0.012)))
    chop = crash[-1] * np.exp(np.cumsum(rng.normal(0, 0.01, 780)))
    closes = np.concatenate([up, crash, chop])

    sweep = ma_breakout_sweep(closes, 5.0)
    assert sweep["best_by_return"] is not None, "돌파 스윕 결과 없음"
    _log(f"[self-test] 통과: ma_breakout_sweep 배선 정상(최우수 {sweep['best_by_return']})")

    gc = golden_cross_series(closes, 50, 200)
    assert gc.shape == closes.shape
    assert np.isnan(gc[:199]).all(), "슬로우선(200) 형성 전엔 NaN이어야 함"
    assert not np.isnan(gc[199])
    _log("[self-test] 통과: golden_cross_series 배선 정상(200일 미만 구간 NaN)")

    boot = bootstrap_vs_bh(closes, gc, 5.0, n_boot=200)
    assert "prob_beats_buyhold_pct" in boot and 0.0 <= boot["prob_beats_buyhold_pct"] <= 100.0
    _log(f"[self-test] 통과: bootstrap_vs_bh 배선 정상(P(이김)={boot['prob_beats_buyhold_pct']}%)")

    # 급락을 피하는 필터라면 매수후보유보다 MDD가 개선돼야 함(정성 확인)
    exp_120 = regime_series(closes, 120, 0.0, 1)
    m120 = simulate(closes, exp_120, 5.0)
    assert m120["mdd"] > m120["bh_mdd"], f"120일선 돌파가 급락 방어를 못함: {m120['mdd']} vs {m120['bh_mdd']}"
    _log(f"[self-test] 통과: 120일선 MDD {m120['mdd']:.1f}% > 매수후보유 {m120['bh_mdd']:.1f}%(방어 확인)")

    _log("[self-test] 전부 통과")


def main():
    ap = argparse.ArgumentParser(description="N일선 돌파/골든크로스 매매 vs 매수후보유")
    ap.add_argument("--ticker")
    ap.add_argument("--name")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--cache")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not args.ticker or not args.name:
        ap.error("--ticker와 --name이 필요합니다(또는 --self-test)")
    run(args.name, args.ticker, args.cost_bps, args.cache)


if __name__ == "__main__":
    main()
