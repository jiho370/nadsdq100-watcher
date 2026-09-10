#!/usr/bin/env python3
"""
circuit_breaker_validation.py — "급락 즉시 방어"를 위한 이산적 서킷브레이커 규칙 대대적
그리드 검증 (2026-09-06, 지호 님 요청). vol_target_fast_response.py에서 확인된 문제
(짧은 창 변동성타깃팅은 폭락 직전 "잠잠했다"는 착시로 오히려 비중을 키워놓는 역효과)를
피하려고, 연속적 변동성 재조정이 아니라 명확한 이벤트 트리거 방식을 시도한다.

규칙: 라이브 레짐신호(추세 ON/OFF, 이미 검증됨)는 그대로 두고, 그 위에
  "당일 등락률이 임계치(threshold) 아래로 떨어지면 → floor 비중으로 즉시 축소하고
   cooldown일 동안 유지, 이후 해제(레짐이 여전히 ON이면 100% 복귀)"
를 얹는다. 다른 백테스트와 동일하게 1봉 지연 실행(오늘 종가로 계산한 신호는 내일부터
반영) — 이 방식도 폭락 당일 자체는 못 피한다(그날 손실은 이미 확정), 폭락 다음날부터의
2차 낙폭을 방어하는 게 목적이라는 점을 분명히 해둔다.

그리드: threshold {-5%,-7%,-10%,-15%} × floor {0%,25%,50%} × cooldown {3,5,10일} = 36조합.
CAGR손실예산 적용 Ulcer개선 스코어(backtest_regime_assets.composite_score)로 순위,
쌍대 블록부트스트랩(최우수 vs 레짐온리 기준선), PBO/DSR 게이트까지 동일 방법론 재사용.

실행: python circuit_breaker_validation.py --asset btc   (또는 eth, all)
결과: output/circuit_breaker_{asset}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

from research.regime.backtest_regime_assets import fetch, regime_series, _ulcer, _mdd, _cagr, composite_score
from research.legacy_cost_30bp.vol_target_validation import paired_bootstrap
import overfit_stats as OS

TRADING_DAYS = 252

# 2026-09-06(지호 님 실제 계정 수수료, 업비트 KRW마켓): 일반주문(시장가) 0.05%·예약주문 0.139%.
# 기존 backtest_regime_assets.COST_BPS["btc"]=30bp는 Fable5의 일반적 가정치였을 뿐 실제
# 계정값이 아니었음 — 서킷브레이커는 급락 감지 즉시 시장가로 던지는 게 목적이라 일반주문
# 수수료(5bp)를 기본으로 쓰고, 슬리피지(급락장 스프레드 확대) 스트레스는 별도 민감도로 확인.
COST_BPS_MARKET = 5.0     # 일반주문(시장가) 0.05%
COST_BPS_RESERVE = 13.9   # 예약주문 0.139%

ASSETS = {
    "btc": ("BTC-USD", {"trend_ma": 120, "band": 0.03, "confirm": 3}, COST_BPS_MARKET),
    "eth": ("ETH-USD", {"trend_ma": 30, "band": 0.0, "confirm": 1}, COST_BPS_MARKET),
}
THRESHOLDS = [-0.05, -0.07, -0.10, -0.15]
FLOORS = [0.0, 0.25, 0.50]
COOLDOWNS = [3, 5, 10]


def _log(m): print(f"[서킷브레이커검증] {m}", file=sys.stderr)


def circuit_breaker_exposure(closes: np.ndarray, regime: np.ndarray, threshold: float,
                             floor: float, cooldown: int) -> np.ndarray:
    """당일수익률<threshold → 다음날부터 floor로 cooldown일간 강제 축소(레짐 ON일 때만
    의미, OFF면 어차피 0). 1봉 지연은 simulate_weighted가 처리하므로 여기선 '오늘 계산해
    오늘자 exposure[i]에 반영'만 하면 된다(그게 자동으로 다음날 수익에 적용됨)."""
    n = len(closes)
    exposure = np.copy(regime)
    remaining = 0
    for i in range(1, n):
        r = closes[i] / closes[i - 1] - 1 if closes[i - 1] else 0.0
        if r < threshold:
            remaining = cooldown
        if remaining > 0 and not np.isnan(exposure[i]):
            exposure[i] = min(exposure[i], floor)
            remaining -= 1
    return exposure


def simulate_weighted(closes, exposure, cost_bps) -> dict:
    ret = np.diff(closes) / closes[:-1]
    exp_lag = exposure[:-1]
    strat_ret = np.nan_to_num(exp_lag, nan=0.0) * ret
    exp_filled = np.nan_to_num(exposure, nan=0.0)
    turnover = np.abs(np.diff(exp_filled))
    cost = turnover[:len(strat_ret)] * (cost_bps / 10000.0)
    strat_ret = np.nan_to_num(strat_ret - cost, nan=0.0)
    nav = np.cumprod(1 + strat_ret)
    bh_nav = closes[1:] / closes[0]
    sd = np.std(strat_ret, ddof=1)
    sharpe = float(np.mean(strat_ret) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    return {"nav": nav, "bh_nav": bh_nav, "cagr": _cagr(nav, len(nav)),
            "bh_cagr": _cagr(bh_nav, len(bh_nav)), "ulcer": _ulcer(nav),
            "bh_ulcer": _ulcer(bh_nav), "mdd": _mdd(nav), "bh_mdd": _mdd(bh_nav),
            "sharpe": round(sharpe, 3), "strat_ret": strat_ret,
            "avg_turnover": round(float(np.mean(turnover)), 4)}


def pbo_gate_cb(closes, regime, grid, cost_bps, month=21, n_blocks=12):
    bh_ret = np.diff(closes) / closes[:-1]
    trials, matrix, dates0 = [], [], None
    for (th, fl, cd) in grid:
        exp = circuit_breaker_exposure(closes, regime, th, fl, cd)
        m = simulate_weighted(closes, exp, cost_bps)
        r = m["strat_ret"]
        n = min(len(r), len(bh_ret))
        excess = r[:n] - bh_ret[:n]
        d, ex = [], []
        for t in range(0, n - month, month):
            d.append(str(t)); ex.append(round(float(excess[t:t + month].sum()), 6))
        if dates0 is None:
            dates0 = d
        matrix.append(ex[:len(dates0)])
        trials.append(f"th{th}_f{fl}_c{cd}")
    trial_data = {"horizon": "circuit_breaker_grid", "universe": "asset", "cost": f"{cost_bps}bp",
                 "rebal_days": month, "hold_days": month, "dates": dates0,
                 "trials": trials, "excess_returns": matrix}
    return OS.analyze(trial_data, n_blocks=n_blocks, save=False)


def run_asset(name: str) -> dict:
    ticker, regime_params, cost_bps = ASSETS[name]
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    regime = regime_series(closes, **regime_params)

    baseline = simulate_weighted(closes, regime, cost_bps)
    baseline_score = composite_score(baseline)

    grid = [(th, fl, cd) for th in THRESHOLDS for fl in FLOORS for cd in COOLDOWNS]
    rows = []
    for th, fl, cd in grid:
        exp = circuit_breaker_exposure(closes, regime, th, fl, cd)
        m = simulate_weighted(closes, exp, cost_bps)
        rows.append({"threshold": th, "floor": fl, "cooldown": cd,
                    "sharpe": m["sharpe"], "cagr": round(m["cagr"], 2), "ulcer": round(m["ulcer"], 2),
                    "mdd": round(m["mdd"], 1), "turnover_month_pct": round(m["avg_turnover"] * 21 * 100, 1),
                    "score": composite_score(m)})
    rankable = [r for r in rows if r["score"] != float("-inf")]
    rankable.sort(key=lambda r: r["score"], reverse=True)
    best = rankable[0] if rankable else None

    boot = None
    cost_sens = None
    if best:
        exp_best = circuit_breaker_exposure(closes, regime, best["threshold"], best["floor"], best["cooldown"])
        ra = simulate_weighted(closes, exp_best, cost_bps)["strat_ret"]
        rb = baseline["strat_ret"]
        boot = paired_bootstrap(ra, rb)
        cost_sens = []
        for cb in (COST_BPS_MARKET, COST_BPS_RESERVE, 30.0, 50.0):
            m = simulate_weighted(closes, exp_best, cb)
            cost_sens.append({"cost_bps": cb, "sharpe": m["sharpe"], "cagr": round(m["cagr"], 2),
                             "ulcer": round(m["ulcer"], 2), "mdd": round(m["mdd"], 1)})

    try:
        pbo = pbo_gate_cb(closes, regime, grid, cost_bps)
    except Exception as e:
        _log(f"[{name}] PBO 게이트 실패({type(e).__name__}: {e}) — 생략")
        pbo = None

    _log(f"[{name}] 기준선(레짐온리) 샤프={baseline['sharpe']:.3f} CAGR={baseline['cagr']:.1f}% "
        f"Ulcer={baseline['ulcer']:.1f} MDD={baseline['mdd']:.1f}% score={baseline_score:.4f}")
    if best:
        _log(f"[{name}] 최우수 threshold={best['threshold']} floor={best['floor']} "
            f"cooldown={best['cooldown']} 샤프={best['sharpe']:.3f} CAGR={best['cagr']:.1f}% "
            f"Ulcer={best['ulcer']:.1f} MDD={best['mdd']:.1f}% score={best['score']:.4f}")
    if boot:
        _log(f"[{name}] 최우수 vs 기준선 부트스트랩: {boot}")
    if pbo:
        _log(f"[{name}] PBO={pbo['pbo']['pbo']:.1%} DSR={pbo['dsr'].get('dsr')} passed={pbo['passed']}")

    return {"asset": name, "ticker": ticker, "n_days": len(closes), "cost_bps": cost_bps,
            "baseline": {"sharpe": baseline["sharpe"], "cagr": round(baseline["cagr"], 2),
                        "ulcer": round(baseline["ulcer"], 2), "mdd": round(baseline["mdd"], 1),
                        "score": baseline_score},
            "grid": rows, "best": best, "bootstrap_best_vs_baseline": boot,
            "cost_sensitivity_best": cost_sens, "pbo_gate": pbo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["btc", "eth", "all"], default="all")
    args = ap.parse_args()
    os.makedirs("output", exist_ok=True)
    targets = ["btc", "eth"] if args.asset == "all" else [args.asset]
    for name in targets:
        _log(f"=== {name} 시작 ({len(THRESHOLDS)}x{len(FLOORS)}x{len(COOLDOWNS)}={len(THRESHOLDS)*len(FLOORS)*len(COOLDOWNS)}조합) ===")
        result = run_asset(name)
        path = f"output/circuit_breaker_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log(f"[{name}] 저장: {path}")


if __name__ == "__main__":
    main()
