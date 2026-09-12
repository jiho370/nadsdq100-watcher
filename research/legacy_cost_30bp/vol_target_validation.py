#!/usr/bin/env python3
"""
⚠ 2026-09-10 이관: 이 파일은 COST_BPS["btc"]=30bp(또는 자체 COST_BPS=30.0) — 실제
업비트 시장가 수수료 5bp의 6배인 낡은 가정을 쓴다(2026-09-06 발견, 상세는
docs/playbook/03-method/EXECUTION.md §3-1). 판정을 뒤집을 후보가 있는지 5bp 기준으로
재계산해 확인 완료 — 전부 DSR이 채택기준(0.95)에서 멀고 OOS 샤프가 음수/0이라 비용
문제가 아니라 과최적화로 기각됨(같은 문서 §3-1 표 참고). 그대로 legacy 처리.

vol_target_validation.py — market_signals.PARAMS의 vol_target(코인 40%·주식 15%)이
실제로 이 프로젝트 데이터로 백테스트된 값인지 검증 (2026-09-06, 지호 님 요청).

배경: market_signals.py의 '변동성 참고 노출' 표시값
  exposure = min(1, 목표변동성 / 최근60일실현변동성)
은 지금까지 화면 표시용일 뿐 실제 매매 사이징에 쓰인 적이 없고, 목표변동성 숫자 자체
(코인 40%·주식 15%)도 이 프로젝트 데이터로 검증된 적이 없었다(코드 전체 grep 확인).
HISTORY.md §0의 "변동성 타깃팅 — 강함"(Moreira-Muir 2017·Barroso-Santa-Clara 2015)은
'기법 자체'에 대한 문헌 근거이지, 이 프로젝트가 쓰는 구체적 숫자를 검증한 게 아니다.
이 스크립트가 그 공백을 처음 메운다.

방법(backtest_regime_assets.py와 동일 철학 재사용 — Ulcer 중심 composite_score,
쌍대 블록부트스트랩, PBO/DSR 게이트):
  · 이미 검증된 라이브 레짐신호(추세선 ON/OFF)는 그대로 두고, 그 위에 변동성타깃
    가중치 w_t = min(1, 목표변동성/실현변동성60일)를 21거래일(월간)마다 재계산해
    곱한다(매일 재계산하면 비현실적으로 회전율이 치솟아 월간 리밸런싱으로 제한).
  · 목표변동성을 그리드로 스윕(코인 12단계·주식 10단계 + "타깃팅 안 함"=현재 라이브
    실질 동작)해 Ulcer 개선 점수로 순위를 매기고, 현재 기본값이 그 안에서 어디쯤인지
    확인.
  · 최우수 후보 vs "타깃팅 안 함" vs "현재 기본값" 쌍대 블록부트스트랩 + 그리드 전체
    PBO/DSR 게이트.

실행: python vol_target_validation.py
결과: output/vol_target_validation.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

from research.regime.backtest_regime_assets import fetch, regime_series, _ulcer, _mdd, _cagr, composite_score, COST_BPS
import overfit_stats as OS

TRADING_DAYS = 252
REBAL_DAYS = 21          # 월간 리밸런싱(매일 재계산 시 회전율 비현실적으로 급등)
EQUITY_COST_BPS = 5      # equity_index_regime_validation.py와 동일


def _log(m): print(f"[변동성타깃검증] {m}", file=sys.stderr)


# ------------------------- 실현변동성·타깃가중치 -------------------------
def realized_vol_series(closes: np.ndarray, w: int = 60) -> np.ndarray:
    """market_signals._realized_vol과 동일 정의(60일 일별수익 표준편차*sqrt(252))를
    시계열 전체에 벡터화. 길이는 closes와 동일(앞부분 NaN)."""
    rets = np.diff(closes) / closes[:-1]
    roll = pd.Series(rets).rolling(w).std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
    out = np.full(len(closes), np.nan)
    out[1:] = roll
    return out


def vol_target_weight(closes: np.ndarray, target: float | None, w: int = 60,
                      rebal_days: int = REBAL_DAYS) -> np.ndarray:
    """target=None → 항상 1.0(타깃팅 없음 = 현재 라이브 실질 동작과 동일, exposure는
    표시만 되고 매매에 반영 안 되므로). 아니면 rebal_days(기본 월간)마다 그 시점
    실현변동성으로 가중치를 재계산해 다음 리밸런싱까지 고정. 실현변동성 계산 전
    구간(워밍업)은 1.0(정보 없음 → 무타깃팅 취급)."""
    n = len(closes)
    if target is None:
        return np.ones(n)
    rv = realized_vol_series(closes, w)
    out = np.ones(n)
    cur = 1.0
    for i in range(n):
        if np.isnan(rv[i]):
            out[i] = 1.0
            continue
        if i % rebal_days == 0:
            cur = min(1.0, target / rv[i]) if rv[i] > 0 else 1.0
        out[i] = cur
    return out


# ------------------------- 성과 시뮬레이션(연속가중치 — 비용은 회전율 비례) -------------------------
def simulate_weighted(closes: np.ndarray, exposure: np.ndarray, cost_bps: float) -> dict:
    """backtest_regime_assets.simulate와 동일 실행규칙(전일 노출로 당일 수익, 1봉 지연)이되,
    비용은 flip(0/1) 여부가 아니라 |Δ노출| 비례(연속 가중치를 다루므로) — 노출이 이진일
    때는 원래 flip 비용과 동일해지는 상위호환."""
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


# ------------------------- 쌍대 블록부트스트랩 -------------------------
def paired_bootstrap(ra: np.ndarray, rb: np.ndarray, block=60, n_boot=2000, seed=7) -> dict:
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    d_ulcer, d_cagr, d_sharpe = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(i * block, (i + 1) * block) for i in idx])
        nav_a, nav_b = np.cumprod(1 + ra[sel]), np.cumprod(1 + rb[sel])
        d_ulcer.append(_ulcer(nav_b) - _ulcer(nav_a))   # 양수 = a(후보)가 Ulcer 더 낮음(개선)
        d_cagr.append(_cagr(nav_a, n) - _cagr(nav_b, n))
        sda, sdb = np.std(ra[sel], ddof=1), np.std(rb[sel], ddof=1)
        sa = np.mean(ra[sel]) / sda * np.sqrt(TRADING_DAYS) if sda > 0 else 0.0
        sb = np.mean(rb[sel]) / sdb * np.sqrt(TRADING_DAYS) if sdb > 0 else 0.0
        d_sharpe.append(sa - sb)
    d_ulcer, d_cagr, d_sharpe = np.array(d_ulcer), np.array(d_cagr), np.array(d_sharpe)
    ci = lambda x: (round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3))
    return {"delta_ulcer_ci90": ci(d_ulcer), "delta_cagr_ci90": ci(d_cagr),
            "delta_sharpe_ci90": ci(d_sharpe),
            "delta_ulcer_excludes_zero": bool(ci(d_ulcer)[0] > 0 or ci(d_ulcer)[1] < 0),
            "delta_sharpe_excludes_zero": bool(ci(d_sharpe)[0] > 0 or ci(d_sharpe)[1] < 0)}


# ------------------------- PBO/DSR 게이트(목표변동성 그리드) -------------------------
def pbo_gate_targets(closes, regime, grid, cost_bps, month=21, n_blocks=12) -> dict | None:
    bh_ret = np.diff(closes) / closes[:-1]
    trials, matrix, dates0 = [], [], None
    for target in grid:
        w = vol_target_weight(closes, target)
        exp = regime * w if target is not None else regime
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
        trials.append(f"vt_{target}")
    trial_data = {"horizon": "vol_target_grid", "universe": "asset", "cost": f"{cost_bps}bp",
                 "rebal_days": month, "hold_days": month, "dates": dates0,
                 "trials": trials, "excess_returns": matrix}
    return OS.analyze(trial_data, n_blocks=n_blocks, save=False)


# ------------------------- 자산별 실행 -------------------------
def run_asset(name: str, ticker: str, regime_params: dict, grid: list, current_target: float,
             cost_bps: float) -> dict:
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    regime = regime_series(closes, **regime_params)

    def exp_for(target):
        w = vol_target_weight(closes, target)
        return regime * w if target is not None else regime

    rows = []
    for target in grid:
        m = simulate_weighted(closes, exp_for(target), cost_bps)
        score = composite_score(m)
        rows.append({"target": target, "cagr": round(m["cagr"], 2), "bh_cagr": round(m["bh_cagr"], 2),
                    "ulcer": round(m["ulcer"], 2), "bh_ulcer": round(m["bh_ulcer"], 2),
                    "mdd": round(m["mdd"], 1), "sharpe": m["sharpe"],
                    "avg_turnover": m["avg_turnover"], "score": score})
    rankable = [r for r in rows if r["score"] != float("-inf")]
    rankable.sort(key=lambda r: r["score"], reverse=True)
    best = rankable[0] if rankable else None
    off_row = next(r for r in rows if r["target"] is None)
    cur_row = next(r for r in rows if r["target"] == current_target)

    boot_best_vs_off, boot_best_vs_cur = None, None
    if best and best["target"] != off_row["target"]:
        ra = simulate_weighted(closes, exp_for(best["target"]), cost_bps)["strat_ret"]
        rb = simulate_weighted(closes, exp_for(off_row["target"]), cost_bps)["strat_ret"]
        boot_best_vs_off = paired_bootstrap(ra, rb)
    if best and cur_row and best["target"] != cur_row["target"]:
        ra = simulate_weighted(closes, exp_for(best["target"]), cost_bps)["strat_ret"]
        rc = simulate_weighted(closes, exp_for(cur_row["target"]), cost_bps)["strat_ret"]
        boot_best_vs_cur = paired_bootstrap(ra, rc)

    try:
        pbo = pbo_gate_targets(closes, regime, grid, cost_bps)
    except Exception as e:
        _log(f"[{name}] PBO 게이트 실패({type(e).__name__}: {e}) — 생략")
        pbo = None

    _log(f"[{name}] 최우수 target={best['target'] if best else None} "
        f"(score={best['score']:.4f}) · 현재기본값 target={current_target}(score={cur_row['score']:.4f}) "
        f"· 무타깃팅(score={off_row['score']:.4f})")
    if boot_best_vs_off:
        _log(f"[{name}] 최우수 vs 무타깃팅 부트스트랩: {boot_best_vs_off}")
    if boot_best_vs_cur:
        _log(f"[{name}] 최우수 vs 현재기본값 부트스트랩: {boot_best_vs_cur}")
    if pbo:
        _log(f"[{name}] PBO={pbo['pbo']['pbo']:.1%} DSR={pbo['dsr'].get('dsr')} passed={pbo['passed']}")

    return {"asset": name, "ticker": ticker, "n_days": len(closes), "regime_params": regime_params,
            "cost_bps": cost_bps, "grid": rows, "best": best, "off": off_row,
            "current_default": cur_row, "bootstrap_best_vs_off": boot_best_vs_off,
            "bootstrap_best_vs_current": boot_best_vs_cur, "pbo_gate": pbo}


def main():
    os.makedirs("output", exist_ok=True)
    crypto_grid = [None, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.00]
    equity_grid = [None, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

    assets = [
        ("btc", "BTC-USD", {"trend_ma": 120, "band": 0.03, "confirm": 3}, crypto_grid, 0.40, COST_BPS["btc"]),
        ("eth", "ETH-USD", {"trend_ma": 30, "band": 0.0, "confirm": 1}, crypto_grid, 0.40, COST_BPS["btc"]),
        ("spx", "^GSPC", {"trend_ma": 200, "band": 0.01, "confirm": 3}, equity_grid, 0.15, EQUITY_COST_BPS),
    ]
    out = {}
    for name, ticker, params, grid, cur_target, cost in assets:
        _log(f"=== {name} 시작 ===")
        out[name] = run_asset(name, ticker, params, grid, cur_target, cost)
    with open("output/vol_target_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/vol_target_validation.json")


if __name__ == "__main__":
    main()
