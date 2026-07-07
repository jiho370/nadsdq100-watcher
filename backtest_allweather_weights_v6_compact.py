#!/usr/bin/env python3
"""
backtest_allweather_weights_v6.py — GLOBAL/REIT 제외, BTC 포함, KR/GOLD/BTC 제약 민감도 정적 자산배분 탐색기.

핵심 변화(v3 대비)
  1) 단순 CAGR 최고가 아니라 하드 위험 필터 -> percentile rank 위험조정 점수 -> robust 집계 순서로 후보를 고른다.
  2) allweather 제약 모드와 open 정적 최적화 모드를 분리한다.
  3) 기본 선택은 OOS 1위가 아니라 OOS + Full + rolling 5/10/15년 + 위기구간을 섞은 robust balanced다.
  4) 결과는 단일 우승자가 아니라 위험성향/기간/한국주식 버킷/pareto 후보군으로 출력한다.
  5) 주식총비중(equity_total)을 핵심 실험축으로 보고, 주식비중 버킷별 최고 후보와 위험지표를 별도 출력한다.

자산 순서
  US_STOCK / KR_STOCK / BTC / BOND / GOLD / REIT(고정 0)

실행 예시
  python backtest_allweather_weights_v3.py --step 5 --fine
  python backtest_allweather_weights_v6.py --step 5 --fine --mode both --output-dir output_v6
  python backtest_allweather_weights_v3.py --synthetic --step 10 --mode both --output-dir v5_test

출력
  output_v6/allweather_weight_results.csv
  output_v6/allweather_weight_winners.csv
  output_v6/allweather_weight_kr_buckets.csv
  output_v6/allweather_weight_pareto.csv
  output_v6/allweather_weight_filter_summary.csv
  output_v6/ALLWEATHER_WEIGHT_RESULT.md
  output_v6/best_allweather_weight_v6.json
  output_v6/best_static_open_v6.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ------------------------- 유니버스 -------------------------
TICKERS = {
    "US_STOCK": "SPY",
    "KR_STOCK": "^KS11",
    "BTC": "BTC-USD",
    "BOND": "IEF",
    "GOLD": "GLD",
    "REIT": "VNQ",
}
FX_TICKER = "KRW=X"
CASH_TICKER = "SHY"
KRW_CASH_RATE = 0.025
ASSET_KEYS = ["US_STOCK", "KR_STOCK", "BTC", "BOND", "GOLD", "REIT"]
# 주식비중 경로는 말 그대로 주식만 본다. BTC는 별도의 대체/고위험 자산으로 분리한다.
EQUITY_KEYS = ["US_STOCK", "KR_STOCK"]
RISK_ASSET_KEYS = ["US_STOCK", "KR_STOCK", "BTC"]

ORIGINAL_ALLWEATHER = {
    "US_STOCK": 35,
    "KR_STOCK": 10,
    "BTC": 0,
    "BOND": 30,
    "GOLD": 25,
    "REIT": 0,
}

ANCHOR_ALLOCS = {
    "kr10_btc0_gold25_35_10_0_30_25": {"US_STOCK": 35, "KR_STOCK": 10, "BTC": 0, "BOND": 30, "GOLD": 25, "REIT": 0},
    "kr10_btc3_gold25_35_10_3_27_25": {"US_STOCK": 35, "KR_STOCK": 10, "BTC": 3, "BOND": 27, "GOLD": 25, "REIT": 0},
    "kr15_btc3_gold22_35_15_3_25_22": {"US_STOCK": 35, "KR_STOCK": 15, "BTC": 3, "BOND": 25, "GOLD": 22, "REIT": 0},
    "kr15_btc3_gold25_32_15_3_25_25": {"US_STOCK": 32, "KR_STOCK": 15, "BTC": 3, "BOND": 25, "GOLD": 25, "REIT": 0},
    "kr20_btc3_gold22_30_20_3_25_22": {"US_STOCK": 30, "KR_STOCK": 20, "BTC": 3, "BOND": 25, "GOLD": 22, "REIT": 0},
    "growth_kr15_btc5_gold20_40_15_5_20_20": {"US_STOCK": 40, "KR_STOCK": 15, "BTC": 5, "BOND": 20, "GOLD": 20, "REIT": 0},
}

CRISES = [
    ("2008", "2007-10-31", "2009-03-31"),
    ("2020", "2020-01-31", "2020-04-30"),
    ("2022", "2022-01-31", "2022-10-31"),
]

PROFILE_CONFIG = {
    "min_drawdown": {
        "label": "낙폭최소형",
        "max_krw_mdd_abs": 10.0,
        "max_worst12_abs": 12.0,
        "max_krw_vol": 12.0,
        "max_turnover": 1.20,
        "min_effective_n_aw": 4.0,
        "min_effective_n_open": 3.0,
        "min_active_aw": 4,
        "min_active_open": 3,
        "max_single_aw": 35.0,
        "max_single_open": 45.0,
        "bond_min_aw": 25.0,
        "gold_max_aw": 25.0,
        "equity_max_aw": 50.0,
        "roll_p10_floor": 0.0,
        "roll_positive_min": 90.0,
        "score_weights": {"return": 0.10, "risk_adjusted": 0.25, "risk": 0.55, "turnover": 0.05, "diversification": 0.05},
        "description": "원화 기준 MDD, worst 12M, 변동성 방어를 최우선",
    },
    "defensive": {
        "label": "방어형",
        "max_krw_mdd_abs": 12.0,
        "max_worst12_abs": 15.0,
        "max_krw_vol": 14.0,
        "max_turnover": 1.50,
        "min_effective_n_aw": 3.5,
        "min_effective_n_open": 2.8,
        "min_active_aw": 4,
        "min_active_open": 3,
        "max_single_aw": 40.0,
        "max_single_open": 50.0,
        "bond_min_aw": 22.0,
        "gold_max_aw": 30.0,
        "equity_max_aw": 58.0,
        "roll_p10_floor": 0.0,
        "roll_positive_min": 85.0,
        "score_weights": {"return": 0.15, "risk_adjusted": 0.30, "risk": 0.45, "turnover": 0.05, "diversification": 0.05},
        "description": "MDD 약 -12% 이내를 우선하면서 위험조정수익 반영",
    },
    "balanced": {
        "label": "균형형",
        "max_krw_mdd_abs": 15.0,
        "max_worst12_abs": 18.0,
        "max_krw_vol": 17.0,
        "max_turnover": 1.80,
        "min_effective_n_aw": 3.3,
        "min_effective_n_open": 2.5,
        "min_active_aw": 4,
        "min_active_open": 3,
        "max_single_aw": 40.0,
        "max_single_open": 55.0,
        "bond_min_aw": 20.0,
        "gold_max_aw": 30.0,
        "equity_max_aw": 65.0,
        "roll_p10_floor": -1.0,
        "roll_positive_min": 80.0,
        "score_weights": {"return": 0.25, "risk_adjusted": 0.35, "risk": 0.30, "turnover": 0.05, "diversification": 0.05},
        "description": "수익률, 샤프/칼마/소르티노, 낙폭을 균형 있게 반영",
    },
    "growth": {
        "label": "성장형",
        "max_krw_mdd_abs": 22.0,
        "max_worst12_abs": 25.0,
        "max_krw_vol": 22.0,
        "max_turnover": 2.20,
        "min_effective_n_aw": 2.8,
        "min_effective_n_open": 2.2,
        "min_active_aw": 3,
        "min_active_open": 3,
        "max_single_aw": 45.0,
        "max_single_open": 65.0,
        "bond_min_aw": 15.0,
        "gold_max_aw": 35.0,
        "equity_max_aw": 75.0,
        "roll_p10_floor": -3.0,
        "roll_positive_min": 70.0,
        "score_weights": {"return": 0.45, "risk_adjusted": 0.30, "risk": 0.15, "turnover": 0.05, "diversification": 0.05},
        "description": "낙폭을 감수하되 위험조정 성과와 하방위험을 함께 반영",
    },
    "aggressive": {
        "label": "공격형",
        "max_krw_mdd_abs": 30.0,
        "max_worst12_abs": 35.0,
        "max_krw_vol": 30.0,
        "max_turnover": 3.00,
        "min_effective_n_aw": 2.5,
        "min_effective_n_open": 2.0,
        "min_active_aw": 3,
        "min_active_open": 2,
        "max_single_aw": 50.0,
        "max_single_open": 80.0,
        "bond_min_aw": 10.0,
        "gold_max_aw": 40.0,
        "equity_max_aw": 85.0,
        "roll_p10_floor": -5.0,
        "roll_positive_min": 60.0,
        "score_weights": {"return": 0.60, "risk_adjusted": 0.25, "risk": 0.10, "turnover": 0.03, "diversification": 0.02},
        "description": "수익률 비중이 가장 크지만 재앙적 낙폭은 필터링",
    },
}

ROBUST_CONTEXT_WEIGHTS = {
    "oos": 0.30,
    "full": 0.15,
    "roll_5y": 0.20,
    "roll_10y": 0.20,
    "roll_15y": 0.10,
    "crisis": 0.05,
}


def _log(msg: str) -> None:
    print(f"[AW-v6] {msg}", file=sys.stderr)


def _round_float(x: Any, ndigits: int = 4) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except TypeError:
        pass
    if isinstance(x, (np.floating, float, int, np.integer)):
        return round(float(x), ndigits)
    return x


# ------------------------- 데이터 -------------------------
def _add_synthetic_cash(df: pd.DataFrame) -> pd.DataFrame:
    n = np.arange(len(df))
    accr = (1 + KRW_CASH_RATE) ** (n / 252)
    df["KRW_CASH"] = (1.0 / df["FX"]) * accr * 1000.0
    return df


def fetch_prices(period: str = "25y") -> pd.DataFrame:
    import yfinance as yf

    tickers = list(TICKERS.values()) + [CASH_TICKER, FX_TICKER]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance에서 데이터를 받지 못했습니다. 네트워크 또는 티커 상태를 확인하세요.")
    close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
    df = close.rename(columns={v: k for k, v in TICKERS.items()})
    df = df.rename(columns={CASH_TICKER: "CASH", FX_TICKER: "FX"})
    if "FX" not in df.columns:
        raise RuntimeError("환율 KRW=X 데이터를 받지 못했습니다.")
    fx = df["FX"].ffill()
    df["KR_STOCK"] = df["KR_STOCK"] / fx
    df = df.drop(columns=["FX"]).join(fx.rename("FX"))
    needed = ASSET_KEYS + ["FX"]
    df = df[needed + (["CASH"] if "CASH" in df.columns else [])].dropna(how="any")
    df = _add_synthetic_cash(df)
    _log(f"데이터 {df.index[0].date()} ~ {df.index[-1].date()} · {len(df)}일")
    return df


def synthetic_prices(days: int = 5200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=dt.date.today(), periods=days)
    cols: dict[str, np.ndarray] = {}
    regime = np.sign(np.sin(np.arange(days) / 260 * math.pi)) * 0.0005
    for k in ASSET_KEYS + ["CASH", "FX"]:
        drift = 0.00022 if k in EQUITY_KEYS + ["REIT"] else 0.00008
        if k == "GOLD":
            drift = 0.00012
        if k == "BTC":
            drift = 0.00035
        if k == "FX":
            drift = 0.0
        vol = 0.012 if k in EQUITY_KEYS + ["REIT"] else 0.004
        if k == "GOLD":
            vol = 0.009
        if k == "BTC":
            vol = 0.035
        if k == "FX":
            vol = 0.006
        shock = regime * (1.5 if k in EQUITY_KEYS + ["REIT"] else -0.3)
        if k == "FX":
            shock = -0.2 * regime
        r = drift + shock + rng.normal(0, vol, days)
        cols[k] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(cols, index=idx)
    df["FX"] = 1300 * df["FX"] / df["FX"].iloc[0]
    return _add_synthetic_cash(df)


def monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    mret = df[ASSET_KEYS].pct_change().add(1).groupby(df.index.to_period("M")).prod().sub(1)
    mret.index = mret.index.to_timestamp("M")
    return mret.iloc[1:].copy()


def monthly_fx(df: pd.DataFrame) -> pd.Series:
    fx_m = df["FX"].groupby(df.index.to_period("M")).last()
    fx_m.index = fx_m.index.to_timestamp("M")
    return fx_m


# ------------------------- 비중 후보 생성 -------------------------
@dataclass(frozen=True)
class Constraints:
    mins: dict[str, int]
    maxs: dict[str, int]
    equity_min: int
    equity_max: int


def normalize_weights(w: dict[str, int]) -> dict[str, int]:
    return {k: int(w.get(k, 0)) for k in ASSET_KEYS}


def weight_key(w: dict[str, int]) -> tuple[int, ...]:
    ww = normalize_weights(w)
    return tuple(ww[k] for k in ASSET_KEYS)


def is_valid_weight(w: dict[str, int], c: Constraints) -> bool:
    ww = normalize_weights(w)
    if sum(ww.values()) != 100:
        return False
    for k in ASSET_KEYS:
        if ww[k] < c.mins[k] or ww[k] > c.maxs[k]:
            return False
    equity = sum(ww[k] for k in EQUITY_KEYS)
    return c.equity_min <= equity <= c.equity_max


def generate_weight_grid(step: int, c: Constraints) -> list[dict[str, int]]:
    if step <= 0 or 100 % step != 0:
        raise ValueError("--step은 100을 나눌 수 있는 양의 정수여야 합니다. 예: 1, 2, 4, 5, 10")
    ranges = {k: range(c.mins[k], c.maxs[k] + 1, step) for k in ASSET_KEYS}
    out: list[dict[str, int]] = []
    first_five = ASSET_KEYS[:-1]
    last = ASSET_KEYS[-1]
    for vals in itertools.product(*(ranges[k] for k in first_five)):
        w = dict(zip(first_five, vals))
        remain = 100 - sum(vals)
        if remain % step != 0:
            continue
        w[last] = remain
        if is_valid_weight(w, c):
            out.append(normalize_weights(w))
    return out


def add_anchor_allocs(weights: list[dict[str, int]], c: Constraints) -> tuple[list[dict[str, int]], dict[tuple[int, ...], str]]:
    seen = {weight_key(w) for w in weights}
    anchor_names: dict[tuple[int, ...], str] = {}
    for name, w in ANCHOR_ALLOCS.items():
        ww = normalize_weights(w)
        if is_valid_weight(ww, c):
            k = weight_key(ww)
            anchor_names[k] = name
            if k not in seen:
                weights.append(ww)
                seen.add(k)
    return weights, anchor_names


def local_integer_grid(center: dict[str, int], radius: int, c: Constraints) -> list[dict[str, int]]:
    if radius <= 0:
        return []
    center = normalize_weights(center)
    first_five = ASSET_KEYS[:-1]
    last = ASSET_KEYS[-1]
    ranges = []
    for k in first_five:
        lo = max(c.mins[k], center[k] - radius)
        hi = min(c.maxs[k], center[k] + radius)
        ranges.append(range(lo, hi + 1))
    out: list[dict[str, int]] = []
    for vals in itertools.product(*ranges):
        w = dict(zip(first_five, vals))
        remain = 100 - sum(vals)
        if abs(remain - center[last]) > radius:
            continue
        w[last] = remain
        if is_valid_weight(w, c):
            out.append(normalize_weights(w))
    return out


# ------------------------- 성과 계산 -------------------------
def curve_metrics_matrix(ret: np.ndarray, prefix: str = "") -> dict[str, np.ndarray]:
    ret = np.asarray(ret, dtype=float)
    if ret.ndim == 1:
        ret = ret.reshape(-1, 1)
    t, n = ret.shape
    nan_arr = np.full(n, np.nan, dtype=float)
    if t < 12:
        return {f"{prefix}{k}": nan_arr.copy() for k in [
            "cagr", "vol", "sharpe", "sortino", "mdd", "calmar", "worst_12m", "best_12m", "positive_month_rate", "final_multiple"
        ]}
    curve = np.cumprod(1.0 + ret, axis=0)
    years = t / 12.0
    cagr = np.power(curve[-1], 1.0 / years) - 1.0
    vol = np.std(ret, axis=0, ddof=1) * math.sqrt(12.0)
    ann_ret = np.mean(ret, axis=0) * 12.0
    sharpe = np.divide(ann_ret, vol, out=np.zeros_like(ann_ret), where=vol > 0)
    neg = np.where(ret < 0, ret, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        downside = np.nanstd(neg, axis=0, ddof=1) * math.sqrt(12.0)
        sortino = np.divide(ann_ret, downside, out=np.full_like(ann_ret, np.nan), where=downside > 0)
    running_max = np.maximum.accumulate(curve, axis=0)
    mdd = np.min(curve / running_max - 1.0, axis=0)
    calmar = np.divide(cagr, np.abs(mdd), out=np.full_like(cagr, np.nan), where=np.abs(mdd) > 0)
    if t > 12:
        roll12 = curve[12:] / curve[:-12] - 1.0
        worst12 = np.min(roll12, axis=0)
        best12 = np.max(roll12, axis=0)
    else:
        worst12 = nan_arr.copy()
        best12 = nan_arr.copy()
    positive = np.mean(ret > 0, axis=0)
    return {
        f"{prefix}cagr": 100.0 * cagr,
        f"{prefix}vol": 100.0 * vol,
        f"{prefix}sharpe": sharpe,
        f"{prefix}sortino": sortino,
        f"{prefix}mdd": 100.0 * mdd,
        f"{prefix}calmar": calmar,
        f"{prefix}worst_12m": 100.0 * worst12,
        f"{prefix}best_12m": 100.0 * best12,
        f"{prefix}positive_month_rate": 100.0 * positive,
        f"{prefix}final_multiple": curve[-1],
    }


def combined_metrics_matrix(usd_ret: np.ndarray, krw_ret: np.ndarray) -> dict[str, np.ndarray]:
    out = curve_metrics_matrix(usd_ret, prefix="")
    out.update(curve_metrics_matrix(krw_ret, prefix="krw_"))
    return out


def metric_dict_at(metric_arrays: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    return {k: _round_float(v[i], 4) for k, v in metric_arrays.items()}


def rolling_summary_matrix(krw_ret: np.ndarray, months: int) -> dict[str, np.ndarray]:
    krw_ret = np.asarray(krw_ret, dtype=float)
    if krw_ret.ndim == 1:
        krw_ret = krw_ret.reshape(-1, 1)
    t, n = krw_ret.shape
    if t < months:
        return {"n_windows": np.zeros(n, dtype=int)}
    curve = np.cumprod(1.0 + krw_ret, axis=0)
    n_windows = t - months + 1
    end_vals = curve[months - 1:]
    start_vals = np.ones((n_windows, n), dtype=float)
    if n_windows > 1:
        start_vals[1:] = curve[: t - months]
    period_growth = end_vals / start_vals
    years = months / 12.0
    cagrs = 100.0 * (np.power(period_growth, 1.0 / years) - 1.0)
    mdds = np.empty((n_windows, n), dtype=float)
    worst_12s = np.empty((n_windows, n), dtype=float) if months > 12 else None
    for start in range(n_windows):
        if start == 0:
            seg_curve = curve[:months]
        else:
            seg_curve = curve[start:start + months] / curve[start - 1]
        running_max = np.maximum.accumulate(seg_curve, axis=0)
        mdds[start] = 100.0 * np.min(seg_curve / running_max - 1.0, axis=0)
        if worst_12s is not None:
            twelve = seg_curve[12:] / seg_curve[:-12] - 1.0
            worst_12s[start] = 100.0 * np.min(twelve, axis=0) if len(twelve) else np.nan
    return {
        "n_windows": np.full(n, n_windows, dtype=int),
        "median_krw_cagr": np.median(cagrs, axis=0),
        "p10_krw_cagr": np.percentile(cagrs, 10, axis=0),
        "p25_krw_cagr": np.percentile(cagrs, 25, axis=0),
        "worst_krw_cagr": np.min(cagrs, axis=0),
        "best_krw_cagr": np.max(cagrs, axis=0),
        "positive_rate": 100.0 * np.mean(cagrs > 0, axis=0),
        "median_krw_mdd": np.median(mdds, axis=0),
        "worst_krw_mdd": np.min(mdds, axis=0),
        "p10_krw_mdd": np.percentile(mdds, 10, axis=0),
        "worst_12m": np.nanmin(worst_12s, axis=0) if worst_12s is not None else np.full(n, np.nan),
    }


def rolling_dict_at(rolling_arrays: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rolling_arrays.items():
        val = v[i]
        out[k] = int(val) if k == "n_windows" else _round_float(val, 4)
    return out


def crisis_returns_matrix(index: pd.DatetimeIndex, usd_ret: np.ndarray, krw_ret: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, a, b in CRISES:
        mask = (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
        if mask.any():
            out[name] = {
                "usd": 100.0 * (np.prod(1.0 + usd_ret[mask], axis=0) - 1.0),
                "krw": 100.0 * (np.prod(1.0 + krw_ret[mask], axis=0) - 1.0),
            }
    return out


def crisis_dict_at(crisis_arrays: dict[str, dict[str, np.ndarray]], i: int) -> dict[str, dict[str, float]]:
    return {name: {"usd": round(float(vals["usd"][i]), 2), "krw": round(float(vals["krw"][i]), 2)} for name, vals in crisis_arrays.items()}


def evaluate_weight_list(
    weights: list[dict[str, int]],
    mret: pd.DataFrame,
    fx_m: pd.Series,
    horizons: list[int],
    oos_frac: float,
    cost_oneway: float,
    anchor_names: dict[tuple[int, ...], str] | None = None,
) -> list[dict[str, Any]]:
    anchor_names = anchor_names or {}
    n = len(weights)
    if n == 0:
        return []
    w_mat = np.array([[w[k] for k in ASSET_KEYS] for w in weights], dtype=float) / 100.0
    r_assets = np.nan_to_num(mret[ASSET_KEYS].to_numpy(dtype=float), nan=0.0)
    gross = r_assets @ w_mat.T
    net = gross.copy()
    if len(r_assets) > 1 and cost_oneway > 0:
        grown = (1.0 + r_assets[:-1])[:, None, :] * w_mat[None, :, :]
        prev = grown / grown.sum(axis=2, keepdims=True)
        turn = np.abs(w_mat[None, :, :] - prev).sum(axis=2)
        net[1:] -= turn * cost_oneway
        turnover_yr = turn.mean(axis=0) * 12.0
    else:
        turnover_yr = np.zeros(n, dtype=float)
    fx = fx_m.reindex(mret.index, method="ffill")
    fx_ratio = (fx / fx.shift(1)).fillna(1.0).to_numpy(dtype=float)
    krw_net = (1.0 + net) * fx_ratio.reshape(-1, 1) - 1.0
    split = int(len(mret) * (1.0 - oos_frac))
    split = max(12, min(split, len(mret) - 12))
    metric_arrays = {
        "is": combined_metrics_matrix(net[:split], krw_net[:split]),
        "oos": combined_metrics_matrix(net[split:], krw_net[split:]),
        "full": combined_metrics_matrix(net, krw_net),
    }
    rolling_arrays = {str(h): rolling_summary_matrix(krw_net, h * 12) for h in horizons}
    crisis_arrays = crisis_returns_matrix(mret.index, net, krw_net)
    results: list[dict[str, Any]] = []
    for i, w in enumerate(weights):
        if (i + 1) % 25000 == 0:
            _log(f"결과 조립 중 {i + 1:,}/{n:,}")
        results.append({
            "weights": normalize_weights(w),
            "anchor": anchor_names.get(weight_key(w)),
            "equity_total": sum(w[k] for k in EQUITY_KEYS),
            "foreign_equity": w["US_STOCK"],
            "btc_weight": w["BTC"],
            "risk_asset_total": sum(w[k] for k in RISK_ASSET_KEYS),
            "turnover_yr": round(float(turnover_yr[i]), 4),
            "is": metric_dict_at(metric_arrays["is"], i),
            "oos": metric_dict_at(metric_arrays["oos"], i),
            "full": metric_dict_at(metric_arrays["full"], i),
            "rolling": {str(h): rolling_dict_at(rolling_arrays[str(h)], i) for h in horizons},
            "crisis": crisis_dict_at(crisis_arrays, i),
        })
    return results


# ------------------------- DataFrame 변환/스코어링 -------------------------
def flatten_result(result: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    w = result["weights"]
    row: dict[str, Any] = {k: w[k] for k in ASSET_KEYS}
    row.update({
        "weight_str": format_weights(w),
        "anchor": result.get("anchor"),
        "equity_total": result["equity_total"],
        "foreign_equity": result["foreign_equity"],
        "btc_weight": result.get("btc_weight", 0),
        "risk_asset_total": result.get("risk_asset_total", result["equity_total"]),
        "turnover_yr": result["turnover_yr"],
    })
    for ctx in ["is", "oos", "full"]:
        for k, v in result[ctx].items():
            row[f"{ctx}_{k}"] = v
    for h in horizons:
        for k, v in result["rolling"].get(str(h), {}).items():
            row[f"roll_{h}y_{k}"] = v
    crisis_krw = []
    crisis_usd = []
    for name, _, _ in CRISES:
        vals = result["crisis"].get(name, {})
        row[f"crisis_{name}_krw"] = vals.get("krw")
        row[f"crisis_{name}_usd"] = vals.get("usd")
        if vals.get("krw") is not None:
            crisis_krw.append(float(vals["krw"]))
        if vals.get("usd") is not None:
            crisis_usd.append(float(vals["usd"]))
    row["crisis_worst_krw"] = min(crisis_krw) if crisis_krw else np.nan
    row["crisis_avg_krw"] = float(np.mean(crisis_krw)) if crisis_krw else np.nan
    row["crisis_worst_usd"] = min(crisis_usd) if crisis_usd else np.nan
    row["crisis_avg_usd"] = float(np.mean(crisis_usd)) if crisis_usd else np.nan
    return row


def format_weights(w: dict[str, int] | pd.Series | dict[str, Any]) -> str:
    return f"{int(w['US_STOCK'])}/{int(w['KR_STOCK'])}/{int(w['BTC'])}/{int(w['BOND'])}/{int(w['GOLD'])}/{int(w['REIT'])}"


def add_composition_columns(df: pd.DataFrame, constraints: Constraints) -> pd.DataFrame:
    out = df.copy()
    vals = out[ASSET_KEYS].astype(float)
    out["max_weight"] = vals.max(axis=1)
    out["min_weight"] = vals.min(axis=1)
    out["active_assets_ge3"] = (vals >= 3.0).sum(axis=1)
    out["active_assets_gt0"] = (vals > 0.0).sum(axis=1)
    hhi = ((vals / 100.0) ** 2).sum(axis=1)
    out["hhi"] = hhi
    out["effective_n"] = 1.0 / hhi.replace(0, np.nan)
    out["btc_reit_total"] = out["BTC"] + out["REIT"]
    out["btc_gold_total"] = out["BTC"] + out["GOLD"]
    out["risk_asset_total"] = out[[k for k in RISK_ASSET_KEYS if k in out.columns]].sum(axis=1)
    out["real_asset_total"] = out["GOLD"] + out["REIT"]
    out["bond_gold_total"] = out["BOND"] + out["GOLD"]
    boundary = np.zeros(len(out), dtype=int)
    for k in ASSET_KEYS:
        boundary += ((out[k] <= constraints.mins[k]) | (out[k] >= constraints.maxs[k])).astype(int).to_numpy()
    out["boundary_hits"] = boundary
    # 경계값/집중 벌점은 출력용. 점수에서는 percentile rank로 다시 반영한다.
    out["concentration_raw"] = (
        20.0 * out["hhi"]
        + 0.12 * np.maximum(out["max_weight"] - 40.0, 0.0)
        + 0.18 * np.maximum(4.0 - out["effective_n"], 0.0)
        + 0.20 * out["boundary_hits"]
    )
    return out


def mode_base_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "open":
        return pd.Series(True, index=df.index)
    if mode == "allweather":
        return (
            (df["BOND"] >= 20)
            & (df["GOLD"] >= 5)
            & (df["active_assets_ge3"] >= 4)
            & (df["effective_n"] >= 2.8)
            & (df["max_weight"] <= 45)
        )
    raise ValueError(f"알 수 없는 mode: {mode}")


def profile_composition_mask(df: pd.DataFrame, mode: str, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    if mode == "allweather":
        return (
            (df["max_weight"] <= cfg["max_single_aw"])
            & (df["effective_n"] >= cfg["min_effective_n_aw"])
            & (df["active_assets_ge3"] >= cfg["min_active_aw"])
            & (df["BOND"] >= cfg["bond_min_aw"])
            & (df["GOLD"] <= cfg["gold_max_aw"])
            & (df["equity_total"] <= cfg["equity_max_aw"])
        )
    if mode == "open":
        return (
            (df["max_weight"] <= cfg["max_single_open"])
            & (df["effective_n"] >= cfg["min_effective_n_open"])
            & (df["active_assets_gt0"] >= cfg["min_active_open"])
        )
    raise ValueError(f"알 수 없는 mode: {mode}")


def path_filter_mask(df: pd.DataFrame, context: str, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    return (
        (df[f"{context}_krw_mdd"] >= -cfg["max_krw_mdd_abs"])
        & (df[f"{context}_krw_worst_12m"] >= -cfg["max_worst12_abs"])
        & (df[f"{context}_krw_vol"] <= cfg["max_krw_vol"])
        & (df["turnover_yr"] <= cfg["max_turnover"])
    )


def rolling_filter_mask(df: pd.DataFrame, context: str, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    required = [
        f"{context}_worst_krw_mdd",
        f"{context}_p10_krw_cagr",
        f"{context}_positive_rate",
        "turnover_yr",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # BTC 등 짧은 데이터로 인해 roll_15y_* 같은 컬럼이 없을 수 있다.
        # 이 context는 필터 통과 후보 없음으로 처리하고, 호출부에서 점수도 건너뛴다.
        return pd.Series(False, index=df.index)
    return (
        (df[f"{context}_worst_krw_mdd"] >= -cfg["max_krw_mdd_abs"])
        & (df[f"{context}_p10_krw_cagr"] >= cfg["roll_p10_floor"])
        & (df[f"{context}_positive_rate"] >= cfg["roll_positive_min"])
        & (df["turnover_yr"] <= cfg["max_turnover"])
    )


def crisis_filter_mask(df: pd.DataFrame, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    # 위기구간은 전체 포트폴리오보다 짧고 강하므로 path mdd보다 약간 느슨하게 본다.
    cap = cfg["max_krw_mdd_abs"] + 8.0
    return (df["crisis_worst_krw"] >= -cap) & (df["turnover_yr"] <= cfg["max_turnover"])


def pct_rank(s: pd.Series, high_good: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)
    return (100.0 * x.rank(pct=True, ascending=True if high_good else False)).fillna(0.0)


def score_path_dataframe(pool: pd.DataFrame, context: str, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    w = cfg["score_weights"]
    ret_score = pct_rank(pool[f"{context}_krw_cagr"], True)
    risk_adj_score = pd.concat([
        pct_rank(pool[f"{context}_krw_sharpe"], True),
        pct_rank(pool[f"{context}_krw_sortino"], True),
        pct_rank(pool[f"{context}_krw_calmar"], True),
    ], axis=1).mean(axis=1)
    risk_score = pd.concat([
        pct_rank(pool[f"{context}_krw_mdd"], True),
        pct_rank(pool[f"{context}_krw_worst_12m"], True),
        pct_rank(pool[f"{context}_krw_vol"], False),
    ], axis=1).mean(axis=1)
    turnover_score = pct_rank(pool["turnover_yr"], False)
    div_score = pd.concat([
        pct_rank(pool["effective_n"], True),
        pct_rank(pool["concentration_raw"], False),
        pct_rank(pool["boundary_hits"], False),
    ], axis=1).mean(axis=1)
    return (
        w["return"] * ret_score
        + w["risk_adjusted"] * risk_adj_score
        + w["risk"] * risk_score
        + w["turnover"] * turnover_score
        + w["diversification"] * div_score
    )


def score_rolling_dataframe(pool: pd.DataFrame, context: str, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    w = cfg["score_weights"]
    ret_score = pd.concat([
        pct_rank(pool[f"{context}_median_krw_cagr"], True),
        pct_rank(pool[f"{context}_p10_krw_cagr"], True),
        pct_rank(pool[f"{context}_worst_krw_cagr"], True),
    ], axis=1).mean(axis=1)
    risk_adj_score = pd.concat([
        pct_rank(pool[f"{context}_p10_krw_cagr"], True),
        pct_rank(pool[f"{context}_positive_rate"], True),
    ], axis=1).mean(axis=1)
    risk_score = pd.concat([
        pct_rank(pool[f"{context}_worst_krw_mdd"], True),
        pct_rank(pool[f"{context}_p10_krw_mdd"], True),
        pct_rank(pool[f"{context}_worst_12m"], True),
    ], axis=1).mean(axis=1)
    turnover_score = pct_rank(pool["turnover_yr"], False)
    div_score = pd.concat([
        pct_rank(pool["effective_n"], True),
        pct_rank(pool["concentration_raw"], False),
        pct_rank(pool["boundary_hits"], False),
    ], axis=1).mean(axis=1)
    return (
        w["return"] * ret_score
        + w["risk_adjusted"] * risk_adj_score
        + w["risk"] * risk_score
        + w["turnover"] * turnover_score
        + w["diversification"] * div_score
    )


def score_crisis_dataframe(pool: pd.DataFrame, profile: str) -> pd.Series:
    cfg = PROFILE_CONFIG[profile]
    w = cfg["score_weights"]
    crisis_return = pd.concat([
        pct_rank(pool["crisis_worst_krw"], True),
        pct_rank(pool["crisis_avg_krw"], True),
    ], axis=1).mean(axis=1)
    div_score = pd.concat([
        pct_rank(pool["effective_n"], True),
        pct_rank(pool["concentration_raw"], False),
    ], axis=1).mean(axis=1)
    # 위기에서는 수익률보다 worst crisis 방어 비중을 더 크게 본다.
    return 0.75 * crisis_return + 0.15 * pct_rank(pool["turnover_yr"], False) + 0.10 * div_score + 5.0 * w["risk"]


def score_context_inplace(df: pd.DataFrame, context: str, profile: str, modes: list[str]) -> list[dict[str, Any]]:
    """mode/profile/context별 필터와 점수를 계산해 df에 컬럼을 추가한다.

    점수는 전체 후보에 대한 절대 점수가 아니라 해당 mode/profile/context의 필터 통과 후보 안에서의
    percentile rank 기반 점수다. 따라서 서로 다른 context 간 점수 크기 비교보다 순위 비교에 적합하다.
    """
    summaries: list[dict[str, Any]] = []
    for mode in modes:
        base = mode_base_mask(df, mode) & profile_composition_mask(df, mode, profile)
        if context in ("is", "oos", "full"):
            hard = base & path_filter_mask(df, context, profile)
            scorer = lambda p: score_path_dataframe(p, context, profile)
        elif context.startswith("roll_"):
            required = [
                f"{context}_median_krw_cagr",
                f"{context}_p10_krw_cagr",
                f"{context}_worst_krw_cagr",
                f"{context}_worst_krw_mdd",
                f"{context}_p10_krw_mdd",
                f"{context}_worst_12m",
                f"{context}_positive_rate",
            ]
            missing = [c for c in required if c not in df.columns]
            col_score = f"score_{mode}_{context}_{profile}"
            col_pass = f"pass_{mode}_{context}_{profile}"
            if missing:
                df[col_pass] = False
                df[col_score] = -1.0
                summaries.append({
                    "mode": mode,
                    "context": context,
                    "profile": profile,
                    "n_total": int(len(df)),
                    "n_mode_base": int(base.sum()),
                    "n_pass": 0,
                    "cap_relaxed": True,
                    "skipped_missing_columns": ",".join(missing),
                })
                continue
            hard = base & rolling_filter_mask(df, context, profile)
            scorer = lambda p: score_rolling_dataframe(p, context, profile)
        elif context == "crisis":
            hard = base & crisis_filter_mask(df, profile)
            scorer = lambda p: score_crisis_dataframe(p, profile)
        else:
            raise KeyError(context)

        col_score = f"score_{mode}_{context}_{profile}"
        col_pass = f"pass_{mode}_{context}_{profile}"
        df[col_pass] = hard
        df[col_score] = np.nan
        relaxed = False
        pool_idx = df.index[hard]
        if len(pool_idx) == 0:
            relaxed = True
            pool_idx = df.index[base]
        if len(pool_idx) == 0:
            relaxed = True
            pool_idx = df.index
        pool = df.loc[pool_idx]
        df.loc[pool_idx, col_score] = scorer(pool)
        # 통과 못 한 후보도 fallback 순위 확인용으로 매우 낮은 보조 점수를 둔다.
        df[col_score] = df[col_score].fillna(-1.0)
        summaries.append({
            "mode": mode,
            "context": context,
            "profile": profile,
            "n_total": int(len(df)),
            "n_mode_base": int(base.sum()),
            "n_pass": int(hard.sum()),
            "cap_relaxed": bool(relaxed),
        })
    return summaries


def add_v3_scores(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    contexts = ["oos", "full"] + [f"roll_{h}y" for h in horizons] + ["crisis"]
    filter_rows: list[dict[str, Any]] = []
    for profile in PROFILE_CONFIG:
        for ctx in contexts:
            filter_rows.extend(score_context_inplace(out, ctx, profile, modes))
    for mode in modes:
        for profile in PROFILE_CONFIG:
            pieces = []
            total_w = 0.0
            for ctx, weight in ROBUST_CONTEXT_WEIGHTS.items():
                if ctx.startswith("roll_") and int(ctx.split("_")[1].replace("y", "")) not in horizons:
                    continue
                col = f"score_{mode}_{ctx}_{profile}"
                if col in out.columns:
                    pieces.append((col, weight))
                    total_w += weight
            robust_col = f"score_{mode}_robust_{profile}"
            pass_col = f"pass_{mode}_robust_{profile}"
            if pieces and total_w > 0:
                score = sum((w / total_w) * out[col] for col, w in pieces)
                # robust는 OOS와 Full 둘 다 하드필터 통과를 우선한다. 없으면 점수는 남기되 pass는 False.
                p_oos = out.get(f"pass_{mode}_oos_{profile}", pd.Series(False, index=out.index))
                p_full = out.get(f"pass_{mode}_full_{profile}", pd.Series(False, index=out.index))
                out[pass_col] = p_oos & p_full & mode_base_mask(out, mode) & profile_composition_mask(out, mode, profile)
                out[robust_col] = score
            else:
                out[pass_col] = False
                out[robust_col] = -1.0
    return out, pd.DataFrame(filter_rows)


def prepare_dataframe(results: list[dict[str, Any]], horizons: list[int], constraints: Constraints, modes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame([flatten_result(r, horizons) for r in results])
    df = add_composition_columns(df, constraints)
    df, filter_summary = add_v3_scores(df, horizons, modes)
    return df, filter_summary


# ------------------------- 후보 선택 -------------------------
def row_to_winner(row: pd.Series, *, mode: str, context: str, profile: str, score: float, cap_relaxed: bool) -> dict[str, Any]:
    out = {
        "mode": mode,
        "context": context,
        "profile": profile,
        "profile_label": PROFILE_CONFIG[profile]["label"],
        "cap_relaxed": bool(cap_relaxed),
        "score": round(float(score), 4),
        "anchor": row.get("anchor"),
        "US_STOCK": int(row["US_STOCK"]),
        "KR_STOCK": int(row["KR_STOCK"]),
        "BTC": int(row["BTC"]),
        "BOND": int(row["BOND"]),
        "GOLD": int(row["GOLD"]),
        "REIT": int(row["REIT"]),
        "weight_str": format_weights(row),
        "equity_total": _round_float(row.get("equity_total"), 2),
        "effective_n": _round_float(row.get("effective_n"), 3),
        "max_weight": _round_float(row.get("max_weight"), 2),
        "active_assets_ge3": int(row.get("active_assets_ge3", 0)),
        "boundary_hits": int(row.get("boundary_hits", 0)),
        "turnover_yr": _round_float(row.get("turnover_yr"), 4),
    }
    if context in ("oos", "full"):
        out.update({
            "krw_cagr": row.get(f"{context}_krw_cagr"),
            "krw_mdd": row.get(f"{context}_krw_mdd"),
            "krw_sharpe": row.get(f"{context}_krw_sharpe"),
            "krw_calmar": row.get(f"{context}_krw_calmar"),
            "krw_worst_12m": row.get(f"{context}_krw_worst_12m"),
            "cagr": row.get(f"{context}_cagr"),
            "mdd": row.get(f"{context}_mdd"),
            "sharpe": row.get(f"{context}_sharpe"),
        })
    elif context.startswith("roll_"):
        out.update({
            "median_krw_cagr": row.get(f"{context}_median_krw_cagr"),
            "p10_krw_cagr": row.get(f"{context}_p10_krw_cagr"),
            "worst_krw_cagr": row.get(f"{context}_worst_krw_cagr"),
            "positive_rate": row.get(f"{context}_positive_rate"),
            "median_krw_mdd": row.get(f"{context}_median_krw_mdd"),
            "worst_krw_mdd": row.get(f"{context}_worst_krw_mdd"),
            "n_windows": row.get(f"{context}_n_windows"),
        })
    elif context == "crisis":
        out.update({
            "crisis_worst_krw": row.get("crisis_worst_krw"),
            "crisis_avg_krw": row.get("crisis_avg_krw"),
            "crisis_2008_krw": row.get("crisis_2008_krw"),
            "crisis_2020_krw": row.get("crisis_2020_krw"),
            "crisis_2022_krw": row.get("crisis_2022_krw"),
        })
    elif context == "robust":
        out.update({
            "oos_krw_cagr": row.get("oos_krw_cagr"),
            "oos_krw_mdd": row.get("oos_krw_mdd"),
            "oos_krw_sharpe": row.get("oos_krw_sharpe"),
            "full_krw_cagr": row.get("full_krw_cagr"),
            "full_krw_mdd": row.get("full_krw_mdd"),
            "crisis_worst_krw": row.get("crisis_worst_krw"),
        })
    return {k: _round_float(v, 4) for k, v in out.items()}


def choose_winners(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contexts = ["oos", "full"] + [f"roll_{h}y" for h in horizons] + ["crisis", "robust"]
    for mode in modes:
        for context in contexts:
            for profile in PROFILE_CONFIG:
                score_col = f"score_{mode}_{context}_{profile}"
                pass_col = f"pass_{mode}_{context}_{profile}"
                if score_col not in df.columns:
                    continue
                pool = df[df[pass_col]] if pass_col in df.columns and df[pass_col].any() else df[mode_base_mask(df, mode)]
                cap_relaxed = False
                if pool.empty:
                    pool = df
                    cap_relaxed = True
                elif pass_col in df.columns and not df[pass_col].any():
                    cap_relaxed = True
                best_idx = pool[score_col].idxmax()
                rows.append(row_to_winner(df.loc[best_idx], mode=mode, context=context, profile=profile, score=df.loc[best_idx, score_col], cap_relaxed=cap_relaxed))
    return pd.DataFrame(rows)


def kr_bucket(kr_weight: int) -> str:
    if kr_weight <= 5:
        return "00-05"
    if kr_weight <= 10:
        return "06-10"
    if kr_weight <= 15:
        return "11-15"
    if kr_weight <= 20:
        return "16-20"
    if kr_weight <= 25:
        return "21-25"
    if kr_weight <= 30:
        return "26-30"
    return "31+"


def build_kr_bucket_summary(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contexts = ["oos", "full", "robust"] + (["roll_5y"] if 5 in horizons else [])
    temp = df.copy()
    temp["kr_bucket"] = temp["KR_STOCK"].apply(lambda x: kr_bucket(int(x)))
    for mode in modes:
        for context in contexts:
            for profile in ["defensive", "balanced", "growth"]:
                score_col = f"score_{mode}_{context}_{profile}"
                if score_col not in temp.columns:
                    continue
                for b, pool in temp.groupby("kr_bucket"):
                    pool = pool[mode_base_mask(pool, mode)]
                    if pool.empty:
                        continue
                    idx = pool[score_col].idxmax()
                    row = row_to_winner(temp.loc[idx], mode=mode, context=context, profile=profile, score=temp.loc[idx, score_col], cap_relaxed=False)
                    row["kr_bucket"] = b
                    rows.append(row)
    return pd.DataFrame(rows)





def btc_bucket(btc_weight: int) -> str:
    if btc_weight == 0:
        return "00"
    if btc_weight <= 3:
        return "01-03"
    if btc_weight <= 5:
        return "04-05"
    if btc_weight <= 10:
        return "06-10"
    if btc_weight <= 15:
        return "11-15"
    if btc_weight <= 20:
        return "16-20"
    return "21+"


def build_btc_bucket_summary(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> pd.DataFrame:
    """BTC 비중별 robust balanced 최선 후보를 비교한다."""
    rows: list[dict[str, Any]] = []
    contexts = ["oos", "full", "robust"] + (["roll_5y"] if 5 in horizons else [])
    temp = df.copy()
    temp["btc_bucket"] = temp["BTC"].apply(lambda x: btc_bucket(int(x)))
    for mode in modes:
        for context in contexts:
            for profile in ["defensive", "balanced", "growth"]:
                score_col = f"score_{mode}_{context}_{profile}"
                if score_col not in temp.columns:
                    continue
                for b, pool in temp.groupby("btc_bucket"):
                    pool = pool[mode_base_mask(pool, mode)]
                    if pool.empty:
                        continue
                    pass_col = f"pass_{mode}_{context}_{profile}"
                    if pass_col in pool.columns and pool[pass_col].any():
                        sel_pool = pool[pool[pass_col]]
                    else:
                        sel_pool = pool
                    idx = sel_pool[score_col].idxmax()
                    row = row_to_winner(temp.loc[idx], mode=mode, context=context, profile=profile, score=temp.loc[idx, score_col], cap_relaxed=False)
                    row["btc_bucket"] = b
                    row.update({
                        "oos_krw_cagr": temp.loc[idx].get("oos_krw_cagr"),
                        "oos_krw_mdd": temp.loc[idx].get("oos_krw_mdd"),
                        "full_krw_cagr": temp.loc[idx].get("full_krw_cagr"),
                        "full_krw_mdd": temp.loc[idx].get("full_krw_mdd"),
                        "crisis_worst_krw": temp.loc[idx].get("crisis_worst_krw"),
                    })
                    rows.append({k: _round_float(v, 4) for k, v in row.items()})
    return pd.DataFrame(rows)


def reit_bucket(reit_weight: int) -> str:
    if reit_weight == 0:
        return "00"
    if reit_weight <= 5:
        return "01-05"
    if reit_weight <= 10:
        return "06-10"
    if reit_weight <= 15:
        return "11-15"
    return "16+"


def build_reit_bucket_summary(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> pd.DataFrame:
    """REIT 비중별로 robust balanced 최선 후보를 비교한다."""
    rows: list[dict[str, Any]] = []
    contexts = ["oos", "full", "robust"] + (["roll_5y"] if 5 in horizons else [])
    temp = df.copy()
    temp["reit_bucket"] = temp["REIT"].apply(lambda x: reit_bucket(int(x)))
    for mode in modes:
        for context in contexts:
            for profile in ["defensive", "balanced", "growth"]:
                score_col = f"score_{mode}_{context}_{profile}"
                if score_col not in temp.columns:
                    continue
                for b, pool in temp.groupby("reit_bucket"):
                    pool = pool[mode_base_mask(pool, mode)]
                    if pool.empty:
                        continue
                    pass_col = f"pass_{mode}_{context}_{profile}"
                    if pass_col in pool.columns and pool[pass_col].any():
                        sel_pool = pool[pool[pass_col]]
                    else:
                        sel_pool = pool
                    idx = sel_pool[score_col].idxmax()
                    row = row_to_winner(temp.loc[idx], mode=mode, context=context, profile=profile, score=temp.loc[idx, score_col], cap_relaxed=False)
                    row["reit_bucket"] = b
                    row.update({
                        "oos_krw_cagr": temp.loc[idx].get("oos_krw_cagr"),
                        "oos_krw_mdd": temp.loc[idx].get("oos_krw_mdd"),
                        "full_krw_cagr": temp.loc[idx].get("full_krw_cagr"),
                        "full_krw_mdd": temp.loc[idx].get("full_krw_mdd"),
                        "crisis_worst_krw": temp.loc[idx].get("crisis_worst_krw"),
                    })
                    rows.append({k: _round_float(v, 4) for k, v in row.items()})
    return pd.DataFrame(rows)

def equity_bucket(eq_weight: float) -> str:
    x = float(eq_weight)
    if x < 30:
        return "20-29"
    if x < 40:
        return "30-39"
    if x < 50:
        return "40-49"
    if x < 60:
        return "50-59"
    if x < 70:
        return "60-69"
    if x < 80:
        return "70-79"
    return "80-90"


def build_equity_bucket_summary(df: pd.DataFrame, horizons: list[int], modes: list[str]) -> pd.DataFrame:
    """주식총비중 버킷별로 최선 후보와 위험지표를 뽑는다.

    이 표가 v5의 핵심이다. 주식 30/50/70/80% 근처에서 원화 기준 CAGR, MDD,
    Sharpe, rolling 하위 10% 성과가 어떻게 바뀌는지 직접 비교할 수 있다.
    """
    rows: list[dict[str, Any]] = []
    temp = df.copy()
    temp["equity_bucket"] = temp["equity_total"].apply(equity_bucket)
    contexts = ["oos", "full", "robust"]
    for h in [5, 10, 15]:
        if h in horizons:
            contexts.append(f"roll_{h}y")
    for mode in modes:
        base_mode = temp[mode_base_mask(temp, mode)].copy()
        if base_mode.empty:
            continue
        for context in contexts:
            for profile in PROFILE_CONFIG:
                score_col = f"score_{mode}_{context}_{profile}"
                if score_col not in base_mode.columns:
                    continue
                for b, pool in base_mode.groupby("equity_bucket"):
                    if pool.empty:
                        continue
                    # 해당 profile의 하드 필터 통과 후보가 있으면 우선 사용한다.
                    pass_col = f"pass_{mode}_{context}_{profile}"
                    if pass_col in pool.columns and pool[pass_col].any():
                        sel_pool = pool[pool[pass_col]]
                    else:
                        sel_pool = pool
                    idx = sel_pool[score_col].idxmax()
                    r = temp.loc[idx]
                    row = row_to_winner(r, mode=mode, context=context, profile=profile, score=r.get(score_col, np.nan), cap_relaxed=False)
                    row["equity_bucket"] = b
                    # 비교용 공통 지표를 항상 붙인다.
                    row.update({
                        "oos_krw_cagr": r.get("oos_krw_cagr"),
                        "oos_krw_mdd": r.get("oos_krw_mdd"),
                        "oos_krw_sharpe": r.get("oos_krw_sharpe"),
                        "oos_krw_calmar": r.get("oos_krw_calmar"),
                        "oos_krw_worst_12m": r.get("oos_krw_worst_12m"),
                        "full_krw_cagr": r.get("full_krw_cagr"),
                        "full_krw_mdd": r.get("full_krw_mdd"),
                        "full_krw_sharpe": r.get("full_krw_sharpe"),
                        "crisis_worst_krw": r.get("crisis_worst_krw"),
                    })
                    for h in [5, 10, 15]:
                        if h in horizons:
                            row[f"roll_{h}y_median_krw_cagr"] = r.get(f"roll_{h}y_median_krw_cagr")
                            row[f"roll_{h}y_p10_krw_cagr"] = r.get(f"roll_{h}y_p10_krw_cagr")
                            row[f"roll_{h}y_worst_krw_mdd"] = r.get(f"roll_{h}y_worst_krw_mdd")
                    rows.append({k: _round_float(v, 4) for k, v in row.items()})
    return pd.DataFrame(rows)


def build_equity_risk_curve(df: pd.DataFrame, modes: list[str], profile: str = "balanced") -> pd.DataFrame:
    """주식비중 버킷별 위험-수익 곡선 요약.

    robust balanced 점수가 가장 높은 후보를 버킷별 대표로 뽑고,
    같은 버킷 안의 최저 MDD/최고 CAGR 후보도 참고값으로 보여준다.
    """
    rows: list[dict[str, Any]] = []
    temp = df.copy()
    temp["equity_bucket"] = temp["equity_total"].apply(equity_bucket)
    for mode in modes:
        base = temp[mode_base_mask(temp, mode)].copy()
        if base.empty:
            continue
        score_col = f"score_{mode}_robust_{profile}"
        pass_col = f"pass_{mode}_robust_{profile}"
        for b, pool in base.groupby("equity_bucket"):
            if pool.empty or score_col not in pool.columns:
                continue
            pass_pool = pool[pool[pass_col]] if pass_col in pool.columns and pool[pass_col].any() else pool
            best = pass_pool.loc[pass_pool[score_col].idxmax()]
            min_mdd = pool.loc[pool["oos_krw_mdd"].idxmax()]  # 덜 음수일수록 좋다.
            max_cagr = pool.loc[pool["oos_krw_cagr"].idxmax()]
            rows.append({
                "mode": mode,
                "profile": profile,
                "equity_bucket": b,
                "best_weight": best["weight_str"],
                "best_equity_total": _round_float(best["equity_total"], 2),
                "best_score": _round_float(best[score_col], 4),
                "best_oos_krw_cagr": _round_float(best["oos_krw_cagr"], 4),
                "best_oos_krw_mdd": _round_float(best["oos_krw_mdd"], 4),
                "best_oos_krw_sharpe": _round_float(best["oos_krw_sharpe"], 4),
                "best_full_krw_cagr": _round_float(best["full_krw_cagr"], 4),
                "best_full_krw_mdd": _round_float(best["full_krw_mdd"], 4),
                "best_crisis_worst_krw": _round_float(best["crisis_worst_krw"], 4),
                "best_effective_n": _round_float(best["effective_n"], 3),
                "best_boundary_hits": int(best.get("boundary_hits", 0)),
                "min_mdd_weight": min_mdd["weight_str"],
                "min_mdd_oos_krw_mdd": _round_float(min_mdd["oos_krw_mdd"], 4),
                "min_mdd_oos_krw_cagr": _round_float(min_mdd["oos_krw_cagr"], 4),
                "max_cagr_weight": max_cagr["weight_str"],
                "max_cagr_oos_krw_cagr": _round_float(max_cagr["oos_krw_cagr"], 4),
                "max_cagr_oos_krw_mdd": _round_float(max_cagr["oos_krw_mdd"], 4),
                "n_candidates": int(len(pool)),
                "n_pass": int(len(pass_pool)),
            })
    order = {"20-29": 0, "30-39": 1, "40-49": 2, "50-59": 3, "60-69": 4, "70-79": 5, "80-90": 6}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_ord"] = out["equity_bucket"].map(order)
        out = out.sort_values(["mode", "_ord"]).drop(columns=["_ord"])
    return out

def build_pareto_frontier(df: pd.DataFrame, modes: list[str], profile: str = "balanced", max_rows: int = 200) -> pd.DataFrame:
    rows = []
    for mode in modes:
        base = df[mode_base_mask(df, mode)].copy()
        if base.empty:
            continue
        # OOS 원화 CAGR 높고 OOS 원화 MDD 덜 나쁜 조합 중 비지배 후보.
        base = base.sort_values(["oos_krw_cagr", "oos_krw_mdd"], ascending=[False, False])
        frontier_idx = []
        best_mdd = -1e9
        for idx, r in base.iterrows():
            mdd = float(r.get("oos_krw_mdd", -1e9))
            if mdd > best_mdd:
                frontier_idx.append(idx)
                best_mdd = mdd
        f = base.loc[frontier_idx].copy()
        f["mode"] = mode
        f["pareto_basis"] = "oos_krw_cagr_vs_oos_krw_mdd"
        score_col = f"score_{mode}_robust_{profile}"
        keep_cols = ["mode", "pareto_basis", "weight_str"] + ASSET_KEYS + [
            "equity_total", "effective_n", "max_weight", "active_assets_ge3", "turnover_yr",
            "oos_krw_cagr", "oos_krw_mdd", "oos_krw_sharpe", "oos_krw_calmar",
            "full_krw_cagr", "full_krw_mdd", "crisis_worst_krw", score_col,
        ]
        rows.append(f[[c for c in keep_cols if c in f.columns]].head(max_rows))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_fine_centers_v3(scored: pd.DataFrame, n: int, modes: list[str]) -> list[dict[str, int]]:
    if n <= 0 or scored.empty:
        return []
    candidates: dict[tuple[int, ...], dict[str, int]] = {}
    for w in [ORIGINAL_ALLWEATHER] + list(ANCHOR_ALLOCS.values()):
        candidates[weight_key(w)] = normalize_weights(w)
    for mode in modes:
        for profile in ["balanced", "defensive", "growth"]:
            for ctx in ["robust", "oos", "full", "roll_5y"]:
                col = f"score_{mode}_{ctx}_{profile}"
                if col not in scored.columns:
                    continue
                pool = scored[mode_base_mask(scored, mode)]
                if pool.empty:
                    continue
                for _, row in pool.nlargest(max(1, n), col).iterrows():
                    w = {k: int(row[k]) for k in ASSET_KEYS}
                    candidates[weight_key(w)] = w
    return list(candidates.values())[: max(10, n * 6)]



# ------------------------- v6 제약 민감도 -------------------------
def safe_score_col(mode: str = "allweather", context: str = "robust", profile: str = "balanced") -> str:
    return f"score_{mode}_{context}_{profile}"


def row_to_sensitivity_record(row: pd.Series, label: str, mode: str, kr_min: int | None, gold_max: int | None, btc_max: int | None) -> dict[str, Any]:
    return {
        "label": label,
        "mode": mode,
        "kr_min_test": kr_min,
        "gold_max_test": gold_max,
        "btc_max_test": btc_max,
        "weight_str": row.get("weight_str"),
        "US_STOCK": int(row.get("US_STOCK", 0)),
        "KR_STOCK": int(row.get("KR_STOCK", 0)),
        "BTC": int(row.get("BTC", 0)),
        "BOND": int(row.get("BOND", 0)),
        "GOLD": int(row.get("GOLD", 0)),
        "REIT": int(row.get("REIT", 0)),
        "equity_total": _round_float(row.get("equity_total"), 2),
        "risk_asset_total": _round_float(row.get("risk_asset_total"), 2),
        "effective_n": _round_float(row.get("effective_n"), 3),
        "max_weight": _round_float(row.get("max_weight"), 2),
        "boundary_hits": _round_float(row.get("boundary_hits"), 2),
        "score": _round_float(row.get(safe_score_col(mode)), 4),
        "oos_krw_cagr": _round_float(row.get("oos_krw_cagr"), 4),
        "oos_krw_mdd": _round_float(row.get("oos_krw_mdd"), 4),
        "oos_krw_sharpe": _round_float(row.get("oos_krw_sharpe"), 4),
        "full_krw_cagr": _round_float(row.get("full_krw_cagr"), 4),
        "full_krw_mdd": _round_float(row.get("full_krw_mdd"), 4),
        "crisis_worst_krw": _round_float(row.get("crisis_worst_krw"), 4),
    }


def build_v6_constraint_sensitivity(scored: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    """KR 최소, GOLD 상한, BTC 상한을 실제 채택 후보 관점에서 비교한다."""
    rows: list[dict[str, Any]] = []
    kr_tests = [0, 10, 15, 20, 25]
    gold_tests = [15, 20, 25, 30]
    btc_tests = [0, 3, 5, 10]
    for mode in modes:
        score_col = safe_score_col(mode)
        if score_col not in scored.columns:
            continue
        base = mode_base_mask(scored, mode) & (scored["REIT"] == 0)
        # 1D tests
        for kr in kr_tests:
            sub = scored[base & (scored["KR_STOCK"] >= kr)].copy()
            sub = sub[sub[score_col].notna()]
            if not sub.empty:
                rows.append(row_to_sensitivity_record(sub.sort_values(score_col, ascending=False).iloc[0], f"KR>={kr}", mode, kr, None, None))
        for gm in gold_tests:
            sub = scored[base & (scored["GOLD"] <= gm)].copy()
            sub = sub[sub[score_col].notna()]
            if not sub.empty:
                rows.append(row_to_sensitivity_record(sub.sort_values(score_col, ascending=False).iloc[0], f"GOLD<={gm}", mode, None, gm, None))
        for bm in btc_tests:
            sub = scored[base & (scored["BTC"] <= bm)].copy()
            sub = sub[sub[score_col].notna()]
            if not sub.empty:
                rows.append(row_to_sensitivity_record(sub.sort_values(score_col, ascending=False).iloc[0], f"BTC<={bm}", mode, None, None, bm))
        # combined practical grid
        for kr in [10, 15, 20]:
            for gm in [20, 25, 30]:
                for bm in [3, 5, 10]:
                    sub = scored[base & (scored["KR_STOCK"] >= kr) & (scored["GOLD"] <= gm) & (scored["BTC"] <= bm)].copy()
                    sub = sub[sub[score_col].notna()]
                    if not sub.empty:
                        rows.append(row_to_sensitivity_record(sub.sort_values(score_col, ascending=False).iloc[0], f"KR>={kr}|GOLD<={gm}|BTC<={bm}", mode, kr, gm, bm))
    return pd.DataFrame(rows)


def build_v6_anchor_comparison(scored: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    rows=[]
    anchors = {
        "candidate_35_15_3_25_22": {"US_STOCK":35,"KR_STOCK":15,"BTC":3,"BOND":25,"GOLD":22,"REIT":0},
        "candidate_32_15_3_25_25": {"US_STOCK":32,"KR_STOCK":15,"BTC":3,"BOND":25,"GOLD":25,"REIT":0},
        "candidate_30_20_3_25_22": {"US_STOCK":30,"KR_STOCK":20,"BTC":3,"BOND":25,"GOLD":22,"REIT":0},
        "candidate_40_10_3_25_22": {"US_STOCK":40,"KR_STOCK":10,"BTC":3,"BOND":25,"GOLD":22,"REIT":0},
        "candidate_35_10_5_25_25": {"US_STOCK":35,"KR_STOCK":10,"BTC":5,"BOND":25,"GOLD":25,"REIT":0},
    }
    for label, w in anchors.items():
        mask = pd.Series(True, index=scored.index)
        for k, v in w.items():
            mask &= (scored[k] == v)
        sub=scored[mask]
        if sub.empty:
            continue
        row=sub.iloc[0]
        for mode in modes:
            rows.append(row_to_sensitivity_record(row, label, mode, None, None, None))
    return pd.DataFrame(rows)

# ------------------------- 출력 -------------------------
def default_winner(winners: pd.DataFrame, mode: str, profile: str = "balanced") -> pd.Series | None:
    sub = winners[(winners["mode"] == mode) & (winners["context"] == "robust") & (winners["profile"] == profile)]
    if sub.empty:
        return None
    return sub.iloc[0]


def export_outputs(
    out_dir: str,
    df: pd.DataFrame,
    winners: pd.DataFrame,
    bucket_df: pd.DataFrame,
    btc_bucket_df: pd.DataFrame,
    equity_bucket_df: pd.DataFrame,
    reit_bucket_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    filter_summary: pd.DataFrame,
    horizons: list[int],
    metadata: dict[str, Any],
    modes: list[str],
    save_detail: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    v6_sensitivity_df = build_v6_constraint_sensitivity(df, modes)
    v6_anchor_df = build_v6_anchor_comparison(df, modes)

    if save_detail:
        df.to_csv(os.path.join(out_dir, "allweather_weight_results.csv"), index=False, encoding="utf-8-sig")
        winners.to_csv(os.path.join(out_dir, "allweather_weight_winners.csv"), index=False, encoding="utf-8-sig")
        bucket_df.to_csv(os.path.join(out_dir, "allweather_weight_kr_buckets.csv"), index=False, encoding="utf-8-sig")
        btc_bucket_df.to_csv(os.path.join(out_dir, "allweather_weight_btc_buckets.csv"), index=False, encoding="utf-8-sig")
        reit_bucket_df.to_csv(os.path.join(out_dir, "allweather_weight_reit_buckets.csv"), index=False, encoding="utf-8-sig")
        equity_bucket_df.to_csv(os.path.join(out_dir, "allweather_weight_equity_buckets.csv"), index=False, encoding="utf-8-sig")
        equity_curve_df.to_csv(os.path.join(out_dir, "allweather_weight_equity_curve.csv"), index=False, encoding="utf-8-sig")
        pareto_df.to_csv(os.path.join(out_dir, "allweather_weight_pareto.csv"), index=False, encoding="utf-8-sig")
        filter_summary.to_csv(os.path.join(out_dir, "allweather_weight_filter_summary.csv"), index=False, encoding="utf-8-sig")
        v6_sensitivity_df.to_csv(os.path.join(out_dir, "allweather_weight_v6_constraint_sensitivity.csv"), index=False, encoding="utf-8-sig")
        v6_anchor_df.to_csv(os.path.join(out_dir, "allweather_weight_v6_anchor_comparison.csv"), index=False, encoding="utf-8-sig")

    payload = {
        "metadata": metadata,
        "profile_config": PROFILE_CONFIG,
        "robust_context_weights": ROBUST_CONTEXT_WEIGHTS,
        "winners": winners.to_dict(orient="records"),
        "equity_bucket_summary": equity_bucket_df.to_dict(orient="records"),
        "btc_bucket_summary": btc_bucket_df.to_dict(orient="records"),
        "reit_bucket_summary": reit_bucket_df.to_dict(orient="records"),
        "equity_curve": equity_curve_df.to_dict(orient="records"),
        "filter_summary": filter_summary.to_dict(orient="records"),
        "v6_constraint_sensitivity": v6_sensitivity_df.to_dict(orient="records"),
        "v6_anchor_comparison": v6_anchor_df.to_dict(orient="records"),
        "top_results_sample": df.sort_values([c for c in df.columns if c.startswith("score_allweather_robust_balanced") or c.startswith("score_open_robust_balanced")][0], ascending=False).head(200).to_dict(orient="records") if any(c.startswith("score_") for c in df.columns) else [],
    }
    if save_detail:
        with open(os.path.join(out_dir, "allweather_weight_search.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else x)

        aw = default_winner(winners, "allweather") if "allweather" in modes else None
        op = default_winner(winners, "open") if "open" in modes else None
        if aw is not None:
            with open(os.path.join(out_dir, "best_allweather_weight_v6.json"), "w", encoding="utf-8") as f:
                json.dump(best_payload_from_winner(aw, metadata, mode="allweather"), f, ensure_ascii=False, indent=2)
        if op is not None:
            with open(os.path.join(out_dir, "best_static_open_v6.json"), "w", encoding="utf-8") as f:
                json.dump(best_payload_from_winner(op, metadata, mode="open"), f, ensure_ascii=False, indent=2)

    md = build_markdown(df, winners, bucket_df, btc_bucket_df, equity_bucket_df, reit_bucket_df, equity_curve_df, pareto_df, filter_summary, horizons, metadata, modes)
    with open(os.path.join(out_dir, "ALLWEATHER_WEIGHT_RESULT.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if save_detail:
        _log(f"저장: {out_dir}/ALLWEATHER_WEIGHT_RESULT.md + 상세 CSV/JSON 파일")
    else:
        _log(f"저장: {out_dir}/ALLWEATHER_WEIGHT_RESULT.md (단일 파일 모드)")


def best_payload_from_winner(row: pd.Series, metadata: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "as_of": metadata["as_of"],
        "source": "backtest_allweather_weights_v6.py",
        "selection_note": "기본 선택은 robust balanced: OOS+Full+Rolling+위기구간, 하드 위험필터와 percentile rank 점수 기반.",
        "mode": mode,
        "context": "robust",
        "profile": "balanced",
        "base_name": f"{mode}_robust_balanced",
        "base_weights": {k: int(row[k]) for k in ASSET_KEYS},
        "summary_metrics": {k: row.get(k) for k in ["oos_krw_cagr", "oos_krw_mdd", "oos_krw_sharpe", "full_krw_cagr", "full_krw_mdd", "crisis_worst_krw"]},
        "dynamic_rules": {"trend_window": None, "mom_mode": "none", "cut": "none", "dest": "none"},
    }


def md_winner_table(winners: pd.DataFrame, mode: str, context: str) -> list[str]:
    sub = winners[(winners["mode"] == mode) & (winners["context"] == context)]
    lines: list[str] = []
    if sub.empty:
        return ["- 해당 후보 없음"]
    if context == "robust":
        lines.append("| 성향 | 비중 | 주식합계 | 유효자산수 | OOS 원화CAGR | OOS 원화MDD | Full 원화CAGR | Full 원화MDD | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['profile_label']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('effective_n')} | "
                f"{r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | "
                f"{r.get('full_krw_cagr')}% | {r.get('full_krw_mdd')}% | {r.get('score')} |"
            )
    elif context in ("oos", "full"):
        lines.append("| 성향 | 비중 | 주식합계 | 유효자산수 | 원화CAGR | 원화MDD | 원화Sharpe | 원화Calmar | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['profile_label']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('effective_n')} | "
                f"{r.get('krw_cagr')}% | {r.get('krw_mdd')}% | {r.get('krw_sharpe')} | {r.get('krw_calmar')} | {r.get('score')} |"
            )
    elif context.startswith("roll_"):
        lines.append("| 성향 | 비중 | 주식합계 | 중앙 원화CAGR | 하위10% CAGR | 최악 원화MDD | 양수 비율 | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['profile_label']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('median_krw_cagr')}% | {r.get('p10_krw_cagr')}% | {r.get('worst_krw_mdd')}% | {r.get('positive_rate')}% | {r.get('score')} |")
    return lines


def build_markdown(
    df: pd.DataFrame,
    winners: pd.DataFrame,
    bucket_df: pd.DataFrame,
    btc_bucket_df: pd.DataFrame,
    equity_bucket_df: pd.DataFrame,
    reit_bucket_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    filter_summary: pd.DataFrame,
    horizons: list[int],
    metadata: dict[str, Any],
    modes: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# 올웨더/정적배분 v6 KR/GOLD/BTC 제약 민감도 백테스트 결과 ({metadata['as_of']})")
    lines.append("")
    lines.append(f"- 기간: {metadata['period'][0]} ~ {metadata['period'][1]}")
    lines.append(f"- 평가 조합 수: {metadata['n_results']:,}개 · step {metadata['step']}%p · fine {metadata['fine']} · 편도 비용 {100*metadata['cost_oneway']:.2f}%")
    lines.append(f"- 모드: {', '.join(modes)}")
    lines.append("- 자산 순서: US_STOCK / KR_STOCK / BTC / BOND / GOLD / REIT(항상 0, 제외)(고정 0)")
    lines.append("- 선택 방식: 하드 위험필터 → percentile rank 위험조정 점수 → OOS+Full+rolling+위기 robust 집계")
    lines.append("")

    for mode in modes:
        d = default_winner(winners, mode)
        if d is not None:
            lines.append(f"## 기본 선택: {mode} robust 균형형")
            lines.append(f"- 비중: **{d['weight_str']}**")
            lines.append(f"- OOS: 원화 CAGR {d.get('oos_krw_cagr')}% · 원화 MDD {d.get('oos_krw_mdd')}% · 원화 Sharpe {d.get('oos_krw_sharpe')}")
            lines.append(f"- Full: 원화 CAGR {d.get('full_krw_cagr')}% · 원화 MDD {d.get('full_krw_mdd')}% · 위기 최악 {d.get('crisis_worst_krw')}%")
            lines.append(f"- 유효자산수 {d.get('effective_n')} · 최대 단일비중 {d.get('max_weight')}% · 경계값 hit {d.get('boundary_hits')}개")
            lines.append("")

    orig = df[df["weight_str"] == "25/5/0/40/25/5"]
    lines.append("## 기준 올웨더 변형: GLOBAL/REIT 제거, BTC 0% 기준안")
    if not orig.empty:
        r = orig.iloc[0]
        lines.append(f"- OOS: 원화 CAGR {r.get('oos_krw_cagr')}% · 원화 MDD {r.get('oos_krw_mdd')}% · 원화 Sharpe {r.get('oos_krw_sharpe')} · USD MDD {r.get('oos_mdd')}%")
        lines.append(f"- Full: 원화 CAGR {r.get('full_krw_cagr')}% · 원화 MDD {r.get('full_krw_mdd')}% · 위기 최악 {r.get('crisis_worst_krw')}%")
    else:
        lines.append("- 현재 제약조건에서는 기준 올웨더 변형이 후보에 포함되지 않았습니다.")
    lines.append("")

    for mode in modes:
        lines.append(f"## {mode} 모드 위험성향별 후보")
        for ctx in ["robust", "oos", "full"] + (["roll_5y"] if 5 in horizons else []):
            lines.append("")
            lines.append(f"### {ctx}")
            lines.extend(md_winner_table(winners, mode, ctx))
        lines.append("")

    if not equity_curve_df.empty:
        lines.append("## 주식총비중별 위험-수익 곡선: robust balanced")
        lines.append("| 모드 | 주식비중 | 대표 비중 | 점수 | OOS 원화CAGR | OOS 원화MDD | OOS Sharpe | Full CAGR | Full MDD | 위기 최악 | 후보수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in equity_curve_df.iterrows():
            lines.append(
                f"| {r.get('mode')} | {r.get('equity_bucket')} | {r.get('best_weight')} | {r.get('best_score')} | "
                f"{r.get('best_oos_krw_cagr')}% | {r.get('best_oos_krw_mdd')}% | {r.get('best_oos_krw_sharpe')} | "
                f"{r.get('best_full_krw_cagr')}% | {r.get('best_full_krw_mdd')}% | {r.get('best_crisis_worst_krw')}% | {r.get('n_candidates')} |"
            )
        lines.append("")
        lines.append("- 이 표가 v5의 핵심입니다. 주식비중을 미리 50%나 80%로 고정하지 않고, 각 주식비중 구간에서 위험조정 성과가 가장 좋은 후보를 비교합니다.")
        lines.append("- `min_mdd_weight`, `max_cagr_weight`까지 보려면 `allweather_weight_equity_curve.csv`를 확인하세요.")
        lines.append("")

    if not equity_bucket_df.empty:
        lines.append("## 주식총비중 버킷별 후보 상세: robust balanced")
        lines.append("| 모드 | 주식비중 | 최선 비중 | 유효자산수 | OOS 원화CAGR | OOS 원화MDD | Full 원화CAGR | 5년 p10 CAGR | 10년 p10 CAGR | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        esub = equity_bucket_df[(equity_bucket_df["context"] == "robust") & (equity_bucket_df["profile"] == "balanced")]
        for _, r in esub.iterrows():
            lines.append(
                f"| {r['mode']} | {r['equity_bucket']} | {r['weight_str']} | {r.get('effective_n')} | "
                f"{r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | "
                f"{r.get('roll_5y_p10_krw_cagr', '')}% | {r.get('roll_10y_p10_krw_cagr', '')}% | {r.get('score')} |"
            )
        lines.append("")

    if not bucket_df.empty:
        lines.append("## 한국주식 비중 버킷별 후보: robust balanced")
        lines.append("| 모드 | 한국주식 비중 | 최선 비중 | 주식합계 | OOS 원화CAGR | OOS 원화MDD | Full 원화CAGR | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        bsub = bucket_df[(bucket_df["context"] == "robust") & (bucket_df["profile"] == "balanced")]
        for _, r in bsub.iterrows():
            lines.append(f"| {r['mode']} | {r['kr_bucket']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('score')} |")
        lines.append("")


    if not btc_bucket_df.empty:
        lines.append("## BTC 비중 버킷별 후보: robust balanced")
        lines.append("| 모드 | BTC 비중 | 최선 비중 | 주식합계 | OOS 원화CAGR | OOS 원화MDD | Full 원화CAGR | Full 원화MDD | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        csub = btc_bucket_df[(btc_bucket_df["context"] == "robust") & (btc_bucket_df["profile"] == "balanced")]
        for _, r in csub.iterrows():
            lines.append(f"| {r['mode']} | {r['btc_bucket']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('full_krw_mdd')}% | {r.get('score')} |")
        lines.append("")


    if not reit_bucket_df.empty:
        lines.append("## REIT 비중 버킷별 후보: robust balanced")
        lines.append("| 모드 | REIT 비중 | 최선 비중 | 주식합계 | BTC | OOS 원화CAGR | OOS 원화MDD | Full 원화CAGR | 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        rsub = reit_bucket_df[(reit_bucket_df["context"] == "robust") & (reit_bucket_df["profile"] == "balanced")]
        for _, r in rsub.iterrows():
            lines.append(f"| {r['mode']} | {r['reit_bucket']} | {r['weight_str']} | {r.get('equity_total')}% | {r.get('BTC')}% | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('score')} |")
        lines.append("")

    if not pareto_df.empty:
        lines.append("## Pareto 후보 예시: OOS 원화 CAGR ↔ OOS 원화 MDD")
        lines.append("| 모드 | 비중 | 원화CAGR | 원화MDD | 원화Sharpe | robust balanced 점수 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in pareto_df.head(20).iterrows():
            score_col = f"score_{r.get('mode')}_robust_balanced"
            score = r.get(score_col, "")
            lines.append(f"| {r.get('mode')} | {r.get('weight_str')} | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('oos_krw_sharpe')} | {score} |")
        lines.append("")

    # v6 constraint sensitivity
    sens = build_v6_constraint_sensitivity(df, modes)
    if not sens.empty:
        lines.append("## v6 핵심 제약 민감도: KR 최소 / GOLD 상한 / BTC 상한")
        lines.append("| 모드 | 테스트 | 비중 | 점수 | 주식합계 | BTC | 금 | OOS CAGR | OOS MDD | Full CAGR | Full MDD |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        show = sens[(sens["label"].str.contains("KR>=", regex=False) | sens["label"].str.contains("GOLD<=", regex=False) | sens["label"].str.contains("BTC<=", regex=False))]
        for _, r in show.head(80).iterrows():
            lines.append(f"| {r['mode']} | {r['label']} | {r['weight_str']} | {r.get('score')} | {r.get('equity_total')}% | {r.get('BTC')}% | {r.get('GOLD')}% | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('full_krw_mdd')}% |")
        lines.append("")
        lines.append("## v6 실전 조합 그리드: KR>=10/15/20 × GOLD<=20/25/30 × BTC<=3/5/10")
        lines.append("| 모드 | 제약 | 최선 비중 | 점수 | OOS CAGR | OOS MDD | Full CAGR | Full MDD |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        practical = sens[sens["label"].str.contains("|", regex=False)]
        for _, r in practical.head(120).iterrows():
            lines.append(f"| {r['mode']} | {r['label']} | {r['weight_str']} | {r.get('score')} | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('full_krw_mdd')}% |")
        lines.append("")
    anchors = build_v6_anchor_comparison(df, modes)
    if not anchors.empty:
        lines.append("## v6 사전 후보 직접 비교")
        lines.append("| 후보 | 모드 | 비중 | 점수 | OOS CAGR | OOS MDD | Full CAGR | Full MDD |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in anchors.iterrows():
            lines.append(f"| {r['label']} | {r['mode']} | {r['weight_str']} | {r.get('score')} | {r.get('oos_krw_cagr')}% | {r.get('oos_krw_mdd')}% | {r.get('full_krw_cagr')}% | {r.get('full_krw_mdd')}% |")
        lines.append("")
    lines.append("## v6 해석 주의")
    lines.append("- 이 결과는 정적 비중 탐색입니다. 추세/모멘텀/컷 신호는 꺼져 있습니다.")
    lines.append("- `allweather` 모드는 분산·채권·금·유효자산수 제약을 더 강하게 적용합니다. `open` 모드는 자산을 0으로 보내는 자유 최적화 결과도 허용합니다.")
    lines.append("- 기본 선택은 OOS 1위가 아니라 robust balanced입니다. 실제 채택 전에는 winners, kr_buckets, pareto, filter_summary를 함께 보세요.")
    lines.append("- 제약의 끝에 붙은 후보는 점수가 높아도 과최적화 신호일 수 있으므로 boundary_hits와 effective_n을 같이 확인하세요.")
    return "\n".join(lines) + "\n"


# ------------------------- 메인 -------------------------
def parse_horizons(s: str) -> list[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        val = int(x)
        if val <= 0:
            raise ValueError("horizons는 양의 정수 연수여야 합니다.")
        out.append(val)
    return sorted(set(out))


def parse_modes(s: str) -> list[str]:
    if s == "both":
        return ["allweather", "open"]
    modes = [x.strip() for x in s.split(",") if x.strip()]
    valid = {"allweather", "open"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"알 수 없는 mode: {bad}. 가능한 값: allweather, open, both")
    return modes


def build_constraints(args: argparse.Namespace) -> Constraints:
    mins = {
        "US_STOCK": args.us_min,
        "KR_STOCK": args.kr_min,
        "BTC": args.btc_min,
        "BOND": args.bond_min,
        "GOLD": args.gold_min,
        "REIT": args.reit_min,
    }
    maxs = {
        "US_STOCK": args.us_max,
        "KR_STOCK": args.kr_max,
        "BTC": args.btc_max,
        "BOND": args.bond_max,
        "GOLD": args.gold_max,
        "REIT": args.reit_max,
    }
    for k in ASSET_KEYS:
        if mins[k] < 0 or maxs[k] > 100 or mins[k] > maxs[k]:
            raise ValueError(f"{k} min/max 제약이 잘못되었습니다: {mins[k]}~{maxs[k]}")
    return Constraints(mins=mins, maxs=maxs, equity_min=args.equity_min, equity_max=args.equity_max)


def main() -> None:
    ap = argparse.ArgumentParser(description="v6 GLOBAL/REIT 제외, BTC 포함, KR/GOLD/BTC 제약 민감도 정적 자산배분 탐색")
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터로 파이프라인 점검")
    ap.add_argument("--period", default="25y", help="yfinance 다운로드 기간. 기본 25y")
    ap.add_argument("--output-dir", default="output_v6", help="출력 폴더")
    ap.add_argument("--save-detail", action="store_true", help="CSV/JSON 상세 파일까지 저장합니다. 기본은 md 단일 파일만 저장합니다.")
    ap.add_argument("--mode", default="both", help="allweather, open, both 중 선택. 기본 both")
    ap.add_argument("--step", type=int, default=5, help="기본 그리드 간격(%%p). 기본 5")
    ap.add_argument("--fine", action="store_true", help="상위 후보 주변을 1%%p 단위로 추가 탐색")
    ap.add_argument("--fine-radius", type=int, default=2, help="--fine 사용 시 주변 탐색 반경(%%p). 기본 2")
    ap.add_argument("--fine-top", type=int, default=4, help="--fine 중심 후보 수 계수. 기본 4")
    ap.add_argument("--max-combos", type=int, default=300000, help="최대 평가 조합 수 안전장치")
    ap.add_argument("--horizons", default="3,5,10,15", help="rolling 보유기간 연수. 예: 3,5,10,15")
    ap.add_argument("--oos-frac", type=float, default=0.4, help="뒤쪽 OOS 비율. 기본 0.4")
    ap.add_argument("--cost-oneway", type=float, default=0.0025, help="편도 거래비용. 기본 0.0025")

    # 탐색 그리드는 넓게 열고, allweather/open 여부는 scoring/filter 단계에서 분리한다.
    ap.add_argument("--us-min", type=int, default=10)
    ap.add_argument("--us-max", type=int, default=70)
    ap.add_argument("--kr-min", type=int, default=0)
    ap.add_argument("--kr-max", type=int, default=45)
    ap.add_argument("--btc-min", type=int, default=0)
    ap.add_argument("--btc-max", type=int, default=10, help="BTC 최대 비중. 기본 10. 실전 분석은 0/3/5/10 버킷 및 상한 민감도로 비교")
    ap.add_argument("--bond-min", type=int, default=0)
    ap.add_argument("--bond-max", type=int, default=60)
    ap.add_argument("--gold-min", type=int, default=0)
    ap.add_argument("--gold-max", type=int, default=30)
    ap.add_argument("--reit-min", type=int, default=0)
    ap.add_argument("--reit-max", type=int, default=0)
    ap.add_argument("--equity-min", type=int, default=20)
    ap.add_argument("--equity-max", type=int, default=90)

    args = ap.parse_args()
    if not (0.1 <= args.oos_frac <= 0.8):
        raise ValueError("--oos-frac는 0.1~0.8 사이가 적절합니다.")
    horizons = parse_horizons(args.horizons)
    modes = parse_modes(args.mode)
    constraints = build_constraints(args)

    df_prices = synthetic_prices() if args.synthetic else fetch_prices(period=args.period)
    mret = monthly_returns(df_prices)
    fx_m = monthly_fx(df_prices)
    _log(f"월간 수익률 {mret.index[0].date()} ~ {mret.index[-1].date()} · {len(mret)}개월")

    # BTC 포함 시 데이터가 2014년 이후로 짧아질 수 있다.
    # 요청 horizon(예: 15년)이 실제 월간 데이터보다 길면 roll_15y_* 컬럼이 생성되지 않으므로
    # 해당 rolling horizon은 자동 제외한다.
    requested_horizons = list(horizons)
    horizons = [h for h in horizons if len(mret) >= h * 12]
    skipped_horizons = [h for h in requested_horizons if h not in horizons]
    if skipped_horizons:
        _log("rolling horizon 제외: " + ", ".join(f"{h}y" for h in skipped_horizons) + " (데이터 기간 부족)")
    if not horizons:
        _log("사용 가능한 rolling horizon이 없습니다. OOS/Full/위기구간 점수만 사용합니다.")

    weights = generate_weight_grid(args.step, constraints)
    weights, anchor_names = add_anchor_allocs(weights, constraints)
    seen = {weight_key(w) for w in weights}
    _log(f"기본 후보 {len(weights):,}개 생성")
    if len(weights) > args.max_combos:
        raise RuntimeError(f"후보가 {len(weights):,}개로 max-combos를 초과했습니다. step을 키우거나 제약을 좁히세요.")

    coarse_results = evaluate_weight_list(weights, mret, fx_m, horizons, args.oos_frac, args.cost_oneway, anchor_names)
    results = coarse_results
    scored_df, filter_summary = prepare_dataframe(coarse_results, horizons, constraints, modes)

    if args.fine:
        centers = select_fine_centers_v3(scored_df, args.fine_top, modes)
        fine_weights: list[dict[str, int]] = []
        for c in centers:
            fine_weights.extend(local_integer_grid(c, args.fine_radius, constraints))
        new_weights = []
        for w in fine_weights:
            k = weight_key(w)
            if k not in seen:
                seen.add(k)
                new_weights.append(w)
        _log(f"fine 후보 {len(new_weights):,}개 추가")
        if len(weights) + len(new_weights) > args.max_combos:
            raise RuntimeError(
                f"fine 포함 후보가 {len(weights) + len(new_weights):,}개로 max-combos를 초과했습니다. "
                "fine-radius/fine-top을 줄이거나 max-combos를 늘리세요."
            )
        if new_weights:
            fine_results = evaluate_weight_list(new_weights, mret, fx_m, horizons, args.oos_frac, args.cost_oneway, anchor_names)
            results = coarse_results + fine_results
            scored_df, filter_summary = prepare_dataframe(results, horizons, constraints, modes)

    winners = choose_winners(scored_df, horizons, modes)
    bucket_df = build_kr_bucket_summary(scored_df, horizons, modes)
    btc_bucket_df = build_btc_bucket_summary(scored_df, horizons, modes)
    reit_bucket_df = build_reit_bucket_summary(scored_df, horizons, modes)
    equity_bucket_df = build_equity_bucket_summary(scored_df, horizons, modes)
    equity_curve_df = build_equity_risk_curve(scored_df, modes, profile="balanced")
    pareto_df = build_pareto_frontier(scored_df, modes)
    metadata = {
        "as_of": dt.date.today().isoformat(),
        "period": [str(df_prices.index[0].date()), str(df_prices.index[-1].date())],
        "monthly_period": [str(mret.index[0].date()), str(mret.index[-1].date())],
        "n_results": int(len(scored_df)),
        "step": args.step,
        "fine": bool(args.fine),
        "fine_radius": args.fine_radius if args.fine else None,
        "fine_top": args.fine_top if args.fine else None,
        "cost_oneway": args.cost_oneway,
        "oos_frac": args.oos_frac,
        "horizons": horizons,
        "modes": modes,
        "constraints": {
            "mins": constraints.mins,
            "maxs": constraints.maxs,
            "equity_min": constraints.equity_min,
            "equity_max": constraints.equity_max,
        },
    }
    export_outputs(args.output_dir, scored_df, winners, bucket_df, btc_bucket_df, equity_bucket_df, reit_bucket_df, equity_curve_df, pareto_df, filter_summary, horizons, metadata, modes, save_detail=args.save_detail)

    for mode in modes:
        d = default_winner(winners, mode)
        if d is not None:
            _log(
                f"기본 선택({mode} robust 균형형): {d['weight_str']} · "
                f"OOS 원화 CAGR {d.get('oos_krw_cagr')}% · OOS 원화 MDD {d.get('oos_krw_mdd')}% · "
                f"Full 원화 CAGR {d.get('full_krw_cagr')}%"
            )


if __name__ == "__main__":
    main()
