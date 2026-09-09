#!/usr/bin/env python3
"""
bond_data_quality_check.py — IEF/TLT/SHY 캐시 데이터 품질 점검 (2026-08-22, 지호 님 요청:
IEF 채권 백테스트 재검증 세션의 일부).

bond_trend_filter_grid.py/bond_asym_band.py가 쓰는 output/regime_price_cache_{ief,tlt,shy}.pkl
3개가: ①공백/결측 없이 예상 구간을 커버하는지 ②서로 다른 실제 시계열인지(캐시 버그로 3개가
같은 데이터를 가리키는 사고 방지) ③듀레이션이 긴 자산일수록 변동성이 크다는 상식적 순서
(SHY<IEF<TLT)를 실제로 만족하는지를 확인한다. bond_trend_filter_grid.py 등을 먼저 실행해
캐시가 최신인 상태여야 의미 있다.

실행: python bond_data_quality_check.py
결과: output/bond_data_quality.json + stderr 로그
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ASSETS = ["ief", "tlt", "shy"]
MAX_GAP_DAYS = 7   # 연속 거래일 사이 7일(주말+공휴일 감안) 넘게 비면 결측 의심


def _log(m): print(f"[채권데이터품질] {m}", file=sys.stderr)


def check_one(name: str) -> dict:
    path = f"output/regime_price_cache_{name}.pkl"
    s = pd.read_pickle(path)
    gaps = s.index.to_series().diff().dt.days
    big_gaps = gaps[gaps > MAX_GAP_DAYS]
    vol_ann = float(s.pct_change().std() * np.sqrt(252) * 100)
    row = {
        "path": path, "start": s.index.min().date().isoformat(), "end": s.index.max().date().isoformat(),
        "n_days": len(s), "n_nan": int(s.isna().sum()), "n_nonpositive": int((s <= 0).sum()),
        "n_gaps_over_7d": int(len(big_gaps)),
        "gap_examples": [{"date": d.date().isoformat(), "gap_days": int(g)} for d, g in big_gaps.head(5).items()],
        "annualized_vol_pct": round(vol_ann, 2),
    }
    _log(f"{name.upper()}: {row['start']}~{row['end']} ({row['n_days']}일) · NaN {row['n_nan']} · "
        f"7일초과갭 {row['n_gaps_over_7d']}개 · 연변동성 {row['annualized_vol_pct']}%")
    return row, s


def check_distinctness(series: dict) -> dict:
    common = pd.DataFrame(series).dropna()
    rets = common.pct_change().dropna()
    corr = rets.corr().round(3)
    navs = common / common.iloc[0]
    pairs = [("ief", "tlt"), ("ief", "shy"), ("tlt", "shy")]
    max_diff = {f"{a}_vs_{b}": round(float((navs[a] - navs[b]).abs().max()), 4) for a, b in pairs}
    result = {"n_common_obs": len(common), "return_corr": corr.to_dict(), "max_abs_nav_diff": max_diff}
    _log(f"공통구간 {len(common)}일 · 상관 IEF-TLT={corr.loc['ief','tlt']} IEF-SHY={corr.loc['ief','shy']} "
        f"TLT-SHY={corr.loc['tlt','shy']} (전부 1.0 아니면 서로 다른 시계열 확인됨)")
    return result


def run() -> dict:
    rows, series = {}, {}
    for name in ASSETS:
        row, s = check_one(name)
        rows[name] = row
        series[name] = s
    distinct = check_distinctness(series)
    vol_order_ok = rows["shy"]["annualized_vol_pct"] < rows["ief"]["annualized_vol_pct"] < rows["tlt"]["annualized_vol_pct"]
    _log(f"변동성 순서(SHY<IEF<TLT) 만족: {vol_order_ok}")
    payload = {"assets": rows, "distinctness": distinct, "volatility_order_shy_lt_ief_lt_tlt": vol_order_ok}
    os.makedirs("output", exist_ok=True)
    with open("output/bond_data_quality.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log("저장: output/bond_data_quality.json")
    return payload


if __name__ == "__main__":
    run()
