#!/usr/bin/env python3
"""
gold_regime_data.py — 금 배분타이밍 레짐신호 1단계: 데이터 수집 + 주간 피처.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md 참고.

실행: python gold_regime_data.py            # 데이터셋 캐시 갱신 후 tail 출력
      python gold_regime_data.py --self-test
결과: output/gold_regime_dataset.pkl
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

from backtest_regime_assets import fetch as fetch_yf
from fx_hedge_validation import fetch_fred

GOLD_TICKER = "GC=F"
DXY_TICKER = "DX-Y.NYB"
IEF_TICKER = "IEF"
FRED_REAL_RATE = "DFII10"
FRED_BREAKEVEN = "T10YIE"
DATASET_PATH = "output/gold_regime_dataset.pkl"
CORR_WINDOW = 60


def _log(m): print(f"[금레짐데이터] {m}", file=sys.stderr)


def _align_calendar(base: pd.Series, others: dict[str, pd.Series]) -> pd.DataFrame:
    """base(Gold) 거래일 캘린더 위에 others를 ffill로 정렬. 정렬 전 시작일 이전(전부
    NaN)인 앞부분은 제거."""
    cal = base.index
    out = pd.DataFrame({"gold": base})
    for name, s in others.items():
        aligned = s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
        out[name] = aligned
    return out.dropna()


def fetch_all(max_stale_days: int = 5) -> pd.DataFrame:
    gold = fetch_yf(GOLD_TICKER, "output/regime_price_cache_gold_dxy.pkl", max_stale_days)
    dxy = fetch_yf(DXY_TICKER, "output/regime_price_cache_dxy.pkl", max_stale_days)
    ief = fetch_yf(IEF_TICKER, "output/regime_price_cache_ief_dxy.pkl", max_stale_days)
    real_rate = fetch_fred(FRED_REAL_RATE)
    breakeven = fetch_fred(FRED_BREAKEVEN)
    df = _align_calendar(gold, {"dxy": dxy, "ief": ief, "real_rate": real_rate,
                                 "breakeven": breakeven})
    _log(f"정렬 완료: {df.index.min().date()}~{df.index.max().date()}, {len(df)}행")
    return df


def self_test():
    """합성 시리즈로 fetch_all의 정렬(ffill) 로직만 별도 함수로 분리해 검증
    (네트워크 호출 없이). _align_calendar가 아직 없으므로 이 단계는 실패해야 정상."""
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", periods=10)
    gold = pd.Series(range(10), index=idx, dtype=float)
    dxy = pd.Series([100.0, 101.0], index=[idx[0], idx[5]])  # 듬성듬성 — ffill 확인용
    out = _align_calendar(gold, {"dxy": dxy})
    assert list(out.columns) == ["gold", "dxy"]
    assert out["dxy"].iloc[3] == 100.0, "ffill이 안 됐음"
    assert out["dxy"].iloc[7] == 101.0
    print("[self-test] 통과: _align_calendar 정렬/ffill 정상", file=__import__("sys").stderr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        df = fetch_all()
        print(df.tail())
