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


MOM_WEEKS = {"3m": 13, "6m": 26, "12m": 52}


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    gold_ret = df["gold"].pct_change()
    dxy_ret = df["dxy"].pct_change()
    real_rate_chg = df["real_rate"].diff()
    corr_dxy = gold_ret.rolling(CORR_WINDOW).corr(dxy_ret)
    corr_rr = gold_ret.rolling(CORR_WINDOW).corr(real_rate_chg)

    weekly = df.resample("W-FRI").last()
    corr_dxy_w = corr_dxy.resample("W-FRI").last()
    corr_rr_w = corr_rr.resample("W-FRI").last()

    out = pd.DataFrame(index=weekly.index)
    for col in ("gold", "dxy", "ief"):
        for label, w in MOM_WEEKS.items():
            out[f"{col}_mom_{label}"] = weekly[col].pct_change(w)
    for label, w in MOM_WEEKS.items():
        out[f"real_rate_mom_{label}"] = weekly["real_rate"].diff(w)
    out["gold_dxy_corr60"] = corr_dxy_w
    out["gold_realrate_corr60"] = corr_rr_w
    return out.dropna()


def load_or_build(force: bool = False, max_stale_days: int = 5) -> pd.DataFrame:
    """캐시가 있으면 재사용하되, 캐시 파일이 max_stale_days일(달력일) 이상 전에
    빌드됐으면 재구축한다(backtest_regime_assets.fetch()와 동일한 신선도 관례,
    5일 여유). force=True면 무조건 재구축.

    신선도는 캐시 파일의 mtime(빌드 시각) 기준이지 인덱스 최댓값 기준이 아니다:
    W-FRI 리샘플은 진행 중인 주를 그 주의 금요일 날짜로 라벨링하므로
    (예: 화요일에 빌드해도 마지막 행은 이번 주 금요일로 찍힘), 인덱스 최댓값은
    빌드 시점보다 미래일 수 있어 "오늘 - 인덱스 최댓값" 방식은 주중에 stale_days가
    음수가 되어 절대 재구축을 트리거하지 못하는 결함이 있다."""
    if not force and os.path.exists(DATASET_PATH):
        mtime = pd.Timestamp.fromtimestamp(os.path.getmtime(DATASET_PATH))
        stale_days = (pd.Timestamp.now() - mtime).days
        if stale_days <= max_stale_days:
            return pd.read_pickle(DATASET_PATH)
        _log(f"캐시 stale(빌드 후 {stale_days}일 경과) — 재구축")
    df = fetch_all(max_stale_days)
    feat = build_weekly_features(df)
    os.makedirs("output", exist_ok=True)
    feat.to_pickle(DATASET_PATH)
    _log(f"저장: {DATASET_PATH} ({len(feat)}주, {feat.index.min().date()}~{feat.index.max().date()})")
    return feat


def self_test_features():
    """합성 60주치 일간 데이터(추세 있는 gold, 반대추세 dxy, 하락하는 real_rate)로
    build_weekly_features 배선 확인. 상관 계산을 위해 현실적 변동성 포함.

    real_rate는 역사적으로 음수 영역을 통과했으므로(DFII10 2020-2022),
    합성 데이터도 양수→음수로 전환되도록 해서 diff() 메서드가 실제로 사용됨을 검증한다."""
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=300)  # ~60주
    # 추세 + 소음: 금은 상승, DXY는 하락, 음의 상관 가능하게 구조화
    gold_drift = np.full(300, 0.002)
    gold_noise = np.random.normal(0, 0.005, 300)
    gold = pd.Series(100 * np.exp(np.cumsum(gold_drift + gold_noise)), index=idx)

    dxy_drift = np.full(300, -0.001)
    dxy_noise = np.random.normal(0, 0.003, 300)
    dxy = pd.Series(100 * np.exp(np.cumsum(dxy_drift + dxy_noise)), index=idx)

    ief_drift = np.full(300, 0.0005)
    ief_noise = np.random.normal(0, 0.002, 300)
    ief = pd.Series(100 * np.exp(np.cumsum(ief_drift + ief_noise)), index=idx)

    # real_rate: 레벨 값, 양수(~1.0)에서 음수(~-0.7)로 전환되는 추세 + 소음.
    # 이는 역사적 DFII10 데이터와 일치(2020-2022 음의 실질금리).
    # pct_change 대신 diff 사용을 증명하기 위해 0을 통과하는 시리즈 필요.
    real_rate_drift = -np.cumsum(np.full(300, 0.0065))  # 1.0 → -0.95 정도로 하락
    real_rate_base = 1.0 + real_rate_drift
    real_rate = pd.Series(real_rate_base + np.random.normal(0, 0.02, 300), index=idx)

    breakeven = pd.Series(np.full(300, 2.0) + np.random.normal(0, 0.05, 300), index=idx)

    df = pd.DataFrame({"gold": gold, "dxy": dxy, "ief": ief, "real_rate": real_rate,
                       "breakeven": breakeven})

    # build_weekly_features 실행
    feat = build_weekly_features(df)
    assert feat.index.freqstr in ("W-FRI",) or feat.index.freq is not None or len(feat) < len(df)

    # 예상 real_rate_mom_12m 계산: 주간 리샘플 후 52주 차이.
    # diff를 사용함을 검증하기 위해 정확한 값과 비교.
    weekly_rr = real_rate.resample("W-FRI").last()
    expected_rr_mom_12m = weekly_rr.iloc[-1] - weekly_rr.iloc[-53]  # last - 52주 전

    last = feat.iloc[-1]
    assert last["gold_mom_12m"] > 0, "금 12개월 모멘텀은 양수여야 함(꾸준 상승 합성데이터)"
    assert last["dxy_mom_12m"] < 0, "DXY 12개월 모멘텀은 음수여야 함"

    # real_rate_mom_12m이 정확히 예상값과 일치하는지 확인 (diff 사용 증명).
    assert np.isclose(last["real_rate_mom_12m"], expected_rr_mom_12m, atol=1e-10), \
        f"실질금리 12개월 모멘텀은 diff를 사용한 {expected_rr_mom_12m:.6f}여야 하는데 {last['real_rate_mom_12m']:.6f} (pct_change 사용 증거)"
    assert last["real_rate_mom_12m"] < 0, "음의 추세 → 음수 레벨변화"

    # pct_change를 쓰면 음수/양수 전환 시리즈에서 NaN이나 이상 값 생성됨을 추가 검증.
    # pct_change 계산: (마지막값 - 52주전) / |52주전값|이므로,
    # 시작(양수)에서 끝(음수)로 가면 극단적 음수 값 생성.
    weekly_rr_pct = weekly_rr.pct_change(52)
    wrong_formula_result = weekly_rr_pct.iloc[-1]
    assert abs(wrong_formula_result) > abs(last["real_rate_mom_12m"]), \
        "pct_change는 zero-crossing 시리즈에서 극단적 값을 생성 (diff가 정답)"

    assert last["gold_dxy_corr60"] < 0, "반대추세로 만들었으니 상관은 음수여야 함"
    print("[self-test] 통과: build_weekly_features 배선 정상 (diff 메서드 검증 포함)", file=sys.stderr)


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
    self_test_features()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        feat = load_or_build(force=True)
        print(feat.tail())
