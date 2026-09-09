#!/usr/bin/env python3
"""
us_spmo_vs_spy_detailed.py — SPMO 상장 이후 SPY 대비 정밀 분석(2026-07-24, 지호 님 요청).

us_spmo_vs_spy_prereg.py가 "SPMO가 SPY를 유의하게 이기는가"라는 단일 가설만 검정했다면,
이건 그 반대 — 가설검정 없이 서술적으로 "언제 좋았고 언제 안 좋았는지"를 세밀히 뜯어본다.

산출:
  1. SPMO 실제 상장일(첫 거래일) 확인
  2. 연도별 수익률(SPY·SPMO·차이)
  3. 상대강도(SPMO/SPY 비율) 지그재그 국면 분해 — 임계값(기본 10%) 이상 반전마다
     구간을 나눠 "이 구간엔 SPMO가 SPY 대비 몇 %p 우위/열위였는가"를 나열
  4. 롤링 12개월 상대수익률 시계열의 최고·최악 구간 top5

실행: python us_spmo_vs_spy_detailed.py
결과: output/us_spmo_vs_spy_detailed.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import sp500_daily_report as R

ZIGZAG_THRESHOLD = 0.10   # 상대강도 비율의 10% 반전을 국면 전환으로 인정


def _log(m): print(f"[SPMO정밀분석] {m}", file=sys.stderr)


def _load():
    hist = R.download_histories(["SPY", "SPMO"], period="max")
    spy, spmo = hist.get("SPY"), hist.get("SPMO")
    if spy is None or spmo is None or spy.empty or spmo.empty:
        raise RuntimeError("SPY/SPMO 시세 조회 실패")
    inception = spmo.index[0]
    idx = spy.index.intersection(spmo.index)
    idx = idx[idx >= inception]
    spy_nav = spy.reindex(idx); spy_nav = spy_nav / spy_nav.iloc[0]
    spmo_nav = spmo.reindex(idx); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    _log(f"SPMO 상장일(첫 거래일): {inception.date()} · 공통구간 {idx[0].date()}~{idx[-1].date()} "
         f"({len(idx)}거래일)")
    return spy_nav, spmo_nav, inception


def _annual_returns(spy_nav: pd.Series, spmo_nav: pd.Series) -> list:
    years = sorted(set(spy_nav.index.year))
    rows = []
    for y in years:
        mask = spy_nav.index.year == y
        s_spy, s_spmo = spy_nav[mask], spmo_nav[mask]
        if len(s_spy) < 2:
            continue
        r_spy = float(s_spy.iloc[-1] / s_spy.iloc[0] - 1) * 100
        r_spmo = float(s_spmo.iloc[-1] / s_spmo.iloc[0] - 1) * 100
        partial = not (s_spy.index[0].month == 1 and s_spy.index[-1].month == 12 and
                       s_spy.index[-1].day >= 28)
        rows.append({"year": int(y), "spy_pct": round(r_spy, 1), "spmo_pct": round(r_spmo, 1),
                    "diff_pct": round(r_spmo - r_spy, 1), "partial_year": partial})
        _log(f"{y}{'(부분)' if partial else ''}: SPY {r_spy:+.1f}% · SPMO {r_spmo:+.1f}% · "
             f"차이 {r_spmo-r_spy:+.1f}%p")
    return rows


def _zigzag_regimes(ratio: pd.Series, threshold: float) -> list:
    """상대강도(SPMO/SPY) 비율의 지그재그 국면 분해 — threshold(예: 0.10=10%) 이상
    반전마다 피벗을 찍어 그 사이 구간을 '우위/열위 국면'으로 나눈다(고전적 zigzag
    지표와 동일 원리, 외부 라이브러리 없이 직접 구현). direction=None(방향 미확정)
    동안은 첫 반전이 나올 때까지 그냥 대기 — 확정 전에는 피벗을 찍지 않는다."""
    dates = ratio.index
    vals = ratio.values
    n = len(vals)
    pivots = [0]
    direction = None            # None(미확정) → "up"/"down"
    extreme_idx = 0             # 현재 추세의 극값 인덱스(direction 확정 후에만 유효)

    for i in range(1, n):
        if direction is None:
            if vals[i] >= vals[0] * (1 + threshold):
                direction = "up"; extreme_idx = i
            elif vals[i] <= vals[0] * (1 - threshold):
                direction = "down"; extreme_idx = i
            continue
        if direction == "up":
            if vals[i] > vals[extreme_idx]:
                extreme_idx = i
            elif vals[i] <= vals[extreme_idx] * (1 - threshold):
                pivots.append(extreme_idx)
                direction = "down"; extreme_idx = i
        else:
            if vals[i] < vals[extreme_idx]:
                extreme_idx = i
            elif vals[i] >= vals[extreme_idx] * (1 + threshold):
                pivots.append(extreme_idx)
                direction = "up"; extreme_idx = i

    last_idx = extreme_idx if direction is not None else n - 1
    if pivots[-1] != last_idx:
        pivots.append(last_idx)
    if pivots[-1] != n - 1:
        pivots.append(n - 1)

    regimes = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        if a == b:
            continue
        chg = float(vals[b] / vals[a] - 1) * 100
        regimes.append({
            "start": dates[a].date().isoformat(), "end": dates[b].date().isoformat(),
            "trading_days": int(b - a),
            "regime": "SPMO 우위 국면" if chg > 0 else "SPMO 열위 국면",
            "relative_change_pct": round(chg, 1),
        })
    return regimes


def _rolling_12m_relative(spy_nav: pd.Series, spmo_nav: pd.Series, window=252) -> pd.Series:
    r_spy = spy_nav / spy_nav.shift(window) - 1
    r_spmo = spmo_nav / spmo_nav.shift(window) - 1
    return (r_spmo - r_spy).dropna() * 100


def run(save=True):
    spy_nav, spmo_nav, inception = _load()
    full_spy_ret = float(spy_nav.iloc[-1] / spy_nav.iloc[0] - 1) * 100
    full_spmo_ret = float(spmo_nav.iloc[-1] / spmo_nav.iloc[0] - 1) * 100
    yrs = len(spy_nav) / 252
    cagr_spy = float(spy_nav.iloc[-1] ** (1 / yrs) - 1) * 100
    cagr_spmo = float(spmo_nav.iloc[-1] ** (1 / yrs) - 1) * 100
    _log(f"전체({yrs:.1f}년): SPY 누적 {full_spy_ret:+.1f}%(CAGR{cagr_spy:.2f}%) vs "
         f"SPMO 누적 {full_spmo_ret:+.1f}%(CAGR{cagr_spmo:.2f}%)")

    annual = _annual_returns(spy_nav, spmo_nav)

    ratio = spmo_nav / spy_nav
    regimes = _zigzag_regimes(ratio, ZIGZAG_THRESHOLD)
    _log(f"지그재그 국면(임계값 {int(ZIGZAG_THRESHOLD*100)}%): {len(regimes)}개")
    for r in regimes:
        _log(f"  {r['start']}~{r['end']}({r['trading_days']}거래일): {r['regime']} "
             f"{r['relative_change_pct']:+.1f}%p")

    roll = _rolling_12m_relative(spy_nav, spmo_nav)
    best5 = roll.nlargest(5)
    worst5 = roll.nsmallest(5)
    _log("롤링 12개월 상대수익률 최고 5구간(종료일 기준): " +
         " · ".join(f"{d.date()}:{v:+.1f}%p" for d, v in best5.items()))
    _log("롤링 12개월 상대수익률 최악 5구간(종료일 기준): " +
         " · ".join(f"{d.date()}:{v:+.1f}%p" for d, v in worst5.items()))

    payload = {
        "spmo_inception": inception.date().isoformat(),
        "as_of": spy_nav.index[-1].date().isoformat(),
        "years": round(yrs, 2),
        "full_period": {"spy_cum_pct": round(full_spy_ret, 1), "spmo_cum_pct": round(full_spmo_ret, 1),
                        "spy_cagr_pct": round(cagr_spy, 2), "spmo_cagr_pct": round(cagr_spmo, 2)},
        "annual_returns": annual,
        "zigzag_regimes": {"threshold_pct": ZIGZAG_THRESHOLD * 100, "regimes": regimes},
        "rolling_12m_relative_return_pct": {
            "best5_end_dates": {str(d.date()): round(float(v), 1) for d, v in best5.items()},
            "worst5_end_dates": {str(d.date()): round(float(v), 1) for d, v in worst5.items()},
        },
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_vs_spy_detailed.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    run()
