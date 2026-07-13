#!/usr/bin/env python3
"""
core_satellite_kr.py — KR_STRATEGY_OPTIONS.md §2-F: 코어-새틀라이트 재구성 시뮬레이션.

"삼전·하이닉스를 퀀트로 맞추려 하지 말고 코어(지수)로 그냥 들고, 알파는 새틀라이트에서"
— B1(시총가중 지수)을 이기는 책임을 코어+새틀라이트 합산에 지운다(§1 판정 규칙과 세트).

구성(사전 등록 — 팩터 탐색 아님, 포트폴리오 설계 비교 4종):
  P1 B1 보유(대조군)          : 코스피200 매수후보유
  P2 코어 단독(레짐 타이밍)    : B1 × STRATEGY.md §1 레짐(200일선 ±1% 히스테리시스·3일 확인,
                                OFF 시 현금 0% — 무이자 보수적 가정)
  P3 65/35 코어+새틀라이트     : P2 코어 65% + valuediv_flow(Phase 3 생존자) 35%, 월간 리밸
  P4 새틀라이트 단독           : valuediv_flow 100% (backtest_kr_strategies 산출 재사용)

실행: python core_satellite_kr.py          # output/core_satellite_kr.json
      python core_satellite_kr.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

OUT_PATH = "output/core_satellite_kr.json"
SATELLITE = "valuediv_flow"
CORE_W = 0.65
MONTH = 21


def _log(m): print(f"[코어새틀KR] {m}", file=sys.stderr)


def regime_series(b1: pd.Series, band=0.01, confirm=3) -> pd.Series:
    """STRATEGY.md §1: 200일선 ±band 히스테리시스 + confirm일 연속 확인. True=ON."""
    ma = b1.rolling(200, min_periods=200).mean()
    above = b1 > ma * (1 + band)
    below = b1 < ma * (1 - band)
    state = pd.Series(True, index=b1.index)   # 초기 ON(패널 시작이 상승기라는 가정 아님 —
    cur, na, nb = True, 0, 0                  # MA 미형성 구간은 ON 유지, 결과 해석 시 감안)
    for i in range(len(b1)):
        if pd.isna(ma.iloc[i]):
            state.iloc[i] = cur; continue
        na = na + 1 if above.iloc[i] else 0
        nb = nb + 1 if below.iloc[i] else 0
        if not cur and na >= confirm:
            cur = True
        elif cur and nb >= confirm:
            cur = False
        state.iloc[i] = cur
    return state


def timed_nav(b1: pd.Series, regime: pd.Series) -> pd.Series:
    """레짐 ON일 다음날 수익 적용(종가 신호·익일 실행), OFF면 0%."""
    r = b1.pct_change().fillna(0)
    exposed = regime.shift(1, fill_value=True)
    return (1 + r.where(exposed, 0.0)).cumprod()


def mix_nav(core: pd.Series, sat: pd.Series, w_core: float, rebal=MONTH) -> pd.Series:
    """월간 목표비중 리밸런싱 혼합 NAV."""
    idx = core.index.intersection(sat.index)
    rc, rs = core.reindex(idx).pct_change().fillna(0), sat.reindex(idx).pct_change().fillna(0)
    nav, vc, vs = [], w_core, 1 - w_core
    for i in range(len(idx)):
        vc *= 1 + rc.iloc[i]; vs *= 1 + rs.iloc[i]
        tot = vc + vs
        nav.append(tot)
        if i % rebal == 0:
            vc, vs = tot * w_core, tot * (1 - w_core)
    return pd.Series(nav, index=idx)


def stats(nav: pd.Series, a=None, b=None) -> dict | None:
    w = nav.loc[a:b] if (a or b) else nav
    if len(w) < 60:
        return None
    r = w.pct_change().dropna()
    yrs = len(w) / 252
    return {"cagr_pct": round(100 * float((w.iloc[-1] / w.iloc[0]) ** (1 / yrs) - 1), 2),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() else 0.0,
            "mdd_pct": round(100 * float((w / w.cummax() - 1).min()), 1)}


SUBS = [("full", None, None), ("2018-2021", None, "2021-12-31"),
        ("2022-2023", "2022-01-01", "2023-12-31"), ("2024+", "2024-01-01", None)]


def run(save=True):
    from benchmarks_kr import load_benchmarks
    bm = load_benchmarks()
    b1 = bm["B1_kospi200"].dropna()
    with open("output/kr_strategy_navs.json", encoding="utf-8") as f:
        navs = json.load(f)
    sat = pd.Series(navs[SATELLITE]); sat.index = pd.to_datetime(sat.index)
    sat = sat.sort_index()
    reg = regime_series(b1)
    core = timed_nav(b1, reg)
    ports = {"P1_B1_hold": b1 / b1.iloc[0],
             "P2_core_timed": core,
             "P3_core65_sat35": mix_nav(core, sat, CORE_W),
             "P4_satellite": sat / sat.iloc[0]}
    rows = {}
    for name, nav in ports.items():
        rows[name] = {tag: stats(nav, a, b) for tag, a, b in SUBS}
        f = rows[name]["full"]
        _log(f"{name:18s} CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} MDD {f['mdd_pct']:6.1f}%")
    off_share = round(100 * float((~reg).mean()), 1)
    payload = {"as_of": b1.index[-1].date().isoformat(), "satellite": SATELLITE,
               "core_weight": CORE_W, "regime_off_days_pct": off_share,
               "rows": rows,
               "note": ("§2-F 설계 비교(팩터 탐색 아님). P3가 B1 대비 샤프·MDD를 개선하면 "
                        "'B1을 이기는 책임'을 합산 구조가 감당한다는 §1 논리의 실증. "
                        "코어 OFF 시 현금 0% 가정(보수적 — CMA 금리 미반영).")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH} (레짐 OFF 일수 {off_share}%)")
    return payload


def self_test():
    _log("[self-test] 합성: 급락 구간에서 레짐 타이밍이 MDD를 줄이는지 검증")
    rng = np.random.default_rng(31)
    idx = pd.bdate_range("2019-01-01", periods=1200)
    r = np.full(1200, 0.0008)
    r[500:650] = -0.006                     # 5개월 약세장
    b1 = pd.Series(100 * np.exp(np.cumsum(r + rng.normal(0, 0.008, 1200))), index=idx)
    reg = regime_series(b1)
    core = timed_nav(b1, reg)
    s_b1, s_core = stats(b1), stats(core)
    assert s_core["mdd_pct"] > s_b1["mdd_pct"] + 5, f"MDD 개선 실패: {s_b1} vs {s_core}"
    sat = pd.Series(np.linspace(1, 2, 1200), index=idx)   # 무변동 상승 새틀라이트
    mixed = mix_nav(core, sat, 0.65)
    s_mix = stats(mixed)
    assert s_mix["mdd_pct"] >= s_core["mdd_pct"], (s_mix, s_core)
    assert abs(len(mixed) - len(idx)) < 3
    _log(f"[self-test] 통과: B1 MDD {s_b1['mdd_pct']}% → 코어 {s_core['mdd_pct']}% → "
         f"혼합 {s_mix['mdd_pct']}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="코어-새틀라이트(§2-F) 설계 비교")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else run()
