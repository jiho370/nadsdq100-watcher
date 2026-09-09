#!/usr/bin/env python3
"""
verify_champion_nav_stats.py — 2026-08-22 알고리즘 재검증 세션: topn8 라이브 챔피언
(가중치 1:2:2·shareholder_yield 클립±5·섹터캡2·ma200_backup=False) CAGR/샤프/MDD를
output/_algo_vs_spy_nav_cache.csv(algo_vs_spy_winrate_by_horizon.py가 만든 15년 NAV,
"라이브 그대로" 재현 — us_spmo_blend_prereg._us_decisions_live_clip 사용)에서 직접
계산한다. 새로 패널을 내려받지 않고 이미 만들어진 NAV를 재사용하므로 야후 요청을
중복하지 않는다(algo_vs_index_lenient.py는 별도 10년 패널을 새로 받고, backtest_
portfolio.py CLI 기본값은 ma200_backup=True 구식 설정이라 라이브를 재현하지 않음 —
둘 다 이 재검증 목적엔 부적합해 이 스크립트를 새로 작성).

비교 대상(둘 다 이 저장소에 이미 기록됨):
  - output/us_algo_15y_champion_check.json (§6-O, 2026-07-24 as_of): CAGR 43.35%·
    샤프 1.24·MDD -55.8%(SPY 14.99%/0.92/-33.7%), 9y/11y/15y 3구간 비교도 포함.
  - output/algo_vs_spy_winrate_by_horizon.json (2026-07-24 as_of): 승률-보유기간 곡선.

이 스크립트는 그 두 파일을 덮어쓰지 않고 output/us_algo_champion_recheck.json에
새로 저장한다(원본은 이력 보존용으로 남김).

실행: python verify_champion_nav_stats.py  (algo_vs_spy_winrate_by_horizon.py를 먼저 실행해
      output/_algo_vs_spy_nav_cache.csv가 있어야 함)
결과: output/us_algo_champion_recheck.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

NAV_CACHE = "output/_algo_vs_spy_nav_cache.csv"
OUT_PATH = "output/us_algo_champion_recheck.json"


def _log(m): print(f"[챔피언재검증] {m}", file=sys.stderr)


def _stats(nav: pd.Series) -> dict:
    nav = nav / nav.iloc[0]
    r = nav.pct_change().dropna()
    yrs = len(nav) / 252
    cagr = 100 * float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else 0.0
    mdd = 100 * float((nav / nav.cummax() - 1).min())
    return {"cagr_pct": round(cagr, 2), "sharpe": round(sharpe, 3), "mdd_pct": round(mdd, 1),
            "n_days": len(nav), "years": round(yrs, 2)}


def _yearly_returns(nav: pd.Series) -> list:
    nav = nav / nav.iloc[0]
    out = []
    for y, grp in nav.groupby(nav.index.year):
        if len(grp) < 2:
            continue
        out.append({"year": int(y), "return_pct": round(100 * float(grp.iloc[-1] / grp.iloc[0] - 1), 1)})
    return out


def _subwindow(algo: pd.Series, spy: pd.Series, years: float) -> dict:
    """전체 NAV 시리즈의 마지막 years년만 잘라 재베이스(리베이스) — 독립적으로 패널을
    다시 받아 재구성한 것은 아니라 §6-O의 9y/11y 수치와 완전히 동일한 방법론은 아님
    (§6-O는 --long-window로 그 기간만큼만 PIT 유니버스를 별도로 다시 뽑았음 — 유니버스
    크기 자체가 다를 수 있음). 여기서는 "같은 15년 시행의 부분구간을 보면 결론이
    바뀌는가"라는 저비용 정합성 체크로만 쓴다(방향성 참고, 원 방법론의 대체 아님)."""
    cutoff = algo.index[-1] - pd.DateOffset(years=years)
    a = algo.loc[algo.index >= cutoff]
    s = spy.loc[spy.index >= cutoff]
    return {"algo": _stats(a), "spy": _stats(s)}


def run(save=True) -> dict:
    if not os.path.exists(NAV_CACHE):
        raise RuntimeError(f"{NAV_CACHE} 없음 — 먼저 python algo_vs_spy_winrate_by_horizon.py 실행")
    df = pd.read_csv(NAV_CACHE, parse_dates=["date"]).set_index("date")
    algo_nav, spy_nav = df["algo_nav"], df["spy_nav"]
    _log(f"NAV 캐시 로드: {algo_nav.index[0].date()} ~ {algo_nav.index[-1].date()} ({len(algo_nav)}거래일)")

    full_algo = _stats(algo_nav)
    full_spy = _stats(spy_nav)
    excess_cagr = round(full_algo["cagr_pct"] - full_spy["cagr_pct"], 2)
    _log(f"[전체 {full_algo['years']}년] 알고리즘 CAGR {full_algo['cagr_pct']}% 샤프 "
         f"{full_algo['sharpe']} MDD {full_algo['mdd_pct']}% | SPY CAGR {full_spy['cagr_pct']}% "
         f"샤프 {full_spy['sharpe']} MDD {full_spy['mdd_pct']}% | 초과CAGR {excess_cagr:+.2f}%p")

    subs = {}
    for yrs, label in ((9, "9y"), (11, "11y")):
        subs[label] = _subwindow(algo_nav, spy_nav, yrs)
        sa, ss = subs[label]["algo"], subs[label]["spy"]
        _log(f"[부분구간 {label}, 참고용] 알고리즘 CAGR {sa['cagr_pct']}% 샤프 {sa['sharpe']} "
             f"MDD {sa['mdd_pct']}% | SPY CAGR {ss['cagr_pct']}% MDD {ss['mdd_pct']}%")

    payload = {
        "as_of": algo_nav.index[-1].date().isoformat(),
        "data_start": algo_nav.index[0].date().isoformat(),
        "config": "topn8 라이브 그대로(best_weights.json 1:2:2, shareholder_yield 클립 ±5, "
                  "sector_cap=2, ma200_backup=False) — algo_vs_spy_winrate_by_horizon.py와 동일 NAV",
        "source_cache": NAV_CACHE,
        "full_period": {"algo": full_algo, "spy": full_spy, "excess_cagr_pct": excess_cagr},
        "yearly_returns_algo": _yearly_returns(algo_nav),
        "subwindows_same_panel_reference_only": subs,
        "compare_to": {
            "us_algo_15y_champion_check.json (2026-07-24 as_of)":
                {"algo_cagr_pct": 43.35, "algo_sharpe": 1.24, "algo_mdd_pct": -55.8,
                 "spy_cagr_pct": 14.99, "spy_sharpe": 0.92, "spy_mdd_pct": -33.7}},
        "caveat": "EDGAR 펀더멘털 커버리지가 15년 전 구간(2012~2015)에서 62~66%로 낮아 "
                  "결측 종목은 팩터 z=0(중립) 처리됨(§6-N/§6-O) — 2012~2015가 포함된 낙폭 수치는 "
                  "데이터결측 왜곡이 섞였을 가능성이 있음(방향성 신호로만 읽을 것). "
                  "subwindows는 15년 전체 패널(796종목 유니버스)의 부분구간 리베이스이지 "
                  "독립적인 --long-window 재구성이 아니므로 §6-O 수치와 소수점까지 일치하진 않음.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
