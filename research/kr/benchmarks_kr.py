#!/usr/bin/env python3
"""
benchmarks_kr.py — KR_STRATEGY_OPTIONS.md §1: 3중 벤치마크 재정의.

삼전·하이닉스 2종목이 끌어올린 시총가중 지수(B1)만으로 종목선정 스킬을 판정하는 왜곡을
해소하기 위해 두 벤치마크를 추가 산출한다:

  B1: 코스피200 지수 (시총가중 — 현행 backtest_kr 벤치마크 그대로, 최종 성적표)
  B2: 코스피200 동일가중 — **종목선정 스킬의 진짜 잣대** (일별 동일가중 평균수익률,
      멤버십은 kr_bt_cache의 PIT 조회분을 구간별 적용)
  B3: 코스피200 ex-Top2 — 삼전(005930)·하이닉스(000660) 제외 시총가중
      (리밸 시점 시총 고정 가중 — 구간 내 드리프트 무시 근사)

한계(정직 고지): TR(배당 포함) 아님 — 세 벤치마크 모두 가격지수 기준이라 상호 비교는 공정.
B2·B3는 상장폐지 종목 시세 누락(yfinance)만큼 생존편향 잔존 — B1과의 갭 해석 시 감안.

실행: python benchmarks_kr.py            # output/benchmarks_kr.json
      python benchmarks_kr.py --self-test
다른 스크립트에서: from benchmarks_kr import load_research_data, build_benchmarks
"""
from __future__ import annotations
import os, sys, json, argparse, pickle
import numpy as np
import pandas as pd

PANEL_CACHE = "output/kr_panel_cache.pkl"
OUT_PATH = "output/benchmarks_kr.json"
TOP2 = ("005930", "000660")


def _log(m): print(f"[벤치마크KR] {m}", file=sys.stderr)


def load_research_data(years=8, rebal_days=63):
    """(panel, membership, fundamentals, flows, mktcaps, bench) — 패널은 pkl 캐시 우선."""
    import backtest_kr as BK
    cache = BK._load_cache()
    if os.path.exists(PANEL_CACHE):
        with open(PANEL_CACHE, "rb") as f:
            d = pickle.load(f)
        panel, bench = d["panel"], d["bench"]
        _log(f"패널 캐시 로드: {panel.shape[1]}종목 × {panel.shape[0]}일 "
             f"({panel.index[0].date()} ~ {panel.index[-1].date()})")
        return (panel, cache.get("membership", {}), cache.get("fundamentals", {}),
                cache.get("flows", {}), cache.get("mktcap", {}), bench)
    panel, membership, fundamentals, flows, mktcaps, bench = BK.prepare_kr_data(years, rebal_days)
    with open(PANEL_CACHE, "wb") as f:
        pickle.dump({"panel": panel, "bench": bench}, f)
    return panel, membership, fundamentals, flows, mktcaps, bench


def _membership_segments(panel: pd.DataFrame, membership: dict):
    """캐시된 PIT 멤버십 날짜들을 구간 경계로 → [(시작 i, 끝 i, members)] (끝 미포함)."""
    keys = sorted(k for k in membership if k != "_current" and membership[k])
    bounds = []
    for k in keys:
        ts = pd.Timestamp(k)
        i = panel.index.searchsorted(ts)
        if i < len(panel.index):
            bounds.append((i, membership[k]))
    if not bounds:
        cur = membership.get("_current") or list(panel.columns)
        return [(0, len(panel), cur)]
    segs = []
    first_i, first_m = bounds[0]
    if first_i > 0:
        segs.append((0, first_i, first_m))          # 첫 조회 이전은 첫 멤버십으로 소급
    for j, (i, m) in enumerate(bounds):
        end = bounds[j + 1][0] if j + 1 < len(bounds) else len(panel)
        segs.append((i, end, m))
    return segs


def build_benchmarks(panel: pd.DataFrame, membership: dict, mktcaps: dict,
                     bench: pd.Series) -> pd.DataFrame:
    """일별 NAV(시작 1.0) DataFrame: columns=[B1_kospi200, B2_equal, B3_ex_top2]."""
    rets = panel.pct_change()
    segs = _membership_segments(panel, membership)
    mc_keys = sorted(mktcaps.keys())

    ew = pd.Series(np.nan, index=panel.index)
    cw_ex = pd.Series(np.nan, index=panel.index)
    for i0, i1, members in segs:
        cols = [c for c in panel.columns if c in set(members)]
        if len(cols) < 20:
            continue
        seg = rets.iloc[i0:i1][cols]
        ew.iloc[i0:i1] = seg.mean(axis=1)           # 일별 동일가중(매일 리밸 근사)
        # B3: 구간 시작일 이전 가장 가까운 시총 스냅샷으로 고정 가중
        d8 = panel.index[i0].strftime("%Y%m%d")
        k = None
        for key in mc_keys:
            if key <= d8 and mktcaps.get(key):
                k = key
        mc = mktcaps.get(k) or {}
        w = pd.Series({c: mc.get(c, np.nan) for c in cols if c not in TOP2}).dropna()
        if len(w) >= 20:
            w = w / w.sum()
            cw_ex.iloc[i0:i1] = seg[w.index].mul(w, axis=1).sum(axis=1, min_count=10)

    b1 = bench.reindex(panel.index).ffill()
    out = pd.DataFrame({
        "B1_kospi200": b1 / b1.iloc[0],
        "B2_equal": (1 + ew.fillna(0)).cumprod(),
        "B3_ex_top2": (1 + cw_ex.fillna(0)).cumprod(),
    }, index=panel.index)
    return out


def _cagr(nav: pd.Series) -> float:
    nav = nav.dropna()
    yrs = len(nav) / 252
    return round(100 * float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1), 2) if yrs > 0.2 else np.nan


def summarize(navs: pd.DataFrame) -> dict:
    subs = [("full", None, None), ("2018-2021", None, "2021-12-31"),
            ("2022-2023", "2022-01-01", "2023-12-31"), ("2024+", "2024-01-01", None)]
    rows = {}
    for name, a, b in subs:
        w = navs.loc[a:b] if (a or b) else navs
        if len(w) < 60:
            continue
        rows[name] = {c: _cagr(w[c]) for c in navs.columns}
    r1 = navs["B1_kospi200"].pct_change()
    corr = {c: round(float(r1.corr(navs[c].pct_change())), 3) for c in navs.columns if c != "B1_kospi200"}
    return {"cagr_pct": rows, "corr_daily_vs_B1": corr}


def run(save=True):
    panel, membership, _, _, mktcaps, bench = load_research_data()
    navs = build_benchmarks(panel, membership, mktcaps, bench)
    s = summarize(navs)
    payload = {"as_of": panel.index[-1].date().isoformat(),
               "n_days": len(navs), "summary": s,
               "usage": ("판정 규칙(§1): 알파 전략은 B2를 위험조정 기준으로 이기면 채택 후보. "
                         "B1은 코어+새틀라이트 합산의 목표."),
               "caveat": "가격지수 기준(배당 제외)·B2/B3는 yfinance 시세 생존편향 잔존",
               "nav": {c: {d.date().isoformat(): round(float(v), 6)
                           for d, v in navs[c].dropna().items()} for c in navs.columns}}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        _log(f"저장: {OUT_PATH}")
    for name, row in s["cagr_pct"].items():
        _log(f"  {name:10s} " + "  ".join(f"{c} {v}%" for c, v in row.items()))
    _log(f"  일별수익 상관(vs B1): {s['corr_daily_vs_B1']}")
    return payload, navs


def load_benchmarks() -> pd.DataFrame:
    """저장된 벤치마크 NAV 로드(Phase 3에서 사용)."""
    with open(OUT_PATH, encoding="utf-8") as f:
        p = json.load(f)
    df = pd.DataFrame({c: pd.Series(v) for c, v in p["nav"].items()})
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def self_test():
    _log("[self-test] 합성 데이터: Top2 쏠림을 심고 B1>B3, B2가 중앙값 종목을 따르는지 검증")
    rng = np.random.default_rng(3)
    n, m = 800, 40
    idx = pd.bdate_range("2021-01-01", periods=n)
    drift = np.full(m, 0.0002); drift[:2] = 0.0015          # Top2만 강한 상승
    cols = ["005930", "000660"] + [f"{i:06d}" for i in range(2, m)]
    panel = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, 0.015, (n, m)), axis=0)),
                         index=idx, columns=cols)
    mc = {c: (5e14 if c in TOP2 else 1e13) for c in cols}
    membership = {idx[0].strftime("%Y%m%d"): cols}
    mktcaps = {idx[0].strftime("%Y%m%d"): mc}
    # B1 근사: 시총가중(Top2 포함)
    w = pd.Series(mc, dtype=float); w /= w.sum()
    b1 = (1 + panel.pct_change()[w.index].mul(w, axis=1).sum(axis=1)).cumprod() * 100
    navs = build_benchmarks(panel, membership, mktcaps, b1)
    c1, c2, c3 = (_cagr(navs[c]) for c in navs.columns)
    assert c1 > c3 + 5, f"Top2 쏠림 재현 실패: B1 {c1}% vs B3 {c3}%"
    assert abs(c2 - c3) < abs(c1 - c3), f"B2는 B3에 가까워야 함: {c1}/{c2}/{c3}"
    s = summarize(navs)
    # 합성 데이터는 B1의 71%가 Top2(B2·B3에서 제외)라 상관 크기 자체는 검증 대상 아님
    assert all(np.isfinite(v) for v in s["corr_daily_vs_B1"].values()), s
    _log(f"[self-test] 통과: B1 {c1}% > B3 {c3}% · B2 {c2}% · corr {s['corr_daily_vs_B1']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="코스피200 3중 벤치마크(B1/B2/B3) 산출")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else run()
