#!/usr/bin/env python3
"""
bond_asym_band.py — 채권 추세필터 비대칭 밴드(상승진입=넉넉히·하락이탈=타이트하게) 검증
(2026-07-23, 지호 님 아이디어)

배경: bond_trend_filter_grid.py의 대칭 밴드 그리드에서 "밴드가 넓을수록 나쁘다"(IEF)가
확인됐는데, 지호 님 지적: 상승(재진입)과 하락(이탈)에 같은 밴드를 쓸 이유가 없다 —
"내려갈 땐 타이트하게(빨리 방어), 올라갈 땐 넉넉하게(신중하게 재진입)"가 위험관리
철학(이 프로젝트가 추세필터를 쓰는 이유 자체)과 더 맞는다.

방법: regime_series를 band_up(ON 진입 임계, MA 위)·band_down(OFF 이탈 임계, MA 아래)
독립으로 받도록 확장. band_down < band_up 조합(비대칭, 요청대로) + band_down == band_up
(기존 대칭, 대조군)을 같은 그리드에 등록해 직접 비교. trend_ma는 각 자산의 stage1
최우수 근방값으로 고정(밴드 비대칭 효과만 순수 비교하기 위해 — 안 그러면 두 변수가
섞임), confirm은 기존에 확인된 "길수록 유리" 패턴 반영해 [3,5,10] 유지.

실행: python bond_asym_band.py
결과: output/bond_asym_band_{ief,tlt,shy}.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import fetch, simulate, composite_score, _ulcer, _cagr

COST_BPS = 5
MIN_OFF_EPISODES = 8

# 각 자산 stage1 최우수 근방(bond_trend_filter_grid.py 결과) — 밴드 효과만 보려고 고정
TREND_MA_BY_ASSET = {"ief": 150, "tlt": 100, "shy": 300}

BAND_DOWN = [0.0, 0.005, 0.01]        # 하락 이탈(OFF) 임계 — 타이트
BAND_UP = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]   # 상승 진입(ON) 임계 — 넉넉(0 포함=대칭 기준선 확보)
CONFIRM = [1, 3, 5, 10]


def _log(m): print(f"[비대칭밴드] {m}", file=sys.stderr)


def regime_series_asym(closes: np.ndarray, trend_ma: int, band_up: float, band_down: float,
                       confirm: int) -> np.ndarray:
    """market_signals.regime_state와 동일 히스테리시스 구조이되 상승(ON) 진입 임계와
    하락(OFF) 이탈 임계를 독립으로 받는다. band_up==band_down이면 기존 대칭 로직과 동일."""
    import pandas as pd
    n = len(closes)
    ma = pd.Series(closes).rolling(trend_ma).mean().to_numpy()
    out = np.full(n, np.nan)
    state = None
    streak_dir, streak = None, 0
    for i in range(n):
        if np.isnan(ma[i]):
            continue
        c = closes[i]
        if c > ma[i] * (1 + band_up):
            raw = "ON"
        elif c < ma[i] * (1 - band_down):
            raw = "OFF"
        else:
            raw = None
        if raw and raw != state:
            if raw == streak_dir:
                streak += 1
            else:
                streak_dir, streak = raw, 1
            if streak >= confirm:
                state = raw
                streak_dir, streak = None, 0
        else:
            streak_dir, streak = None, 0
        out[i] = np.nan if state is None else (1.0 if state == "ON" else 0.0)
    return out


def run_asset(name: str, ticker: str) -> dict:
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    trend_ma = TREND_MA_BY_ASSET[name]
    always_on = simulate(closes, np.ones(len(closes)), COST_BPS)

    rows = []
    for bd in BAND_DOWN:
        for bu in BAND_UP:
            if bu < bd:
                continue   # "올라갈 때 넉넉·내려갈 때 타이트" 취지상 band_up >= band_down만
            for cf in CONFIRM:
                exp = regime_series_asym(closes, trend_ma, bu, bd, cf)
                m = simulate(closes, exp, COST_BPS)
                score = composite_score(m)
                rows.append({"band_up": bu, "band_down": bd, "confirm": cf,
                            "symmetric": bu == bd,
                            "cagr": round(m["cagr"], 2), "ulcer": round(m["ulcer"], 2),
                            "mdd": round(m["mdd"], 1), "off_episodes": m["off_episodes"],
                            "score": score, "rankable": m["off_episodes"] >= MIN_OFF_EPISODES})

    rankable = [r for r in rows if r["rankable"] and r["score"] != float("-inf")]
    asym = [r for r in rankable if not r["symmetric"]]
    sym = [r for r in rankable if r["symmetric"]]
    asym_scores = [r["score"] for r in asym]
    sym_scores = [r["score"] for r in sym]

    rankable.sort(key=lambda r: r["score"], reverse=True)
    best = rankable[0] if rankable else None
    best_sym = max(sym, key=lambda r: r["score"]) if sym else None

    _log(f"[{name}] trend_ma={trend_ma} 고정 · 비대칭조합 {len(asym)}개(평균score "
         f"{np.mean(asym_scores) if asym_scores else float('nan'):.3f}) vs "
         f"대칭조합 {len(sym)}개(평균score {np.mean(sym_scores) if sym_scores else float('nan'):.3f})")
    _log(f"[{name}] 전체 최우수: {best}")
    _log(f"[{name}] 대칭 중 최우수: {best_sym}")

    payload = {
        "asset": name, "ticker": ticker, "trend_ma_fixed": trend_ma,
        "baseline_no_filter": {"cagr": round(always_on["cagr"], 2), "ulcer": round(always_on["ulcer"], 2),
                               "mdd": round(always_on["mdd"], 1)},
        "n_rankable_total": len(rankable), "n_asymmetric": len(asym), "n_symmetric": len(sym),
        "asym_score_mean": round(float(np.mean(asym_scores)), 3) if asym_scores else None,
        "sym_score_mean": round(float(np.mean(sym_scores)), 3) if sym_scores else None,
        "asym_score_median": round(float(np.median(asym_scores)), 3) if asym_scores else None,
        "sym_score_median": round(float(np.median(sym_scores)), 3) if sym_scores else None,
        "best_overall": best, "best_symmetric": best_sym,
        "all_rows": rows,
    }
    os.makedirs("output", exist_ok=True)
    with open(f"output/bond_asym_band_{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"[{name}] 저장: output/bond_asym_band_{name}.json")
    return payload


def main():
    for name, ticker in [("ief", "IEF"), ("tlt", "TLT"), ("shy", "SHY")]:
        run_asset(name, ticker)


if __name__ == "__main__":
    main()
