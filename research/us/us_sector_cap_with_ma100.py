#!/usr/bin/env python3
"""
us_sector_cap_with_ma100.py — 지호 님 질문(2026-09-23): "100일선 필터 이후의 섹터캡"
재검증. 원래 us_sector_cap_sweep.py(2026-07-17)는 floor·100일선 필터가 없던 시절
검증이라, 이번에 floor=3.25+100일선 필터까지 적용한 뒤(오늘 실측: 후보가 26→16으로
줄면서 Health Care 쏠림이 더 세짐) 섹터캡을 무제한/1/2/3/4로 다시 스윕한다.

가벼운 이벤트평균 프레임(backtest_portfolio.py의 무거운 NAV 시뮬레이터 대신 이번 세션
전체에서 쓴 것과 동일한 방식) + overfit_stats._mdd로 이벤트 수익률을 복리연결한 낙폭
근사치를 같이 낸다 — 원래 트레이드오프(CAGR↑ vs MDD↓)를 이 프레임으로 재현.

실행: python -m research.us.us_sector_cap_with_ma100 [--years 10]
결과: output/us_sector_cap_with_ma100.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import sp500_daily_report as R
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features

TOPN = 8
FLOOR = 3.25
CAP_GRID = [None, 1, 2, 3, 4]
OUT_PATH = "output/us_sector_cap_with_ma100.json"


def _log(m): print(f"[US섹터캡+100일선]  {m}", file=sys.stderr)


def _pick_with_cap(score: pd.Series, sector_map: dict, cap: int | None, topn: int) -> list:
    ranked = score.sort_values(ascending=False).index.tolist()
    if cap is None:
        return ranked[:topn]
    out, per_sec = [], {}
    for sym in ranked:
        sec = sector_map.get(sym) or "(기타)"
        if per_sec.get(sec, 0) >= cap:
            continue
        out.append(sym)
        per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(out) >= topn:
            break
    return out


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")
    sector_map = R.fetch_wikipedia_sectors()
    _log(f"섹터맵 {len(sector_map)}종목(현재 시점 GICS — PIT 아님, 원 검증과 동일 근사 한계)")

    pools = []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score >= FLOOR]
        if len(score) < TOPN:
            pools.append(None)
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        above = mom["ma100_gap"] > 0 if "ma100_gap" in mom.columns else pd.Series(True, index=score.index)
        score = score[above.reindex(score.index).fillna(False)]
        pools.append({"score": score, "fwd": fwd, "bench": bench})

    rows, trial_labels, trial_returns, pool_sizes = [], [], [], []
    for cap in CAP_GRID:
        ex = []
        n_secs_used = []
        for p in pools:
            if p is None or len(p["score"]) < 1:
                continue
            top = _pick_with_cap(p["score"], sector_map, cap, TOPN)
            r = p["fwd"].reindex(top).dropna()
            if len(r) == 0:
                continue
            ex.append(float(r.mean()) - p["bench"])
            n_secs_used.append(len({sector_map.get(s) or "(기타)" for s in top}))
        if len(ex) < 8:
            continue
        a = np.array(ex)
        mdd = OS._mdd(a)
        row = {"sector_cap": cap if cap is not None else "무제한", "n_events": len(a),
              "mean_excess_pct": round(100 * float(a.mean()), 3),
              "win_rate_pct": round(100 * float((a > 0).mean()), 1),
              "mdd_proxy_pct": round(100 * mdd, 2),
              "avg_n_sectors_in_top8": round(float(np.mean(n_secs_used)), 2)}
        rows.append(row)
        trial_labels.append(f"cap{cap}")
        trial_returns.append(ex)
        _log(f"cap={row['sector_cap']}: 초과 {row['mean_excess_pct']:+.2f}%p 승률 {row['win_rate_pct']}% "
            f"낙폭근사 {row['mdd_proxy_pct']}% 평균섹터수 {row['avg_n_sectors_in_top8']} n={row['n_events']}")

    lens = [len(ex) for ex in trial_returns]
    pbo_report = None
    if lens:
        common_len = max(set(lens), key=lens.count)
        keep_idx = [i for i, l in enumerate(lens) if l == common_len]
        if len(keep_idx) >= 3 and common_len >= 8:
            data = {"trials": [trial_labels[i] for i in keep_idx],
                   "excess_returns": [trial_returns[i] for i in keep_idx],
                   "rebal_days": 63, "hold_days": 126, "horizon": "6m",
                   "universe": "US_floor_ma100_top8", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)

    payload = {"n_snaps": len(snaps), "topn": TOPN, "floor": FLOOR, "cap_grid": CAP_GRID,
              "rows": rows, "pbo_dsr": pbo_report,
              "method": ("floor=3.25+100일선필터 통과 후보에서 섹터캡 무제한/1/2/3/4 스윕. "
                        "mdd_proxy_pct = 이벤트 초과수익을 순서대로 복리연결한 누적곡선의 "
                        "최대낙폭(overfit_stats._mdd, 몬테카를로 검정과 동일 정의) — 전체 NAV "
                        "재시뮬레이션 아닌 근사치. sector_map은 현재 시점 위키피디아 GICS "
                        "(point-in-time 아님, 원 검증 us_sector_cap_sweep.py와 동일 한계).")}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    run(years=args.years)


if __name__ == "__main__":
    main()
