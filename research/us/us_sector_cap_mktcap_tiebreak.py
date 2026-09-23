#!/usr/bin/env python3
"""
us_sector_cap_mktcap_tiebreak.py — 지호 님 질문(2026-09-23): "섹터캡 설정시 시총순으로
나열하는건?" → "그니까 섹터캡 걸린 종목만 시총순으로 필터"

즉 섹터 구성(어느 섹터가 몇 자리를 먼저 채우는가)은 그대로 종합점수 순위가 결정하되,
캡 때문에 한 섹터 안에서 "누가 그 자리를 차지하는가"만 종합점수 대신 시총순으로 고르면
어떤지 검증. us_sector_cap_with_ma100.py의 _pick_with_cap(섹터 내부도 점수순)과 정확히
섹터 구성은 같고 섹터 내부 선택 기준만 다른 변형을 페어드 비교.

CAP_GRID=[1,2,3,4] (무제한은 섹터캡 자체가 안 걸리므로 비교 대상 아님 — us_sector_cap_with_ma100.py
에서 이미 무제한이 최선이라는 결론 남음, 이번엔 "캡을 쓴다면"이라는 조건부 질문에 답함).

실행: python -m research.us.us_sector_cap_mktcap_tiebreak [--years 10]
결과: output/us_sector_cap_mktcap_tiebreak.json
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
import sp500_daily_report as R
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features
from research.us.us_topn_by_mktcap import _mktcap_for

TOPN = 10
FLOOR = 3.25
CAP_GRID = [1, 2, 3, 4]
OUT_PATH = "output/us_sector_cap_mktcap_tiebreak.json"


def _log(m): print(f"[US섹터캡+시총타이브레이크]  {m}", file=sys.stderr)


def _pick_score_order(score: pd.Series, sector_map: dict, cap: int, topn: int) -> list:
    """기준(us_sector_cap_with_ma100.py와 동일): 섹터 구성도 내부 선택도 전부 점수순."""
    ranked = score.sort_values(ascending=False).index.tolist()
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


def _pick_mktcap_tiebreak(score: pd.Series, mktcap: pd.Series, sector_map: dict,
                          cap: int, topn: int) -> list:
    """섹터가 채워지는 순서(어느 섹터가 우선인가)는 점수순 그대로 — 다만 그 섹터 안에서
    실제로 뽑히는 종목은 점수순 선착순이 아니라 시총 내림차순으로 고른다."""
    ranked = score.sort_values(ascending=False).index.tolist()
    sec_members = {}
    for sym in ranked:
        sec = sector_map.get(sym) or "(기타)"
        sec_members.setdefault(sec, []).append(sym)
    sec_by_mktcap = {}
    for sec, syms in sec_members.items():
        with_cap = [s for s in syms if pd.notna(mktcap.get(s))]
        without_cap = [s for s in syms if s not in with_cap]
        sec_by_mktcap[sec] = sorted(with_cap, key=lambda s: mktcap[s], reverse=True) + without_cap

    out, per_sec, used = [], {}, set()
    for sym in ranked:
        sec = sector_map.get(sym) or "(기타)"
        if per_sec.get(sec, 0) >= cap:
            continue
        pick = next((s for s in sec_by_mktcap.get(sec, []) if s not in used), None)
        if pick is None or pick in used:
            continue
        out.append(pick)
        used.add(pick)
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
        if len(score) < TOPN:
            pools.append(None)
            continue
        price_row = panel.loc[pd.Timestamp(snap["date"])]
        mktcap = pd.Series({s: _mktcap_for(funds.get(s), snap["date"], float(price_row.get(s, np.nan)))
                            for s in score.index})
        pools.append({"score": score, "mktcap": mktcap, "fwd": fwd, "bench": bench})

    rows = []
    for cap in CAP_GRID:
        ex_a, ex_b, overlap = [], [], []
        for p in pools:
            if p is None:
                continue
            sel_a = _pick_score_order(p["score"], sector_map, cap, TOPN)
            sel_b = _pick_mktcap_tiebreak(p["score"], p["mktcap"], sector_map, cap, TOPN)
            ra = p["fwd"].reindex(sel_a).dropna()
            rb = p["fwd"].reindex(sel_b).dropna()
            if len(ra) == 0 or len(rb) == 0:
                continue
            ex_a.append(float(ra.mean()) - p["bench"])
            ex_b.append(float(rb.mean()) - p["bench"])
            overlap.append(len(set(sel_a) & set(sel_b)) / TOPN)
        if len(ex_a) < 8:
            continue
        a, b = np.array(ex_a), np.array(ex_b)
        diff = b - a
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
        t = float(diff.mean() / se) if se else None
        row = {"sector_cap": cap, "n_events": len(a),
              "score_order_mean_excess_pct": round(100 * float(a.mean()), 3),
              "score_order_win_rate_pct": round(100 * float((a > 0).mean()), 1),
              "mktcap_tiebreak_mean_excess_pct": round(100 * float(b.mean()), 3),
              "mktcap_tiebreak_win_rate_pct": round(100 * float((b > 0).mean()), 1),
              "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
              "paired_t_stat": round(t, 3) if t is not None else None,
              "avg_overlap_frac": round(float(np.mean(overlap)), 3)}
        rows.append(row)
        _log(f"cap={cap}: 점수순 {row['score_order_mean_excess_pct']:+.2f}%p(승률"
            f"{row['score_order_win_rate_pct']}%) vs 시총타이브레이크 "
            f"{row['mktcap_tiebreak_mean_excess_pct']:+.2f}%p(승률"
            f"{row['mktcap_tiebreak_win_rate_pct']}%) diff={row['paired_diff_mean_pct']}%p "
            f"t={row['paired_t_stat']} 겹침률={row['avg_overlap_frac']} n={row['n_events']}")

    payload = {"n_snaps": len(snaps), "topn": TOPN, "floor": FLOOR, "cap_grid": CAP_GRID, "rows": rows,
              "method": ("floor=3.25+100일선필터 통과 풀에서 섹터캡을 적용하되, 섹터 구성 우선순위"
                        "(어느 섹터가 먼저 자리를 채우는가)는 점수순으로 동일하게 두고, 캡 때문에 "
                        "한 섹터 안에서 실제로 어떤 종목이 뽑히는지만 점수순 vs 시총순으로 바꿔 비교. "
                        "sector_map은 현재 시점 위키피디아 GICS(point-in-time 아님).")}
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
