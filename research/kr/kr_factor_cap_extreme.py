#!/usr/bin/env python3
"""
kr_factor_cap_extreme.py — us_factor_cap_extreme.py의 한국판(2026-09-22). 한국은 점수
6점 이상에서 초과수익이 역전되는 패턴이 나왔으므로(kr_composite_score_vs_index.py), 이
극단 구간을 실제로 topN 선정에서 제외하면 라이브 성과가 개선되는지 확인한다.
지호 님 후속 질문: "캡 x topn 5~7 스윕해서 비교" — topN 자체도 5/6/7로 바꿔가며 같은
캡 그리드를 반복.

방법: backtest_kr.build_kr_snaps 재사용, 종합점수(valuediv 동일가중)로 정렬 →
(a) 그냥 topN(현행 5) vs (b) 점수 상한 C 이상 제외 후 topN, 를 스냅샷(6개월 forward)
이벤트평균으로 비교.

실행: python -m research.kr.kr_factor_cap_extreme  (topn 5/6/7 전부 스윕)
결과: output/kr_factor_cap_extreme.json

추가(2026-09-23, 지호 님 질문 — "상한캡 건 상태에서 top방식이 수익률이 더 좋았는지,
선별종목 수익률이 여러 변수와 어떤 상관인지, 변수 결합하면 어떤지"): 최적 캡(=6, 아래
결과 참고)을 고정한 뒤 그 밴드 내에서 (a) top5(밴드 내 최고점) vs 밴드 전체평균 vs
bottom5(밴드 내 최저점) 비교, (b) 밴드 내에서 개별 변수(value·pbr_inv·div_yield·roe·
mom6·mom12_1)별 순위상관(IC)+t검정, (c) 팩터+모멘텀 결합점수의 밴드 내 IC. band_analysis()
가 이를 output/kr_factor_cap_extreme.json의 "band_analysis" 키에 추가로 저장한다."""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from research.kr.kr_factor_value_vs_rank import _composite

CAP_GRID = [None, 8, 7, 6, 5, 4, 3, 2]
TOPN_GRID = [5, 6, 7]
HORIZON = "6m"
OUT_PATH = "output/kr_factor_cap_extreme.json"


def _log(m): print(f"[KR극단제외] {m}", file=sys.stderr)


def _one_topn(snaps, topn: int, cap_grid=CAP_GRID) -> list[dict]:
    rows = []
    for cap in cap_grid:
        ev, ex, sels = [], [], []
        for snap in snaps:
            raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
            score = _composite(raw).reindex(fwd.index).dropna()
            pool = score[score <= cap] if cap is not None else score
            if len(pool) < topn:
                continue
            top = pool.sort_values(ascending=False).index[:topn]
            r = fwd.reindex(top).dropna()
            if len(r) == 0:
                continue
            ev.append(float(r.mean())); ex.append(float(r.mean()) - bnc)
            sels.append(set(top))
        if not ev:
            continue
        ev_a, ex_a = np.array(ev), np.array(ex)
        turns = [1 - len(sels[j] & sels[j - 1]) / max(len(sels[j]), 1) for j in range(1, len(sels))]
        rows.append({"cap": cap, "n_events": len(ev),
                    "mean_fwd_ret_pct": round(100 * float(ev_a.mean()), 3),
                    "mean_excess_pct": round(100 * float(ex_a.mean()), 3),
                    "excess_sharpe": round(float(ex_a.mean() / ex_a.std()) * (252 / 126) ** 0.5, 3) if ex_a.std() else None,
                    "win_rate_pct": round(100 * float((ex_a > 0).mean()), 1),
                    "turnover_pct": round(100 * float(np.mean(turns)), 1) if turns else None})
        _log(f"cap={cap}: 초과 {rows[-1]['mean_excess_pct']:+.2f}%p 승률 {rows[-1]['win_rate_pct']}% "
            f"n={rows[-1]['n_events']}")
    return rows


BAND_CAP = 6.0     # kr_factor_cap_extreme.json 실측 최적치(topn5 기준 초과수익 최대)
BAND_TOPN = 5
BAND_SIGNALS = ["value", "pbr_inv", "div_yield", "roe", "mom6", "mom12_1"]


def _band_selection_check(snaps, cap=BAND_CAP, topn=BAND_TOPN) -> dict:
    """상한캡(cap)을 건 밴드 안에서 top(밴드 내 최고점) vs 밴드전체평균 vs
    bottom(밴드 내 최저점) 선택 방식을 비교 — "캡을 걸었으면 그 안에서도 top방식이
    맞는지"에 대한 답."""
    top_ex, whole_ex, bot_ex = [], [], []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = score[score <= cap]
        if len(band) < topn * 2:
            continue
        ranked = band.sort_values(ascending=False)
        top = ranked.index[:topn]
        bot = ranked.index[-topn:]
        r_top = fwd.reindex(top).dropna()
        r_bot = fwd.reindex(bot).dropna()
        r_whole = fwd.reindex(band.index).dropna()
        if len(r_top) == 0 or len(r_bot) == 0 or len(r_whole) == 0:
            continue
        top_ex.append(float(r_top.mean()) - bnc)
        bot_ex.append(float(r_bot.mean()) - bnc)
        whole_ex.append(float(r_whole.mean()) - bnc)

    if len(top_ex) < 8:
        return {"note": "표본 부족"}
    t, w, b = np.array(top_ex), np.array(whole_ex), np.array(bot_ex)
    diff_tw = t - w
    se = diff_tw.std(ddof=1) / np.sqrt(len(diff_tw)) if len(diff_tw) > 1 else None
    t_stat = float(diff_tw.mean() / se) if se else None
    return {"n_events": len(t), "cap": cap, "topn": topn,
           "top_of_band_mean_excess_pct": round(100 * float(t.mean()), 3),
           "top_of_band_win_rate_pct": round(100 * float((t > 0).mean()), 1),
           "whole_band_mean_excess_pct": round(100 * float(w.mean()), 3),
           "whole_band_win_rate_pct": round(100 * float((w > 0).mean()), 1),
           "bottom_of_band_mean_excess_pct": round(100 * float(b.mean()), 3),
           "bottom_of_band_win_rate_pct": round(100 * float((b > 0).mean()), 1),
           "paired_t_top_vs_whole": round(t_stat, 3) if t_stat is not None else None}


def _band_ic(snaps, panel, cap=BAND_CAP) -> dict:
    """캡 이하(밴드 내) 후보만으로 개별 변수별 순위상관(IC) — 팩터·모멘텀 각각과
    결합점수(팩터z+mom6z)까지. kr_momentum_overlay._extra_mom_features 재사용."""
    from research.kr.kr_momentum_overlay import _extra_mom_features

    per_snap_corr = {c: [] for c in BAND_SIGNALS + ["combo_factor_plus_mom6"]}
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        band_idx = score[score <= cap].index
        if len(band_idx) < 10:
            continue
        extra = _extra_mom_features(panel, snap["date"], band_idx)
        vals = pd.concat([raw.reindex(band_idx)[["value", "pbr_inv", "div_yield", "roe",
                                                  "mom6", "mom12_1"]], extra[["ma100_gap"]]],
                         axis=1) if "ma100_gap" in extra.columns else raw.reindex(band_idx)
        r = (fwd.reindex(band_idx) - bnc)

        sd = score.reindex(band_idx).std(ddof=0)
        fz = ((score.reindex(band_idx) - score.reindex(band_idx).mean()) / sd
              if sd else score.reindex(band_idx) * 0.0)
        mret = raw.reindex(band_idx)["mom6"]
        msd = mret.std(ddof=0)
        mz = (((mret - mret.mean()) / msd).fillna(0.0) if msd else mret * 0.0)
        combo = fz + mz

        for col in BAND_SIGNALS:
            if col not in vals.columns:
                continue
            pair = pd.concat([vals[col], r], axis=1, keys=["v", "ex"]).dropna()
            if len(pair) >= 5:
                c = pair["v"].corr(pair["ex"], method="spearman")
                if pd.notna(c):
                    per_snap_corr[col].append(float(c))
        pair = pd.concat([combo, r], axis=1, keys=["v", "ex"]).dropna()
        if len(pair) >= 5:
            c = pair["v"].corr(pair["ex"], method="spearman")
            if pd.notna(c):
                per_snap_corr["combo_factor_plus_mom6"].append(float(c))

    out = {}
    for col, vals_ in per_snap_corr.items():
        if len(vals_) < 5:
            continue
        a = np.array(vals_)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None
        t = float(a.mean() / se) if se else None
        out[col] = {"mean_spearman": round(float(a.mean()), 4),
                    "t_stat": round(t, 3) if t is not None else None, "n_snaps": len(a)}
    return out


def band_analysis(snaps, panel, cap=BAND_CAP, topn=BAND_TOPN) -> dict:
    sel = _band_selection_check(snaps, cap, topn)
    ic = _band_ic(snaps, panel, cap)
    _log(f"[밴드분석 cap={cap}] top {sel.get('top_of_band_mean_excess_pct')}%p 대 "
        f"전체 {sel.get('whole_band_mean_excess_pct')}%p 대 bottom "
        f"{sel.get('bottom_of_band_mean_excess_pct')}%p (t={sel.get('paired_t_top_vs_whole')})")
    for col, row in ic.items():
        _log(f"[밴드IC] {col}: spearman {row['mean_spearman']:+.4f}(t={row['t_stat']}, n={row['n_snaps']})")
    return {"cap": cap, "topn": topn, "selection_method_check": sel, "ic": ic}


def run(topn_grid=TOPN_GRID, save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개 · topn 스윕 {topn_grid}")

    by_topn = {}
    for topn in topn_grid:
        _log(f"--- topn={topn} ---")
        rows = _one_topn(snaps, topn)
        baseline = next((r for r in rows if r["cap"] is None), None)
        by_topn[str(topn)] = {"baseline_cap_none": baseline, "rows": rows}

    band = band_analysis(snaps, panel)

    payload = {"n_snaps": len(snaps), "topn_grid": topn_grid, "band_analysis": band,
              "method": ("매 스냅샷 종합점수(valuediv 동일가중) 상위 topn(cap 이하 종목만 "
                        "후보) 선정 후 6개월 forward return 이벤트평균. 전체 NAV "
                        "재시뮬레이션 아님(가벼운 이벤트 프레임). topN 자체도 5/6/7로 "
                        "바꿔가며 캡 그리드를 반복."),
              "by_topn": by_topn}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    run()


if __name__ == "__main__":
    main()
