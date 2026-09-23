#!/usr/bin/env python3
"""
us_topn_by_mktcap.py — 지호 님 질문(2026-09-23): "최종 선별된건 팩터랭킹이 아니라
시총 순위로 나열하는 거 어때? 수익률에 영향 없나?"

방법: 현행 라이브와 동일한 후보풀(floor=3.25 + 100일선 필터 통과, export_data.py 재현)에서
"최종 topN"을 뽑는 두 가지 방식을 페어드 비교한다:
  a) 기준(현행): 종합점수(가중합성) 내림차순 topN
  b) 시총 내림차순 topN — 같은 풀 안에서 순서만 다르게
시총은 fundamentals_edgar.factor_values 내부와 동일한 근사(EPS>0인 분기의 순이익/EPS로
발행주식수 역산 × 가격 — 실시간 marketCap 대신 PIT 재현 가능한 유일한 값, 라이브
rd_mktcap·shareholder_yield 계산에 이미 쓰이는 것과 동일한 근사).

topN=10(현재 라이브 REPORT_POOL/US_MAX_HOLD과 동일). 시총 데이터가 topN보다 적게 남는
스냅샷은 페어드 비교에서 제외(두 변형 다 같은 스냅샷 집합으로 비교).

실행: python -m research.us.us_topn_by_mktcap [--years 10]
결과: output/us_topn_by_mktcap.json
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
import fundamentals_edgar as F
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features

TOPN = 10
FLOOR = 3.25
OUT_PATH = "output/us_topn_by_mktcap.json"


def _log(m): print(f"[US시총정렬]  {m}", file=sys.stderr)


def _mktcap_for(fd: dict, date_iso: str, price: float) -> float:
    """fundamentals_edgar.factor_values와 동일한 근사(순이익/EPS로 발행주식수 역산)."""
    rec = fd or {}
    eps = F.asof(rec.get("eps"), date_iso)
    ni = F.asof(rec.get("ni"), date_iso)
    if ni is None or eps in (None, 0) or not price or price <= 0:
        return np.nan
    shares = ni / eps
    if shares is None or shares <= 0:
        return np.nan
    return price * shares


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    ex_score, ex_mktcap, overlap, ic_vals = [], [], [], []
    n_skip_thin = 0
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score >= FLOOR]
        if len(score) < TOPN:
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        above = mom["ma100_gap"] > 0 if "ma100_gap" in mom.columns else pd.Series(True, index=score.index)
        score = score[above.reindex(score.index).fillna(False)]
        if len(score) < TOPN:
            continue

        price_row = panel.loc[pd.Timestamp(snap["date"])]
        mktcap = pd.Series({s: _mktcap_for(funds.get(s), snap["date"], float(price_row.get(s, np.nan)))
                            for s in score.index})
        mktcap = mktcap.dropna()
        if len(mktcap) < TOPN:
            n_skip_thin += 1
            continue

        sel_score = score.sort_values(ascending=False).index[:TOPN]
        sel_mktcap = mktcap.sort_values(ascending=False).index[:TOPN]

        r1 = fwd.reindex(sel_score).dropna()
        r2 = fwd.reindex(sel_mktcap).dropna()
        if len(r1) == 0 or len(r2) == 0:
            continue
        ex_score.append(float(r1.mean()) - bench)
        ex_mktcap.append(float(r2.mean()) - bench)
        overlap.append(len(set(sel_score) & set(sel_mktcap)) / TOPN)

        common = mktcap.index
        if len(common) >= 10:
            corr = mktcap.reindex(common).rank().corr(fwd.reindex(common), method="spearman")
            if pd.notna(corr):
                ic_vals.append(float(corr))

    a = np.array(ex_score)
    b = np.array(ex_mktcap)
    diff = b - a
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None

    ic_arr = np.array(ic_vals)
    ic_se = ic_arr.std(ddof=1) / np.sqrt(len(ic_arr)) if len(ic_arr) > 1 else None
    ic_t = float(ic_arr.mean() / ic_se) if ic_se else None

    payload = {
        "n_events": len(a), "n_skipped_thin_mktcap": n_skip_thin, "topn": TOPN, "floor": FLOOR,
        "baseline_score_mean_excess_pct": round(100 * float(a.mean()), 3) if len(a) else None,
        "baseline_score_win_rate_pct": round(100 * float((a > 0).mean()), 1) if len(a) else None,
        "mktcap_order_mean_excess_pct": round(100 * float(b.mean()), 3) if len(b) else None,
        "mktcap_order_win_rate_pct": round(100 * float((b > 0).mean()), 1) if len(b) else None,
        "paired_diff_mean_pct": round(100 * float(diff.mean()), 3) if len(diff) else None,
        "paired_t_stat": round(t, 3) if t is not None else None,
        "avg_overlap_frac": round(float(np.mean(overlap)), 3) if overlap else None,
        "mktcap_rank_ic_spearman_mean": round(float(ic_arr.mean()), 4) if len(ic_arr) else None,
        "mktcap_rank_ic_t_stat": round(ic_t, 3) if ic_t is not None else None,
        "mktcap_rank_ic_n_snaps": len(ic_arr),
        "method": ("floor=3.25+100일선필터 통과 풀에서 topN을 종합점수순 vs 시총(EPS 역산 근사)순"
                  "으로 뽑아 페어드 비교. mktcap_rank_ic = 풀 내 시총순위와 개별종목 향후수익의 "
                  "스피어만 상관(양수면 대형주가 유리, 음수면 소형주가 유리 — 부호 자체가 힌트).")}
    _log(f"[결과] n={payload['n_events']} 기준(점수순) {payload['baseline_score_mean_excess_pct']}%p"
        f"(승률{payload['baseline_score_win_rate_pct']}%) vs 시총순 "
        f"{payload['mktcap_order_mean_excess_pct']}%p(승률{payload['mktcap_order_win_rate_pct']}%) "
        f"페어드diff={payload['paired_diff_mean_pct']}%p t={payload['paired_t_stat']} "
        f"겹침률={payload['avg_overlap_frac']} IC={payload['mktcap_rank_ic_spearman_mean']}"
        f"(t={payload['mktcap_rank_ic_t_stat']})")

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
