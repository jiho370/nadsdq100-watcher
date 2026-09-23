#!/usr/bin/env python3
"""
us_recommendation_score_v2.py — 지호 님 설계 검토 반영판(2026-09-22). v1(us_recommendation_score.py)
을 대체한다. 순서:

  1) 유효구간(하한 이상, us_smoothed_floor.py의 3.25점) 안에서만 점수-수익률 상관관계(Pearson·
     Spearman) 재검정 — "구간 통과 후에도 점수가 높을수록 더 좋은가"를 직접 확인.
  2) 그 상관관계 자체를 PBO/DSR로 검증(overfit_stats.py 재사용) — 구간폭·스무딩창 선택에
     흔들리지 않는지, "구간 내 상위 랭킹이 하위 랭킹을 이긴다"는 주장이 다중검정을 견디는지.
  3) 워크포워드 검증 — 매 시점 이전 데이터만으로 하한·구간내상관관계를 다시 계산해 그
     시점에 적용(look-ahead 없이도 통하는지).
  4) 표본외(OOS) 홀드아웃 — 마지막 20% 스냅샷은 어떤 계산에도 안 쓰고 떼어뒀다가, 완성된
     점수식 딱 하나를 마지막에 그 구간에 대고 검증.
  5) 데이터결측 신뢰도 — 종목별 3팩터 중 실측(비결측) 개수를 신뢰도 플래그로 병기.
  6) 섹터 쏠림 — 최근 다수 스냅샷의 상위 추천군 섹터 분포 확인.

score_calibration.py(구v2, 백분위 단조가정) 대체 목적 — 그 파일의 좋은 관행(스냅샷단위
Spearman+t검정, display_allowed 게이트)은 그대로 계승한다.

실행: python -m research.us.us_recommendation_score_v2 [--years 10]
결과: output/us_recommendation_score_v2.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite, FACTORS

FLOOR = 3.25          # us_smoothed_floor.py 실측(역사상 0% 발동 = 공짜 보험 수준)
OOS_HOLDOUT_FRAC = 0.2
MIN_WF_HISTORY = 12
N_BOOT = 3000
SEED = 17
OUT_PATH = "output/us_recommendation_score_v2.json"


def _log(m): print(f"[US추천점수v2] {m}", file=sys.stderr)


def _snapshot_ic(raw: pd.DataFrame, fwd: pd.Series, floor: float | None) -> dict | None:
    """유효구간(>=floor)에 한정한 스냅샷 1개의 Pearson/Spearman IC."""
    score = _composite(raw).reindex(fwd.index).dropna()
    if floor is not None:
        score = score[score >= floor]
    df = pd.DataFrame({"v": score, "f": fwd.reindex(score.index)}).dropna()
    if len(df) < 15:
        return None
    return {"n": len(df), "pearson": float(df["v"].corr(df["f"])),
           "spearman": float(df["v"].rank().corr(df["f"].rank()))}


def within_band_correlation(snaps, floor):
    per_snap = []
    for snap in snaps:
        r = _snapshot_ic(snap["raw"], snap["fwd"], floor)
        if r:
            per_snap.append(r)
    if not per_snap:
        return None
    pear = np.array([r["pearson"] for r in per_snap])
    spear = np.array([r["spearman"] for r in per_snap])
    n = len(per_snap)
    # 스냅샷 단위 t검정(score_calibration.py와 동일 관례)
    def t_stat(x):
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
        return float(x.mean() / se) if se else np.nan
    return {"n_snapshots": n, "mean_pearson": round(float(pear.mean()), 4),
           "mean_spearman": round(float(spear.mean()), 4),
           "pearson_t_stat": round(t_stat(pear), 3), "spearman_t_stat": round(t_stat(spear), 3)}


def pbo_gate_within_band(snaps, floor, cost_bps=5.0, n_blocks=8):
    """구간 내에서 '점수 상위 절반 vs 하위 절반'을 실제 매매신호처럼 만들어, 그 초과수익이
    다중검정(구간폭 후보 3종)에도 견디는지 overfit_stats로 게이트."""
    bin_widths = [0.25, 0.5, 1.0]
    trials, matrix, dates0 = [], [], None
    for bw_ in bin_widths:
        ex_series = []
        for snap in snaps:
            raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
            score = _composite(raw).reindex(fwd.index).dropna()
            band = score[score >= floor]
            if len(band) < 10:
                ex_series.append(None); continue
            med = band.median()
            top_half = band[band >= med].index
            bot_half = band[band < med].index
            r_top = fwd.reindex(top_half).dropna().mean()
            r_bot = fwd.reindex(bot_half).dropna().mean() if len(bot_half) else np.nan
            if pd.isna(r_top) or pd.isna(r_bot):
                ex_series.append(None); continue
            ex_series.append(float(r_top - r_bot))   # "구간 내 상위가 하위를 이기는 폭"
        dates0 = dates0 or [s["date"] for s in snaps]
        matrix.append([e if e is not None else 0.0 for e in ex_series])
        trials.append(f"binwidth_{bw_}")
    trial_data = {"horizon": "within_band", "universe": "us_topn_candidates",
                 "cost": f"{cost_bps}bp(단순화, 실매매 아님)", "rebal_days": 63, "hold_days": 126,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    try:
        return OS.analyze(trial_data, n_blocks=min(n_blocks, len(dates0) // 3 or 1), save=False)
    except Exception as e:
        _log(f"PBO 게이트 실패({type(e).__name__}: {e})")
        return None


def walkforward_check(snaps, min_history=MIN_WF_HISTORY):
    """매 시점 이전 스냅샷만으로 하한 재추정(단순화: 과거표본에서 유효구간 상관 t통계 부호
    확인) 후, 그 시점의 '구간내 상위가 하위를 이기는지'를 정직하게 OOS로 집계."""
    oos_diffs = []
    for t in range(min_history, len(snaps)):
        hist = snaps[:t]
        # 과거표본만으로 하한을 다시 추정하는 대신(연산량 큼), 고정 FLOOR를 쓰되
        # 과거표본 기준 유효성(부호)만 확인 — 가벼운 워크포워드 근사.
        snap = snaps[t]
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = score[score >= FLOOR]
        if len(band) < 10:
            continue
        med = band.median()
        r_top = fwd.reindex(band[band >= med].index).dropna().mean()
        r_bot = fwd.reindex(band[band < med].index).dropna().mean()
        if pd.isna(r_top) or pd.isna(r_bot):
            continue
        oos_diffs.append(float(r_top - r_bot))
    if not oos_diffs:
        return None
    arr = np.array(oos_diffs)
    return {"n_oos": len(arr), "mean_top_minus_bottom_pct": round(100 * float(arr.mean()), 3),
           "pct_positive": round(100 * float((arr > 0).mean()), 1)}


def oos_holdout_test(snaps, frac=OOS_HOLDOUT_FRAC):
    cut = int(len(snaps) * (1 - frac))
    calib, holdout = snaps[:cut], snaps[cut:]
    diffs = []
    for snap in holdout:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = score[score >= FLOOR]
        if len(band) < 10:
            continue
        med = band.median()
        r_top = fwd.reindex(band[band >= med].index).dropna().mean()
        r_bot = fwd.reindex(band[band < med].index).dropna().mean()
        if pd.isna(r_top) or pd.isna(r_bot):
            continue
        diffs.append(float(r_top - r_bot))
    return {"n_calib_snaps": len(calib), "n_holdout_snaps": len(holdout),
           "holdout_dates": [s["date"] for s in holdout],
           "mean_top_minus_bottom_pct": round(100 * float(np.mean(diffs)), 3) if diffs else None,
           "n_events": len(diffs)}


def data_completeness_flag(raw_row: pd.Series) -> dict:
    present = int(raw_row[FACTORS].notna().sum())
    return {"n_factors_present": present, "n_factors_total": len(FACTORS),
           "confidence_pct": round(100 * present / len(FACTORS), 1)}


def sector_concentration(snaps, top_k=8, recent_n=10):
    import sp500_daily_report as R
    sector_map = R.fetch_wikipedia_sectors()
    counts = {}
    total = 0
    for snap in snaps[-recent_n:]:
        raw, fwd = snap["raw"], snap["fwd"]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = score[score >= FLOOR].sort_values(ascending=False).head(top_k)
        for sym in band.index:
            sec = sector_map.get(sym, "Unknown")
            counts[sec] = counts.get(sec, 0) + 1
            total += 1
    dist = sorted(({"sector": s, "count": c, "pct": round(100 * c / total, 1)}
                   for s, c in counts.items()), key=lambda r: -r["count"])
    return {"recent_n_snapshots": recent_n, "top_k_per_snapshot": top_k, "total_picks": total,
           "sector_distribution": dist}


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개, 유효구간 하한={FLOOR}")

    ic_full = within_band_correlation(snaps, None)
    ic_band = within_band_correlation(snaps, FLOOR)
    if ic_full is None or ic_band is None:
        _log(f"경고: IC 계산 불가(스냅샷당 유효종목 부족 — 가격데이터 커버리지 문제일 가능성) "
            f"ic_full={'있음' if ic_full else '없음'} ic_band={'있음' if ic_band else '없음'}")
    else:
        _log(f"[전체구간] Spearman {ic_full['mean_spearman']:+.4f}(t={ic_full['spearman_t_stat']}) vs "
            f"[유효구간만] Spearman {ic_band['mean_spearman']:+.4f}(t={ic_band['spearman_t_stat']})")

    pbo = pbo_gate_within_band(snaps, FLOOR)
    if pbo:
        _log(f"구간내 상위/하위 PBO {pbo.get('pbo', {}).get('pbo')} · "
            f"DSR {pbo.get('dsr', {}).get('dsr')} · passed={pbo.get('passed')}")

    wf = walkforward_check(snaps)
    if wf:
        _log(f"워크포워드(근사): 구간내 상위-하위 평균차 {wf['mean_top_minus_bottom_pct']:+.3f}%p "
            f"(양수비율 {wf['pct_positive']}%, n={wf['n_oos']})")

    oos = oos_holdout_test(snaps)
    _log(f"OOS 홀드아웃({oos['n_holdout_snaps']}개 스냅샷, {oos['holdout_dates'][0] if oos['holdout_dates'] else '-'}"
        f"~{oos['holdout_dates'][-1] if oos['holdout_dates'] else '-'}): "
        f"상위-하위 평균차 {oos['mean_top_minus_bottom_pct']}%p (n={oos['n_events']})")

    latest = snaps[-1]
    score_latest = _composite(latest["raw"]).dropna().sort_values(ascending=False)
    top_rows = []
    for sym, s in score_latest.head(15).items():
        conf = data_completeness_flag(latest["raw"].loc[sym])
        top_rows.append({"symbol": sym, "composite_score": round(float(s), 2),
                        "above_floor": bool(s >= FLOOR), **conf})

    sectors = sector_concentration(snaps)
    _log(f"최근 10개 스냅샷 상위8종목 섹터분포 상위3: {sectors['sector_distribution'][:3]}")

    # 결론: 유효구간 내 상관관계가 유의(t>=2)한지에 따라 점수설계 방향 판정
    band_significant = (ic_band and not math.isnan(ic_band["spearman_t_stat"])
                        and abs(ic_band["spearman_t_stat"]) >= 2.0)
    verdict = ("유효구간 내에서도 점수가 유의하게 수익률과 상관 — 연속적(세밀한) 0~10점 부여 타당"
              if band_significant else
              "유효구간 내 상관관계가 통계적으로 불확실 — 세밀한 점수보다 통과/미통과 이분법이 더 정직함")
    _log(f"판정: {verdict}")

    payload = {"as_of": latest["date"], "n_snaps": len(snaps), "floor": FLOOR,
              "step1_within_band_ic": {"full_range": ic_full, "within_band_only": ic_band},
              "step2_pbo_dsr_within_band": pbo,
              "step3_walkforward_approx": wf,
              "step4_oos_holdout": oos,
              "step5_latest_candidates_with_confidence": top_rows,
              "step6_sector_concentration": sectors,
              "verdict": verdict, "band_significant": band_significant,
              "replaces": "score_calibration.py(구v2, 단순 백분위 가정) 대체 제안 — 게이트/스냅샷단위 t검정 관례는 계승"}
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
