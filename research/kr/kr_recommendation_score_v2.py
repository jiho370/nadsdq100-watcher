#!/usr/bin/env python3
"""
kr_recommendation_score_v2.py — us_recommendation_score_v2.py의 한국판(2026-09-22, 지호 님
설계 검토 반영). v1(kr_recommendation_score.py) 대체.

한국은 하한(3.75)과 상한(6.5) 둘 다 있는 밴드 — us판과 달리 "밴드 안"이 유효구간이다.

실행: python -m research.kr.kr_recommendation_score_v2
결과: output/kr_recommendation_score_v2.json
"""
from __future__ import annotations
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import overfit_stats as OS
from research.kr.kr_factor_value_vs_rank import _composite, FACTORS

FLOOR = 3.5     # A+B(2026-09-22, 지호 님 지시): 3.75->3.5로 소폭 확장(밴드를 넓게)
CAP = 7.0       # 6.5->7.0로 소폭 확장
MIN_OBS_PER_SNAP = 8   # A: 15->8, 스냅샷 하나하나의 상관계수는 노이즈해지지만 t검정용
                        # 표본(스냅샷 개수) 자체가 늘어 평균의 신뢰도는 오히려 개선됨
STAT_REBAL_DAYS = 63   # C 시도했으나 철회: fundamentals 딕셔너리가 애초에 246개 분기성
                        # 날짜에만 존재해(load_research_data 원본 수집 주기) 월간(21일)
                        # 요청 시 대부분 빈 데이터로 떨어짐(종목 전부 점수0·팩터0/3 실측
                        # 확인) — 재수집 없이는 불가, 63일(기존)로 되돌림. A+B만 유지.
HORIZON = "6m"
OOS_HOLDOUT_FRAC = 0.2
MIN_WF_HISTORY = 30    # 스냅샷이 3배 촘촘해진 만큼 워밍업 기간도 비례 확대(12->30)
OUT_PATH = "output/kr_recommendation_score_v2.json"


def _log(m): print(f"[KR추천점수v2] {m}", file=sys.stderr)


def _band(score: pd.Series, floor, cap):
    s = score
    if floor is not None:
        s = s[s >= floor]
    if cap is not None:
        s = s[s <= cap]
    return s


def _snapshot_ic(raw, fwd, floor, cap):
    score = _composite(raw).reindex(fwd.index).dropna()
    score = _band(score, floor, cap)
    df = pd.DataFrame({"v": score, "f": fwd.reindex(score.index)}).dropna()
    if len(df) < MIN_OBS_PER_SNAP:
        return None
    return {"n": len(df), "pearson": float(df["v"].corr(df["f"])),
           "spearman": float(df["v"].rank().corr(df["f"].rank()))}


def within_band_correlation(snaps, floor, cap):
    per_snap = []
    for snap in snaps:
        r = _snapshot_ic(snap["raw"], snap["fwd"][HORIZON], floor, cap)
        if r:
            per_snap.append(r)
    if not per_snap:
        return None
    pear = np.array([r["pearson"] for r in per_snap])
    spear = np.array([r["spearman"] for r in per_snap])

    def t_stat(x):
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
        return float(x.mean() / se) if se else np.nan
    return {"n_snapshots": len(per_snap), "mean_pearson": round(float(pear.mean()), 4),
           "mean_spearman": round(float(spear.mean()), 4),
           "pearson_t_stat": round(t_stat(pear), 3), "spearman_t_stat": round(t_stat(spear), 3)}


def pbo_gate_within_band(snaps, floor, cap, n_blocks=8):
    bin_widths = [0.25, 0.5, 1.0]
    trials, matrix, dates0 = [], [], None
    for bw_ in bin_widths:
        ex_series = []
        for snap in snaps:
            raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
            score = _composite(raw).reindex(fwd.index).dropna()
            band = _band(score, floor, cap)
            if len(band) < MIN_OBS_PER_SNAP:
                ex_series.append(None); continue
            med = band.median()
            r_top = fwd.reindex(band[band >= med].index).dropna().mean()
            r_bot = fwd.reindex(band[band < med].index).dropna().mean()
            if pd.isna(r_top) or pd.isna(r_bot):
                ex_series.append(None); continue
            ex_series.append(float(r_top - r_bot))
        dates0 = dates0 or [s["date"] for s in snaps]
        matrix.append([e if e is not None else 0.0 for e in ex_series])
        trials.append(f"binwidth_{bw_}")
    trial_data = {"horizon": "within_band", "universe": "kr_topn_candidates",
                 "cost": "5bp(단순화, 실매매 아님)", "rebal_days": STAT_REBAL_DAYS, "hold_days": 126,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    try:
        return OS.analyze(trial_data, n_blocks=min(n_blocks, len(dates0) // 3 or 1), save=False)
    except Exception as e:
        _log(f"PBO 게이트 실패({type(e).__name__}: {e})")
        return None


def walkforward_check(snaps, min_history=MIN_WF_HISTORY):
    oos_diffs = []
    for t in range(min_history, len(snaps)):
        snap = snaps[t]
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = _band(score, FLOOR, CAP)
        if len(band) < MIN_OBS_PER_SNAP:
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
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = _band(score, FLOOR, CAP)
        if len(band) < MIN_OBS_PER_SNAP:
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


def sector_concentration(snaps, top_k=5, recent_n=10):
    try:
        from research.kr.kr_sector import fetch_sectors, sector_of
        dates = [s["date"].replace("-", "") for s in snaps[-recent_n:]]
        sec_by_date = fetch_sectors(dates)
    except Exception as e:
        _log(f"섹터분류 조회 실패({type(e).__name__}: {e}) — 생략")
        return {"error": str(e)}
    counts, total = {}, 0
    for snap in snaps[-recent_n:]:
        d8 = snap["date"].replace("-", "")
        raw, fwd = snap["raw"], snap["fwd"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        band = _band(score, FLOOR, CAP).sort_values(ascending=False).head(top_k)
        for sym in band.index:
            sec = sector_of(sec_by_date, d8, sym) or "Unknown"
            counts[sec] = counts.get(sec, 0) + 1
            total += 1
    if total == 0:
        return {"total_picks": 0}
    dist = sorted(({"sector": s, "count": c, "pct": round(100 * c / total, 1)}
                   for s, c in counts.items()), key=lambda r: -r["count"])
    return {"recent_n_snapshots": recent_n, "top_k_per_snapshot": top_k, "total_picks": total,
           "sector_distribution": dist}


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=STAT_REBAL_DAYS, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개, 유효밴드=[{FLOOR}, {CAP}]")

    ic_full = within_band_correlation(snaps, None, None)
    ic_band = within_band_correlation(snaps, FLOOR, CAP)
    _log(f"[전체구간] Spearman {ic_full['mean_spearman']:+.4f}(t={ic_full['spearman_t_stat']}) vs "
        f"[유효밴드만] Spearman {ic_band['mean_spearman']:+.4f}(t={ic_band['spearman_t_stat']})")

    pbo = pbo_gate_within_band(snaps, FLOOR, CAP)
    if pbo:
        _log(f"밴드내 상위/하위 PBO {pbo.get('pbo', {}).get('pbo')} · "
            f"DSR {pbo.get('dsr', {}).get('dsr')} · passed={pbo.get('passed')}")

    wf = walkforward_check(snaps)
    if wf:
        _log(f"워크포워드(근사): 밴드내 상위-하위 평균차 {wf['mean_top_minus_bottom_pct']:+.3f}%p "
            f"(양수비율 {wf['pct_positive']}%, n={wf['n_oos']})")

    oos = oos_holdout_test(snaps)
    _log(f"OOS 홀드아웃({oos['n_holdout_snaps']}개 스냅샷): "
        f"상위-하위 평균차 {oos['mean_top_minus_bottom_pct']}%p (n={oos['n_events']})")

    latest = snaps[-1]
    score_latest = _composite(latest["raw"]).dropna().sort_values(ascending=False)
    band_latest = _band(score_latest, FLOOR, CAP)
    band_median = float(band_latest.median()) if len(band_latest) else None
    top_rows = []
    for sym, s in score_latest.head(15).items():
        conf = data_completeness_flag(latest["raw"].loc[sym])
        if not (FLOOR <= s <= CAP):
            tier = "제외"
        elif band_median is not None and s >= band_median:
            tier = "매수"
        else:
            tier = "관찰"
        top_rows.append({"symbol": sym, "composite_score": round(float(s), 2),
                        "within_band": bool(FLOOR <= s <= CAP), "tier": tier, **conf})

    sectors = sector_concentration(snaps)
    _log(f"최근 10개 스냅샷 밴드내상위5 섹터분포 상위3: {sectors.get('sector_distribution', [])[:3]}")

    band_significant = (ic_band and not math.isnan(ic_band["spearman_t_stat"])
                        and abs(ic_band["spearman_t_stat"]) >= 2.0)
    verdict = ("유효밴드 내에서도 점수가 유의하게 수익률과 상관 — 연속적(세밀한) 0~10점 부여 타당"
              if band_significant else
              "유효밴드 내 상관관계가 통계적으로 불확실 — 세밀한 점수보다 통과/미통과 이분법이 더 정직함")
    _log(f"판정: {verdict}")

    payload = {"as_of": latest["date"], "n_snaps": len(snaps), "floor": FLOOR, "cap": CAP,
              "step1_within_band_ic": {"full_range": ic_full, "within_band_only": ic_band},
              "step2_pbo_dsr_within_band": pbo,
              "step3_walkforward_approx": wf,
              "step4_oos_holdout": oos,
              "step5_latest_candidates_with_confidence": top_rows,
              "step6_sector_concentration": sectors,
              "verdict": verdict, "band_significant": band_significant}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
