#!/usr/bin/env python3
"""
us_momentum_overlay.py — 지호 님 요청(2026-09-22): "선별된거 중에서 모멘텀이나 몇일선,
또는 6개월/3개월/1개월 수익률로 또 선별해서 추천강도 조정하는 방식"을 백테스트로 검증.

이전 pbo_gate_within_band()의 "밴드 내 위그룹 vs 아래그룹" 트라이얼 구성은 같은 팩터
점수를 반으로 가른 것이라 순환논리(위그룹이 이기는 게 당연)였다는 문제 제기(지호 님)에
대한 대안이기도 하다 — 여기서는 팩터 점수로 이미 뽑힌 topn8 후보군에, 팩터와는 독립인
모멘텀 신호(이동평균 이격도·후행수익률)를 실제로 다르게 설계될 수 있었던 여러 방식으로
얹어보고, 그 결과들을 PBO/DSR 트라이얼로 쓴다 — 팩터 자체를 반으로 가르는 게 아니므로
순환논리 문제가 없다.

방법:
  1) us_factor_formula_pit_sweep.build_snaps() 그대로 재사용(팩터 계산 로직 안 건드림).
  2) 같은 PIT 패널에서 스냅샷 날짜의 가격 인덱스를 찾아 모멘텀 신호를 별도 계산
     (20/50/100/150/200일선 이격도, 1/3/6개월 후행수익률).
  3) 팩터 상위 topn8(라이브 조건)을 기준 후보군으로 고정하고 그 안에서:
     a) MA필터: N일선 아래 종목 제외(5가지)
     b) 수익률틸트: 후보군을 그 후행수익률로 다시 상위/하위 절반만 남김(추세추종 3가지 +
        역추세 3가지 — 1차 결과에서 단기수익률일수록 효과가 갈릴 수 있어 반대 방향도 확인)
     c) 결합점수: 팩터 z + 6개월수익률 z 합산으로 전체 유니버스에서 재선정
     d) 복합: ma100일 필터 통과 종목만 남긴 뒤 그 안에서 다시 수익률틸트(지호 님 질문 —
        "100일이랑 n개월 수익률 같이 하는건?")
  4) 기준(팩터 단독 topn8) 대비 이벤트(스냅샷) 평균 초과수익 개선폭 + 페어드 t검정.
  5) 이벤트 수가 같은 시행들을 묶어 overfit_stats.OS.analyze()로 PBO/DSR 판정.
  6) 모멘텀 신호 자체와 개별종목 forward 초과수익의 스냅샷별 순위상관(IC)+t검정, 그리고
     신호 5분위 구간별 평균 초과수익 테이블(지호 님 요청 — "그냥 +-로 하지말고 상관관계
     분석해서 몇 이상이면 유리한지") — topn8 후보군 내에서만 계산(전체 유니버스 아님).

실행: python -m research.us.us_momentum_overlay [--years 10]
결과: output/us_momentum_overlay.json
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
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite

TOPN = 8
TOPN_WIDE = 15
MA_GRID = [20, 50, 100, 150, 200]
RET_WINDOWS = {"1m": 21, "3m": 63, "6m": 126}
OUT_PATH = "output/us_momentum_overlay.json"


def _log(m): print(f"[US모멘텀오버레이] {m}", file=sys.stderr)


def _mom_features(panel: pd.DataFrame, date_iso: str, tickers) -> pd.DataFrame:
    """스냅샷 날짜 기준 이동평균 이격도·후행수익률 — build_snaps는 안 건드리고 같은
    panel에서 직접 재계산(빠르고 팩터 로직과 독립)."""
    p = panel.index.get_indexer([pd.Timestamp(date_iso)], method="pad")[0]
    px = panel.iloc[: p + 1]
    cur = px.iloc[-1]
    out = {}
    for n in MA_GRID:
        if len(px) > n:
            ma = px.iloc[-n:].mean()
            out[f"ma{n}_gap"] = cur / ma - 1
    for label, d in RET_WINDOWS.items():
        if len(px) > d:
            out[f"ret_{label}"] = cur / px.iloc[-d - 1] - 1
    return pd.DataFrame(out).reindex(tickers)


def _make_variants():
    def base_sel(score, mom):
        return score.sort_values(ascending=False).index[:TOPN]

    def make_ma_filter(n):
        def sel(score, mom):
            top = score.sort_values(ascending=False).index[:TOPN]
            col = f"ma{n}_gap"
            if col not in mom.columns:
                return top
            keep = [t for t in top if pd.notna(mom.loc[t, col]) and mom.loc[t, col] > 0]
            return keep if len(keep) >= 3 else top
        return sel

    def make_ret_tilt(label, reverse=False):
        def sel(score, mom):
            top = score.sort_values(ascending=False).index[:TOPN]
            col = f"ret_{label}"
            if col not in mom.columns:
                return top
            m = mom.loc[top, col].dropna()
            if len(m) < 4:
                return top
            half = max(len(m) // 2, 3)
            return m.sort_values(ascending=reverse).index[:half]
        return sel

    def make_ma_then_tilt(n, label):
        ma_sel = make_ma_filter(n)
        def sel(score, mom):
            survivors = ma_sel(score, mom)
            col = f"ret_{label}"
            if col not in mom.columns or len(survivors) < 4:
                return survivors
            m = mom.loc[list(survivors), col].dropna()
            if len(m) < 4:
                return survivors
            half = max(len(m) // 2, 3)
            return m.sort_values(ascending=False).index[:half]
        return sel

    def combo_sel(score, mom):
        sd = score.std(ddof=0)
        fz = (score - score.mean()) / sd if sd else score * 0.0
        mret = mom["ret_6m"] if "ret_6m" in mom.columns else pd.Series(dtype=float)
        msd = mret.std(ddof=0) if len(mret) else 0.0
        mz = (((mret - mret.mean()) / msd).reindex(score.index).fillna(0.0)
              if msd else pd.Series(0.0, index=score.index))
        combo = fz + mz
        return combo.sort_values(ascending=False).index[:TOPN]

    variants = {"baseline_factor_only": base_sel}
    for n in MA_GRID:
        variants[f"ma{n}_filter"] = make_ma_filter(n)
    for label in RET_WINDOWS:
        variants[f"tilt_top_half_{label}"] = make_ret_tilt(label)
        variants[f"tilt_bottom_half_{label}(역추세)"] = make_ret_tilt(label, reverse=True)
    variants["combo_factor_plus_mom6m"] = combo_sel
    for label in RET_WINDOWS:
        variants[f"ma100_then_top_half_{label}"] = make_ma_then_tilt(100, label)
    return variants


SIGNAL_COLS = [f"ma{n}_gap" for n in MA_GRID] + [f"ret_{l}" for l in RET_WINDOWS]


def _momentum_ic_and_buckets(snaps, panel) -> dict:
    """topn8 후보군 내에서 모멘텀 신호별로: (a) 스냅샷별 개별종목 순위상관(신호 vs forward
    초과수익) 평균+t검정, (b) 전체 스냅샷 풀링한 5분위 구간별 평균 초과수익 테이블.
    지호 님 요청 — "그냥 +-로 하지말고 상관관계 분석해서 몇 이상이면 유리한지"."""
    per_snap_corr = {c: [] for c in SIGNAL_COLS}
    pooled = {c: [] for c in SIGNAL_COLS}
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        mom = _mom_features(panel, snap["date"], top)
        r = fwd.reindex(top) - bench   # 종목별 초과수익
        for col in SIGNAL_COLS:
            pair = pd.concat([mom[col], r], axis=1, keys=["sig", "ex"]).dropna()
            if len(pair) >= 4:
                c = pair["sig"].corr(pair["ex"], method="spearman")
                if pd.notna(c):
                    per_snap_corr[col].append(float(c))
            for sig, ex in pair.itertuples(index=False):
                pooled[col].append((float(sig), float(ex)))

    ic_table = {}
    for col, vals in per_snap_corr.items():
        if len(vals) < 5:
            continue
        a = np.array(vals)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None
        t = float(a.mean() / se) if se else None
        ic_table[col] = {"mean_spearman": round(float(a.mean()), 4),
                         "t_stat": round(t, 3) if t is not None else None, "n_snaps": len(a)}

    bucket_table = {}
    for col, pairs in pooled.items():
        if len(pairs) < 20:
            continue
        df = pd.DataFrame(pairs, columns=["sig", "ex"])
        try:
            df["q"] = pd.qcut(df["sig"], 5, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for q, g in df.groupby("q", observed=True):
            rows.append({"range": f"[{q.left:.3f}, {q.right:.3f}]",
                        "mean_excess_pct": round(100 * float(g["ex"].mean()), 3), "n": len(g)})
        bucket_table[col] = sorted(rows, key=lambda r: r["range"])

    return {"ic": ic_table, "buckets": bucket_table}


def _combo_wide_check(snaps, panel) -> dict:
    """지호 님 요청: 이진 필터 대신 팩터z+모멘텀z 연속결합을 더 넓은 후보군(topn15)에서
    — topn8은 표본이 작아 개별종목 IC로는 신호가 있어도 필터/틸트로는 덜 잡힐 수 있어
    더 넓은 풀에서 연속결합 방식을 별도 확인."""
    base_ex, combo_ex = [], []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN_WIDE:
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        ret6m = mom["ret_6m"] if "ret_6m" in mom.columns else pd.Series(dtype=float)
        base_top = score.sort_values(ascending=False).index[:TOPN_WIDE]
        r_base = fwd.reindex(base_top).dropna()

        sd = score.std(ddof=0)
        fz = (score - score.mean()) / sd if sd else score * 0.0
        msd = ret6m.std(ddof=0) if len(ret6m) else 0.0
        mz = (((ret6m - ret6m.mean()) / msd).reindex(score.index).fillna(0.0)
              if msd else pd.Series(0.0, index=score.index))
        combo = (fz + mz).sort_values(ascending=False).index[:TOPN_WIDE]
        r_combo = fwd.reindex(combo).dropna()

        if len(r_base) == 0 or len(r_combo) == 0:
            continue
        base_ex.append(float(r_base.mean()) - bench)
        combo_ex.append(float(r_combo.mean()) - bench)

    if len(base_ex) < 8:
        return {"note": "표본 부족"}
    b, c = np.array(base_ex), np.array(combo_ex)
    diff = c - b
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None
    return {"n_events": len(b), "topn": TOPN_WIDE,
           "baseline_factor_only_mean_excess_pct": round(100 * float(b.mean()), 3),
           "baseline_win_rate_pct": round(100 * float((b > 0).mean()), 1),
           "combo_factor_plus_mom6m_mean_excess_pct": round(100 * float(c.mean()), 3),
           "combo_win_rate_pct": round(100 * float((c > 0).mean()), 1),
           "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
           "paired_t_stat": round(t, 3) if t is not None else None}


def _event_stats(sel_fn, snaps, panel) -> list:
    ex = []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            ex.append(None)
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        sel = sel_fn(score, mom)
        r = fwd.reindex(sel).dropna() if sel is not None and len(sel) else pd.Series(dtype=float)
        ex.append(float(r.mean()) - bench if len(r) else None)
    return ex


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    variants = _make_variants()
    trial_labels, trial_returns, summary = [], [], []
    baseline_ex = None
    for label, fn in variants.items():
        ex_raw = _event_stats(fn, snaps, panel)
        ex = [x for x in ex_raw if x is not None]
        if len(ex) < 8:
            _log(f"{label}: 이벤트 부족({len(ex)}) — 스킵")
            continue
        ex_a = np.array(ex)
        if label == "baseline_factor_only":
            baseline_ex = ex_a
        row = {"label": label, "n_events": len(ex),
              "mean_excess_pct": round(100 * float(ex_a.mean()), 3),
              "win_rate_pct": round(100 * float((ex_a > 0).mean()), 1)}
        summary.append(row)
        trial_labels.append(label)
        trial_returns.append(ex)
        _log(f"{label}: 초과 {row['mean_excess_pct']:+.2f}%p 승률 {row['win_rate_pct']}% n={row['n_events']}")

    paired = []
    if baseline_ex is not None:
        for label, ex in zip(trial_labels, trial_returns):
            if label == "baseline_factor_only" or len(ex) != len(baseline_ex):
                continue
            diff = np.array(ex) - baseline_ex
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
            t = float(diff.mean() / se) if se else None
            paired.append({"label": label, "mean_diff_pct": round(100 * float(diff.mean()), 3),
                          "t_stat": round(t, 3) if t is not None else None, "n": len(diff)})

    mom_ic = _momentum_ic_and_buckets(snaps, panel)
    for col, row in mom_ic["ic"].items():
        _log(f"IC[{col}]: spearman {row['mean_spearman']:+.4f} (t={row['t_stat']}, n={row['n_snaps']})")

    combo_wide = _combo_wide_check(snaps, panel)
    _log(f"[넓은후보군결합점수 top{TOPN_WIDE}] 팩터단독 {combo_wide.get('baseline_factor_only_mean_excess_pct')}%p"
        f"(승률{combo_wide.get('baseline_win_rate_pct')}%) vs 결합 "
        f"{combo_wide.get('combo_factor_plus_mom6m_mean_excess_pct')}%p(승률"
        f"{combo_wide.get('combo_win_rate_pct')}%) 페어드t={combo_wide.get('paired_t_stat')}")

    mc = OS.monte_carlo_test(baseline_ex.tolist(), n=1000, seed=0) if baseline_ex is not None else None
    if mc and "note" not in mc:
        _log(f"[몬테카를로 순열검정] 기준전략 MDD {mc['observed_mdd_pct']}%p — 무작위 순서 중 "
            f"{mc['pct_permutations_with_mdd_at_least_as_bad']}%가 이만큼 나쁨")

    lens = [len(ex) for ex in trial_returns]
    pbo_report = None
    if lens:
        common_len = max(set(lens), key=lens.count)
        keep_idx = [i for i, l in enumerate(lens) if l == common_len]
        if len(keep_idx) >= 3 and common_len >= 8:
            data = {"trials": [trial_labels[i] for i in keep_idx],
                   "excess_returns": [trial_returns[i] for i in keep_idx],
                   "rebal_days": 63, "hold_days": 126, "horizon": "6m",
                   "universe": "US_top8", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)

    payload = {"n_snaps": len(snaps), "topn": TOPN,
              "method": ("팩터종합점수(1:2:2) topn8 기준 후보군에, 팩터와 독립인 모멘텀 신호를 "
                        "실제로 다르게 설계될 수 있었던 변형(MA필터·수익률틸트·결합점수)으로 "
                        "적용 — 같은 점수를 반으로 가르는 것과 달리 순환논리 없음. 이 변형들을 "
                        "시행으로 삼아 PBO/DSR 판정."),
              "summary": summary, "paired_vs_baseline": paired,
              "pbo_dsr": pbo_report, "momentum_ic_and_buckets": mom_ic,
              "combo_wide_topn15": combo_wide, "monte_carlo_baseline": mc}
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
