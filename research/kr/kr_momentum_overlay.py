#!/usr/bin/env python3
"""
kr_momentum_overlay.py — us_momentum_overlay.py의 한국판(2026-09-22, 지호 님 요청):
"선별된거 중에서 모멘텀이나 몇일선, 또는 6개월/3개월/1개월 수익률로 또 선별해서 추천강도
조정"을 백테스트로 검증.

한국 raw factor(backtest_kr._price_factors)엔 이미 mom12_1(12-1개월 모멘텀)·mom6(6개월
수익률)·hi52_prox(52주 신고가 근접도)가 들어있다(현행 라이브가 mom12_1×0.6+hi52×0.4로
core-satellite 없이 쓰는 그 팩터들). 여기선 그 값을 그대로 재사용하고, 1개월 수익률과
N일선 이격도만 같은 panel에서 추가로 계산한다.

방법:
  1) backtest_kr.build_kr_snaps() 그대로 재사용(팩터 계산 로직 안 건드림).
  2) 같은 panel에서 스냅샷 날짜의 가격 인덱스를 찾아 20/50/100/150/200일선 이격도 +
     1개월 후행수익률 추가 계산(mom6·mom12_1은 raw에 이미 있음).
  3) 팩터(valuediv) 상위 topn5(라이브 조건)를 기준 후보군으로 고정하고 그 안에서:
     a) MA필터: N일선 아래 종목 제외(5가지)
     b) 수익률틸트: 후보군을 후행수익률(1개월/mom6/mom12_1)로 상위/하위 절반만 남김
        (추세추종 3가지 + 역추세 3가지 — 지호 님 질문: "-가 심하면 반대로 하면 안되나,
        수익률 -일때 가점 주는 식으로")
     c) 결합점수: 팩터 z + mom6 z 합산으로 전체 유니버스에서 재선정
  4) 기준(팩터 단독 topn5) 대비 이벤트 평균 초과수익 개선폭 + 페어드 t검정.
  5) 이벤트 수가 같은 시행들을 묶어 overfit_stats.OS.analyze()로 PBO/DSR 판정.
  6) 모멘텀 신호와 개별종목 forward 초과수익의 순위상관(IC)+5분위 버킷 테이블(지호 님
     요청 — "그냥 +-로 하지말고 상관관계 분석해서 몇 이상이면 유리한지").
  7) 레짐 진단(지호 님 질문 — "국장 초과수익이 -야? 반도체 급등 때문 아닌가"): 벤치마크
     (KOSPI) 자체의 후행 6개월 수익률과 기준전략(팩터단독 topn5) 초과수익의 스냅샷별
     상관 + 최근 4개 스냅샷 vs 그 이전 구간 초과수익 비교를 별도로 찍는다(데이터를 몰래
     제외하는 게 아니라 원인을 진단하는 목적).
  8) 삼성전자(005930)·SK하이닉스(000660) 제외 벤치마크(지호 님 요청 — "삼전/하닉 최근
     상승세는 제외하고 승률을 계산"): panel+mktcaps로 그 시점 유니버스 시총가중수익률을
     직접 재구성해(공식 KOSPI 지수 자체를 대체하는 게 아니라 근사치 — 방법 한계는 결과에
     명시) 그 두 종목 포함/제외 버전을 비교, 기준전략(topn5) 초과수익·승률을 재계산.
  9) 이진 필터/틸트 대신 "점수 자체에 모멘텀을 연속적으로 섞는" 결합점수를 더 넓은 후보군
     (topn12)에서 테스트(지호 님 요청 — topn5는 표본이 너무 작아 IC로는 유의한 신호가
     필터/틸트로는 안 잡혔던 문제의 대안). 결합점수 자체 topn12 vs 팩터단독 topn12를
     별도 기준으로 비교(topn5 트라이얼과는 섞지 않음).

실행: python -m research.kr.kr_momentum_overlay
결과: output/kr_momentum_overlay.json
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

import overfit_stats as OS
from research.kr.kr_factor_value_vs_rank import _composite

TOPN = 5
TOPN_WIDE = 12
HORIZON = "6m"
MA_GRID = [20, 50, 100, 150, 200]
GIANTS = ["005930", "000660"]   # 삼성전자·SK하이닉스
OUT_PATH = "output/kr_momentum_overlay.json"


def _log(m): print(f"[KR모멘텀오버레이] {m}", file=sys.stderr)


def _extra_mom_features(panel: pd.DataFrame, date_iso: str, tickers) -> pd.DataFrame:
    """N일선 이격도 + 1개월 후행수익률(mom6·mom12_1은 snap["raw"]에 이미 있음)."""
    p = panel.index.get_indexer([pd.Timestamp(date_iso)], method="pad")[0]
    px = panel.iloc[: p + 1]
    cur = px.iloc[-1]
    out = {}
    for n in MA_GRID:
        if len(px) > n:
            ma = px.iloc[-n:].mean()
            out[f"ma{n}_gap"] = cur / ma - 1
    if len(px) > 21:
        out["ret_1m"] = cur / px.iloc[-22] - 1
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
            return keep if len(keep) >= 2 else top
        return sel

    def make_ret_tilt(col, reverse=False):
        def sel(score, mom):
            top = score.sort_values(ascending=False).index[:TOPN]
            if col not in mom.columns:
                return top
            m = mom.loc[top, col].dropna()
            if len(m) < 4:
                return top
            half = max(len(m) // 2, 2)
            return m.sort_values(ascending=reverse).index[:half]
        return sel

    def combo_sel(score, mom):
        sd = score.std(ddof=0)
        fz = (score - score.mean()) / sd if sd else score * 0.0
        mret = mom["mom6"] if "mom6" in mom.columns else pd.Series(dtype=float)
        msd = mret.std(ddof=0) if len(mret) else 0.0
        mz = (((mret - mret.mean()) / msd).reindex(score.index).fillna(0.0)
              if msd else pd.Series(0.0, index=score.index))
        combo = fz + mz
        return combo.sort_values(ascending=False).index[:TOPN]

    variants = {"baseline_factor_only": base_sel}
    for n in MA_GRID:
        variants[f"ma{n}_filter"] = make_ma_filter(n)
    for col in ("ret_1m", "mom6", "mom12_1"):
        variants[f"tilt_top_half_{col}"] = make_ret_tilt(col)
        variants[f"tilt_bottom_half_{col}(역추세)"] = make_ret_tilt(col, reverse=True)
    variants["combo_factor_plus_mom6"] = combo_sel
    return variants


SIGNAL_COLS = [f"ma{n}_gap" for n in MA_GRID] + ["ret_1m", "mom6", "mom12_1"]


def _momentum_ic_and_buckets(snaps, panel) -> dict:
    """topn5 후보군 내에서 모멘텀 신호별로: (a) 스냅샷별 개별종목 순위상관(신호 vs forward
    초과수익) 평균+t검정, (b) 전체 스냅샷 풀링한 5분위 구간별 평균 초과수익 테이블."""
    per_snap_corr = {c: [] for c in SIGNAL_COLS}
    pooled = {c: [] for c in SIGNAL_COLS}
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        extra = _extra_mom_features(panel, snap["date"], top)
        mom = pd.concat([raw.reindex(top)[["mom6", "mom12_1"]], extra], axis=1)
        r = fwd.reindex(top) - bnc
        for col in SIGNAL_COLS:
            pair = pd.concat([mom[col], r], axis=1, keys=["sig", "ex"]).dropna()
            if len(pair) >= 3:
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


def _regime_diagnosis(snaps) -> dict:
    """지호 님 질문 진단용: 벤치마크(KOSPI) forward 6개월 수익률 vs 기준전략(팩터단독
    topn5) 초과수익의 스냅샷별 상관 + 최근 구간 vs 이전 구간 비교. 데이터를 빼는 게
    아니라 '왜 최근이 나쁜지'를 설명하기 위한 진단 전용 — 필터/선정 로직에는 안 쓴다."""
    rows = []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        r = fwd.reindex(top).dropna()
        if len(r) == 0:
            continue
        rows.append({"date": snap["date"], "bench_fwd_6m_pct": round(100 * bnc, 2),
                    "strategy_excess_pct": round(100 * (float(r.mean()) - bnc), 2)})

    if len(rows) < 6:
        return {"note": "표본 부족", "rows": rows}
    tail = rows[-4:]
    head = rows[:-4]
    ic = None
    if len(rows) >= 8:
        b = np.array([x["bench_fwd_6m_pct"] for x in rows])
        e = np.array([x["strategy_excess_pct"] for x in rows])
        ic = float(pd.Series(b).corr(pd.Series(e), method="spearman"))
    return {"rows": rows,
           "spearman_bench_fwdret_vs_strategy_excess": round(ic, 4) if ic is not None else None,
           "recent_4_mean_excess_pct": round(float(np.mean([x["strategy_excess_pct"] for x in tail])), 3),
           "earlier_mean_excess_pct": round(float(np.mean([x["strategy_excess_pct"] for x in head])), 3),
           "note": ("bench_fwd_6m_pct가 아주 크면(예: 2025년 이후처럼 +60%대) 벤치마크 자체가 "
                   "소수 종목 쏠림으로 급등했다는 신호 — 음의 상관이면 '지수가 쏠려서 뜨거울수록 "
                   "밸류·배당 팩터가 더 뒤처진다'는 구조적 설명이 성립. 데이터를 제외하지 않고 "
                   "그대로 둔 채 원인만 진단.")}


def _mktcap_asof(mktcaps: dict, date_iso: str) -> dict:
    """mktcaps는 분기 안팎 간격이라 스냅샷 날짜와 정확히 안 맞을 수 있어 그 이전 최신
    날짜를 찾는다(공식 지수 재현이 아니라 근사 가중치용이라 이 정도면 충분)."""
    d8 = date_iso.replace("-", "")
    keys = sorted(k for k in mktcaps if k <= d8)
    return mktcaps[keys[-1]] if keys else {}


def _capw_fwd_return(panel: pd.DataFrame, members, weights: dict, p: int, hold_days: int = 126):
    """그 시점 시총가중치로 [p+1, p+1+hold_days] 구간 수익률 재구성(공식 KOSPI 대체가 아닌
    근사치 — 재조정 없이 시작시점 가중치 고정, 리밸 주기 내 근사)."""
    e = p + 1
    if e + hold_days >= len(panel):
        return None
    p0 = panel.iloc[e][members]
    p1 = panel.iloc[e + hold_days][members]
    ret = (p1 / p0 - 1).dropna()
    w = pd.Series({t: weights.get(t, 0.0) for t in ret.index})
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    return float((ret * w).sum())


def ex_giants_benchmark_check(snaps, panel, membership, mktcaps) -> dict:
    """지호 님 요청: "삼전/하닉 최근 상승세는 제외하고 승률을 계산해보자". 공식 KOSPI를
    바꿀 순 없으니 panel+mktcaps로 그 시점 유니버스 시총가중수익률을 직접 재구성해
    (근사치) 두 종목 포함/제외 버전을 비교하고, 기준전략(팩터단독 topn5) 초과수익·승률을
    재계산한다."""
    rows = []
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]
        r = fwd.reindex(top).dropna()
        if len(r) == 0:
            continue
        p = panel.index.get_indexer([pd.Timestamp(snap["date"])], method="pad")[0]
        d8 = snap["date"].replace("-", "")
        members = membership.get(d8) or membership.get("_current") or list(panel.columns)
        members = [t for t in panel.columns.intersection(members)
                  if pd.notna(panel.iloc[p][t])]
        w = _mktcap_asof(mktcaps, snap["date"])
        capw_incl = _capw_fwd_return(panel, members, w, p)
        members_ex = [t for t in members if t not in GIANTS]
        capw_excl = _capw_fwd_return(panel, members_ex, w, p)
        if capw_incl is None or capw_excl is None:
            continue
        strat_ret = float(r.mean())
        rows.append({"date": snap["date"],
                    "official_bench_fwd_pct": round(100 * snap["bench"][HORIZON], 2),
                    "reconstructed_capw_incl_giants_pct": round(100 * capw_incl, 2),
                    "reconstructed_capw_excl_giants_pct": round(100 * capw_excl, 2),
                    "strategy_fwd_pct": round(100 * strat_ret, 2),
                    "excess_vs_official_pct": round(100 * (strat_ret - snap["bench"][HORIZON]), 2),
                    "excess_vs_excl_giants_pct": round(100 * (strat_ret - capw_excl), 2)})
    if len(rows) < 8:
        return {"note": "표본 부족", "rows": rows}
    off_ex = np.array([r["excess_vs_official_pct"] for r in rows])
    exg_ex = np.array([r["excess_vs_excl_giants_pct"] for r in rows])
    sanity_gap = np.array([r["reconstructed_capw_incl_giants_pct"] - r["official_bench_fwd_pct"]
                           for r in rows])
    return {"n_events": len(rows), "rows": rows,
           "sanity_check_mean_gap_reconstructed_vs_official_pct": round(float(sanity_gap.mean()), 2),
           "mean_excess_vs_official_bench_pct": round(float(off_ex.mean()), 3),
           "win_rate_vs_official_pct": round(100 * float((off_ex > 0).mean()), 1),
           "mean_excess_vs_excl_giants_bench_pct": round(float(exg_ex.mean()), 3),
           "win_rate_vs_excl_giants_pct": round(100 * float((exg_ex > 0).mean()), 1),
           "note": ("재구성 벤치마크는 시작시점 시총가중 고정(재조정 없음) 근사치 — sanity_check가 "
                   "official과 크게 벌어지면 재구성 자체의 신뢰도가 낮다는 뜻이니 함께 확인.")}


def _combo_wide_check(snaps, panel) -> dict:
    """지호 님 요청: 이진 필터 대신 팩터z+모멘텀z 연속결합을 더 넓은 후보군(topn12)에서
    — topn5는 표본이 작아 IC로는 유의했던 신호가 필터/틸트로는 안 잡히는 문제의 대안."""
    base_ex, combo_ex = [], []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN_WIDE:
            continue
        extra = _extra_mom_features(panel, snap["date"], score.index)
        mom6 = raw.reindex(score.index)["mom6"]
        base_top = score.sort_values(ascending=False).index[:TOPN_WIDE]
        r_base = fwd.reindex(base_top).dropna()

        sd = score.std(ddof=0)
        fz = (score - score.mean()) / sd if sd else score * 0.0
        msd = mom6.std(ddof=0)
        mz = ((mom6 - mom6.mean()) / msd).reindex(score.index).fillna(0.0) if msd else score * 0.0
        combo = (fz + mz).sort_values(ascending=False).index[:TOPN_WIDE]
        r_combo = fwd.reindex(combo).dropna()

        if len(r_base) == 0 or len(r_combo) == 0:
            continue
        base_ex.append(float(r_base.mean()) - bnc)
        combo_ex.append(float(r_combo.mean()) - bnc)

    if len(base_ex) < 8:
        return {"note": "표본 부족"}
    b, c = np.array(base_ex), np.array(combo_ex)
    diff = c - b
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None
    return {"n_events": len(b), "topn": TOPN_WIDE,
           "baseline_factor_only_mean_excess_pct": round(100 * float(b.mean()), 3),
           "baseline_win_rate_pct": round(100 * float((b > 0).mean()), 1),
           "combo_factor_plus_mom6_mean_excess_pct": round(100 * float(c.mean()), 3),
           "combo_win_rate_pct": round(100 * float((c > 0).mean()), 1),
           "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
           "paired_t_stat": round(t, 3) if t is not None else None}


def _event_stats(sel_fn, snaps, panel) -> list:
    ex = []
    for snap in snaps:
        raw, fwd, bnc = snap["raw"], snap["fwd"][HORIZON], snap["bench"][HORIZON]
        score = _composite(raw).reindex(fwd.index).dropna()
        if len(score) < TOPN:
            ex.append(None)
            continue
        extra = _extra_mom_features(panel, snap["date"], score.index)
        mom = pd.concat([raw.reindex(score.index)[["mom6", "mom12_1"]], extra], axis=1)
        sel = sel_fn(score, mom)
        r = fwd.reindex(sel).dropna() if sel is not None and len(sel) else pd.Series(dtype=float)
        ex.append(float(r.mean()) - bnc if len(r) else None)
    return ex


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
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

    mom_ic = _momentum_ic_and_buckets(snaps, panel)
    for col, row in mom_ic["ic"].items():
        _log(f"IC[{col}]: spearman {row['mean_spearman']:+.4f} (t={row['t_stat']}, n={row['n_snaps']})")

    regime = _regime_diagnosis(snaps)
    _log(f"[레짐진단] bench초과 상관 {regime.get('spearman_bench_fwdret_vs_strategy_excess')} · "
        f"최근4개 평균초과 {regime.get('recent_4_mean_excess_pct')}%p vs 이전 평균 "
        f"{regime.get('earlier_mean_excess_pct')}%p")

    ex_giants = ex_giants_benchmark_check(snaps, panel, membership, mktcaps)
    _log(f"[삼전하닉제외] sanity갭 {ex_giants.get('sanity_check_mean_gap_reconstructed_vs_official_pct')}%p · "
        f"공식벤치 대비 초과 {ex_giants.get('mean_excess_vs_official_bench_pct')}%p(승률 "
        f"{ex_giants.get('win_rate_vs_official_pct')}%) vs 제외벤치 대비 초과 "
        f"{ex_giants.get('mean_excess_vs_excl_giants_bench_pct')}%p(승률 "
        f"{ex_giants.get('win_rate_vs_excl_giants_pct')}%)")

    combo_wide = _combo_wide_check(snaps, panel)
    _log(f"[넓은후보군결합점수 top{TOPN_WIDE}] 팩터단독 {combo_wide.get('baseline_factor_only_mean_excess_pct')}%p"
        f"(승률{combo_wide.get('baseline_win_rate_pct')}%) vs 결합 "
        f"{combo_wide.get('combo_factor_plus_mom6_mean_excess_pct')}%p(승률"
        f"{combo_wide.get('combo_win_rate_pct')}%) 페어드t={combo_wide.get('paired_t_stat')}")

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
                   "rebal_days": 63, "hold_days": 126, "horizon": HORIZON,
                   "universe": "KR_top5", "cost": "gross(이벤트 평균)"}
            pbo_report = OS.analyze(data, save=False)

    payload = {"n_snaps": len(snaps), "topn": TOPN,
              "method": ("팩터종합점수(valuediv 동일가중) topn5 기준 후보군에, 팩터와 독립인 "
                        "모멘텀 신호(mom6·mom12_1·1개월수익률·N일선이격도)를 MA필터·수익률틸트· "
                        "결합점수로 적용 — 같은 점수를 반으로 가르는 것과 달리 순환논리 없음. "
                        "이 변형들을 시행으로 삼아 PBO/DSR 판정."),
              "summary": summary, "paired_vs_baseline": paired,
              "pbo_dsr": pbo_report, "momentum_ic_and_buckets": mom_ic,
              "regime_diagnosis": regime, "ex_giants_benchmark_check": ex_giants,
              "combo_wide_topn12": combo_wide, "monte_carlo_baseline": mc}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    run()
