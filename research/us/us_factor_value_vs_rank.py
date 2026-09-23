#!/usr/bin/env python3
"""
us_factor_value_vs_rank.py — 지호 님 질문(2026-09-22): "팩터는 절대값(z-score 크기)이
중요한가 상대순위가 중요한가? 수익률에 어느 쪽이 더 영향을 주는지 상관관계가 궁금하다.
또 어느 정도 이상에서는 수익률 차이가 없는지(포화) 아니면 다다익선인지 — 백테스트로
확인해달라."

3개 라이브 팩터(int_gp_assets·rd_mktcap·shareholder_yield)와 라이브 가중치(1:2:2 raw)
합성점수에 대해:
  (A) 절대값 IC — Pearson(원값 또는 라이브와 동일하게 클립된 z-score, 6개월 forward
      return), 스냅샷마다 계산 후 평균.
  (B) 순위 IC — Spearman(팩터 순위, forward-return 순위) = backtest_weights.py가 이미
      쓰는 정의(순위-순위 상관)와 동일. (A)-(B) 차이가 "크기가 순위를 넘어서는 추가
      정보를 주는가"에 대한 직접적 답.
  (C) 10분위(decile) 스프레드 — 팩터값으로 10등분해 분위별 평균 forward return을 전체
      스냅샷에서 집계, 단조증가(다다익선)인지 특정 분위 이후 평평해지는지(포화)를
      스냅샷-블록 부트스트랩 90%CI와 함께 확인.

재구현 금지(MAINTENANCE.md §1) — us_factor_formula_pit_sweep.build_snaps()를 그대로
재사용해 PIT 스냅샷(원값 raw·6개월 forward return fwd)을 얻는다. 이 스냅샷은 이미 그
시점 실제 S&P500 구성종목(생존편향 없음)·EDGAR 원값 기준.

실행: python -m research.us.us_factor_value_vs_rank [--years 10]
결과: output/us_factor_value_vs_rank.json
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
import research.us.us_factor_formula_pit_sweep as PS

FACTORS = PS.FACTORS  # ["int_gp_assets", "rd_mktcap", "shareholder_yield"]
COMPOSITE_WEIGHTS = {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}  # 라이브(1:2:2 raw)
N_DECILES = 10
N_BOOT = 2000
SEED = 11
OUT_PATH = "output/us_factor_value_vs_rank.json"


def _log(m): print(f"[US팩터값vs순위] {m}", file=sys.stderr)


def _zclip(s: pd.Series, cap=3.0) -> pd.Series:
    sd = s.std()
    z = (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0
    return z.clip(-cap, cap).fillna(0.0)


def _composite(raw: pd.DataFrame) -> pd.Series:
    """라이브(export_data.select_by_weights)와 동일: shareholder_yield만 ±5, 나머지 ±3."""
    z_gp = _zclip(raw["int_gp_assets"], 3.0)
    z_rd = _zclip(raw["rd_mktcap"], 3.0)
    z_sy = _zclip(raw["shareholder_yield"], 5.0)
    return (COMPOSITE_WEIGHTS["int_gp_assets"] * z_gp + COMPOSITE_WEIGHTS["rd_mktcap"] * z_rd
            + COMPOSITE_WEIGHTS["shareholder_yield"] * z_sy)


def per_snap_ics(snaps: list, factor_name: str, get_values) -> list[dict]:
    """스냅샷마다 (원값 Pearson IC, 클립z Pearson IC, 순위 Spearman IC, n종목) 계산."""
    out = []
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"]
        vals_raw = get_values(raw).reindex(fwd.index)
        df = pd.DataFrame({"v": vals_raw, "f": fwd}).dropna()
        if len(df) < 20:
            continue
        pearson_raw = float(df["v"].corr(df["f"]))
        z = _zclip(df["v"])
        pearson_z = float(z.corr(df["f"])) if z.std() else float("nan")
        spearman = float(df["v"].rank().corr(df["f"].rank()))
        out.append({"date": snap["date"], "n": len(df),
                    "pearson_raw": pearson_raw, "pearson_zclip": pearson_z, "spearman": spearman})
    return out


def decile_spread(snaps: list, factor_name: str, get_values, n_dec=N_DECILES) -> dict:
    """스냅샷마다 팩터값 10분위 → 분위별 forward return, 스냅샷 전체 평균 + 블록부트스트랩."""
    bucket_rets = {d: [] for d in range(n_dec)}  # decile -> [snap별 평균수익률, ...]
    n_events = 0
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"]
        vals = get_values(raw).reindex(fwd.index)
        df = pd.DataFrame({"v": vals, "f": fwd}).dropna()
        if len(df) < n_dec * 3:
            continue
        try:
            df["decile"] = pd.qcut(df["v"], n_dec, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_events += 1
        for d, g in df.groupby("decile"):
            bucket_rets[int(d)].append(float(g["f"].mean()))

    rows = []
    for d in range(n_dec):
        r = bucket_rets[d]
        if not r:
            rows.append({"decile": d + 1, "n_snaps": 0, "mean_fwd_ret_pct": None})
            continue
        rows.append({"decile": d + 1, "n_snaps": len(r),
                    "mean_fwd_ret_pct": round(100 * float(np.mean(r)), 3),
                    "std_fwd_ret_pct": round(100 * float(np.std(r, ddof=1)), 3) if len(r) > 1 else None})

    # 블록부트스트랩(스냅샷 단위로 리샘플 — 같은 날짜 안 종목들은 독립이 아니므로)
    valid_deciles = [d for d in range(n_dec) if bucket_rets[d]]
    if len(valid_deciles) >= 2:
        rng = np.random.default_rng(SEED)
        arrs = {d: np.array(bucket_rets[d]) for d in valid_deciles}
        min_len = min(len(a) for a in arrs.values())
        boot_means = {d: np.empty(N_BOOT) for d in valid_deciles}
        for i in range(N_BOOT):
            idx = rng.integers(0, min_len, min_len)
            for d in valid_deciles:
                boot_means[d][i] = arrs[d][idx].mean() if len(arrs[d]) == min_len else \
                    arrs[d][rng.integers(0, len(arrs[d]), min_len)].mean()
        top, bot = max(valid_deciles), min(valid_deciles)
        spread = boot_means[top] - boot_means[bot]
        ci = (round(100 * float(np.percentile(spread, 5)), 3), round(100 * float(np.percentile(spread, 95)), 3))
        top_vs_bottom_ci90 = {"top_decile": top + 1, "bottom_decile": bot + 1,
                              "spread_pct_ci90": ci, "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    else:
        top_vs_bottom_ci90 = None

    # 포화(saturation) 진단: 상위 절반(6~10분위) 내에서 인접분위 평균차이가 작아지는지
    means = [r["mean_fwd_ret_pct"] for r in rows if r["mean_fwd_ret_pct"] is not None]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1)) if len(means) > 1 else None

    return {"factor": factor_name, "n_events": n_events, "deciles": rows,
            "top_vs_bottom_ci90": top_vs_bottom_ci90, "strictly_monotonic": monotonic}


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    if not funds:
        raise RuntimeError("output/fundamentals_cache.json 없음 — python fundamentals_edgar.py 먼저 실행")
    _log(f"패널 {panel.shape[1]}종목 x {panel.shape[0]}거래일")
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"PIT 스냅샷 {len(snaps)}개(6개월 forward, 분기 리밸런싱 시점)")

    targets = {f: (lambda raw, f=f: raw[f]) for f in FACTORS}
    targets["composite_1_2_2"] = lambda raw: _composite(raw)

    payload = {"n_snaps": len(snaps), "factors": list(targets.keys()),
              "method": ("절대값IC=Pearson(원값 또는 라이브클립z, 6개월fwd) vs 순위IC="
                        "Spearman(팩터순위,수익률순위) — 스냅샷별 계산 후 평균. 10분위 "
                        "스프레드는 스냅샷별 qcut 후 분위 평균수익률을 전체에서 집계, "
                        "상위-하위 분위 차이는 스냅샷 단위 블록부트스트랩 90%CI."),
              "results": {}}

    for name, getter in targets.items():
        ics = per_snap_ics(snaps, name, getter)
        if not ics:
            _log(f"[{name}] 유효 스냅샷 없음"); continue
        df = pd.DataFrame(ics)
        ic_summary = {
            "n_snaps_used": len(df),
            "mean_pearson_raw": round(float(df["pearson_raw"].mean()), 4),
            "mean_pearson_zclip": round(float(df["pearson_zclip"].mean()), 4),
            "mean_spearman_rank": round(float(df["spearman"].mean()), 4),
            "pearson_raw_minus_spearman": round(float(df["pearson_raw"].mean() - df["spearman"].mean()), 4),
        }
        dec = decile_spread(snaps, name, getter)
        payload["results"][name] = {"ic": ic_summary, "decile_spread": dec}
        _log(f"[{name}] Pearson(원값) IC {ic_summary['mean_pearson_raw']:+.4f} · "
            f"Pearson(클립z) IC {ic_summary['mean_pearson_zclip']:+.4f} · "
            f"Spearman(순위) IC {ic_summary['mean_spearman_rank']:+.4f} · "
            f"단조증가={dec['strictly_monotonic']}")

    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def self_test():
    _log("[self-test] 합성 스냅샷으로 배선 확인")
    rng = np.random.default_rng(3)
    snaps = []
    for t in range(15):
        n = 150
        v = rng.normal(0, 1, n)
        fwd = 0.02 * v + rng.normal(0, 0.05, n)  # v가 fwd를 선형예측(양의 IC 기대)
        raw = pd.DataFrame({"int_gp_assets": v, "rd_mktcap": rng.normal(0, 1, n),
                            "shareholder_yield": rng.normal(0, 1, n)},
                           index=[f"S{i}" for i in range(n)])
        snaps.append({"date": f"2020-{t+1:02d}-01", "raw": raw,
                     "fwd": pd.Series(fwd, index=raw.index), "bench": 0.01})
    ics = per_snap_ics(snaps, "int_gp_assets", lambda raw: raw["int_gp_assets"])
    assert len(ics) == 15
    mean_pearson = float(np.mean([r["pearson_raw"] for r in ics]))
    assert mean_pearson > 0.2, f"양의 선형관계를 못 잡음: {mean_pearson}"
    _log(f"[self-test] 통과: per_snap_ics 배선 정상(평균 Pearson {mean_pearson:.3f})")

    dec = decile_spread(snaps, "int_gp_assets", lambda raw: raw["int_gp_assets"])
    means = [r["mean_fwd_ret_pct"] for r in dec["deciles"] if r["mean_fwd_ret_pct"] is not None]
    assert means[-1] > means[0], f"상위분위가 하위분위보다 수익률이 높아야 함: {means}"
    assert dec["top_vs_bottom_ci90"]["excludes_zero"], "이 정도 신호면 CI가 0을 배제해야 함"
    _log(f"[self-test] 통과: decile_spread 배선 정상(하위 {means[0]:.2f}% -> 상위 {means[-1]:.2f}%, "
        f"CI {dec['top_vs_bottom_ci90']['spread_pct_ci90']})")
    _log("[self-test] 전부 통과")


def main():
    ap = argparse.ArgumentParser(description="US 팩터 절대값 vs 순위 IC + 10분위 포화 분석")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    run(years=args.years)


if __name__ == "__main__":
    main()
