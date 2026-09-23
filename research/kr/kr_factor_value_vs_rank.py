#!/usr/bin/env python3
"""
kr_factor_value_vs_rank.py — us_factor_value_vs_rank.py의 한국판(2026-09-22, 지호 님
"한국 쪽도 같은 그리드로" 패턴을 이어받은 질문: "팩터는 절대값이 중요한가 상대순위가
중요한가 + 다다익선인가 포화인가, 백테스트로 확인").

라이브 valuediv 3팩터(value=1/PER · pbr_inv=1/PBR · div_yield=배당수익률, 동일가중)와
합성점수에 대해 (A) Pearson(원값/클립z, 6개월fwd) vs (B) Spearman(순위,순위) IC 비교 +
(C) 10분위 스프레드(포화 여부, 스냅샷 블록부트스트랩)를 확인한다.

재구현 금지 — backtest_kr.build_kr_snaps()를 그대로 재사용(코스피200 PIT 멤버십·
pykrx 펀더멘탈 원값·다구간 forward return 이미 계산됨). live_ok(EPS/ROE/PER/200MA
필터) 마스크는 적용하지 않음 — "팩터 자체의 예측력"을 보는 게 목적이라 필터로 표본을
줄이지 않는다(라이브 선정 자체를 재현하는 게 목적이 아님, 참고로 명시).

실행: python -m research.kr.kr_factor_value_vs_rank
결과: output/kr_factor_value_vs_rank.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

FACTORS = ["value", "pbr_inv", "div_yield"]   # 1/PER, 1/PBR, 배당수익률
COMPOSITE_WEIGHTS = {"value": 1, "pbr_inv": 1, "div_yield": 1}   # 라이브(동일가중)
HORIZON = "6m"
N_DECILES = 10
N_BOOT = 2000
SEED = 11
OUT_PATH = "output/kr_factor_value_vs_rank.json"


def _log(m): print(f"[KR팩터값vs순위] {m}", file=sys.stderr)


def _zclip(s: pd.Series, cap=3.0) -> pd.Series:
    sd = s.std()
    z = (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0
    return z.clip(-cap, cap).fillna(0.0)


def _composite(raw: pd.DataFrame) -> pd.Series:
    z_v = _zclip(raw["value"]); z_p = _zclip(raw["pbr_inv"]); z_d = _zclip(raw["div_yield"])
    return (COMPOSITE_WEIGHTS["value"] * z_v + COMPOSITE_WEIGHTS["pbr_inv"] * z_p
            + COMPOSITE_WEIGHTS["div_yield"] * z_d)


def per_snap_ics(snaps: list, get_values, horizon=HORIZON) -> list[dict]:
    out = []
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"][horizon]
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


def decile_spread(snaps: list, get_values, horizon=HORIZON, n_dec=N_DECILES) -> dict:
    bucket_rets = {d: [] for d in range(n_dec)}
    n_events = 0
    for snap in snaps:
        raw, fwd = snap["raw"], snap["fwd"][horizon]
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

    valid_deciles = [d for d in range(n_dec) if bucket_rets[d]]
    top_vs_bottom_ci90 = None
    if len(valid_deciles) >= 2:
        rng = np.random.default_rng(SEED)
        top, bot = max(valid_deciles), min(valid_deciles)
        a_top, a_bot = np.array(bucket_rets[top]), np.array(bucket_rets[bot])
        n = min(len(a_top), len(a_bot))
        spread = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx_t = rng.integers(0, len(a_top), n)
            idx_b = rng.integers(0, len(a_bot), n)
            spread[i] = a_top[idx_t].mean() - a_bot[idx_b].mean()
        ci = (round(100 * float(np.percentile(spread, 5)), 3), round(100 * float(np.percentile(spread, 95)), 3))
        top_vs_bottom_ci90 = {"top_decile": top + 1, "bottom_decile": bot + 1,
                              "spread_pct_ci90": ci, "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}

    means = [r["mean_fwd_ret_pct"] for r in rows if r["mean_fwd_ret_pct"] is not None]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1)) if len(means) > 1 else None
    return {"n_events": n_events, "deciles": rows,
            "top_vs_bottom_ci90": top_vs_bottom_ci90, "strictly_monotonic": monotonic}


def run(save: bool = True) -> dict:
    from research.kr.benchmarks_kr import load_research_data
    import backtest_kr as BK

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    _log(f"패널 {panel.shape[1]}종목 x {panel.shape[0]}거래일")
    snaps, n_pit_ok, n_ps = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                              rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개(요청 {n_ps}개 중 PIT멤버십 확보 {n_pit_ok}개) · "
        f"6개월 forward, 분기 리밸런싱 시점 · live_ok 필터 미적용(팩터 자체 예측력만 본다)")

    targets = {f: (lambda raw, f=f: raw[f]) for f in FACTORS}
    targets["composite_valuediv"] = lambda raw: _composite(raw)

    payload = {"n_snaps": len(snaps), "factors": list(targets.keys()), "horizon": HORIZON,
              "method": ("us_factor_value_vs_rank.py와 동일 방법론 — 절대값IC=Pearson(원값/"
                        "라이브클립z) vs 순위IC=Spearman, 10분위 스프레드는 스냅샷 단위 "
                        "블록부트스트랩 90%CI. live_ok(EPS/ROE/PER/200MA) 필터는 적용 안 함"
                        "(라이브 재현이 아니라 팩터 자체 예측력 확인이 목적)."),
              "results": {}}

    for name, getter in targets.items():
        ics = per_snap_ics(snaps, getter)
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
        dec = decile_spread(snaps, getter)
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
        fwd6 = 0.02 * v + rng.normal(0, 0.05, n)
        raw = pd.DataFrame({"value": v, "pbr_inv": rng.normal(0, 1, n),
                            "div_yield": rng.normal(0, 1, n)},
                           index=[f"S{i}" for i in range(n)])
        snaps.append({"date": f"2020-{t+1:02d}-01", "raw": raw,
                     "fwd": {"6m": pd.Series(fwd6, index=raw.index)}, "bench": {"6m": 0.01}})
    ics = per_snap_ics(snaps, lambda raw: raw["value"])
    assert len(ics) == 15
    mean_pearson = float(np.mean([r["pearson_raw"] for r in ics]))
    assert mean_pearson > 0.2, f"양의 선형관계를 못 잡음: {mean_pearson}"
    _log(f"[self-test] 통과: per_snap_ics 배선 정상(평균 Pearson {mean_pearson:.3f})")
    dec = decile_spread(snaps, lambda raw: raw["value"])
    means = [r["mean_fwd_ret_pct"] for r in dec["deciles"] if r["mean_fwd_ret_pct"] is not None]
    assert means[-1] > means[0]
    assert dec["top_vs_bottom_ci90"]["excludes_zero"]
    _log(f"[self-test] 통과: decile_spread 배선 정상({means[0]:.2f}% -> {means[-1]:.2f}%, "
        f"CI {dec['top_vs_bottom_ci90']['spread_pct_ci90']})")
    _log("[self-test] 전부 통과")


def main():
    ap = argparse.ArgumentParser(description="KR 팩터 절대값 vs 순위 IC + 10분위 포화 분석")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    run()


if __name__ == "__main__":
    main()
