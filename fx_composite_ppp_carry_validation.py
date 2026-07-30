#!/usr/bin/env python3
"""
fx_composite_ppp_carry_validation.py — PPP(REER)+캐리 복합신호, 타 통화쌍 선행검증 (2026-07-30).

배경: 캐리+VIX 복합신호(fx_composite_external_validation.py)는 외부검증(JPY/BRL/ZAR)에서
0/3 기각됐다(STRATEGY.md §6-S-6, JPY·BRL은 무작위 대조군 하위 7%권). VIX가 위기를
선행경고하지 못하고 동시에 급등하는 지표라 추세추종과 같은 "전환점 지연" 문제를 반복했을
가능성이 지목됐다. 지호 님이 VIX를 빼고 **PPP(실질실효환율 평균회귀)+캐리** 조합으로
재도전하기로 결정.

**사전등록 공식(원달러 데이터를 보기 전에 확정)**:
  signal_ppp_t   = 현지_REER_t (BIS 실질실효환율, FRED) — 자국 REER가 자기 역사 대비
                   높음(실질강세·고평가) → PPP 평균회귀 이론상 이후 그 통화가 약세로
                   되돌아갈 가능성 → 달러보유(h=1) 쪽 (Rogoff PPP 문헌 근거)
  signal_carry_t = US_3M금리_t − 현지_3M금리_t (FX캐리 문헌, Lustig-Menkhoff 계열,
                   fx_composite_external_validation.py와 동일 정의 재사용)
  composite_t(w) = w_ppp·z_expanding(signal_ppp_t) + w_carry·z_expanding(signal_carry_t)
  h_w,t = 1{composite_t(w) > 0}
비율(PPP:캐리) = 3:1/2:1/1:1/1:2/1:3 5개를 argmax 선택 없이 동일가중 앙상블(§1 A안
원칙, fx_composite_external_validation.py와 동일 관례).

**검증 순서(§6 먼저)**: USD/JPY·USD/BRL·USD/ZAR에 재조정 없이 그대로 적용 → 2/3 이상
통과해야 원달러(KRW)로 이식.

REER 데이터: FRED BIS 실질실효환율(월별, 1994~) — RBKRBIS(한국)·RBJPBIS(일본)·
RBBRBIS(브라질)·RBZABIS(남아공).

실행: python fx_composite_ppp_carry_validation.py --external
      python fx_composite_ppp_carry_validation.py --krw
      python fx_composite_ppp_carry_validation.py --self-test
결과: output/fx_composite_ppp_{jpy,brl,zar,krw}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import fx_hedge_validation as FV
import fx_composite_external_validation as CEV

WEIGHT_RATIOS = [(3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]   # (ppp, carry), 앙상블용(argmax 없음)

REER_SERIES = {"jpy": "RBJPBIS", "brl": "RBBRBIS", "zar": "RBZABIS", "krw": "RBKRBIS"}


def _log(m): print(f"[FX-PPP검증] {m}", file=sys.stderr)


def build_frame_ppp(name: str) -> dict:
    cfg = CEV.CURRENCIES[name]
    spy = FV.fetch_spy()
    fx = CEV.fetch_fx_generic(cfg["fx_ticker"])
    us_rate = FV.fetch_fred("DTB3")
    local_rate = FV.fetch_fred(cfg["rate_series"])
    reer = FV.fetch_fred(REER_SERIES[name])

    cal = spy.index
    def align(s):
        return s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
    fx_a, us_a, local_a, reer_a = align(fx), align(us_rate), align(local_rate), align(reer)

    valid = spy.notna() & fx_a.notna() & us_a.notna() & local_a.notna() & reer_a.notna()
    cal = cal[valid]
    spy, fx_a, us_a, local_a, reer_a = (s[valid] for s in (spy, fx_a, us_a, local_a, reer_a))

    carry_daily = (local_a - us_a) / 100.0 / FV.TRADING_DAYS
    signal_carry = (us_a - local_a).to_numpy()
    signal_ppp = reer_a.to_numpy()

    return {"cal": cal, "spy": spy.to_numpy(), "fx": fx_a.to_numpy(), "carry": carry_daily.to_numpy(),
           "signal_ppp": signal_ppp, "signal_carry": signal_carry}


def build_composite_ensemble_ppp(frame: dict) -> dict:
    zp = CEV.z_expanding(frame["signal_ppp"])
    zc = CEV.z_expanding(frame["signal_carry"])
    h_list, combo_stats = [], []
    for wp, wc in WEIGHT_RATIOS:
        composite = wp * zp + wc * zc
        valid = ~np.isnan(composite)
        h = np.where(valid, (composite > 0).astype(float), 0.0)
        h_list.append(h)
        e_valid = h[valid]
        if len(e_valid) < 30:
            combo_stats.append(None); continue
        p = float(e_valid.mean())
        runs, cur, cur_len = [], e_valid[0], 1
        for v in e_valid[1:]:
            if v == cur:
                cur_len += 1
            else:
                runs.append((cur, cur_len)); cur, cur_len = v, 1
        runs.append((cur, cur_len))
        on_runs = [l for v, l in runs if v == 1.0]
        off_runs = [l for v, l in runs if v == 0.0]
        l_on = float(np.mean(on_runs)) if on_runs else 1.0
        l_off = float(np.mean(off_runs)) if off_runs else 1.0
        combo_stats.append({"p": p, "l_on": l_on, "l_off": l_off,
                            "q_on_to_off": min(max(1.0 / l_on, 1e-4), 1.0),
                            "q_off_to_on": min(max(1.0 / l_off, 1e-4), 1.0)})
    h_ensemble = np.mean(h_list, axis=0)
    return {"h_ensemble": h_ensemble, "combo_stats": combo_stats,
           "per_ratio_exposure": [round(float(h.mean()), 4) for h in h_list]}


def run_one_currency(name: str, n_rep_gate1: int = 2000, n_rep_gate2: int = 10000) -> dict:
    frame = build_frame_ppp(name)
    ens = build_composite_ensemble_ppp(frame)
    _log(f"[{name.upper()}] 기간 {frame['cal'][0].date()}~{frame['cal'][-1].date()}"
        f"({len(frame['cal'])}일), 평균노출 {ens['h_ensemble'].mean():.3f}, "
        f"비율별노출 {ens['per_ratio_exposure']}")

    g1 = FV.gate1_matched_random(frame, ens["h_ensemble"], ens["combo_stats"], n_rep=n_rep_gate1)
    g2 = FV.gate2_bootstrap(frame, ens["h_ensemble"], n_rep=n_rep_gate2)

    result = {"currency": name, "reer_series": REER_SERIES[name],
             "date_range": [str(frame["cal"][0].date()), str(frame["cal"][-1].date())],
             "n_days": len(frame["cal"]), "avg_exposure": round(float(ens["h_ensemble"].mean()), 4),
             "per_ratio_exposure": ens["per_ratio_exposure"],
             "gate1": g1, "gate2": g2,
             "gate1_pass": g1["real_filter_percentile_vs_random"]["delta_ulcer"] >= 95,
             "gate2_pass": all(g2[f"block{L}"]["delta_ulcer_excludes_zero"] and
                               g2[f"block{L}"]["delta_ulcer_ci95"][0] > 0 for L in (21, 63, 126)),
             }
    result["overall_pass"] = result["gate1_pass"] and result["gate2_pass"]
    _log(f"[{name.upper()}] Gate1={'통과' if result['gate1_pass'] else '미달'}"
        f"(ΔUlcer {g1['real_filter_percentile_vs_random']['delta_ulcer']}%ile) · "
        f"Gate2={'통과' if result['gate2_pass'] else '미달'} · "
        f"종합={'통과' if result['overall_pass'] else '미달'}")
    return result


def main():
    ap = argparse.ArgumentParser(description="PPP(REER)+캐리 복합신호 — 타 통화쌍 선행검증(§6 먼저)")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--krw", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    os.makedirs("output", exist_ok=True)
    if args.external or not args.krw:
        results = {}
        for name in ("jpy", "brl", "zar"):
            r = run_one_currency(name)
            results[name] = r
            with open(f"output/fx_composite_ppp_{name}.json", "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
        n_pass = sum(1 for r in results.values() if r["overall_pass"])
        majority = n_pass >= 2
        _log(f"\n=== 외부검증 종합: {n_pass}/3 통과 ===")
        _log(f"판정: {'2/3 이상 통과 — 원달러로 이식 가능' if majority else '2/3 미달 — §8 적용, 원달러 이식 보류'}")
        with open("output/fx_composite_ppp_summary.json", "w", encoding="utf-8") as f:
            json.dump({"n_pass": n_pass, "majority_pass": majority,
                      "per_currency": {k: v["overall_pass"] for k, v in results.items()}}, f,
                     ensure_ascii=False, indent=2)

    if args.krw:
        r = run_one_currency("krw")
        with open("output/fx_composite_ppp_krw.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] PPP+캐리 복합공식·앙상블 배선 확인(합성 데이터)")
    rng = np.random.default_rng(13)
    n = 1500

    # PPP 방향성: 자국 REER가 높은(고평가) 구간에서 exposure(h=1, 달러보유)가 더 높아야 함
    reer_step = np.concatenate([np.full(750, 90.0), np.full(750, 130.0)])   # 후반부 REER 급등(고평가)
    carry_flat = rng.normal(0, 0.1, n)
    frame_mini = {"signal_ppp": reer_step, "signal_carry": carry_flat}
    ens = build_composite_ensemble_ppp(frame_mini)
    h = ens["h_ensemble"]
    assert h[800:].mean() > h[300:700].mean(), \
        f"REER 고평가 구간에서 노출이 더 높아야 하는데 아님: 전반 {h[300:700].mean():.3f} vs 후반 {h[800:].mean():.3f}"
    _log(f"[self-test] 통과: PPP(REER고평가→달러보유) 방향 확인(전반 {h[300:700].mean():.3f} → 후반 {h[800:].mean():.3f})")

    # 캐리 방향성 재확인(공유 로직이므로 간단히)
    reer_flat = rng.normal(100, 1, n)
    carry_step = np.concatenate([np.full(750, -5.0), np.full(750, 5.0)])
    frame_mini2 = {"signal_ppp": reer_flat, "signal_carry": carry_step}
    ens2 = build_composite_ensemble_ppp(frame_mini2)
    h2 = ens2["h_ensemble"]
    assert h2[800:].mean() > h2[300:700].mean(), "캐리가 우호적으로 바뀐 후반부에서 노출이 더 높아야 함"
    _log(f"[self-test] 통과: 캐리 방향 확인(전반 {h2[300:700].mean():.3f} → 후반 {h2[800:].mean():.3f})")

    assert len(ens["combo_stats"]) == 5 and all(c is not None for c in ens["combo_stats"])
    _log("[self-test] 통과: 5개 비율 combo_stats 전부 생성됨")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
