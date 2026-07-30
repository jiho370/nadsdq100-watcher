#!/usr/bin/env python3
"""
fx_composite_trend_carry_ppp_validation.py — 추세+캐리+PPP 3팩터, 타 통화쌍 선행검증 (2026-07-30).

배경: 캐리+VIX(§6-S-6, 0/3 기각) → PPP+캐리(0/3 기각, JPY만 근소 미달)에 이은 세 번째
시도. 지호 님 제안: "추세+캐리+PPP를 배율 다양하게" — 이 조합은 AQR류 FX 스타일
프리미엄 문헌(Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere",
Koijen-Moskowitz-Pedersen-Vrugt 2018 "Carry")이 실제로 다루는 FX 3대 정석 스타일
(모멘텀·캐리·밸류)의 조합이라 지금까지 시도 중 문헌적 근거가 가장 탄탄하다.

**사전등록 공식(원달러 데이터를 보기 전에 확정)**:
  signal_trend_t  = (price_t − MA200_t) / MA200_t   (단일 표준 파라미터 200일선 — 재탐색
                    안 함, §6-R의 "지수 기본값 그대로" 앵커와 동일 선택으로 특정 파라미터
                    재선택 편향 회피)
  signal_carry_t  = US_3M금리_t − 현지_3M금리_t     (fx_composite_external_validation.py
                    와 동일 정의)
  signal_ppp_t    = 현지_REER_t                       (fx_composite_ppp_carry_validation.py
                    와 동일 정의)
  composite_t(w)  = w_t·z_expanding(signal_trend_t) + w_c·z_expanding(signal_carry_t)
                    + w_p·z_expanding(signal_ppp_t)
  h_w,t = 1{composite_t(w) > 0}
"배율 다양하게"는 argmax 선택이 아니라 **7개 가중치 조합(동일·각 팩터 2배 강조·각 팩터
2배 약화)을 전부 동일가중 앙상블**(§1 A안 원칙 재적용, 지난 두 시도와 동일 관례):
  (1,1,1) 균등 / (2,1,1)(1,2,1)(1,1,2) 팩터별 강조 / (1,2,2)(2,1,2)(2,2,1) 팩터별 약화
(w_t, w_c, w_p 순서)

**검증 순서(§6 먼저)**: USD/JPY·USD/BRL·USD/ZAR에 재조정 없이 그대로 적용 → 2/3 이상
통과해야 원달러(KRW)로 이식.

실행: python fx_composite_trend_carry_ppp_validation.py --external
      python fx_composite_trend_carry_ppp_validation.py --krw
      python fx_composite_trend_carry_ppp_validation.py --self-test
결과: output/fx_composite_tcp_{jpy,brl,zar,krw}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import fx_hedge_validation as FV
import fx_composite_external_validation as CEV
import fx_composite_ppp_carry_validation as PPV

# (w_trend, w_carry, w_ppp) — 7개, argmax 없이 전부 앙상블
WEIGHT_TRIPLES = [
    (1, 1, 1),
    (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (1, 2, 2), (2, 1, 2), (2, 2, 1),
]
MA_WINDOW = 200   # 단일 표준 파라미터(재탐색 안 함)


def _log(m): print(f"[FX-3팩터검증] {m}", file=sys.stderr)


def build_frame_tcp(name: str) -> dict:
    cfg = CEV.CURRENCIES[name]
    spy = FV.fetch_spy()
    fx = CEV.fetch_fx_generic(cfg["fx_ticker"])
    us_rate = FV.fetch_fred("DTB3")
    local_rate = FV.fetch_fred(cfg["rate_series"])
    reer = FV.fetch_fred(PPV.REER_SERIES[name])

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
    fx_close = fx_a.to_numpy()
    ma200 = pd.Series(fx_close).rolling(MA_WINDOW).mean().to_numpy()
    signal_trend = (fx_close - ma200) / ma200

    return {"cal": cal, "spy": spy.to_numpy(), "fx": fx_close, "carry": carry_daily.to_numpy(),
           "signal_trend": signal_trend, "signal_carry": signal_carry, "signal_ppp": signal_ppp}


def build_composite_ensemble_tcp(frame: dict) -> dict:
    zt = CEV.z_expanding(frame["signal_trend"])
    zc = CEV.z_expanding(frame["signal_carry"])
    zp = CEV.z_expanding(frame["signal_ppp"])
    h_list, combo_stats = [], []
    for wt, wc, wp in WEIGHT_TRIPLES:
        composite = wt * zt + wc * zc + wp * zp
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
    frame = build_frame_tcp(name)
    ens = build_composite_ensemble_tcp(frame)
    _log(f"[{name.upper()}] 기간 {frame['cal'][0].date()}~{frame['cal'][-1].date()}"
        f"({len(frame['cal'])}일), 평균노출 {ens['h_ensemble'].mean():.3f}, "
        f"비율별노출 {ens['per_ratio_exposure']}")

    g1 = FV.gate1_matched_random(frame, ens["h_ensemble"], ens["combo_stats"], n_rep=n_rep_gate1)
    g2 = FV.gate2_bootstrap(frame, ens["h_ensemble"], n_rep=n_rep_gate2)

    result = {"currency": name,
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
    ap = argparse.ArgumentParser(description="추세+캐리+PPP 3팩터 — 타 통화쌍 선행검증(§6 먼저)")
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
            with open(f"output/fx_composite_tcp_{name}.json", "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
        n_pass = sum(1 for r in results.values() if r["overall_pass"])
        majority = n_pass >= 2
        _log(f"\n=== 외부검증 종합: {n_pass}/3 통과 ===")
        _log(f"판정: {'2/3 이상 통과 — 원달러로 이식 가능' if majority else '2/3 미달 — §8 적용, 원달러 이식 보류'}")
        with open("output/fx_composite_tcp_summary.json", "w", encoding="utf-8") as f:
            json.dump({"n_pass": n_pass, "majority_pass": majority,
                      "per_currency": {k: v["overall_pass"] for k, v in results.items()}}, f,
                     ensure_ascii=False, indent=2)

    if args.krw:
        r = run_one_currency("krw")
        with open("output/fx_composite_tcp_krw.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 추세+캐리+PPP 복합공식·앙상블 배선 확인(합성 데이터)")
    rng = np.random.default_rng(21)
    n = 1500

    # 추세 방향성: 후반부에 가격이 MA200 위로 뚜렷이 벌어지면 exposure가 높아져야 함
    base = 100 + np.cumsum(rng.normal(0, 0.3, n))
    fx_up = base.copy()
    fx_up[750:] = fx_up[750:] + np.linspace(0, 15, n - 750)   # 후반부 상승추세 심음
    ma200 = pd.Series(fx_up).rolling(200).mean().to_numpy()
    signal_trend = (fx_up - ma200) / ma200
    flat = rng.normal(0, 0.1, n)
    # PPP는 완전한 상수(분산0)로 두면 z-score 분모가 0이 돼 NaN이 전파되므로 미세한
    # 노이즈만 섞어 사실상 평평하게(추세 성분만 격리 확인).
    ppp_flat = 100.0 + rng.normal(0, 1e-6, n)
    frame_mini = {"signal_trend": signal_trend, "signal_carry": flat, "signal_ppp": ppp_flat}
    ens = build_composite_ensemble_tcp(frame_mini)
    h = ens["h_ensemble"]
    # expanding z-score 특성상 추세가 오래 지속되면 그 값 자체가 새 평균에 흡수돼 "극단"으로
    # 안 보이게 된다(정상 동작) — 그러므로 "추세 시작 직후"(아직 확장평균이 못 따라잡은 구간)와
    # 비교해야 한다. 아주 후반부(오래 지속된 후)와 비교하면 이 self-test처럼 잘못된 기대가 된다.
    assert h[760:900].mean() > h[400:700].mean(), \
        f"추세 시작 직후 노출이 안 늘어남: 이전 {h[400:700].mean():.3f} vs 시작직후 {h[760:900].mean():.3f}"
    _log(f"[self-test] 통과: 추세 방향 확인(이전 {h[400:700].mean():.3f} → 시작직후 {h[760:900].mean():.3f})")

    assert len(ens["combo_stats"]) == 7 and all(c is not None for c in ens["combo_stats"])
    _log("[self-test] 통과: 7개 가중치조합 combo_stats 전부 생성됨")

    # 캐리·PPP 방향성은 기존 두 스크립트에서 이미 검증된 로직 재사용이므로 wiring만 재확인
    carry_step = np.concatenate([np.full(750, -5.0), np.full(750, 5.0)])
    reer_step = np.concatenate([np.full(750, 90.0), np.full(750, 130.0)])
    frame_mini2 = {"signal_trend": rng.normal(0, 0.01, n), "signal_carry": carry_step, "signal_ppp": reer_step}
    ens2 = build_composite_ensemble_tcp(frame_mini2)
    h2 = ens2["h_ensemble"]
    assert h2[800:].mean() > h2[300:700].mean(), "캐리+PPP 우호적 전환 후 노출이 더 높아야 함"
    _log(f"[self-test] 통과: 캐리+PPP 방향 확인({h2[300:700].mean():.3f} → {h2[800:].mean():.3f})")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
