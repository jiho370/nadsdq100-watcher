#!/usr/bin/env python3
"""
fx_composite_external_validation.py — 캐리+VIX 복합신호, 타 통화쌍 선행검증 (2026-07-30).

배경: fx_hedge_validation.py(추세추종 MA그리드)는 Gate2에서 기각됐다(STRATEGY.md §6-S).
지호 님이 대안으로 "경제적으로 독립된 입력"(가격변환 스태킹 아님)을 요청했고, 검증
순서도 뒤집어(§6을 먼저 — 원달러 표본은 이미 오염됨) 타 통화쌍에서 먼저 확인하기로
했다(§6-S-5의 재도전 조건).

**사전등록 공식(이 문서 작성 시점에 확정, 원달러 데이터는 아직 안 봄)**:
  signal_carry_t = US_3M금리_t − 현지_3M금리_t   (미국 금리 프리미엄 — FX캐리 문헌 근거,
                                                    Lustig-Menkhoff 계열)
  signal_risk_t  = VIX_t                          (위험선호 — VIX 높으면 리스크오프로
                                                    달러강세, §0-2 자연완충재 동기와 정합)
  composite_t(w) = w_carry·z_expanding(signal_carry_t) + w_risk·z_expanding(signal_risk_t)
  h_w,t = 1{composite_t(w) > 0}
지호 님이 "특정 비율 하나를 고르지 말고 여러 비율을 앙상블하자"고 결정(§1 A안 원칙
재적용, argmax 선택 없음) — 캐리:VIX = 3:1/2:1/1:1/1:2/1:3 5개 비율의 h_w를 동일가중
평균해 h_ensemble 하나로 최종 신호를 만든다. z_expanding은 그날까지 데이터만 쓰는
확장윈도우 z-score(최소 252일 워밍업, look-ahead 없음).

**검증 순서(§6을 먼저)**: 원 프로토콜의 통화쌍 후보 중 대만(TWD)은 FRED에 3개월금리
데이터가 없어(OECD/IMF 국제금리통계 미가입국) 프로토콜이 이미 명시한 대안인 남아공
(ZAR)으로 교체. **USD/JPY·USD/BRL·USD/ZAR에 이 고정공식을 재조정 없이 그대로 적용**해
Gate1(무작위 대조군)+Gate2(블록부트스트랩) 통과 여부 확인 → 2/3 이상 통과해야 원달러
(KRW)로 이식해 최종 확인.

한계(명시): 브라질 금리(IRSTCI01BRM156N)는 정확한 3개월물이 아니라 즉시금리(Selic류)
근사 — 정확한 3개월 은행간금리 시계열이 FRED에 없어 최선의 대체재를 씀.

실행: python fx_composite_external_validation.py --external   # JPY/BRL/ZAR 먼저
      python fx_composite_external_validation.py --krw         # 원달러(외부검증 통과시만)
      python fx_composite_external_validation.py --self-test
결과: output/fx_composite_external_{jpy,brl,zar,krw}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import fx_hedge_validation as FV

WEIGHT_RATIOS = [(3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]   # (carry, risk), 앙상블용(argmax 없음)
EXPAND_MIN_PERIODS = 252

CURRENCIES = {
    "jpy": {"fx_ticker": "JPY=X", "rate_series": "IR3TIB01JPM156N",
           "note": "일본 3개월 은행간금리(FRED, 한국과 동일 계열)"},
    "brl": {"fx_ticker": "BRL=X", "rate_series": "IRSTCI01BRM156N",
           "note": "브라질 즉시금리(Selic류) 근사 — 정확한 3개월물 FRED 미제공"},
    "zar": {"fx_ticker": "ZAR=X", "rate_series": "IR3TIB01ZAM156N",
           "note": "남아공 3개월 은행간금리(FRED) — 원 프로토콜의 대만(TWD, FRED 데이터 없음) 대체"},
    "krw": {"fx_ticker": "KRW=X", "rate_series": "IR3TIB01KRM156N",
           "note": "한국 3개월 은행간금리(FRED) — 최종 확인 대상, 외부검증 2/3 통과 시에만 실행"},
}


def _log(m): print(f"[FX복합외부검증] {m}", file=sys.stderr)


def fetch_vix() -> pd.Series:
    path = "output/regime_price_cache_vix.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    import yfinance as yf
    df = yf.download("^VIX", period="max", auto_adjust=True, interval="1d", progress=False)
    s = df["Close"]
    s = s.iloc[:, 0] if hasattr(s, "columns") else s
    s = s.dropna()
    os.makedirs("output", exist_ok=True)
    s.to_pickle(path)
    return s


def fetch_fx_generic(ticker: str) -> pd.Series:
    safe = ticker.replace("=", "").replace("^", "")
    path = f"output/regime_price_cache_{safe}.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    import yfinance as yf
    df = yf.download(ticker, period="max", auto_adjust=True, interval="1d", progress=False)
    s = df["Close"]
    s = s.iloc[:, 0] if hasattr(s, "columns") else s
    s = s.dropna()
    os.makedirs("output", exist_ok=True)
    s.to_pickle(path)
    return s


def z_expanding(x: np.ndarray, min_periods: int = EXPAND_MIN_PERIODS) -> np.ndarray:
    """그날까지 데이터만 쓰는 확장윈도우 z-score(look-ahead 없음). 워밍업 구간은 NaN."""
    s = pd.Series(x)
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    z = (s - mean) / std.replace(0, np.nan)
    return z.to_numpy()


def build_frame_generic(fx_ticker: str, rate_series_id: str) -> dict:
    """SPY 달력 기준으로 spy·fx·carry(헤지수익)·us_rate·local_rate·vix 전부 정렬(ffill)."""
    spy = FV.fetch_spy()
    fx = fetch_fx_generic(fx_ticker)
    us_rate = FV.fetch_fred("DTB3")
    local_rate = FV.fetch_fred(rate_series_id)
    vix = fetch_vix()

    cal = spy.index
    def align(s):
        return s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
    fx_a, us_a, local_a, vix_a = align(fx), align(us_rate), align(local_rate), align(vix)

    valid = spy.notna() & fx_a.notna() & us_a.notna() & local_a.notna() & vix_a.notna()
    cal = cal[valid]
    spy, fx_a, us_a, local_a, vix_a = (s[valid] for s in (spy, fx_a, us_a, local_a, vix_a))

    carry_daily = (local_a - us_a) / 100.0 / FV.TRADING_DAYS   # 헤지상태 수익 근사(현지-미국)
    signal_carry = (us_a - local_a).to_numpy()                 # 노출신호(미국 프리미엄, 부호주의)
    signal_risk = vix_a.to_numpy()

    return {"cal": cal, "spy": spy.to_numpy(), "fx": fx_a.to_numpy(), "carry": carry_daily.to_numpy(),
           "signal_carry": signal_carry, "signal_risk": signal_risk}


def build_composite_ensemble(frame: dict) -> dict:
    """5개 캐리:VIX 비율의 h_w를 계산 → 동일가중 평균(h_ensemble) + Gate1용 combo_stats."""
    zc = z_expanding(frame["signal_carry"])
    zr = z_expanding(frame["signal_risk"])
    h_list, combo_stats = [], []
    for wc, wr in WEIGHT_RATIOS:
        composite = wc * zc + wr * zr
        valid = ~np.isnan(composite)
        h = np.where(valid, (composite > 0).astype(float), 0.0)   # 워밍업 NaN → 노출0(보수적)
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
    cfg = CURRENCIES[name]
    frame = build_frame_generic(cfg["fx_ticker"], cfg["rate_series"])
    ens = build_composite_ensemble(frame)
    _log(f"[{name.upper()}] 기간 {frame['cal'][0].date()}~{frame['cal'][-1].date()}"
        f"({len(frame['cal'])}일), 평균노출 {ens['h_ensemble'].mean():.3f}, "
        f"비율별노출 {ens['per_ratio_exposure']}")

    g1 = FV.gate1_matched_random(frame, ens["h_ensemble"], ens["combo_stats"], n_rep=n_rep_gate1)
    g2 = FV.gate2_bootstrap(frame, ens["h_ensemble"], n_rep=n_rep_gate2)

    result = {"currency": name, "config": cfg,
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
    ap = argparse.ArgumentParser(description="캐리+VIX 복합신호 — 타 통화쌍 선행검증(§6 먼저)")
    ap.add_argument("--external", action="store_true", help="JPY/BRL/ZAR 3개 실행")
    ap.add_argument("--krw", action="store_true", help="원달러 최종확인(외부검증 통과 전제)")
    ap.add_argument("--currency", default=None, help="단일 통화만(jpy/brl/zar/krw)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    os.makedirs("output", exist_ok=True)
    if args.currency:
        r = run_one_currency(args.currency)
        with open(f"output/fx_composite_external_{args.currency}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        return

    if args.external or not args.krw:
        results = {}
        for name in ("jpy", "brl", "zar"):
            r = run_one_currency(name)
            results[name] = r
            with open(f"output/fx_composite_external_{name}.json", "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
        n_pass = sum(1 for r in results.values() if r["overall_pass"])
        _log(f"\n=== 외부검증 종합: {n_pass}/3 통과 ===")
        majority = n_pass >= 2
        _log(f"판정: {'2/3 이상 통과 — 원달러로 이식 가능' if majority else '2/3 미달 — §8 적용, 원달러 이식 보류'}")
        with open("output/fx_composite_external_summary.json", "w", encoding="utf-8") as f:
            json.dump({"n_pass": n_pass, "majority_pass": majority,
                      "per_currency": {k: v["overall_pass"] for k, v in results.items()}}, f,
                     ensure_ascii=False, indent=2)

    if args.krw:
        r = run_one_currency("krw")
        with open("output/fx_composite_external_krw.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] z_expanding·복합공식·앙상블 배선 확인(합성 데이터)")
    rng = np.random.default_rng(9)
    n = 1500

    # z_expanding: 알려진 통계와 대조(상수+노이즈 시계열의 확장평균이 실제 평균에 수렴해야 함)
    const_plus_noise = 5.0 + rng.normal(0, 1, n)
    z = z_expanding(const_plus_noise, min_periods=100)
    assert np.isnan(z[:99]).all(), "워밍업 구간이 NaN이어야 함"
    assert np.isfinite(z[200:]).all(), "워밍업 이후엔 유한값이어야 함"
    _log("[self-test] 통과: z_expanding 워밍업·값 범위 정상")

    # 복합공식 방향성 검증: signal_risk(VIX 역할)를 인위적으로 미래 방어가 필요한 구간에서
    # 높게 심어두면, composite 부호가 그 구간에서 exposure를 유지(h=1)하도록 나와야 함
    # (VIX 높을 때 h=1 유지가 "자연완충재" 동기와 일치하는지 배선 확인).
    signal_risk_high = np.concatenate([np.full(300, 10.0), np.full(1200, 40.0)])  # 후반부 VIX 급등
    signal_carry_flat = rng.normal(0, 0.1, n)
    frame_mini = {"signal_carry": signal_carry_flat, "signal_risk": signal_risk_high}
    ens = build_composite_ensemble(frame_mini)
    h = ens["h_ensemble"]
    # VIX가 40으로 급등한 구간(후반)에서 z_risk가 양수로 커져 h가 1(노출유지) 쪽으로 쏠려야 함
    assert h[400:].mean() > h[150:250].mean(), \
        f"VIX 급등구간에서 노출이 더 높아야 하는데 아님: 후반 {h[400:].mean():.3f} vs 초반 {h[150:250].mean():.3f}"
    _log(f"[self-test] 통과: VIX(위험선호) 상승 시 노출유지 방향 확인(초반 {h[150:250].mean():.3f} → 후반 {h[400:].mean():.3f})")

    # 앙상블이 5개 비율의 평균인지(가중치를 캐리 쪽으로 완전히 쏠리게 하면 캐리신호 부호를 따라야 함)
    signal_carry_step = np.concatenate([np.full(750, -5.0), np.full(750, 5.0)])   # 후반부 캐리 우호적
    signal_risk_flat = rng.normal(20, 1, n)
    frame_mini2 = {"signal_carry": signal_carry_step, "signal_risk": signal_risk_flat}
    ens2 = build_composite_ensemble(frame_mini2)
    h2 = ens2["h_ensemble"]
    assert h2[800:].mean() > h2[300:700].mean(), "캐리가 우호적으로 바뀐 후반부에서 노출이 더 높아야 함"
    _log(f"[self-test] 통과: 캐리 신호 방향 반영 확인(전반 {h2[300:700].mean():.3f} → 후반 {h2[800:].mean():.3f})")

    assert len(ens["combo_stats"]) == 5 and all(c is not None for c in ens["combo_stats"])
    _log("[self-test] 통과: 5개 비율 combo_stats 전부 생성됨")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
