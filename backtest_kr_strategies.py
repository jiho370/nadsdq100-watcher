#!/usr/bin/env python3
"""
backtest_kr_strategies.py — KR_STRATEGY_OPTIONS.md Phase 3: 국장 전략 후보 포트폴리오 판정.

Phase 0 진단(regime_kr: broad 레짐에서 저PBR·저변동성·배당 IC 강함, narrow에서 전멸)과
§2-A/C/D의 ex-ante 근거로 **사전 등록한 6개 트라이얼만** 비교한다(추가·수정 금지):

  T1 live            현행 재현: live 필터 + z(mom12_1)·0.6 + z(hi52)·0.4   [베이스라인]
  T2 valuediv        z(pbr_inv)+z(value)+z(div_yield)  — §2-A 저PBR×환원 계열
  T3 valuediv_lowvol T2 + z(low_vol)                   — 레짐 진단 상위 팩터 보강
  T4 valuediv_flow   T2 + z(frgn_flow)                 — §2-C 외국인 수급 결합
  T5 flow            z(frgn_flow) 단독                 — §2-C 단독 검증
  T6 valuediv_regime T2 + R1=narrow(쏠림장)이면 결정 스킵(신규 편입 중단) — §2-E 오버레이

공통: topn=6(현행 상한) · 63일 결정 격자 · 풀 30 · 동일 체결 엔진(backtest_portfolio.simulate
— 동일비중 진입, 6개월 재평가 + 200일선 -3% 매도) · CostModel("kospi").
필터: T1은 live 필터 그대로, T2~T6은 흑자(roe>0)만 — 밸류 전략에 추세 필터를 강제하지
않기 위함(단, 엔진의 200일선 매도는 전 트라이얼 공통 — 비교 공정성 우선).

판정(§1 벤치마크 재정의): **B2(코스피200 동일가중) 대비 월간 초과수익**으로 PBO/DSR.
B1(시총가중) 대비는 참고 표기(B1은 코어+새틀라이트 합산의 목표).

실행: python backtest_kr_strategies.py          # output/backtest_kr_strategies.json
      python backtest_kr_strategies.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP

OUT_PATH = "output/backtest_kr_strategies.json"
PBO_PATH = "output/pbo_report_kr_strategies.json"
POOL_KR = 30
TOPN = 6

TRIALS = {
    "live": {"mom12_1": 0.6, "hi52_prox": 0.4},
    "valuediv": {"pbr_inv": 1, "value": 1, "div_yield": 1},
    "valuediv_lowvol": {"pbr_inv": 1, "value": 1, "div_yield": 1, "low_vol": 1},
    "valuediv_flow": {"pbr_inv": 1, "value": 1, "div_yield": 1, "frgn_flow": 1},
    "flow": {"frgn_flow": 1},
    "valuediv_regime": {"pbr_inv": 1, "value": 1, "div_yield": 1},   # + narrow 스킵
}


def _log(m): print(f"[전략KR] {m}", file=sys.stderr)


def build_decisions(panel, snaps, trial: str, narrow_dates: set[str] | None = None) -> list:
    """스냅샷 → (패널 위치, 순위 상위 POOL_KR) 결정 리스트."""
    pos_by_date = {d.date().isoformat(): i for i, d in enumerate(panel.index)}
    w = TRIALS[trial]
    out = []
    for s in snaps:
        p = pos_by_date.get(s["date"])
        if p is None:
            continue
        if trial == "valuediv_regime" and narrow_dates and s["date"] in narrow_dates:
            continue                                   # 쏠림장: 결정 스킵(보유 유지)
        if trial == "live":
            ok = s["live_ok"]
            pool = ok[ok].index
        else:
            roe = s["raw"]["roe"]
            pool = roe[roe > 0].index                  # 흑자 필터만
        if len(pool) < TOPN + 2:
            continue
        z = s["z"].loc[pool]
        score = sum(z[f] * wt for f, wt in w.items())
        out.append((p, list(score.sort_values(ascending=False).index[:POOL_KR])))
    return out


def load_narrow_dates() -> set[str]:
    try:
        with open("output/regime_kr.json", encoding="utf-8") as f:
            lab = json.load(f)["labels_daily"]
        return {d for d, v in lab.items() if v.get("R1") == "narrow"}
    except Exception:
        _log("regime_kr.json 없음 — T6은 스킵 없이 T2와 동일해짐(주의)")
        return set()


def run(save=True):
    from benchmarks_kr import load_research_data, load_benchmarks
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    navs_bm = load_benchmarks()
    b1 = navs_bm["B1_kospi200"].reindex(panel.index).ffill()
    b2 = navs_bm["B2_equal"].reindex(panel.index).ffill()
    narrow = load_narrow_dates()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    rows, matrix, dates0, navs_out = [], [], None, {}
    for name in TRIALS:
        decisions = build_decisions(panel, snaps, name, narrow)
        nav = BP.simulate(panel, ma200, decisions, TOPN, cost)
        if nav is None:
            _log(f"{name}: NAV 산출 실패"); continue
        navs_out[name] = nav
        m1 = BP.metrics(nav, b1)
        m2 = BP.metrics(nav, b2)
        d, r = BP.monthly_excess(nav, b2)              # 판정은 B2 기준(§1)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        rows.append({"trial": name, "n_decisions": len(decisions),
                     "cagr_pct": m1["cagr_pct"], "vol_pct": m1["vol_pct"],
                     "sharpe": m1["sharpe"], "mdd_pct": m1["mdd_pct"],
                     "excess_vs_B1": m1["excess_cagr_pct"],
                     "excess_vs_B2": m2["excess_cagr_pct"],
                     "bench_B1_cagr": m1["bench_cagr_pct"], "bench_B2_cagr": m2["bench_cagr_pct"]})
        _log(f"{name:18s} CAGR {m1['cagr_pct']:6.2f}% 샤프 {m1['sharpe']:5.2f} "
             f"MDD {m1['mdd_pct']:6.1f}% · vs B1 {m1['excess_cagr_pct']:+6.2f}%p · "
             f"vs B2 {m2['excess_cagr_pct']:+6.2f}%p")
    if len(rows) < 2:
        raise RuntimeError("트라이얼 부족")
    n_ev = min(len(r) for r in matrix)
    trial_data = {"horizon": "kr_strategies", "universe": "kospi200_pit",
                  "cost": cost.describe(), "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                  "dates": dates0[:n_ev], "trials": [r["trial"] for r in rows],
                  "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "pool": POOL_KR,
               "budget": "사전 등록 6트라이얼 — 추가·수정·부활 금지(KR_STRATEGY_OPTIONS §6)",
               "judgment": "B2(동일가중) 대비 샤프·PBO/DSR — B1 대비는 참고(§1 판정 규칙)",
               "rows": rows,
               "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
               "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
               "passed": rpt.get("passed", False)}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(PBO_PATH, "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        with open("output/trial_returns_kr_strategies.json", "w", encoding="utf-8") as f:
            json.dump(trial_data, f, ensure_ascii=False)
        with open("output/kr_strategy_navs.json", "w", encoding="utf-8") as f:
            json.dump({n: {d.date().isoformat(): round(float(v), 6) for d, v in s.items()}
                       for n, s in navs_out.items()}, f, ensure_ascii=False)
        _log(f"저장: {OUT_PATH} · PBO {payload['pbo']} · DSR {payload['dsr']} · "
             f"passed={payload['passed']}")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성: 저PBR 신호를 심은 스냅샷으로 결정 빌더·트라이얼 배선 검증")
    rng = np.random.default_rng(17)
    n_days, n_syms = 700, 30
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_syms)]
    quality = rng.normal(0, 1, n_syms)
    panel = pd.DataFrame(100 * np.exp(np.cumsum(
        rng.normal(0.0002 + 0.0005 * quality, 0.015, (n_days, n_syms)), axis=0)),
        index=idx, columns=cols)
    snaps = []
    for p in range(260, n_days - 130, 63):
        raw = pd.DataFrame({
            "pbr_inv": quality + rng.normal(0, 0.3, n_syms),
            "value": quality + rng.normal(0, 0.3, n_syms),
            "div_yield": rng.normal(0, 1, n_syms),
            "low_vol": rng.normal(0, 1, n_syms),
            "frgn_flow": rng.normal(0, 1, n_syms),
            "mom12_1": rng.normal(0, 1, n_syms),
            "hi52_prox": rng.normal(0, 1, n_syms),
            "roe": np.abs(quality) + 0.01,
        }, index=cols)
        fwd = {h: panel.iloc[min(p + 1 + hd, n_days - 1)] / panel.iloc[p + 1] - 1
               for h, hd in BW.TD.items()}
        snaps.append({"date": idx[p].date().isoformat(), "raw": raw,
                      "z": raw.apply(BW._z).fillna(0.0), "fwd": fwd,
                      "live_ok": pd.Series(True, index=cols)})
    dec = build_decisions(panel, snaps, "valuediv")
    assert len(dec) >= 5 and all(len(r) <= POOL_KR for _, r in dec)
    # 심은 신호: valuediv 상위 풀의 평균 품질 > 전체 평균
    top_q = np.mean([quality[cols.index(t)] for t in dec[0][1][:6]])
    assert top_q > 0.3, f"저PBR 신호 선별 실패: {top_q}"
    # narrow 스킵 배선
    dec_r = build_decisions(panel, snaps, "valuediv_regime",
                            narrow_dates={snaps[0]["date"], snaps[1]["date"]})
    assert len(dec_r) == len(dec) - 2, (len(dec_r), len(dec))
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    nav = BP.simulate(panel, ma200, dec, TOPN, cost)
    assert nav is not None and np.isfinite(nav.iloc[-1])
    _log(f"[self-test] 통과: 결정 {len(dec)}개 · 상위풀 품질 {top_q:+.2f} · NAV {nav.iloc[-1]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="국장 전략 후보 6종 포트폴리오 판정(Phase 3)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else run()
