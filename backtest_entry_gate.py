#!/usr/bin/env python3
"""
backtest_entry_gate.py — "매수/관찰 분류 게이트"(200일선 위 & 52주고점 -25% 이내)가
이미 채택된 퀄리티 바스켓의 성과를 실제로 바꾸는지 A/B 검증 (지호 님 요청, 2026-07).

배경: export_data.split_by_entry()의 entry_ok() 필터는 STRATEGY.md에 문헌(Faber/
Zakamulin의 200일선 추세추종, George-Hwang 2004의 52주고점 근접도)으로만 근거가 있고,
이 시스템 자체 데이터로 "이 게이트를 켰을 때 이미 검증된 퀄리티 바스켓 성과가 실제로
개선되는가"는 한 번도 테스트된 적이 없다. 이걸 직접 검증한다.

방법: backtest_costs.py와 동일한 PIT 이벤트·비용모델·채택 가중치(int_gp_assets1·
rd_mktcap2·shareholder_yield2, output/best_weights.json)를 그대로 재사용. 매 이벤트에서
팩터 점수로 순위를 매긴 뒤,
  [baseline] 상위 N종목 그대로(현재 backtest_costs.py가 검증한 것과 동일)
  [gated]    순위대로 내려가며 entry_ok(종가>200일선 & 52주고점 -25%이내)를 만족하는
             종목만 채택, 부족분은 게이트 없이 순위 다음 종목으로 보충(라이브의
             split_by_entry와 동일 정책 — 필터 통과 종목이 부족할 때만 완화)
두 바스켓의 이벤트별 순초과수익 차이를 이벤트당(paired) 비교. 이건 "여러 후보 중 최선을
고르는" 다중검정 문제가 아니라 이미 정해진 규칙 하나(게이트 on/off)의 A/B 검증이므로
PBO는 해당 없음 — 리밸·보유 중첩만 T_eff로 보정한 paired t-test로 판정.

실행: python backtest_entry_gate.py --years 10
      python backtest_entry_gate.py --self-test
결과: output/backtest_entry_gate.json
"""
from __future__ import annotations
import os, sys, json, math, argparse
from statistics import NormalDist
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC

OUT_PATH = "output/backtest_entry_gate.json"
MA_WINDOW = 200
HI_WINDOW = 252
HI_BAND_PCT = -0.25   # George-Hwang: 52주고점 대비 -25% 이내


def _log(m): print(f"[진입게이트] {m}", file=sys.stderr)


def _load_weights():
    try:
        with open("output/best_weights.json", encoding="utf-8") as f:
            w = (json.load(f).get("weights") or {})
        if any(w.values()):
            return {k: v for k, v in w.items() if v}
    except Exception:
        pass
    return {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}   # 현재 채택 가중치 폴백


def _entry_ok(panel, p, idx):
    """entry_ok = 종가>200일선 & 52주고점 대비 -25% 이내 (export_data.split_by_entry와 동일 정의)."""
    price = panel.iloc[p][idx]
    ma200 = panel.iloc[p - MA_WINDOW + 1:p + 1][idx].mean()
    hi52 = panel.iloc[p - HI_WINDOW + 1:p + 1][idx].max()
    above = price > ma200
    near_high = (price / hi52 - 1) >= HI_BAND_PCT
    return (above & near_high).fillna(False)


def run(panel, spy, funds, pit, cost, weights, topn=30, rebal_days=63, opens=None):
    import tech_factors as T
    cross = T.build_panels(panel)
    if opens is not None:
        opens = opens.reindex_like(panel)
        on_cum = (1 + (opens / panel.shift(1) - 1).fillna(0)).cumprod()
        cross["overnight_mom"] = on_cum.shift(21) / on_cum.shift(252) - 1
    snaps = BC.build_snaps(panel, spy, funds, opens, rebal_days)
    pit_snaps, cov = BC._filter_snaps(snaps, pit, "pit")
    if len(pit_snaps) < 8:
        raise RuntimeError(f"이벤트 부족({len(pit_snaps)})")

    cols = [c for c in weights if weights.get(c)]
    wv = pd.Series(weights)
    events = []
    for s in pit_snaps:
        p = int(panel.index.get_indexer([pd.Timestamp(s["date"])])[0])
        if p < HI_WINDOW:
            continue
        idx = s["raw"].index
        avail_cols = [c for c in cols if c in s["raw"].columns]
        if not avail_cols:
            continue
        score = (s["z"][avail_cols] * wv[avail_cols]).sum(axis=1).sort_values(ascending=False)
        ranked = list(score.index)
        base_top = ranked[:topn]

        ok = _entry_ok(panel, p, idx).reindex(idx).fillna(False)
        gated_survivors = [t for t in ranked if ok.get(t, False)]
        gate_top = gated_survivors[:topn]
        filled_from_gate = len(gate_top)
        if len(gate_top) < topn:   # 라이브 정책과 동일: 부족분은 게이트 없이 순위대로 보충
            gate_top += [t for t in ranked if t not in gate_top][:topn - len(gate_top)]

        row = {"date": s["date"], "gate_pass_pct": round(100 * filled_from_gate / topn, 1),
              "basket_overlap_pct": round(100 * len(set(base_top) & set(gate_top)) / topn, 1)}
        for h in ("6m", "12m"):
            fwd = s["fwd"][h]
            for tag, basket in (("base", base_top), ("gate", gate_top)):
                r = fwd.reindex(basket).dropna()
                net = float(np.mean([cost.net(x) for x in r])) if len(r) else None
                row[f"{tag}_excess_{h}"] = (net - s["bench"][h]) if net is not None else None
        events.append(row)

    if len(events) < 8:
        raise RuntimeError(f"유효 이벤트 부족({len(events)})")

    def _paired_stat(h):
        d = np.array([e[f"gate_excess_{h}"] - e[f"base_excess_{h}"] for e in events
                     if e[f"gate_excess_{h}"] is not None and e[f"base_excess_{h}"] is not None])
        T_raw = len(d)
        t_eff = max(int(round(T_raw * min(rebal_days / BW.TD[h], 1.0))), 3)
        mean_d, sd_d = float(d.mean()), float(d.std(ddof=1)) if T_raw > 1 else 0.0
        se = sd_d / math.sqrt(t_eff) if sd_d > 0 else 0.0
        t_stat = mean_d / se if se > 0 else 0.0
        p_value = 2 * (1 - NormalDist().cdf(abs(t_stat))) if se > 0 else 1.0
        base_mean = float(np.mean([e[f"base_excess_{h}"] for e in events if e[f"base_excess_{h}"] is not None]))
        gate_mean = float(np.mean([e[f"gate_excess_{h}"] for e in events if e[f"gate_excess_{h}"] is not None]))
        return {"base_excess_pct": round(100 * base_mean, 2), "gate_excess_pct": round(100 * gate_mean, 2),
               "diff_pct": round(100 * mean_d, 2), "n_events_raw": T_raw, "n_eff": t_eff,
               "t_stat": round(t_stat, 2), "p_value_approx": round(p_value, 4),
               "significant_at_5pct": bool(se > 0 and p_value < 0.05)}

    result_6m, result_12m = _paired_stat("6m"), _paired_stat("12m")
    avg_gate_pass = round(float(np.mean([e["gate_pass_pct"] for e in events])), 1)
    avg_overlap = round(float(np.mean([e["basket_overlap_pct"] for e in events])), 1)

    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "n_events": len(events),
              "gate_definition": f"종가>{MA_WINDOW}일선 & 52주고점 대비 {HI_BAND_PCT:+.0%} 이내",
              "avg_gate_pass_pct": avg_gate_pass,
              "avg_basket_overlap_pct": avg_overlap,
              "result_6m": result_6m, "result_12m": result_12m,
              "events": events,
              "note": ("baseline=현재 검증된 퀄리티 바스켓(게이트 없음) vs gate=동일 바스켓에 라이브의 "
                      "매수/관찰 분류 필터(200일선·52주고점) 적용. 단일 규칙의 A/B 검증이라 PBO(다중검정 "
                      "과최적화 검사)는 해당 없음 — 리밸·보유 중첩만 T_eff로 보정한 paired t-test로 판정.")}
    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {OUT_PATH} (이벤트 {len(events)}개 · 평균 게이트통과율 {avg_gate_pass}% · "
         f"바스켓 겹침 {avg_overlap}%)")
    _log(f"  6M: base {result_6m['base_excess_pct']}%p vs gate {result_6m['gate_excess_pct']}%p "
         f"(차이 {result_6m['diff_pct']}%p, t={result_6m['t_stat']}, p≈{result_6m['p_value_approx']}, "
         f"유의={result_6m['significant_at_5pct']})")
    _log(f"  12M: base {result_12m['base_excess_pct']}%p vs gate {result_12m['gate_excess_pct']}%p "
         f"(차이 {result_12m['diff_pct']}%p, t={result_12m['t_stat']}, p≈{result_12m['p_value_approx']}, "
         f"유의={result_12m['significant_at_5pct']})")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 게이트 A/B 로직 점검")
    panel, spy, funds, opens = BW._synthetic()
    pit = BC._synthetic_pit(panel)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = {"gross_margin": 1, "shareholder_yield": 1}
    payload = run(panel, spy, funds, pit, cost, weights, topn=15, rebal_days=63, opens=opens)
    assert payload["n_events"] > 4
    assert 0 <= payload["avg_gate_pass_pct"] <= 100
    assert "t_stat" in payload["result_6m"] and "t_stat" in payload["result_12m"]

    # 극단 케이스: 항상 게이트 통과(전부 above_ma200且근접) → base==gate 바스켓 100% 겹침
    n = 1100
    idx = pd.bdate_range("2020-01-01", periods=n)
    up = pd.DataFrame({f"S{i:02d}": 100 * np.exp(np.cumsum(np.full(n, 0.0015 + i * 0.0002)))
                       for i in range(20)}, index=idx)
    spy2 = pd.Series(100 * np.exp(np.cumsum(np.full(n, 0.0005))), index=idx)
    pit2 = [(idx[0].date().isoformat(), frozenset(up.columns))]
    funds2 = None
    p2 = run(up, spy2, funds2, pit2, cost, {"mom6": 1}, topn=5, rebal_days=63)
    assert p2["avg_basket_overlap_pct"] == 100.0, "전종목 상승추세인데 게이트로 바스켓이 바뀜"
    _log(f"[self-test] 통과: 이벤트 {payload['n_events']} · 극단케이스 겹침 {p2['avg_basket_overlap_pct']}%")


def run_kr_gate(snaps, cost, topn=10, rebal_days=63):
    """국장 버전 — 국장 라이브 규칙엔 200일선이 이미 필터로 들어 있어, 여기서는 라이브에 없는
    '52주고점 -25% 이내'(George-Hwang) 게이트만 on/off로 A/B 한다(미장과 정의가 다름을 명기).
    baseline = live 필터+랭킹 topN / gate = 동일 랭킹에서 hi52_prox>=-25% 우선 채택."""
    events = []
    for s in snaps:
        pool = s["live_ok"][s["live_ok"]].index
        if len(pool) < topn:
            continue
        z = s["z"].loc[pool]
        score = (z["mom12_1"] * 0.6 + z["hi52_prox"] * 0.4).sort_values(ascending=False)
        ranked = list(score.index)
        base_top = ranked[:topn]
        near = s["raw"]["hi52_prox"].reindex(pool) >= -0.25
        gate_top = [t for t in ranked if bool(near.get(t, False))][:topn]
        filled = len(gate_top)
        if filled < topn:
            gate_top += [t for t in ranked if t not in gate_top][:topn - filled]
        row = {"date": s["date"], "gate_pass_pct": round(100 * filled / topn, 1),
               "basket_overlap_pct": round(100 * len(set(base_top) & set(gate_top)) / topn, 1)}
        for h in ("6m", "12m"):
            fwd = s["fwd"][h]
            for tag, basket in (("base", base_top), ("gate", gate_top)):
                r = fwd.reindex(basket).dropna()
                net = float(np.mean([cost.net(x) for x in r])) if len(r) else None
                row[f"{tag}_excess_{h}"] = (net - s["bench"][h]) if net is not None else None
        events.append(row)
    if len(events) < 8:
        raise RuntimeError(f"유효 이벤트 부족({len(events)})")

    import backtest_weights as _BW
    def _paired(h):
        d = np.array([e[f"gate_excess_{h}"] - e[f"base_excess_{h}"] for e in events
                      if e[f"gate_excess_{h}"] is not None and e[f"base_excess_{h}"] is not None])
        t_eff = max(int(round(len(d) * min(rebal_days / _BW.TD[h], 1.0))), 3)
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        se = sd / math.sqrt(t_eff) if sd > 0 else 0.0
        t = float(d.mean()) / se if se > 0 else 0.0
        p = 2 * (1 - NormalDist().cdf(abs(t))) if se > 0 else 1.0
        return {"base_excess_pct": round(100 * float(np.mean([e[f"base_excess_{h}"] for e in events])), 2),
                "gate_excess_pct": round(100 * float(np.mean([e[f"gate_excess_{h}"] for e in events])), 2),
                "diff_pct": round(100 * float(d.mean()), 2), "n_events_raw": len(d), "n_eff": t_eff,
                "t_stat": round(t, 2), "p_value_approx": round(p, 4),
                "significant_at_5pct": bool(se > 0 and p < 0.05)}

    payload = {"as_of": events[-1]["date"], "market": "kr", "topn": topn,
               "n_events": len(events),
               "gate_definition": "52주고점 대비 -25% 이내(George-Hwang)만 — 200일선은 라이브 필터에 이미 포함",
               "avg_gate_pass_pct": round(float(np.mean([e["gate_pass_pct"] for e in events])), 1),
               "avg_basket_overlap_pct": round(float(np.mean([e["basket_overlap_pct"] for e in events])), 1),
               "result_6m": _paired("6m"), "result_12m": _paired("12m"), "events": events,
               "note": "국장 라이브 랭킹(mom0.6+hi52 0.4) 고정, hi52 게이트만 on/off한 paired A/B"}
    out = OUT_PATH.replace(".json", "_kr.json")
    os.makedirs("output", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    r6, r12 = payload["result_6m"], payload["result_12m"]
    _log(f"저장: {out} (이벤트 {len(events)})")
    _log(f"  6M: base {r6['base_excess_pct']}%p vs gate {r6['gate_excess_pct']}%p (t={r6['t_stat']}, p≈{r6['p_value_approx']})")
    _log(f"  12M: base {r12['base_excess_pct']}%p vs gate {r12['gate_excess_pct']}%p (t={r12['t_stat']}, p≈{r12['p_value_approx']})")
    return payload


# ------------------------- 게이트 후보 스윕(2026-07: 현행 게이트가 유해 판정 → 대체 탐색) -------------------------
GATE_CANDIDATES = ["ma200_only", "hi52_near", "hi52_far", "pullback20", "no_overheat", "low_vol"]


def _gate_panels(panel):
    ma20 = panel.rolling(20, min_periods=20).mean()
    ma50 = panel.rolling(50, min_periods=50).mean()
    ma200 = panel.rolling(200, min_periods=200).mean()
    hi52 = panel.rolling(252, min_periods=60).max()
    vol60 = panel.pct_change(fill_method=None).rolling(60, min_periods=30).std()
    delta = panel.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return {"ma20": ma20, "ma50": ma50, "ma200": ma200, "hi52": hi52, "vol60": vol60, "rsi": rsi}


def _gate_masks(gp, panel, p, idx):
    price = panel.iloc[p][idx]
    hi_gap = price / gp["hi52"].iloc[p][idx] - 1
    gap50 = price / gp["ma50"].iloc[p][idx] - 1
    vol = gp["vol60"].iloc[p][idx]
    return {
        "ma200_only": (price > gp["ma200"].iloc[p][idx]).fillna(False),
        "hi52_near": (hi_gap >= -0.25).fillna(False),
        "hi52_far": (hi_gap < -0.25).fillna(False),
        "pullback20": (price < gp["ma20"].iloc[p][idx]).fillna(False),
        "no_overheat": ((gp["rsi"].iloc[p][idx] < 72) & (gap50 < 0.15)).fillna(False),
        "low_vol": (vol <= vol.median()).fillna(False),
    }


def run_sweep(panel, spy, funds, pit, cost, weights, topn=30, rebal_days=63, opens=None):
    """후보 게이트 6종을 각각 baseline(게이트 없음) 대비 paired A/B — 부족분 순위 보충 정책 동일.
    ⚠ 6개를 동시에 시험하므로 다중검정: 채택은 t≥2.7(≈본페로니 5%/6) & diff>0 일 때만 논의."""
    import tech_factors as T
    cross = T.build_panels(panel)
    snaps = BC.build_snaps(panel, spy, funds, opens, rebal_days)
    pit_snaps, _ = BC._filter_snaps(snaps, pit, "pit")
    if len(pit_snaps) < 8:
        raise RuntimeError(f"이벤트 부족({len(pit_snaps)})")
    gp = _gate_panels(panel)
    cols = [c for c in weights if weights.get(c)]
    wv = pd.Series(weights)
    acc = {g: {"d6": [], "d12": [], "pass": [], "ovl": []} for g in GATE_CANDIDATES}
    base6, base12 = [], []
    n_ev = 0
    for s in pit_snaps:
        p = int(panel.index.get_indexer([pd.Timestamp(s["date"])])[0])
        if p < HI_WINDOW:
            continue
        idx = s["raw"].index
        avail = [c for c in cols if c in s["raw"].columns]
        if not avail:
            continue
        score = (s["z"][avail] * wv[avail]).sum(axis=1).sort_values(ascending=False)
        ranked = list(score.index)
        base_top = ranked[:topn]

        def _net_excess(basket, h):
            r = s["fwd"][h].reindex(basket).dropna()
            if not len(r):
                return None
            return float(np.mean([cost.net(x) for x in r])) - s["bench"][h]

        b6, b12 = _net_excess(base_top, "6m"), _net_excess(base_top, "12m")
        if b6 is None or b12 is None:
            continue
        masks = _gate_masks(gp, panel, p, idx)
        for g in GATE_CANDIDATES:
            ok = masks[g]
            gate_top = [t for t in ranked if bool(ok.get(t, False))][:topn]
            filled = len(gate_top)
            if filled < topn:
                gate_top += [t for t in ranked if t not in gate_top][:topn - filled]
            g6, g12 = _net_excess(gate_top, "6m"), _net_excess(gate_top, "12m")
            if g6 is None or g12 is None:
                continue
            acc[g]["d6"].append(g6 - b6); acc[g]["d12"].append(g12 - b12)
            acc[g]["pass"].append(100 * filled / topn)
            acc[g]["ovl"].append(100 * len(set(base_top) & set(gate_top)) / topn)
        base6.append(b6); base12.append(b12)
        n_ev += 1

    def _stat(dlist, h):
        d = np.array(dlist)
        t_eff = max(int(round(len(d) * min(rebal_days / BW.TD[h], 1.0))), 3)
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        se = sd / math.sqrt(t_eff) if sd > 0 else 0.0
        t = float(d.mean()) / se if se > 0 else 0.0
        return round(100 * float(d.mean()), 2), round(t, 2), round(2 * (1 - NormalDist().cdf(abs(t))), 4) if se > 0 else 1.0

    rows = []
    for g in GATE_CANDIDATES:
        if len(acc[g]["d6"]) < 8:
            continue
        d6, t6, p6 = _stat(acc[g]["d6"], "6m")
        d12, t12, p12 = _stat(acc[g]["d12"], "12m")
        rows.append({"gate": g, "diff_6m_pct": d6, "t_6m": t6, "p_6m": p6,
                     "diff_12m_pct": d12, "t_12m": t12, "p_12m": p12,
                     "avg_pass_pct": round(float(np.mean(acc[g]["pass"])), 1),
                     "avg_overlap_pct": round(float(np.mean(acc[g]["ovl"])), 1),
                     "adopt_candidate": bool(d6 > 0 and t6 >= 2.7)})
    rows.sort(key=lambda r: r["diff_6m_pct"], reverse=True)
    payload = {"as_of": panel.index[-1].date().isoformat(), "n_events": n_ev, "topn": topn,
               "baseline_excess_6m_pct": round(100 * float(np.mean(base6)), 2),
               "baseline_excess_12m_pct": round(100 * float(np.mean(base12)), 2),
               "rows": rows,
               "adoption_rule": "6개 동시 시험 — 채택 논의는 diff>0 & t≥2.7(본페로니 5%/6 근사)일 때만",
               "note": "게이트별 paired 차이(게이트 적용 − 게이트 없음). 음수=게이트가 해로움."}
    out = OUT_PATH.replace(".json", "_sweep.json")
    os.makedirs("output", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {out} (이벤트 {n_ev} · baseline 6M {payload['baseline_excess_6m_pct']}%p)")
    for r in rows:
        _log(f"  {r['gate']:12s} 6M {r['diff_6m_pct']:+6.2f}%p(t={r['t_6m']:+.2f}) "
             f"12M {r['diff_12m_pct']:+6.2f}%p · 통과율 {r['avg_pass_pct']}% · 후보={r['adopt_candidate']}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="매수/관찰 분류 게이트(200일선·52주고점) A/B 검증")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--market", default="us", choices=["us", "kr", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--sweep", action="store_true", help="게이트 후보 6종 스윕(미장)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.market == "kr":
        import backtest_kr as BK
        panel, membership, fundamentals, flows, mktcaps, bench = BK.prepare_kr_data(
            int(args.years), args.rebal_days)
        snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals, args.rebal_days)
        cost = BC.CostModel("kospi", max(args.commission_bps, 1.5), args.slippage_bps)
        run_kr_gate(snaps, cost, topn=min(args.topn, 10), rebal_days=args.rebal_days)
        return
    weights = _load_weights()
    pit = BC.load_pit(args.pit_file)
    panel, spy, opens = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel(args.market, args.commission_bps, args.slippage_bps)
    if args.sweep:
        run_sweep(panel, spy, funds, pit, cost, weights, topn=args.topn,
                  rebal_days=args.rebal_days, opens=opens)
    else:
        run(panel, spy, funds, pit, cost, weights, topn=args.topn, rebal_days=args.rebal_days, opens=opens)


if __name__ == "__main__":
    main()
