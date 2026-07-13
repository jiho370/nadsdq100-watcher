#!/usr/bin/env python3
"""
factor_kr_flow.py — KR_STRATEGY_OPTIONS.md §2-C: 수급 팩터 정밀 IC 검증.

기존 backtest_kr의 frgn_flow(60일 창, 리밸일 조회)보다 정밀한 **일별 수급 패널**로
사전 등록 시그널 4종만 검증한다(다중검정 예산 — 이 4종 외 추가 금지, 탈락 부활 금지):

  F1 frgn_i20: 최근 20일 외국인 순매수액 / 시총 (강도)
  F2 frgn_p20: 최근 20일 중 외국인 순매수일 비율 (지속성)
  F3 frgn_i60: 최근 60일 강도 (기존 팩터 재확인용 대조군)
  F4 indiv_i20: 최근 20일 개인 순매수액 / 시총 (역신호 가설 — IC<0 기대, §0-S2)

판정(사전 등록): 지평 1m/3m/6m Spearman IC, 서브기간 3분할 중 2구간 이상 동일 부호
∧ 전기간 |t|≥2 일 때만 Phase 3 포트폴리오화 후보로 승격.

데이터: pykrx get_market_trading_value_by_date(종목별 일별, ~17초/종목 — 캐시 필수·재개 가능)
실행: python factor_kr_flow.py --collect     # 일별 수급 수집(중단 후 재실행 시 재개)
      python factor_kr_flow.py               # IC 분석 → output/factor_kr_flow.json
      python factor_kr_flow.py --self-test
"""
from __future__ import annotations
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd

FLOW_CACHE = "output/kr_flow_daily.pkl"
OUT_PATH = "output/factor_kr_flow.json"
SIGNALS = ["frgn_i20", "frgn_p20", "frgn_i60", "indiv_i20"]
HORIZONS = ["1m", "3m", "6m"]


def _log(m): print(f"[수급KR] {m}", file=sys.stderr)


# ------------------------- 수집(재개 가능) -------------------------
def _load_flow_cache() -> dict:
    if os.path.exists(FLOW_CACHE):
        with open(FLOW_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def collect(tickers: list[str], start8: str, end8: str):
    from pykrx import stock as K
    cache = _load_flow_cache()
    todo = [t for t in tickers if t not in cache]
    _log(f"수집 대상 {len(todo)}/{len(tickers)}종목 (캐시 {len(cache)}) — 종목당 ~17초")
    t0 = time.time()
    for i, t in enumerate(todo):
        try:
            df = K.get_market_trading_value_by_date(start8, end8, t)
            cols = {}
            for tag, pat in (("frgn", "외국인"), ("indiv", "개인"), ("inst", "기관")):
                c = next((c for c in df.columns if pat in str(c)), None)
                cols[tag] = df[c].astype(float) if c is not None else pd.Series(dtype=float)
            cache[t] = pd.DataFrame(cols)
        except Exception as e:
            _log(f"{t} 실패: {e} — 건너뜀(재실행 시 재시도)")
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            with open(FLOW_CACHE, "wb") as f:
                pickle.dump(cache, f)
            el = time.time() - t0
            _log(f"{i+1}/{len(todo)} 저장 (경과 {el/60:.1f}분, 잔여 ~{el/(i+1)*(len(todo)-i-1)/60:.0f}분)")
    return cache


# ------------------------- 시그널 -------------------------
def build_signals(flow: dict, panel_index: pd.DatetimeIndex, mktcaps: dict,
                  snap_dates: list[str], tickers_by_date: dict) -> dict:
    """{snap_date: DataFrame(index=tickers, columns=SIGNALS)}."""
    mc_keys = sorted(mktcaps.keys())
    out = {}
    for d in snap_dates:
        d8 = d.replace("-", "")
        mck = None
        for k in mc_keys:
            if k <= d8 and mktcaps.get(k):
                mck = k
        mc = mktcaps.get(mck) or {}
        ts = pd.Timestamp(d)
        rows = {}
        for t in tickers_by_date.get(d, []):
            df = flow.get(t)
            cap = mc.get(t)
            if df is None or df.empty or not cap:
                continue
            w = df.loc[:ts]
            if len(w) < 60:
                continue
            f20, f60 = w["frgn"].iloc[-20:], w["frgn"].iloc[-60:]
            i20 = w["indiv"].iloc[-20:]
            rows[t] = {"frgn_i20": float(f20.sum()) / cap,
                       "frgn_p20": float((f20 > 0).mean()),
                       "frgn_i60": float(f60.sum()) / cap,
                       "indiv_i20": float(i20.sum()) / cap}
        out[d] = pd.DataFrame.from_dict(rows, orient="index")
    return out


# ------------------------- IC 분석 -------------------------
def analyze(snaps, sig_by_date: dict, labels: pd.DataFrame | None = None) -> dict:
    per_snap = []       # [{date, sig|h: ic}]
    for s in snaps:
        sig = sig_by_date.get(s["date"])
        if sig is None or len(sig) < 15:
            continue
        row = {"date": s["date"]}
        for h in HORIZONS:
            fr = s["fwd"][h].rank()
            for f in SIGNALS:
                v = sig[f].rank().corr(fr.reindex(sig.index))
                if pd.notna(v):
                    row[f"{f}|{h}"] = float(v)
        per_snap.append(row)
    df = pd.DataFrame(per_snap).set_index("date").sort_index()
    n = len(df)
    thirds = [df.iloc[: n // 3], df.iloc[n // 3: 2 * n // 3], df.iloc[2 * n // 3:]]
    rows = []
    for f in SIGNALS:
        for h in HORIZONS:
            col = f"{f}|{h}"
            if col not in df.columns:
                continue
            a = df[col].dropna()
            if len(a) < 6:
                continue
            t = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if a.std(ddof=1) > 0 else 0.0
            sub_ics = [round(float(x[col].dropna().mean()), 4) if x[col].notna().sum() >= 2 else None
                       for x in thirds]
            signs = [s2 for s2 in sub_ics if s2 is not None]
            sign_consist = sum(1 for s2 in signs if np.sign(s2) == np.sign(a.mean()))
            promoted = bool(abs(t) >= 2.0 and sign_consist >= 2)
            rows.append({"signal": f, "horizon": h, "ic": round(float(a.mean()), 4),
                         "t": round(t, 2), "n": int(len(a)), "sub_ic": sub_ics,
                         "sign_consistency": f"{sign_consist}/{len(signs)}",
                         "promoted": promoted})
    # 레짐(R1) 분해(진단 — regime_kr 라벨 재사용)
    by_regime = {}
    if labels is not None:
        lbl = labels["R1"].copy()
        lbl.index = [d.date().isoformat() for d in lbl.index]
        reg = lbl.reindex(df.index)
        for g in reg.dropna().unique():
            sub = df[reg == g]
            by_regime[str(g)] = {c: round(float(sub[c].dropna().mean()), 4)
                                 for c in df.columns if sub[c].notna().sum() >= 3}
    return {"rows": rows, "n_snapshots": n, "by_regime_R1": by_regime,
            "per_snapshot": {d: {k: round(v, 4) for k, v in r.dropna().items()}
                             for d, r in df.iterrows()}}


def run(save=True):
    from benchmarks_kr import load_research_data
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    flow = _load_flow_cache()
    if not flow:
        _log("일별 수급 캐시 없음 — 먼저 --collect 실행"); sys.exit(1)
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    snap_dates = [s["date"] for s in snaps]
    tickers_by_date = {s["date"]: list(s["raw"].index) for s in snaps}
    _log(f"시그널 계산: 스냅샷 {len(snap_dates)}개 × 수급 캐시 {len(flow)}종목")
    sig = build_signals(flow, panel.index, mktcaps, snap_dates, tickers_by_date)
    labels = None
    try:
        import regime_kr as RG
        with open(RG.OUT_PATH, encoding="utf-8") as f:
            lab = json.load(f)["labels_daily"]
        labels = pd.DataFrame.from_dict(lab, orient="index")
        labels.index = pd.to_datetime(labels.index)
    except Exception:
        _log("레짐 라벨 없음 — 레짐 분해 생략")
    res = analyze(snaps, sig, labels)
    promoted = [r for r in res["rows"] if r["promoted"]]
    payload = {"as_of": panel.index[-1].date().isoformat(),
               "budget": "사전 등록 시그널 4종 × 지평 3 — 추가·부활 금지",
               "criteria": "|t|>=2 ∧ 서브기간 3분할 중 2+ 동일 부호 → Phase 3 승격",
               "promoted": [{k: r[k] for k in ("signal", "horizon", "ic", "t")} for r in promoted],
               **res}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        _log(f"저장: {OUT_PATH}")
    for r in res["rows"]:
        _log(f"  {r['signal']:10s} {r['horizon']:3s} IC {r['ic']:+.4f} t={r['t']:+.2f} "
             f"sub={r['sub_ic']} 일관성 {r['sign_consistency']} {'★승격' if r['promoted'] else ''}")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성: 외국인이 미래 상승 종목을 사는 신호를 심고 IC 탐지 검증")
    rng = np.random.default_rng(21)
    n_days, n_syms = 900, 40
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_syms)]
    quality = rng.normal(0, 1, n_syms)
    panel = pd.DataFrame(100 * np.exp(np.cumsum(
        rng.normal(0.0002 + 0.0006 * quality, 0.02, (n_days, n_syms)), axis=0)),
        index=idx, columns=cols)
    flow = {}
    for i, t in enumerate(cols):
        frgn = rng.normal(quality[i] * 5e8, 1e9, n_days)      # 품질 종목 순매수
        indiv = -frgn + rng.normal(0, 5e8, n_days)            # 개인은 반대편
        flow[t] = pd.DataFrame({"frgn": frgn, "indiv": indiv, "inst": rng.normal(0, 1e9, n_days)},
                               index=idx)
    mktcaps = {idx[0].strftime("%Y%m%d"): {t: 1e12 for t in cols}}
    # 가짜 스냅샷: 63일 간격, fwd 1m/3m/6m
    import backtest_weights as BW
    snaps = []
    for p in range(260, n_days - 130, 63):
        fwd = {h: panel.iloc[min(p + 1 + hd, n_days - 1)] / panel.iloc[p + 1] - 1
               for h, hd in BW.TD.items()}
        snaps.append({"date": idx[p].date().isoformat(), "raw": pd.DataFrame(index=cols),
                      "fwd": fwd})
    sig = build_signals(flow, idx, mktcaps, [s["date"] for s in snaps],
                        {s["date"]: cols for s in snaps})
    res = analyze(snaps, sig)
    ic = {(r["signal"], r["horizon"]): r["ic"] for r in res["rows"]}
    assert ic[("frgn_i20", "3m")] > 0.15, ic
    assert ic[("indiv_i20", "3m")] < -0.15, ic
    assert any(r["promoted"] for r in res["rows"]), "심은 신호가 승격 기준 미달"
    _log(f"[self-test] 통과: frgn_i20 3m IC {ic[('frgn_i20','3m')]:+.3f} · "
         f"indiv_i20 {ic[('indiv_i20','3m')]:+.3f}")


def main():
    ap = argparse.ArgumentParser(description="한국 수급 팩터 정밀 IC(사전 등록 4종)")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.collect:
        from benchmarks_kr import load_research_data
        panel, *_ = load_research_data()
        collect(list(panel.columns), panel.index[0].strftime("%Y%m%d"),
                panel.index[-1].strftime("%Y%m%d"))
        return
    run()


if __name__ == "__main__":
    main()
