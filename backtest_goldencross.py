#!/usr/bin/env python3
"""
backtest_goldencross.py — 골든크로스 '이벤트 스터디' (시기별 유효성).

질문: 골든크로스가 뜬 뒤, 1일·1주·2주·1달·2달·3달·6달·9달·1년 시점에
      그 종목이 '시장(SPY) 대비' 얼마나 더 올랐나? 승률은? → 언제 유효한지 본다.

방법(미래참조 없음):
  · 각 종목의 이동평균에서 '상향 돌파(fast가 slow를 아래→위로)'가 일어난 날 = 이벤트.
  · 이벤트일 t 기준 여러 보유기간 h 뒤의 (종목수익률 - SPY수익률) = 초과수익 을 모아 평균·승률.
  · 비교 기준선(baseline): 아무 날에나 들어갔을 때의 평균 초과수익(=크로스가 특별한지 대조).

크로스 종류:
  gc_50_200  : 50일선이 200일선 상향돌파 (정석 골든크로스)
  gc_20_60   : 20일선이 60일선 상향돌파
  gc_20_200  : 20일선이 200일선 상향돌파
  gc_60_200  : 60일선이 200일선 상향돌파

실행(PC): python backtest_goldencross.py --years 12
          python backtest_goldencross.py --self-test
출력: 콘솔 표(크로스별 × 보유기간별 평균 초과수익%·승률%) + output/goldencross.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

HORIZONS = {"1d": 1, "1w": 5, "2w": 10, "1m": 21, "2m": 42, "3m": 63,
            "6m": 126, "9m": 189, "12m": 252}
CROSSES = {"gc_50_200": (50, 200), "gc_20_60": (20, 60),
           "gc_20_200": (20, 200), "gc_60_200": (60, 200)}


def cross_up_mask(panel, fast, slow):
    """fast MA가 slow MA를 아래→위로 돌파한 날 = True (dates×syms)."""
    mf = panel.rolling(fast, min_periods=fast).mean()
    ms = panel.rolling(slow, min_periods=slow).mean()
    above = mf > ms
    return above & (~above.shift(1, fill_value=False))


def build_excess(panel, spy):
    """보유기간별 초과수익 패널(종목수익률 - SPY수익률) dict."""
    spy = spy.reindex(panel.index).ffill()
    excess = {}
    for h, hd in HORIZONS.items():
        stock_fwd = panel.shift(-hd) / panel - 1.0
        spy_fwd = spy.shift(-hd) / spy - 1.0
        excess[h] = stock_fwd.sub(spy_fwd, axis=0)   # dates×syms
    return excess


def collect_examples(panel, excess, per=4):
    """크로스별 대표 사례(최고/최악/최근) 실제 종목·날짜·시점별 초과수익 저장."""
    out = {}
    for name, (f, s) in CROSSES.items():
        mask = cross_up_mask(panel, f, s)
        ev = mask.stack()
        idx = ev[ev].index                       # (date, sym) 목록
        if len(idx) == 0:
            out[name] = {}; continue
        rec = pd.DataFrame(index=idx)
        for h in HORIZONS:
            rec[h] = excess[h].stack().reindex(idx)
        rec = rec.dropna(subset=["1m"])          # 최소 1달 이후 데이터 있는 것만
        if rec.empty:
            out[name] = {}; continue

        def pack(df):
            items = []
            for (dt, sym), row in df.iterrows():
                items.append({"symbol": sym, "date": str(pd.Timestamp(dt).date()),
                              "excess": {h: (None if pd.isna(row[h]) else round(100 * float(row[h]), 1))
                                         for h in HORIZONS}})
            return items
        best = rec.sort_values("12m", ascending=False).head(per) if "12m" in rec else rec.head(per)
        worst = rec.sort_values("12m").head(per) if "12m" in rec else rec.tail(per)
        recent = rec.sort_index(level=0).tail(per)
        out[name] = {"best": pack(best), "worst": pack(worst), "recent": pack(recent),
                     "events": int(len(rec))}
    return out


def event_study(panel, excess):
    results = {}
    # 기준선: 아무 날 진입(전체 유효 지점) 평균 초과수익
    base = {}
    valid_any = panel.rolling(200, min_periods=200).mean().notna()
    for h in HORIZONS:
        e = excess[h].where(valid_any)
        vals = e.values[np.isfinite(e.values)]
        base[h] = {"mean": round(100 * float(vals.mean()), 2),
                   "win": round(100 * float((vals > 0).mean()), 1),
                   "n": int(vals.size)}
    results["baseline(아무날 진입)"] = base

    for name, (f, s) in CROSSES.items():
        mask = cross_up_mask(panel, f, s)
        row = {}
        for h in HORIZONS:
            e = excess[h].where(mask)
            vals = e.values[np.isfinite(e.values)]
            if vals.size:
                row[h] = {"mean": round(100 * float(vals.mean()), 2),
                          "win": round(100 * float((vals > 0).mean()), 1),
                          "n": int(vals.size)}
        results[name] = row
    return results


def print_table(results):
    hs = list(HORIZONS)
    print("\n=== 골든크로스 이후 시장(SPY) 대비 '평균 초과수익%' (괄호=승률%) ===", file=sys.stderr)
    head = "%-22s" % "cross" + "".join("%12s" % h for h in hs)
    print(head, file=sys.stderr); print("-" * len(head), file=sys.stderr)
    for name, row in results.items():
        line = "%-22s" % name
        for h in hs:
            c = row.get(h)
            line += "%12s" % (f"{c['mean']:+.1f}({c['win']:.0f})" if c else "-")
        print(line, file=sys.stderr)
    # 이벤트 수(첫 크로스 기준)
    any_cross = next((n for n in results if n != "baseline(아무날 진입)"), None)
    if any_cross:
        n1 = results[any_cross].get("1m", {}).get("n")
        print(f"\n(예: {any_cross} 이벤트 수 ≈ {n1})", file=sys.stderr)
    print("해석: baseline 대비 초과수익이 '크고 승률 높은' 보유기간 = 그 크로스가 유효한 구간.", file=sys.stderr)


def build_panel(years):
    import sp500_daily_report as R
    R._require_yf()
    universe, _ = R.get_sp500()
    bad = ("-W", "-WI", "-WS", "-U", "-RT", "-R", ".W", ".U")
    universe = [s for s in universe if not any(s.upper().endswith(x) for x in bad)]
    hist = R.download_histories(universe, period=f"{int(years)}y")
    panel = pd.DataFrame({s: c for s, c in hist.items() if c is not None and len(c)}).sort_index()
    spy = R.download_histories(["SPY"], period=f"{int(years)}y").get("SPY")
    return panel, spy


def _synthetic(n_days=2600, n_syms=120, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n_days)
    data = {f"S{i:02d}": 100 * np.exp(np.cumsum(
        rng.normal(rng.normal(0.0003, 0.0004), rng.uniform(0.01, 0.03), n_days))) for i in range(n_syms)}
    panel = pd.DataFrame(data, index=dates)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n_days))), index=dates)
    return panel, spy


def main():
    ap = argparse.ArgumentParser(description="골든크로스 이벤트 스터디(시기별 유효성)")
    ap.add_argument("--years", type=float, default=12)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        panel, spy = _synthetic(); print("[self-test] 합성 데이터", file=sys.stderr)
    else:
        panel, spy = build_panel(args.years)
        print(f"[골든크로스] 패널 {panel.shape[1]}종목 × {panel.shape[0]}일", file=sys.stderr)
    excess = build_excess(panel, spy)
    results = event_study(panel, excess)
    examples = collect_examples(panel, excess, per=5)
    os.makedirs("output", exist_ok=True)
    with open("output/goldencross.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("output/goldencross_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print_table(results)
    print("\n결과 저장: output/goldencross.json · goldencross_examples.json", file=sys.stderr)


if __name__ == "__main__":
    main()
