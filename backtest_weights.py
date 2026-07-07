#!/usr/bin/env python3
"""
backtest_weights.py — '모멘텀 + 여러 펀더멘탈' 지표를 예측력(IC)으로 선별하고 가중치를 최적화.

2단계:
  (A) 지표 선별 — 각 지표의 IC(Information Coefficient) = 매 시점 지표값과 '6개월 후 수익률'의
      순위상관(Spearman)을 이벤트마다 구해 평균. IC 가 클수록 예측력이 큰 지표 → 상위 K개만 채택.
  (B) 가중치 최적화 — 채택된 지표들의 가중치 조합을 격자탐색, 추천 이벤트별 forward-return 로 평가.

지표(모두 '높을수록 좋다' 방향, 시점정보만 사용 = 미래참조 없음):
  모멘텀:  mom6, mom12_1
  펀더멘탈(EDGAR): value(1/PER), sales_yield(1/PSR), roe, roa, net_margin, op_margin,
                   gross_margin, fcf_yield, leverage(-부채/자본), rev_growth, ni_growth, div_yield

실행(PC): python fundamentals_edgar.py               # (1회/증분) 펀더멘탈 수집
          python backtest_weights.py --years 10 --topn 30 --keep 6
          python backtest_weights.py --self-test
결과: 콘솔에 IC 순위표 + 최적 가중치. output/best_weights.json, ic_report.json, backtest_weights.json
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, math
import numpy as np
import pandas as pd

TD = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}   # 보유기간 스윕(근거 비교용)
LOOKBACK = 252
COST = float(os.environ.get("LONG_COST", "0.001"))  # 왕복 거래비용 10bp(장기는 저회전)
MOM_FACTORS = ["mom6", "mom12_1"]


def _z(col):
    sd = col.std()
    zz = (col - col.mean()) / sd if sd and not np.isnan(sd) else col * 0.0
    return zz.clip(-3, 3)


def _raw_frame(panel, p, funds, use_fund, cross=None):
    import fundamentals_edgar as F
    price = panel.iloc[p]
    valid = [s for s in price.dropna().index if not np.isnan(panel.iloc[p - LOOKBACK][s])]
    if not valid:
        return None
    v = pd.Index(valid)
    cols = {"mom6": (panel.iloc[p] / panel.iloc[p - 126] - 1).reindex(v),
            "mom12_1": (panel.iloc[p - 21] / panel.iloc[p - 252] - 1).reindex(v)}
    if cross:                                   # 이동평균 크로스오버 기술 신호
        for f, df in cross.items():
            cols[f] = df.iloc[p].reindex(v)
    if use_fund:
        date_iso = panel.index[p].date().isoformat()
        fv = {f: {} for f in F.FUND_FACTOR_NAMES}
        for s in v:
            for f, val in F.factor_values(funds.get(s) or {}, date_iso, float(price[s])).items():
                fv[f][s] = val
        for f in F.FUND_FACTOR_NAMES:
            if any(fv[f].values()):
                cols[f] = pd.Series(fv[f]).reindex(v)
    return pd.DataFrame(cols).dropna(subset=["mom6", "mom12_1"])


def _weight_grid(factors, levels=(0, 1, 2)):
    seen, out = set(), []
    for combo in itertools.product(levels, repeat=len(factors)):
        if sum(combo) == 0:
            continue
        g = 0
        for c in combo:
            g = math.gcd(g, c)
        key = tuple(c // g for c in combo)
        if key in seen:
            continue
        seen.add(key); out.append(dict(zip(factors, key)))
    return out


def run(panel, spy, funds=None, topn=30, rebal_days=63, keep=6, levels=(0, 1, 2),
        opens=None, oos_frac=0.0):
    import tech_factors as T
    use_fund = bool(funds)
    spy = spy.reindex(panel.index).ffill()
    n = len(panel); max_h = max(TD.values())
    # 익일 진입: 신호는 p일 종가로 계산, 매수는 다음 거래일(p+1) 종가 → 미래참조 제거
    ps = list(range(LOOKBACK, n - max_h - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")

    cross = T.build_panels(panel)              # 기술 팩터(크로스·잔차모멘텀 등)
    if opens is not None:                      # 오버나이트 모멘텀(Lou-Polk-Skouras): 밤사이 수익 12-1
        opens = opens.reindex_like(panel)
        on_cum = (1 + (opens / panel.shift(1) - 1).fillna(0)).cumprod()
        cross["overnight_mom"] = on_cum.shift(21) / on_cum.shift(252) - 1

    snaps = []; snap_ics = []
    for p in ps:
        raw = _raw_frame(panel, p, funds, use_fund, cross)
        if raw is None or raw.empty:
            continue
        e = p + 1
        fwd = {h: (panel.iloc[e + hd][raw.index] / panel.iloc[e][raw.index] - 1) for h, hd in TD.items()}
        bench = {h: float(spy.iloc[e + hd] / spy.iloc[e] - 1) for h, hd in TD.items()}
        f6r = fwd["6m"].rank()
        snap_ics.append({f: raw[f].rank().corr(f6r) for f in raw.columns})
        snaps.append((raw.apply(_z).fillna(0.0), fwd, bench))

    def agg_ic(idxs):
        acc = {}
        for i in idxs:
            for f, v in snap_ics[i].items():
                if pd.notna(v):
                    acc.setdefault(f, []).append(v)
        return sorted(((f, round(float(np.mean(v)), 4)) for f, v in acc.items() if v),
                      key=lambda kv: kv[1], reverse=True)

    def pick(ic_sorted):
        sel = [f for f, ic in ic_sorted if ic > 0][:keep]
        if "mom12_1" not in sel:
            sel = (["mom12_1"] + sel)[:max(keep, 1)]
        return sel if len(sel) >= 2 else [f for f, _ in ic_sorted[:2]]

    def eval_config(w, idxs):
        wv = pd.Series(w); cols = list(w)
        ev = {h: [] for h in TD}; ex = {h: [] for h in TD}; sels = []
        for i in idxs:
            z, fwd, bench = snaps[i]
            top = (z[cols] * wv).sum(axis=1).sort_values(ascending=False).index[:topn]
            sels.append(set(top))
            for h in TD:
                r = fwd[h].reindex(top).dropna()
                if len(r):
                    net = float(r.mean()) - COST
                    ev[h].append(net); ex[h].append(net - bench[h])
        row = {"weights": w}
        turns = [1 - len(sels[j] & sels[j-1]) / max(len(sels[j]), 1) for j in range(1, len(sels))]
        row["turnover"] = round(100 * float(np.mean(turns)), 1) if turns else None
        for h in TD:
            if ev[h]:
                a = np.array(ev[h]); e2 = np.array(ex[h])
                row[f"ret_{h}"] = round(100 * a.mean(), 2)
                row[f"excess_{h}"] = round(100 * e2.mean(), 2)
                row[f"win_{h}"] = round(100 * float((a > 0).mean()), 1)
                row[f"worst_{h}"] = round(100 * float(a.min()), 1)
                if e2.std() > 0:
                    row[f"sharpe_{h}"] = round(float(e2.mean() / e2.std()) * math.sqrt(252.0 / TD[h]), 2)
        return row

    allidx = list(range(len(snaps)))
    if oos_frac and 0 < oos_frac < 0.9:        # 워크포워드: 앞부분 학습 → 뒷부분 표본외 검증
        cut = int(len(snaps) * (1 - oos_frac))
        train, test = list(range(cut)), list(range(cut, len(snaps)))
        ic_sorted = agg_ic(train); selected = pick(ic_sorted)
        results = [eval_config(w, train) for w in _weight_grid(selected, levels)]
        best = max(results, key=score_config)
        oos = {"train": best, "oos": eval_config(best["weights"], test),
               "n_train": len(train), "n_test": len(test)}
        return results, len(snaps), ic_sorted, selected, oos

    ic_sorted = agg_ic(allidx); selected = pick(ic_sorted)
    results = [eval_config(w, allidx) for w in _weight_grid(selected, levels)]
    return results, len(snaps), ic_sorted, selected, None


def score_config(r):
    return (r.get("excess_6m", -999) * 1.0 + r.get("excess_12m", -999) * 0.5
            + (r.get("win_6m", 0) - 50) * 0.05 + r.get("worst_12m", -50) * 0.10
            - (r.get("turnover") or 100) * 0.03)


def _wstr(w):
    return "·".join(f"{k}{v}" for k, v in w.items() if v)


def report(results, n_events, ic_sorted, selected, oos=None, self_test=False):
    print(f"\n=== (A) 지표 예측력 IC 순위 (6개월 forward-return 순위상관, 이벤트 {n_events}회) ===", file=sys.stderr)
    for f, ic in ic_sorted:
        mark = " ★채택" if f in selected else ""
        print(f"    {f:14s} IC {ic:+.4f}{mark}", file=sys.stderr)
    print(f"  → 채택 지표: {selected}", file=sys.stderr)

    ranked = sorted(results, key=score_config, reverse=True)
    print(f"\n=== (B) 채택 지표 가중치 조합 상위 15 (6개월 초과수익 중심) — 단위 % ===", file=sys.stderr)
    cols = [("weights", 42), ("ret_6m", 8), ("excess_6m", 10), ("excess_12m", 11),
            ("win_6m", 7), ("worst_12m", 10), ("turnover", 9)]
    hdr = "".join(str(c).rjust(w) for c, w in cols); print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for r in ranked[:15]:
        line = _wstr(r["weights"]).rjust(42)
        for c, w in cols[1:]:
            v = r.get(c); line += ("" if v is None else str(v)).rjust(w)
        print(line, file=sys.stderr)

    best = ranked[0]
    # 보유기간 스윕: 최우수 가중치의 기간별 순초과수익·순샤프(근거 비교)
    print(f"\n=== 보유기간별 성과 (최우수 가중치, 거래비용 {COST*100:.2f}% 차감, 익일 진입) ===", file=sys.stderr)
    print("%-6s %10s %8s %8s" % ("보유", "순초과%p", "승률%", "순샤프"), file=sys.stderr)
    hold_sharpe = {}
    for h in TD:
        exc, win, shp = best.get(f"excess_{h}"), best.get(f"win_{h}"), best.get(f"sharpe_{h}")
        if exc is not None:
            hold_sharpe[h] = shp if shp is not None else -9
            print("%-6s %10s %8s %8s" % (h, exc, win, shp), file=sys.stderr)
    rec_hold = max(hold_sharpe, key=hold_sharpe.get) if hold_sharpe else "6m"
    print(f"  → 순샤프 최고 보유기간(추천): {rec_hold}", file=sys.stderr)

    os.makedirs("output", exist_ok=True)
    payload = {"weights": best["weights"], "metrics": {k: v for k, v in best.items() if k != "weights"},
               "selected_factors": selected, "ic": dict(ic_sorted),
               "recommended_hold": rec_hold,
               "events": n_events, "self_test": bool(self_test),
               "criteria": "IC로 지표 선별 후, 6M초과+12M+승률-최악낙폭-회전율로 가중치 선정"}
    with open("output/best_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open("output/ic_report.json", "w", encoding="utf-8") as f:
        json.dump(dict(ic_sorted), f, ensure_ascii=False, indent=2)
    with open("output/backtest_weights.json", "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)
    if oos:
        t, o = oos["train"], oos["oos"]
        print(f"\n=== 워크포워드 표본외 검증 (학습 {oos['n_train']}이벤트 → 검증 {oos['n_test']}이벤트) ===", file=sys.stderr)
        print("%-8s %10s %10s %8s" % ("구간", "6M초과%p", "12M초과%p", "6M순샤프"), file=sys.stderr)
        print("%-8s %10s %10s %8s" % ("학습(IS)", t.get("excess_6m"), t.get("excess_12m"), t.get("sharpe_6m")), file=sys.stderr)
        print("%-8s %10s %10s %8s" % ("검증(OOS)", o.get("excess_6m"), o.get("excess_12m"), o.get("sharpe_6m")), file=sys.stderr)
        try:
            keep_ratio = (o.get("excess_6m") or 0) / (t.get("excess_6m") or 1)
            verdict = "합격(표본외 절반 이상 유지)" if keep_ratio >= 0.5 else "주의(표본외 크게 감쇠 → 과최적화 의심)"
            print(f"  → 표본외 유지율 {keep_ratio*100:.0f}% : {verdict}", file=sys.stderr)
        except Exception:
            pass
        payload["oos"] = {"train": {k: t.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
                          "test": {k: o.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")}}
        with open("output/best_weights.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    note = " (self-test: 참고용)" if self_test else ""
    print(f"\n>>> 최적 가중치: {best['weights']}  → output/best_weights.json{note}", file=sys.stderr)
    print(f"    6M수익 {best.get('ret_6m')}% · 6M초과 {best.get('excess_6m')}%p · "
          f"12M초과 {best.get('excess_12m')}%p · 회전율 {best.get('turnover')}%", file=sys.stderr)


def build_panel(years):
    import sp500_daily_report as R
    R._require_yf()
    universe, _ = R.get_sp500()
    bad = ("-W", "-WI", "-WS", "-U", "-RT", "-R", ".W", ".U")
    universe = [s for s in universe if not any(s.upper().endswith(x) for x in bad)]
    hist = R.download_histories(universe, period=f"{int(years)}y")
    panel = pd.DataFrame({s: c for s, c in hist.items() if c is not None and len(c)}).sort_index()
    spy = R.download_histories(["SPY"], period=f"{int(years)}y").get("SPY")
    opens = None                               # 오버나이트 모멘텀용 시가(있으면)
    try:
        import yfinance as yf
        od = yf.download(list(panel.columns), period=f"{int(years)}y", auto_adjust=True,
                         progress=False, threads=True)
        opens = od["Open"].reindex(panel.index) if "Open" in od else None
    except Exception:
        opens = None
    return panel, spy, opens


def load_funds():
    p = "output/fundamentals_cache.json"
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _synthetic(n_days=2100, n_syms=90, seed=11):
    import fundamentals_edgar as F
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    data = {f"S{i:02d}": 100 * np.exp(np.cumsum(
        rng.normal(rng.normal(0.0003, 0.0003), rng.uniform(0.01, 0.03), n_days))) for i in range(n_syms)}
    panel = pd.DataFrame(data, index=dates)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n_days))), index=dates)
    funds = {}
    for s in data:
        eps = round(float(rng.uniform(1, 15)), 2); ni = eps * 1e8
        funds[s] = {"eps": [{"end": "2015-12-31", "filed": "2016-02-01", "val": eps}],
                    "ni": [{"end": "2015-12-31", "filed": "2016-02-01", "val": ni}],
                    "equity": [{"end": "2015-12-31", "filed": "2016-02-01", "val": float(rng.uniform(5e8, 5e9))}],
                    "assets": [{"end": "2015-12-31", "filed": "2016-02-01", "val": float(rng.uniform(1e9, 1e10))}],
                    "revenue": [{"end": "2014-12-31", "filed": "2015-02-01", "val": ni * 4},
                                {"end": "2015-12-31", "filed": "2016-02-01", "val": ni * 5}],
                    "opinc": [{"end": "2015-12-31", "filed": "2016-02-01", "val": ni * 1.3}],
                    "gross": [{"end": "2015-12-31", "filed": "2016-02-01", "val": ni * 2.5}],
                    "ocf": [{"end": "2015-12-31", "filed": "2016-02-01", "val": ni * 1.2}],
                    "capex": [{"end": "2015-12-31", "filed": "2016-02-01", "val": ni * 0.3}],
                    "debt": [{"end": "2015-12-31", "filed": "2016-02-01", "val": float(rng.uniform(0, 5e9))}],
                    "divs": [{"end": "2015-12-31", "filed": "2016-02-01", "val": -ni * 0.2}]}
    opens = panel.shift(1) * (1 + rng.normal(0, 0.005, panel.shape))   # 합성 시가(갭)
    return panel, spy, funds, opens


def main():
    ap = argparse.ArgumentParser(description="모멘텀+펀더멘탈 지표 선별(IC) 및 가중치 탐색")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--keep", type=int, default=6, help="IC 상위 몇 개 지표를 가중치 탐색에 쓸지")
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--oos", type=float, default=0.0, help="워크포워드 표본외 비율(예: 0.4 = 뒤 40%%로 검증)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    levels = tuple(int(x) for x in args.levels.split(","))
    if args.self_test:
        panel, spy, funds, opens = _synthetic(); print("[self-test] 합성 데이터로 로직 점검", file=sys.stderr)
    else:
        panel, spy, opens = build_panel(args.years); funds = load_funds()
        print(f"[백테스트] 패널 {panel.shape[1]}종목 × {panel.shape[0]}일 · "
              f"펀더멘탈 {'있음('+str(len(funds))+')' if funds else '없음'} · "
              f"시가 {'있음' if opens is not None else '없음(오버나이트 생략)'}", file=sys.stderr)
        if not funds:
            print("  ※ 먼저: python fundamentals_edgar.py", file=sys.stderr)
    results, n_ev, ic_sorted, selected, oos = run(panel, spy, funds=funds, topn=args.topn,
                                                  rebal_days=args.rebal_days, keep=args.keep,
                                                  levels=levels, opens=opens, oos_frac=args.oos)
    report(results, n_ev, ic_sorted, selected, oos=oos, self_test=args.self_test)


if __name__ == "__main__":
    main()
