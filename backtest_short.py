#!/usr/bin/env python3
"""
backtest_short.py — '단기(2주 보유)' 신호 탐색 백테스트.

장기 알고리즘(모멘텀+펀더멘탈)과 별개로, 2주 정도 짧게 보유하는 단기 매매용 신호를 찾는다.
방식: 매 10거래일(≈2주) 리밸런싱, 상위 N종목을 1주·2주·1달 보유했을 때 SPY 대비 초과수익 평가.
     각 신호의 IC(2주 기준)로 예측력 순위 → 상위만 골라 가중치 조합 탐색.

단기 신호(모두 '높을수록 좋다' 방향, 가격+거래량):
  gc2060     : 20일선이 60일선을 '최근' 상향돌파(빠른 골든크로스)
  rvol       : 거래량 급증(당일 거래량 / 20일 평균)
  ret1w      : 1주 모멘텀(최근 5일 수익률)
  rev1m      : 1달 반전(최근 21일 수익률의 음수 = 눌린 종목 반등 기대)
  rsi_low    : 과매도(낮은 RSI = 반등 여지)
  breakout20 : 20일 신고가 근접(현재가 / 20일 최고가)

실행(PC): python backtest_short.py --years 10
          python backtest_short.py --self-test
출력: 콘솔 IC·가중치표 + output/best_short_weights.json
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, math
import numpy as np
import pandas as pd

# 보유기간을 고정하지 않고 여러 기간을 모두 평가 → 거래비용 뺀 '순 샤프' 최고 기간을 데이터가 선택.
HZ = {"1w": 5, "2w": 10, "3w": 15, "1m": 21, "6w": 30, "2m": 42, "3m": 63}
PRIMARY = os.environ.get("SHORT_PRIMARY", "2w")   # 지표 선별(IC) 기준 기간
LOOKBACK = 252   # 52주 신고가 계산에 필요
COST = float(os.environ.get("SHORT_COST", "0.0015"))   # 왕복 거래비용(스프레드+슬리피지) 15bp
# 학술 리포트 반영: 변동성조정 반전(rev5d_vol), 거래량조건부 모멘텀(volmom), 52주 신고가(high52) 추가
FACTORS = ["gc2060", "rvol", "ret1w", "rev1m", "rev5d_vol", "volmom", "rsi_low", "breakout20", "high52"]


def _rsi(panel, period=14):
    d = panel.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_factor_panels(price, vol):
    ma20 = price.rolling(20).mean(); ma60 = price.rolling(60).mean()
    above = ma20 > ma60
    gc2060 = above.astype(float) * (1 - above.rolling(21, min_periods=5).mean())   # 최근 20>60 돌파
    rvol = vol / vol.rolling(20).mean()
    ret1w = price / price.shift(5) - 1
    rev1m = -(price / price.shift(21) - 1)
    dvol20 = price.pct_change().rolling(20).std()
    rev5d_vol = -(price / price.shift(5) - 1) / dvol20            # 변동성조정 1주 반전(과매도)
    volmom = (price / price.shift(20) - 1) * (vol / vol.rolling(20).mean())  # 거래량조건부 모멘텀
    rsi_low = -_rsi(price)                          # 낮은 RSI일수록 큰 값
    breakout20 = price / price.rolling(20).max()    # 1에 가까울수록 20일 신고가
    high52 = price / price.rolling(252, min_periods=60).max()     # 52주 신고가 근접도
    return {"gc2060": gc2060, "rvol": rvol, "ret1w": ret1w, "rev1m": rev1m,
            "rev5d_vol": rev5d_vol, "volmom": volmom, "rsi_low": rsi_low,
            "breakout20": breakout20, "high52": high52}


def _z(col):
    sd = col.std()
    zz = (col - col.mean()) / sd if sd and not np.isnan(sd) else col * 0.0
    return zz.clip(-3, 3)


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


def run(price, vol, spy, topn=20, rebal_days=10, keep=4, levels=(0, 1, 2)):
    spy = spy.reindex(price.index).ffill()
    fp = build_factor_panels(price, vol)
    n = len(price); max_h = max(HZ.values())
    ps = list(range(LOOKBACK, n - max_h - 1, rebal_days))   # 익일 진입 여유
    if not ps:
        raise RuntimeError("기간이 짧음.")
    snaps = []; ic_acc = {}
    for p in ps:
        valid = price.iloc[p].dropna().index
        valid = [s for s in valid if not np.isnan(price.iloc[p - LOOKBACK][s])]
        if not valid:
            continue
        v = pd.Index(valid)
        raw = pd.DataFrame({f: fp[f].iloc[p].reindex(v) for f in FACTORS}).dropna(how="all")
        if raw.empty:
            continue
        e = p + 1                                          # 익일 진입(신호=p 종가, 매수=p+1 종가)
        fwd = {h: (price.iloc[e + hd][raw.index] / price.iloc[e][raw.index] - 1) for h, hd in HZ.items()}
        bench = {h: float(spy.iloc[e + hd] / spy.iloc[e] - 1) for h, hd in HZ.items()}
        f2 = fwd[PRIMARY].rank()
        for f in raw.columns:
            ic = raw[f].rank().corr(f2)
            if pd.notna(ic):
                ic_acc.setdefault(f, []).append(ic)
        snaps.append((raw.apply(_z).fillna(0.0), fwd, bench))

    IC = {f: round(float(np.mean(vv)), 4) for f, vv in ic_acc.items() if vv}
    ic_sorted = sorted(IC.items(), key=lambda kv: kv[1], reverse=True)
    selected = [f for f, ic in ic_sorted if ic > 0][:keep] or [f for f, _ in ic_sorted[:2]]

    results = []
    for w in _weight_grid(selected, levels):
        wv = pd.Series(w); cols = list(w)
        ev = {h: [] for h in HZ}; ex = {h: [] for h in HZ}; sels = []
        for z, fwd, bench in snaps:
            score = (z[cols] * wv).sum(axis=1)
            top = score.sort_values(ascending=False).index[:topn]
            sels.append(set(top))
            for h in HZ:
                r = fwd[h].reindex(top).dropna()
                if len(r):
                    net = float(r.mean()) - COST          # 왕복 거래비용 차감(단기 필수)
                    ev[h].append(net); ex[h].append(net - bench[h])
        row = {"weights": w}
        turns = [1 - len(sels[i] & sels[i-1]) / max(len(sels[i]), 1) for i in range(1, len(sels))]
        row["turnover"] = round(100 * float(np.mean(turns)), 1) if turns else None
        for h in HZ:
            if ev[h]:
                a = np.array(ev[h]); e = np.array(ex[h])
                row[f"ret_{h}"] = round(100 * a.mean(), 2)
                row[f"excess_{h}"] = round(100 * e.mean(), 2)
                row[f"win_{h}"] = round(100 * float((a > 0).mean()), 1)
                row[f"beat_{h}"] = round(100 * float((e > 0).mean()), 1)
                # 순 샤프(이벤트 초과수익 평균/표준편차) — 연율화(연 리밸 횟수 반영)
                if e.std() > 0:
                    ppy = 252.0 / HZ[h]      # 연간 보유 회전 수
                    row[f"sharpe_{h}"] = round(float(e.mean() / e.std()) * math.sqrt(ppy), 2)
        results.append(row)
    return results, len(snaps), ic_sorted, selected


def _score(r):   # 2주 초과수익 우선 + 승률
    return r.get(f"excess_{PRIMARY}", -9) * 1.0 + (r.get(f"beat_{PRIMARY}", 0) - 50) * 0.05


def report(results, n, ic_sorted, selected, self_test=False):
    print(f"\n=== 단기(2주) 신호 IC 순위 (이벤트 {n}회) ===", file=sys.stderr)
    for f, ic in ic_sorted:
        print(f"   {f:11s} IC {ic:+.4f}{'  ★채택' if f in selected else ''}", file=sys.stderr)
    ranked = sorted(results, key=_score, reverse=True)
    print(f"\n=== 가중치 조합 상위 12 (2주 기준) — 초과수익%(시장이긴비율%) ===", file=sys.stderr)
    cols = [("weights", 34), ("ret_1w", 8), ("excess_1w", 10), ("excess_2w", 10),
            ("beat_2w", 8), ("excess_1m", 10), ("turnover", 9)]
    hdr = "".join(str(c).rjust(w) for c, w in cols); print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for r in ranked[:12]:
        line = ("·".join(f"{k}{v}" for k, v in r["weights"].items() if v)).rjust(34)
        for c, w in cols[1:]:
            val = r.get(c); line += ("" if val is None else str(val)).rjust(w)
        print(line, file=sys.stderr)
    best = ranked[0]
    # 보유기간 스윕: 최우수 가중치의 기간별 순초과수익·시장이긴비율·순샤프
    print(f"\n=== 보유기간별 성과 (최우수 가중치, 거래비용 {COST*100:.2f}% 차감) ===", file=sys.stderr)
    print("%-6s %10s %10s %8s" % ("보유", "순초과%p", "이긴%", "순샤프"), file=sys.stderr)
    hold_sharpe = {}
    for h in HZ:
        exc, beat, shp = best.get(f"excess_{h}"), best.get(f"beat_{h}"), best.get(f"sharpe_{h}")
        if exc is not None:
            hold_sharpe[h] = shp if shp is not None else -9
            print("%-6s %10s %10s %8s" % (h, exc, beat, shp), file=sys.stderr)
    rec_hold = max(hold_sharpe, key=hold_sharpe.get) if hold_sharpe else PRIMARY
    print(f"  → 순샤프 최고 보유기간(추천): {rec_hold}", file=sys.stderr)

    os.makedirs("output", exist_ok=True)
    payload = {"weights": best["weights"], "metrics": {k: v for k, v in best.items() if k != "weights"},
               "selected_factors": selected, "ic": dict(ic_sorted),
               "ic_hold": PRIMARY, "recommended_hold": rec_hold,
               "self_test": bool(self_test)}
    with open("output/best_short_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 단기 최적 가중치: {best['weights']}  → output/best_short_weights.json"
          f"{' (self-test)' if self_test else ''}", file=sys.stderr)
    print(f"    추천 보유기간 {rec_hold} · 그 기간 순초과 {best.get('excess_'+rec_hold)}%p · "
          f"순샤프 {best.get('sharpe_'+rec_hold)} · 회전율 {best.get('turnover')}%", file=sys.stderr)


def build_panel(years):
    import sp500_daily_report as R
    R._require_yf()
    universe, _ = R.get_sp500()
    bad = ("-W", "-WI", "-WS", "-U", "-RT", "-R", ".W", ".U")
    universe = [s for s in universe if not any(s.upper().endswith(x) for x in bad)]
    hist, vol = R.download_histories(universe, period=f"{int(years)}y", with_volume=True)
    price = pd.DataFrame({s: c for s, c in hist.items() if c is not None and len(c)}).sort_index()
    volp = pd.DataFrame({s: c for s, c in vol.items() if c is not None and len(c)}).reindex_like(price)
    spy = R.download_histories(["SPY"], period=f"{int(years)}y").get("SPY")
    return price, volp, spy


def _synthetic(n_days=1500, n_syms=100, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    price = pd.DataFrame({f"S{i:02d}": 100 * np.exp(np.cumsum(
        rng.normal(rng.normal(0.0003, 0.0004), rng.uniform(0.01, 0.03), n_days))) for i in range(n_syms)}, index=dates)
    vol = pd.DataFrame(rng.lognormal(15, 0.5, (n_days, n_syms)), index=dates, columns=price.columns)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n_days))), index=dates)
    return price, vol, spy


def main():
    ap = argparse.ArgumentParser(description="단기(2주) 신호 탐색 백테스트")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        price, vol, spy = _synthetic(); print("[self-test] 합성 데이터", file=sys.stderr)
    else:
        price, vol, spy = build_panel(args.years)
        print(f"[단기백테스트] {price.shape[1]}종목 × {price.shape[0]}일", file=sys.stderr)
    results, n, ic_sorted, selected = run(price, vol, spy, topn=args.topn, keep=args.keep)
    report(results, n, ic_sorted, selected, self_test=args.self_test)


if __name__ == "__main__":
    main()
