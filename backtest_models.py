#!/usr/bin/env python3
"""
backtest_models.py — '추천 이벤트별 forward-return' 백테스트(모델 비교).

방식(사용자 설계 그대로):
  · 매 분기말(리밸런싱 시점 t)마다 각 모델이 상위 N종목을 '추천'한다(그 시점까지의 데이터만 사용 = 미래참조 없음).
  · 그 종목군(동일가중)을 3·6·12개월 보유했을 때의 '실제 이후 수익률'을 이벤트마다 기록한다.
  · 모든 추천 이벤트의 성과를 평균해 모델을 비교한다(점수 평균이 아니라 '이후 성과' 평균).
  · 벤치마크(SPY) 대비 초과수익률, 승률, 최악 이벤트, 변동성 대비 성과, 종목 교체율까지 산출.

데이터 한계(정직하게):
  · 무료 yfinance 는 과거 '주가'만 제공하고 과거 '펀더멘탈(ROE·PER·FCF)'은 주지 않는다.
  · 따라서 퀄리티·밸류·현금흐름 모델은 '그 시점의' 값을 알 수 없어 정직한 백테스트가 불가(미래참조).
    → 이 엔진은 **가격기반 모델만** 다룬다: 모멘텀 / 저변동성 / 추세 / 이들의 조합.
  · 유니버스는 '현재' S&P500 구성으로, 과거 편입/퇴출을 반영하지 않는다(생존편향 존재 — 결과는 참고용).

실행(데이터가 되는 PC/GitHub 에서):
  python backtest_models.py                 # 기본 10년·분기 리밸·상위 30
  python backtest_models.py --years 8 --topn 30 --self-test   # 합성데이터 로직 점검(네트워크 불필요)
출력: 콘솔 표 + output/backtest_models.json / .csv
"""
from __future__ import annotations
import os, sys, json, csv, argparse
import numpy as np
import pandas as pd

TD = {"3m": 63, "6m": 126, "12m": 252}   # 보유기간(거래일)
LOOKBACK = 252                            # 모멘텀/추세 계산에 필요한 최소 과거일


# ----------------------------- 모델(가격기반) -----------------------------
def _ret(panel, p, back):
    return panel.iloc[p] / panel.iloc[p - back] - 1.0


def _vol(panel, p, win=126):
    seg = panel.iloc[p - win:p + 1]
    return seg.pct_change().std()


def _ma(panel, p, win):
    return panel.iloc[p - win:p + 1].mean()


def select(model, panel, p, n):
    """시점 p(위치 인덱스)에서 model 이 추천하는 상위 n 종목 리스트. p까지 데이터만 사용."""
    price = panel.iloc[p]
    valid = price.dropna().index
    valid = [s for s in valid if not np.isnan(panel.iloc[p - LOOKBACK][s])]  # 충분한 과거 보유
    if not valid:
        return []
    v = pd.Index(valid)
    mom6 = _ret(panel, p, 126).reindex(v)
    mom12_1 = (_ret(panel, p, 252) - _ret(panel, p, 21)).reindex(v)
    vol = _vol(panel, p).reindex(v)
    ma200 = _ma(panel, p, 200).reindex(v)
    above = price.reindex(v) > ma200

    if model == "momentum":
        s = mom6.sort_values(ascending=False)
    elif model == "momentum_12_1":
        s = mom12_1.sort_values(ascending=False)
    elif model == "lowvol":
        s = vol.sort_values(ascending=True)
    elif model == "trend":
        cand = mom6[(above) & (mom6 > 0)]
        s = cand.sort_values(ascending=False)
    elif model == "mom+lowvol":
        r = (mom6.rank(ascending=False) + vol.rank(ascending=True))
        s = r.sort_values(ascending=True)
    elif model == "mom+trend":
        base = mom6.where(above & (mom6 > 0))
        s = base.sort_values(ascending=False)
    else:
        return []
    return list(s.dropna().index[:n])


MODELS = ["momentum", "momentum_12_1", "lowvol", "trend", "mom+lowvol", "mom+trend"]


# ----------------------------- 백테스트 엔진 -----------------------------
def _fwd(series_row_now, series_row_fut):
    return series_row_fut / series_row_now - 1.0


def run_backtest(panel: pd.DataFrame, spy: pd.Series, topn=30, rebal_days=63):
    """추천 이벤트별 forward-return 수집 → 모델×보유기간 성과 집계."""
    spy = spy.reindex(panel.index).ffill()
    n = len(panel)
    max_h = max(TD.values())
    rebal_ps = list(range(LOOKBACK, n - max_h, rebal_days))
    if not rebal_ps:
        raise RuntimeError("데이터가 짧아 리밸런싱 시점이 없음(기간을 늘리세요).")

    # 이벤트별 선택 저장(교체율 계산용) + 성과
    events = {m: [] for m in MODELS}       # [{p, syms, ret{h}, excess{h}}]
    for m in MODELS:
        prev = None
        for p in rebal_ps:
            syms = select(m, panel, p, topn)
            if not syms:
                continue
            rec = {"p": p, "syms": syms, "ret": {}, "excess": {}}
            for h, hd in TD.items():
                now = panel.iloc[p][syms]
                fut = panel.iloc[p + hd][syms]
                r = _fwd(now, fut).dropna()
                if len(r) == 0:
                    continue
                port = float(r.mean())
                bench = float(spy.iloc[p + hd] / spy.iloc[p] - 1.0)
                rec["ret"][h] = port
                rec["excess"][h] = port - bench
            # 교체율
            if prev is not None:
                inter = len(set(syms) & set(prev))
                rec["turnover"] = 1 - inter / max(len(syms), 1)
            prev = syms
            events[m].append(rec)

    # 집계
    results = {}
    for m in MODELS:
        evs = events[m]
        row = {"events": len(evs)}
        turns = [e["turnover"] for e in evs if "turnover" in e]
        row["turnover"] = round(100 * float(np.mean(turns)), 1) if turns else None
        for h in TD:
            rr = [e["ret"][h] for e in evs if h in e["ret"]]
            ex = [e["excess"][h] for e in evs if h in e["excess"]]
            if not rr:
                continue
            rr = np.array(rr); ex = np.array(ex)
            row[f"ret_{h}"] = round(100 * rr.mean(), 2)
            row[f"excess_{h}"] = round(100 * ex.mean(), 2)
            row[f"win_{h}"] = round(100 * float((rr > 0).mean()), 1)
            row[f"beat_{h}"] = round(100 * float((ex > 0).mean()), 1)   # 벤치마크 초과 비율
            row[f"worst_{h}"] = round(100 * float(rr.min()), 1)
            sd = rr.std()
            row[f"sharpe_{h}"] = round(float(rr.mean() / sd), 2) if sd > 0 else None
        results[m] = row
    return results, rebal_ps


# ----------------------------- 출력 -----------------------------
def print_table(results, primary="6m"):
    cols = [("model", 16), (f"ret_{primary}", 9), (f"excess_{primary}", 10),
            (f"win_{primary}", 7), (f"beat_{primary}", 8), (f"worst_{primary}", 8),
            (f"sharpe_{primary}", 8), ("turnover", 9), ("events", 7)]
    print(f"\n=== 추천 이벤트 백테스트(보유 {primary} 기준 정렬) — 단위 % ===", file=sys.stderr)
    hdr = "".join(str(c).rjust(w) for c, w in cols)
    print(hdr, file=sys.stderr); print("-" * len(hdr), file=sys.stderr)
    order = sorted(results, key=lambda m: results[m].get(f"excess_{primary}", -999), reverse=True)
    for m in order:
        r = results[m]
        line = ""
        for c, w in cols:
            v = m if c == "model" else r.get(c)
            line += (("" if v is None else str(v))).rjust(w)
        print(line, file=sys.stderr)


def choose_best(results: dict) -> tuple[str, dict]:
    """사용자 우선순위로 최우수 모델 선정:
       6개월 초과수익(최우선) + 12개월 초과수익(0.5) + 승률가점 - 최악낙폭 반영 - 회전율 패널티(자주 매매 지양).
       6개월 초과수익이 양(+)인 모델만 후보(시장 못 이기면 탈락)."""
    cand = {m: r for m, r in results.items() if r.get("excess_6m", -999) > 0} or results

    def key(m):
        r = results[m]
        return (r.get("excess_6m", -999) * 1.0
                + r.get("excess_12m", -999) * 0.5
                + (r.get("win_6m", 0) - 50) * 0.05
                + r.get("worst_12m", -50) * 0.10          # 음수 → 깊은 낙폭 감점
                - r.get("turnover", 100) * 0.03)          # 회전율 높을수록 감점
    best = max(cand, key=key)
    return best, results[best]


def save(results, path_json, path_csv):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    keys = sorted({k for r in results.values() for k in r})
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model"] + keys)
        for m, r in results.items():
            w.writerow([m] + [r.get(k, "") for k in keys])


# ----------------------------- 데이터 로딩(PC/GitHub) -----------------------------
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


def _synthetic(n_days=2100, n_syms=80, seed=7):
    """네트워크 없이 로직 점검용 합성 주가(기하 브라운운동)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    data = {}
    for i in range(n_syms):
        drift = rng.normal(0.0003, 0.0003)
        vol = rng.uniform(0.01, 0.03)
        steps = rng.normal(drift, vol, n_days)
        data[f"S{i:02d}"] = 100 * np.exp(np.cumsum(steps))
    panel = pd.DataFrame(data, index=dates)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n_days))), index=dates)
    return panel, spy


def main():
    ap = argparse.ArgumentParser(description="추천 이벤트 forward-return 백테스트(가격기반 모델 비교)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63, help="리밸런싱 간격(거래일, 63≈분기)")
    ap.add_argument("--self-test", action="store_true", help="합성데이터로 로직 점검(네트워크 불필요)")
    args = ap.parse_args()

    if args.self_test:
        panel, spy = _synthetic()
        print("[self-test] 합성 주가로 로직 점검", file=sys.stderr)
    else:
        panel, spy = build_panel(args.years)
        print(f"[백테스트] 패널 {panel.shape[1]}종목 × {panel.shape[0]}일", file=sys.stderr)

    results, rebal_ps = run_backtest(panel, spy, topn=args.topn, rebal_days=args.rebal_days)
    os.makedirs("output", exist_ok=True)
    save(results, "output/backtest_models.json", "output/backtest_models.csv")
    for h in ("3m", "6m", "12m"):
        print_table(results, primary=h)

    # 최우수 모델 선정 → best_model.json 저장. 일일 추천(daily_ai_report)이 이 파일을 읽어 종목을 고른다.
    best, bmetrics = choose_best(results)
    import datetime as _dt
    payload = {"model": best, "generated": _dt.date.today().isoformat(),
               "years": args.years, "topn": args.topn, "events": len(rebal_ps),
               "metrics": bmetrics, "self_test": bool(args.self_test),
               "criteria": "6개월 초과수익 우선 + 12개월 + 승률 - 최악낙폭 - 회전율"}
    with open("output/best_model.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    note = " (self-test: 참고용, 실추천에 쓰지 마세요)" if args.self_test else ""
    print(f"\n>>> 최우수 모델: '{best}' → output/best_model.json 저장{note}", file=sys.stderr)
    print(f"    (6M 초과 {bmetrics.get('excess_6m')}%p · 12M 초과 {bmetrics.get('excess_12m')}%p · "
          f"회전율 {bmetrics.get('turnover')}%)", file=sys.stderr)
    print(f"리밸런싱 이벤트 {len(rebal_ps)}회 · 결과: output/backtest_models.json / .csv / best_model.json", file=sys.stderr)


if __name__ == "__main__":
    main()
