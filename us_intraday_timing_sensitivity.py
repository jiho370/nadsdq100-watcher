#!/usr/bin/env python3
"""
us_intraday_timing_sensitivity.py — 미국장 리포트를 "장마감(현행)" 대신 "시초가/개장+30분/
+1시간/+2시간"에 계산하면 신호값이 얼마나 달라지는지 정량화 (지호 님 요청, 2026-07-28).

배경: 국장 메일을 08:00→09:30(개장 30분 후)으로 이관한 뒤, 미장도 "개장 직후"로 옮기자는
제안이 나왔다. 대화에서 나온 반대 논거 두 가지를 데이터로 검증한다:
  1. 종가 기준으로 검증된 알고리즘(200일선 등)에 시가/장중가를 넣으면 신호가 얼마나 흔들리는가.
  2. 개장 초반(U자형 변동성 패턴)이 실제로 신호를 더 불안정하게 만드는가.

⚠ 데이터 한계(정직하게 명시): yfinance 30분봉은 최대 60일치만 제공한다. 이건 "포트폴리오
수익률 우열"을 묻는 다년간 백테스트(PBO/DSR 프레임)와는 질문 자체가 다르다 — 여기서 묻는 건
"같은 날 다른 시각에 계산하면 신호값이 얼마나 다른가"이므로, 페어드(paired) 비교로 판정한다
(종목-일 단위 표본은 충분히 크다: 지수 2개+종목 30개 × 최대 42거래일 ≈ 1300+ 페어).

방법: 종목별 최근 2년 일별 종가(200일선 계산용)에 최근 60일 30분봉에서 뽑은 시가/+30분/+1시간/
+2시간 스냅샷을 '오늘 가격'으로 갈아끼워 지표를 재계산 → 실제 종가로 계산한 값과 페어드 비교.
지수는 market_signals.analyze()(레짐/모멘텀/RSI/눌림갭), 종목은 sp500_daily_report.
compute_indicators()(RSI/50일선갭/200일선갭 — export_data.hot()과 동일 정의)를 그대로 재사용.

출력: output/us_intraday_timing_sensitivity.json
"""
from __future__ import annotations
import sys, json, datetime
import numpy as np
import pandas as pd
import yfinance as yf

import market_signals as MS
import sp500_daily_report as R

INDEX_TICKERS = ["^GSPC", "^NDX"]
STOCK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    "JNJ", "PG", "HD", "V", "MA", "KO", "PEP", "WMT", "DIS", "NFLX",
    "CRM", "ADBE", "INTC", "CSCO", "PFE", "MRK", "ABBV", "COST", "MCD", "NKE",
]
SNAPSHOTS = ["open", "p30m", "p1h", "p2h"]   # vs "close"(현행, 기준)


def _log(m): print(f"[SENS] {m}", file=sys.stderr)


def _daily_closes(tickers: list[str]) -> dict:
    """{ticker: pandas Series(일별 종가, tz-naive date index)}"""
    df = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                     progress=False, threads=True)
    close = df["Close"] if "Close" in getattr(df, "columns", []) else df
    out = {}
    for t in tickers:
        try:
            s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
        except Exception:
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        if len(s) >= 210:
            out[t] = s
    return out


def _intraday_snapshots(tickers: list[str]) -> dict:
    """{ticker: {date_str: {"open":.., "p30m":.., "p1h":.., "p2h":..}}} — 30분봉 60일."""
    out = {t: {} for t in tickers}
    df = yf.download(tickers, period="60d", interval="30m", auto_adjust=True,
                     progress=False, threads=True, prepost=False)
    for t in tickers:
        try:
            sub = df.xs(t, axis=1, level=1) if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1 else df
        except Exception:
            continue
        if "Close" not in sub.columns or "Open" not in sub.columns:
            continue
        sub = sub.dropna(subset=["Close"])
        idx_et = sub.index.tz_convert("America/New_York") if sub.index.tz is not None else sub.index.tz_localize("UTC").tz_convert("America/New_York")
        by_day = {}
        for ts, row in zip(idx_et, sub.itertuples()):
            day = ts.date().isoformat()
            by_day.setdefault(day, []).append((ts, float(row.Open), float(row.Close)))
        for day, bars in by_day.items():
            bars.sort(key=lambda x: x[0])
            if len(bars) < 4:      # +2시간 마크(4번째 30분봉)까지 있어야 채택
                continue
            out[t][day] = {
                "open": bars[0][1],      # 시초가(첫 봉의 시가)
                "p30m": bars[0][2],      # 09:30+30분 = 첫 봉 종가
                "p1h":  bars[1][2],      # +1시간 = 둘째 봉 종가
                "p2h":  bars[3][2],      # +2시간 = 넷째 봉 종가
            }
    return out


def _swap_last(series: pd.Series, day: str, value: float) -> list[float] | None:
    """series에서 day '이전'까지의 종가 + day의 스냅샷값을 마지막에 붙인 리스트."""
    d = pd.Timestamp(day)
    prior = series[series.index < d]
    if len(prior) < 205:
        return None
    return [float(v) for v in prior.tolist()] + [value]


def _index_metrics(closes: list[float]) -> dict:
    a = MS.analyze(closes, "equity")
    return {"regime": a["regime"], "gap_trend": a["gap_trend"], "gap_dip": a["gap_dip"], "rsi": a["rsi"]}


def _stock_metrics(closes: list[float]) -> dict:
    s = pd.Series(closes)
    ind = R.compute_indicators(s)
    price, ma50, ma200 = ind["price"], ind["ma50"], ind["ma200"]
    gap50 = ((price / ma50 - 1) * 100) if ma50 and not np.isnan(ma50) else None
    gap200 = ((price / ma200 - 1) * 100) if ma200 and not np.isnan(ma200) else None
    rsi = ind["rsi"] if not np.isnan(ind["rsi"]) else None
    hot = bool((rsi is not None and rsi >= 72) or (gap50 is not None and gap50 >= 15))
    return {"rsi": rsi, "gap50": gap50, "gap200": gap200, "hot": hot}


def run():
    _log(f"지수 {len(INDEX_TICKERS)}개 + 종목 {len(STOCK_TICKERS)}개 데이터 수집 중...")
    all_tickers = INDEX_TICKERS + STOCK_TICKERS
    daily = _daily_closes(all_tickers)
    intraday = _intraday_snapshots(all_tickers)
    _log(f"일별 종가 확보 {len(daily)}/{len(all_tickers)} · 30분봉 확보 "
         f"{sum(1 for t in all_tickers if intraday.get(t))}/{len(all_tickers)}")

    rows = []   # 종목-일-스냅샷 단위 레코드
    for t in all_tickers:
        is_index = t in INDEX_TICKERS
        if t not in daily or not intraday.get(t):
            continue
        series = daily[t]
        for day, snaps in intraday[t].items():
            d = pd.Timestamp(day)
            if d not in series.index:
                continue   # 그날 공식 종가 없음(조기폐장·데이터갭 등) → 스킵
            close_actual = float(series.loc[d])
            base_closes = _swap_last(series, day, close_actual)
            if base_closes is None:
                continue
            base_m = _index_metrics(base_closes) if is_index else _stock_metrics(base_closes)
            for snap in SNAPSHOTS:
                val = snaps.get(snap)
                if val is None:
                    continue
                closes_variant = base_closes[:-1] + [val]
                m = _index_metrics(closes_variant) if is_index else _stock_metrics(closes_variant)
                rows.append({"ticker": t, "is_index": is_index, "day": day, "snapshot": snap,
                            "base": base_m, "variant": m})
    _log(f"페어드 레코드 {len(rows)}건 생성")

    # ── 집계: 스냅샷 종류별로 (a) 연속값 평균절대차 + 페어드 t검정, (b) 이산값(레짐/과열) 뒤집힘률
    summary = {}
    for snap in SNAPSHOTS:
        idx_rows = [r for r in rows if r["snapshot"] == snap and r["is_index"]]
        stk_rows = [r for r in rows if r["snapshot"] == snap and not r["is_index"]]

        def _cont_stats(pairs):
            arr = np.array(pairs, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return None
            mean_abs = float(np.mean(np.abs(arr)))
            mean_signed = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            t_stat = (mean_signed / (sd / np.sqrt(len(arr)))) if sd > 0 else 0.0
            return {"n": int(len(arr)), "mean_abs_diff": round(mean_abs, 3),
                    "mean_signed_diff": round(mean_signed, 3), "t_stat": round(float(t_stat), 3)}

        idx_gap_trend = [r["variant"]["gap_trend"] - r["base"]["gap_trend"] for r in idx_rows
                         if r["variant"]["gap_trend"] is not None and r["base"]["gap_trend"] is not None]
        idx_rsi = [r["variant"]["rsi"] - r["base"]["rsi"] for r in idx_rows
                  if r["variant"]["rsi"] is not None and r["base"]["rsi"] is not None]
        idx_regime_flip = [1 if r["variant"]["regime"] != r["base"]["regime"] else 0 for r in idx_rows]

        stk_gap200 = [r["variant"]["gap200"] - r["base"]["gap200"] for r in stk_rows
                     if r["variant"]["gap200"] is not None and r["base"]["gap200"] is not None]
        stk_rsi = [r["variant"]["rsi"] - r["base"]["rsi"] for r in stk_rows
                  if r["variant"]["rsi"] is not None and r["base"]["rsi"] is not None]
        stk_hot_flip = [1 if r["variant"]["hot"] != r["base"]["hot"] else 0 for r in stk_rows]

        summary[snap] = {
            "index": {"n_daypairs": len(idx_rows), "gap_trend_pct": _cont_stats(idx_gap_trend),
                      "rsi": _cont_stats(idx_rsi),
                      "regime_flip_rate_pct": round(100 * np.mean(idx_regime_flip), 1) if idx_regime_flip else None},
            "stock": {"n_daypairs": len(stk_rows), "gap200_pct": _cont_stats(stk_gap200),
                      "rsi": _cont_stats(stk_rsi),
                      "hot_flip_rate_pct": round(100 * np.mean(stk_hot_flip), 1) if stk_hot_flip else None},
        }

    out = {"generated": datetime.datetime.now().isoformat(timespec="minutes"),
          "universe": {"index": INDEX_TICKERS, "stocks": STOCK_TICKERS},
          "data_limitation": "yfinance 30분봉 최대 60일 — 다년간 수익률 백테스트가 아니라 "
                              "같은 날 시각별 신호값 페어드 비교(측정 민감도 검정)",
          "summary": summary,
          "raw_rows_n": len(rows)}
    with open("output/us_intraday_timing_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_intraday_timing_sensitivity.json")
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
