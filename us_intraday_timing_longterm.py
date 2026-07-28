#!/usr/bin/env python3
"""
us_intraday_timing_longterm.py — us_intraday_timing_sensitivity.py(60일 표본)를 다년간으로
확장(지호 님 요청, 2026-07-28). 두 파트로 나눠 각각 가능한 최장 기간의 '진짜' 데이터를 쓴다
(보간·추정 없음 — 이전 스크립트와 같은 원칙):

  A. 시가(Open) vs 종가(Close), 최근 8년 — 일봉 Open은 야후가 티커 역사 전체로 제공하므로
     기간 제약이 없다. 표본이 가장 크고(8년×32종목 ≈ 2000거래일/종목) 가장 신뢰도 높은 파트.
  B. +1시간/+2시간 vs 종가, 최근 2년 — yfinance 60분봉 한도(최대 730일)가 진짜 상한이라
     8년을 못 채운다. 60분봉이라 +30분 마크는 이 파트에서 못 뽑는다(그건 원 스크립트의
     60일 표본에서만 유효 — 그대로 참고용으로 남겨둔다).

방법은 원 스크립트와 동일(오늘의 '현재가'만 스냅샷값으로 갈아끼우고 그 이전 종가 히스토리는
그대로 둔 채 재계산 → 페어드 비교). 레짐(regime) 계산은 O(n) 단일 패스로 최적화했다(원
스크립트처럼 매 스냅샷마다 regime_state()를 처음부터 다시 도는 O(n·w) 재호출을 8년 규모로
하면 너무 느려서, 실제 종가 시퀀스로 한 번만 훑으며 '그날 진입 직전 상태'를 캐시해두고
스왑값 하루치만 상태전이 규칙을 한 스텝 더 적용하는 방식으로 대체).

출력: output/us_intraday_timing_longterm.json
"""
from __future__ import annotations
import sys, json, datetime
import numpy as np
import pandas as pd
import yfinance as yf

import market_signals as MS
import sp500_daily_report as R
import us_intraday_timing_sensitivity as S   # 유니버스·헬퍼 재사용

YEARS_OPEN = 8      # 파트 A(일봉 시가) 기간
DAYS_HOURLY = 730   # 파트 B(60분봉) — yfinance 한도(2년)


def _log(m): print(f"[LONGTERM] {m}", file=sys.stderr)


# ------------------------- 파트 A: 시가 vs 종가, 8년 -------------------------
def _daily_open_close(tickers, years=YEARS_OPEN) -> dict:
    """{ticker: DataFrame(Open, Close), tz-naive date index, 내림차순 아님}"""
    df = yf.download(tickers, period=f"{years}y", interval="1d", auto_adjust=True,
                     progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            sub = df.xs(t, axis=1, level=1) if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1 else df
            sub = sub[["Open", "Close"]].dropna()
        except Exception:
            continue
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        if len(sub) >= 210:
            out[t] = sub
    return out


def _regime_states_single_pass(closes: list[float], trend_ma: int, band: float, confirm: int):
    """market_signals.regime_state()와 동일 로직을 '매일의 진입 직전 상태'까지 기록하며 한 번만
    훑는다. 반환: (state_before[i], streak_dir_before[i], streak_before[i]) — 날짜 i를
    처리하기 '직전'의 누적 상태(스왑 실험에서 그 하루만 다른 값으로 한 스텝 더 진행시키는 데 씀).
    또한 실제 종가 기준 결과(state_after[i])도 같이 반환(베이스라인 비교용)."""
    n = len(closes)
    state, streak_dir, streak = None, None, 0
    before, after = [None] * n, [None] * n
    for i in range(n):
        before[i] = (state, streak_dir, streak)
        if i < trend_ma - 1:
            after[i] = state
            continue
        ma = S.MS._sma(closes, trend_ma, i) if hasattr(S, "MS") else MS._sma(closes, trend_ma, i)
        if ma is None:
            after[i] = state; continue
        c = closes[i]
        raw = "ON" if c > ma * (1 + band) else ("OFF" if c < ma * (1 - band) else None)
        if raw and raw != state:
            if raw == streak_dir:
                streak += 1
            else:
                streak_dir, streak = raw, 1
            if streak >= confirm:
                state = raw
                streak_dir, streak = None, 0
        else:
            streak_dir, streak = None, 0
        after[i] = state
    return before, after


def _one_step_state(prev_state, prev_dir, prev_streak, ma, price, band, confirm):
    """하루치만 다른 가격(price)으로 상태전이 규칙을 한 스텝 진행 — 그날의 결과 state 반환."""
    if ma is None:
        return prev_state
    raw = "ON" if price > ma * (1 + band) else ("OFF" if price < ma * (1 - band) else None)
    state, streak_dir, streak = prev_state, prev_dir, prev_streak
    if raw and raw != state:
        if raw == streak_dir:
            streak += 1
        else:
            streak_dir, streak = raw, 1
        if streak >= confirm:
            state = raw
    return state


def run_part_a():
    tickers = S.INDEX_TICKERS + S.STOCK_TICKERS
    _log(f"[A] 시가 vs 종가 {YEARS_OPEN}년 데이터 수집 중 ({len(tickers)}종목)...")
    data = _daily_open_close(tickers, YEARS_OPEN)
    _log(f"[A] 확보 {len(data)}/{len(tickers)}종목")

    p_equity = MS.PARAMS["equity"]
    idx_gap_diff, idx_rsi_diff, idx_flip = [], [], []
    stk_gap200_diff, stk_rsi_diff, stk_hot_flip = [], [], []

    for t in tickers:
        if t not in data:
            continue
        is_index = t in S.INDEX_TICKERS
        df = data[t]
        closes = df["Close"].tolist()
        opens = df["Open"].tolist()
        n = len(closes)
        if n < 210:
            continue

        if is_index:
            before, after = _regime_states_single_pass(closes, p_equity["trend_ma"], p_equity["band"], p_equity["confirm"])
            for i in range(p_equity["trend_ma"] + p_equity["confirm"], n):
                ma = MS._sma(closes, p_equity["trend_ma"], i)
                if ma is None:
                    continue
                base_gap = (closes[i] / ma - 1) * 100
                # 시가로 스왑했을 때의 MA(200일선엔 오늘값도 포함되므로 O(1) 보정)
                ma_swap = ma + (opens[i] - closes[i]) / p_equity["trend_ma"]
                swap_gap = (opens[i] / ma_swap - 1) * 100 if ma_swap else None
                if swap_gap is None:
                    continue
                idx_gap_diff.append(swap_gap - base_gap)

                base_rsi = MS._rsi(closes[:i + 1])
                rsi_series_swap = closes[:i] + [opens[i]]
                swap_rsi = MS._rsi(rsi_series_swap)
                if base_rsi is not None and swap_rsi is not None:
                    idx_rsi_diff.append(swap_rsi - base_rsi)

                p_state, p_dir, p_streak = before[i]
                swap_state = _one_step_state(p_state, p_dir, p_streak, ma_swap, opens[i],
                                             p_equity["band"], p_equity["confirm"])
                idx_flip.append(1 if swap_state != after[i] else 0)
        else:
            close_s = pd.Series(closes)
            for i in range(210, n):
                base_ind = R.compute_indicators(close_s.iloc[:i + 1])
                if base_ind is None or np.isnan(base_ind["ma200"]) or np.isnan(base_ind["ma50"]):
                    continue
                base_gap200 = (base_ind["price"] / base_ind["ma200"] - 1) * 100
                base_rsi = base_ind["rsi"]
                base_gap50 = (base_ind["price"] / base_ind["ma50"] - 1) * 100
                base_hot = bool((not np.isnan(base_rsi) and base_rsi >= 72) or base_gap50 >= 15)

                swap_series = pd.concat([close_s.iloc[:i], pd.Series([opens[i]])], ignore_index=True)
                swap_ind = R.compute_indicators(swap_series)
                if swap_ind is None or np.isnan(swap_ind["ma200"]):
                    continue
                swap_gap200 = (swap_ind["price"] / swap_ind["ma200"] - 1) * 100
                swap_rsi = swap_ind["rsi"]
                swap_gap50 = (swap_ind["price"] / swap_ind["ma50"] - 1) * 100 if not np.isnan(swap_ind["ma50"]) else 0
                swap_hot = bool((not np.isnan(swap_rsi) and swap_rsi >= 72) or swap_gap50 >= 15)

                stk_gap200_diff.append(swap_gap200 - base_gap200)
                if not np.isnan(base_rsi) and not np.isnan(swap_rsi):
                    stk_rsi_diff.append(swap_rsi - base_rsi)
                stk_hot_flip.append(1 if swap_hot != base_hot else 0)
        _log(f"[A] {t} 처리 완료 (누적 지수쌍 {len(idx_gap_diff)} · 종목쌍 {len(stk_gap200_diff)})")

    def _stats(arr):
        a = np.array(arr, dtype=float); a = a[~np.isnan(a)]
        if len(a) == 0: return None
        sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        t_stat = (float(np.mean(a)) / (sd / np.sqrt(len(a)))) if sd > 0 else 0.0
        return {"n": int(len(a)), "mean_abs_diff": round(float(np.mean(np.abs(a))), 3),
                "mean_signed_diff": round(float(np.mean(a)), 3), "t_stat": round(t_stat, 3)}

    return {
        "years": YEARS_OPEN,
        "index": {"n_daypairs": len(idx_gap_diff), "gap_trend_pct": _stats(idx_gap_diff),
                  "rsi": _stats(idx_rsi_diff),
                  "regime_flip_rate_pct": round(100 * np.mean(idx_flip), 2) if idx_flip else None},
        "stock": {"n_daypairs": len(stk_gap200_diff), "gap200_pct": _stats(stk_gap200_diff),
                  "rsi": _stats(stk_rsi_diff),
                  "hot_flip_rate_pct": round(100 * np.mean(stk_hot_flip), 2) if stk_hot_flip else None},
    }


# ------------------------- 파트 B: +1시간/+2시간 vs 종가, 2년(60분봉 한도) -------------------------
def _hourly_snapshots(tickers) -> dict:
    out = {t: {} for t in tickers}
    df = yf.download(tickers, period=f"{DAYS_HOURLY}d", interval="60m", auto_adjust=True,
                     progress=False, threads=True, prepost=False)
    for t in tickers:
        try:
            sub = df.xs(t, axis=1, level=1) if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1 else df
            sub = sub.dropna(subset=["Close"])
        except Exception:
            continue
        idx_et = sub.index.tz_convert("America/New_York") if sub.index.tz is not None else sub.index.tz_localize("UTC").tz_convert("America/New_York")
        by_day = {}
        for ts, close_v in zip(idx_et, sub["Close"].tolist()):
            by_day.setdefault(ts.date().isoformat(), []).append((ts, float(close_v)))
        for day, bars in by_day.items():
            bars.sort(key=lambda x: x[0])
            if len(bars) < 2:
                continue
            out[t][day] = {"p1h": bars[0][1], "p2h": bars[1][1]}
    return out


def run_part_b():
    tickers = S.INDEX_TICKERS + S.STOCK_TICKERS
    _log(f"[B] +1h/+2h vs 종가 {DAYS_HOURLY}일(60분봉 한도) 데이터 수집 중...")
    daily = S._daily_closes(tickers)
    hourly = _hourly_snapshots(tickers)
    _log(f"[B] 일봉 {len(daily)}/{len(tickers)} · 60분봉 {sum(1 for t in tickers if hourly.get(t))}/{len(tickers)}")

    rows = []
    for t in tickers:
        is_index = t in S.INDEX_TICKERS
        if t not in daily or not hourly.get(t):
            continue
        series = daily[t]
        for day, snaps in hourly[t].items():
            d = pd.Timestamp(day)
            if d not in series.index:
                continue
            close_actual = float(series.loc[d])
            base_closes = S._swap_last(series, day, close_actual)
            if base_closes is None:
                continue
            base_m = S._index_metrics(base_closes) if is_index else S._stock_metrics(base_closes)
            for snap in ("p1h", "p2h"):
                val = snaps.get(snap)
                if val is None:
                    continue
                variant = base_closes[:-1] + [val]
                m = S._index_metrics(variant) if is_index else S._stock_metrics(variant)
                rows.append({"ticker": t, "is_index": is_index, "snapshot": snap, "base": base_m, "variant": m})

    summary = {}
    for snap in ("p1h", "p2h"):
        idx_rows = [r for r in rows if r["snapshot"] == snap and r["is_index"]]
        stk_rows = [r for r in rows if r["snapshot"] == snap and not r["is_index"]]

        def _cs(pairs):
            a = np.array(pairs, dtype=float); a = a[~np.isnan(a)]
            if len(a) == 0: return None
            sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            ts = (float(np.mean(a)) / (sd / np.sqrt(len(a)))) if sd > 0 else 0.0
            return {"n": int(len(a)), "mean_abs_diff": round(float(np.mean(np.abs(a))), 3),
                    "mean_signed_diff": round(float(np.mean(a)), 3), "t_stat": round(ts, 3)}

        idx_gap = [r["variant"]["gap_trend"] - r["base"]["gap_trend"] for r in idx_rows
                  if r["variant"]["gap_trend"] is not None and r["base"]["gap_trend"] is not None]
        idx_rsi = [r["variant"]["rsi"] - r["base"]["rsi"] for r in idx_rows
                  if r["variant"]["rsi"] is not None and r["base"]["rsi"] is not None]
        idx_flip = [1 if r["variant"]["regime"] != r["base"]["regime"] else 0 for r in idx_rows]
        stk_gap = [r["variant"]["gap200"] - r["base"]["gap200"] for r in stk_rows
                  if r["variant"]["gap200"] is not None and r["base"]["gap200"] is not None]
        stk_rsi = [r["variant"]["rsi"] - r["base"]["rsi"] for r in stk_rows
                  if r["variant"]["rsi"] is not None and r["base"]["rsi"] is not None]
        stk_hot_flip = [1 if r["variant"]["hot"] != r["base"]["hot"] else 0 for r in stk_rows]

        summary[snap] = {
            "index": {"n_daypairs": len(idx_rows), "gap_trend_pct": _cs(idx_gap), "rsi": _cs(idx_rsi),
                      "regime_flip_rate_pct": round(100 * np.mean(idx_flip), 2) if idx_flip else None},
            "stock": {"n_daypairs": len(stk_rows), "gap200_pct": _cs(stk_gap), "rsi": _cs(stk_rsi),
                      "hot_flip_rate_pct": round(100 * np.mean(stk_hot_flip), 2) if stk_hot_flip else None},
        }
    return {"days": DAYS_HOURLY, "summary": summary}


def run():
    part_a = run_part_a()
    part_b = run_part_b()
    out = {"generated": datetime.datetime.now().isoformat(timespec="minutes"),
          "universe": {"index": S.INDEX_TICKERS, "stocks": S.STOCK_TICKERS},
          "part_a_open_vs_close_multiyear": part_a,
          "part_b_hourly_vs_close_2y": part_b,
          "note": "part_a는 일봉 Open(실측)이라 기간 제약이 없어 8년 사용, part_b는 yfinance "
                  "60분봉 한도(730일)가 실질 상한 — +30분 마크는 이 장기표본에서 재현 불가"
                  "(그건 원 스크립트의 60일 30분봉 표본에서만 유효)."}
    with open("output/us_intraday_timing_longterm.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_intraday_timing_longterm.json")
    return out


if __name__ == "__main__":
    result = run()
    print(json.dumps({"part_a": result["part_a_open_vs_close_multiyear"],
                      "part_b": result["part_b_hourly_vs_close_2y"]}, ensure_ascii=False, indent=2))
