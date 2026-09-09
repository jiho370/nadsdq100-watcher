#!/usr/bin/env python3
"""
crash_delay_loss_analysis.py — "서킷브레이커가 실제로 트리거된 순간부터, 실행이 지연되면
얼마나 더 손실이 커지는가"를 1분~1주 세밀한 지연시간 그리드로 분석 (2026-09-06, 지호 님
요청). realtime_circuit_breaker_paper.py의 정의(최근 24시간 고점 대비 -5%)와 동일한
트리거 조건을 실제 1분봉 데이터에서 찾아, 그 시점(T0) 이후 지연시간별 가격 변화를 잰다.

데이터: 바이낸스 공개 API(BTCUSDT·ETHUSDT 1분봉, 인증 불필요) — 업비트 KRW 페어의 분봉은
과거 이력 접근이 제한적이라, 크래시의 '형태'(가격이 얼마나 더 빠지는가)는 원화/달러
페어가 사실상 동일하게 움직인다는 전제로 대용(proxy)한다. 주요 역사적 급락 이벤트
5~6건 × 2자산을 대상으로 함.

실행: python crash_delay_loss_analysis.py
결과: output/crash_delay_loss.json, 원본 분봉 캐시는 output/binance_1m_cache_*.pkl
"""
from __future__ import annotations
import os, sys, json, time
import urllib.request
import numpy as np
import pandas as pd

DELAYS_MIN = {
    "1분": 1, "5분": 5, "15분": 15, "30분": 30,
    "1시간": 60, "2시간": 120, "4시간": 240, "8시간": 480, "12시간": 720,
    "1일": 1440, "2일": 2880, "3일": 4320, "5일": 7200, "7일": 10080,
}
TRIGGER_WINDOW_MIN = 1440   # 트리거 정의: 최근 24시간 고점 대비
TRIGGER_THRESHOLD = -0.05

EVENTS = [
    # (자산, 심볼, 라벨, 대략적 시작일 — 이 앞뒤로 데이터를 받아 그 안에서 실제 -5% 최초
    #  교차 시점을 스캔한다)
    ("BTC", "BTCUSDT", "2020-03 코로나 폭락", "2020-03-10", "2020-03-22"),
    ("BTC", "BTCUSDT", "2021-05 중국 채굴금지", "2021-05-17", "2021-05-29"),
    ("BTC", "BTCUSDT", "2022-05 테라루나 붕괴", "2022-05-07", "2022-05-19"),
    ("BTC", "BTCUSDT", "2022-06 셀시우스 사태", "2022-06-11", "2022-06-23"),
    ("BTC", "BTCUSDT", "2022-11 FTX 파산", "2022-11-06", "2022-11-18"),
    ("ETH", "ETHUSDT", "2020-03 코로나 폭락", "2020-03-10", "2020-03-22"),
    ("ETH", "ETHUSDT", "2021-05 중국 채굴금지", "2021-05-17", "2021-05-29"),
    ("ETH", "ETHUSDT", "2022-05 테라루나 붕괴", "2022-05-07", "2022-05-19"),
    ("ETH", "ETHUSDT", "2022-06 셀시우스 사태", "2022-06-11", "2022-06-23"),
    ("ETH", "ETHUSDT", "2022-11 FTX 파산", "2022-11-06", "2022-11-18"),
]


def _log(m): print(f"[급락지연손실분석] {m}", file=sys.stderr)


def fetch_1m_klines(symbol: str, start: str, end: str) -> pd.Series:
    cache_path = f"output/binance_1m_cache_{symbol}_{start}_{end}.pkl"
    if os.path.exists(cache_path):
        _log(f"{symbol} {start}~{end}: 캐시 사용")
        return pd.read_pickle(cache_path)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out_ts, out_px = [], []
    cur = start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m"
              f"&startTime={cur}&limit=1000")
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if not data:
            break
        for row in data:
            out_ts.append(row[0]); out_px.append(float(row[4]))   # close
        cur = data[-1][0] + 60_000
        time.sleep(0.05)
    s = pd.Series(out_px, index=pd.to_datetime(out_ts, unit="ms", utc=True))
    s = s[~s.index.duplicated()]
    os.makedirs("output", exist_ok=True)
    s.to_pickle(cache_path)
    _log(f"{symbol} {start}~{end}: 신규 다운로드 {len(s)}분")
    return s


def find_trigger(prices: np.ndarray) -> int | None:
    """최근 TRIGGER_WINDOW_MIN분 고점 대비 TRIGGER_THRESHOLD 이하로 떨어진 첫 인덱스."""
    n = len(prices)
    roll_max = pd.Series(prices).rolling(TRIGGER_WINDOW_MIN, min_periods=60).max().to_numpy()
    for i in range(60, n):
        if np.isnan(roll_max[i]):
            continue
        if prices[i] / roll_max[i] - 1 < TRIGGER_THRESHOLD:
            return i
    return None


def analyze_event(asset, symbol, label, start, end) -> dict | None:
    s = fetch_1m_klines(symbol, start, end)
    prices = s.to_numpy()
    idx = find_trigger(prices)
    if idx is None:
        _log(f"[{asset}/{label}] 트리거(-5%) 없음 — 건너뜀")
        return None
    t0_price = prices[idx]
    t0_time = s.index[idx]
    result = {"asset": asset, "label": label, "trigger_time": t0_time.isoformat(),
             "trigger_price": float(t0_price), "delays": {}}
    for dname, dmin in DELAYS_MIN.items():
        j = idx + dmin
        if j >= len(prices):
            result["delays"][dname] = None   # 데이터 범위를 벗어남(수집기간 부족)
            continue
        chg = prices[j] / t0_price - 1
        result["delays"][dname] = round(float(chg) * 100, 2)
    _log(f"[{asset}/{label}] 트리거 {t0_time} @ {t0_price:.1f} · "
        f"1일후 {result['delays'].get('1일')}% · 1주후 {result['delays'].get('7일')}%")
    return result


def main():
    os.makedirs("output", exist_ok=True)
    results = []
    for asset, symbol, label, start, end in EVENTS:
        try:
            r = analyze_event(asset, symbol, label, start, end)
            if r:
                results.append(r)
        except Exception as e:
            _log(f"[{asset}/{label}] 실패({type(e).__name__}: {e})")

    # 자산별 지연시간대별 집계(평균·중앙값·최악·최선)
    summary = {}
    for asset in ("BTC", "ETH"):
        rows = [r for r in results if r["asset"] == asset]
        agg = {}
        for dname in DELAYS_MIN:
            vals = [r["delays"][dname] for r in rows if r["delays"].get(dname) is not None]
            if not vals:
                continue
            agg[dname] = {"mean": round(float(np.mean(vals)), 2),
                          "median": round(float(np.median(vals)), 2),
                          "worst": round(float(np.min(vals)), 2),
                          "best": round(float(np.max(vals)), 2), "n_events": len(vals)}
        summary[asset] = agg

    out = {"events": results, "summary": summary,
          "trigger_def": f"최근{TRIGGER_WINDOW_MIN}분 고점 대비 {TRIGGER_THRESHOLD:+.0%}"}
    with open("output/crash_delay_loss.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/crash_delay_loss.json")
    for asset, agg in summary.items():
        _log(f"=== {asset} 지연시간별 평균 추가손익(%) ===")
        for dname in DELAYS_MIN:
            if dname in agg:
                _log(f"  {dname:>4}: 평균 {agg[dname]['mean']:+.2f}% · 중앙값 {agg[dname]['median']:+.2f}% "
                    f"· 최악 {agg[dname]['worst']:+.2f}% · 최선 {agg[dname]['best']:+.2f}% (n={agg[dname]['n_events']})")


if __name__ == "__main__":
    main()
