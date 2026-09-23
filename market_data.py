#!/usr/bin/env python3
"""
market_data.py — 시세 수집 공용 캐시·재시도·폴백 계층(2026-09-23, 지호 님 요청).

이번 세션에서 겪은 문제를 한 곳에서 고친다(HKUDS/Vibe-Trading 저장소의 캐시 정책
설계를 부분 차용):
  1) backtest_costs._purge_yf_cache()가 PIT 패널을 다시 만들 때마다 로컬 캐시를
     통째로 지우고 ~700종목을 매번 새로 받아서, 반복 실행 시 레이트리밋으로 커버리지가
     607/703 → 108/703까지 붕괴하는 사고가 여러 번 있었다. → 확정된 과거 구간은
     디스크에 유지하고, 캐시가 이미 요청 기간만큼 깊으면 최근 10거래일만 다시 받아
     이어붙인다(vibe-trading의 "end_date가 오늘 이전일 때만 캐싱" 정책과 같은 발상).
  2) 상장폐지·리브랜딩으로 영구히 실패하는 티커(SRCL·HCP·CBS 등 ~94개)가 풀 재빌드마다
     매번 재시도됐다. → 실패를 state/known_bad_tickers.json에 기록해 스킵하고,
     아주 드문 재상장 대비 90일에 한 번만 재확인한다.

2026-09-23 시도했다가 제거: stooq.com 무료 폴백(vibe-trading 폴백체인 아이디어 차용)을
넣었었는데, stooq가 JS 챌린지로 봇 요청을 막고 있어(requests로는 통과 불가) 실질적으로
전혀 동작하지 않는 죽은 코드였다 — 지호 님 확인 후 제거.

이 모듈은 야후 SDK를 직접 모른다 — 실제 배치 다운로드 함수는 호출부(현재는
sp500_daily_report.download_histories() 하나)가 콜백으로 넘긴다. 그 함수만 이 계층을
거치면 daily_ai_report.py·weekly_report.py·backtest_costs.py 등 기존 12개 호출부는
전부 자동으로 혜택을 받고 코드 변경이 필요 없다.

캐시(output/price_cache/)는 재생성 가능한 로컬 전용 캐시라 커밋 안 함(.gitignore).
known_bad_tickers.json은 CI "Persist state" 단계가 커밋해 로컬·GitHub Actions 양쪽에서
누적된다.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

import pandas as pd

CACHE_DIR = "output/price_cache"
BAD_TICKER_PATH = "state/known_bad_tickers.json"
BAD_RECHECK_DAYS = 90
TAIL_REFRESH_DAYS = 10
MIN_CALL_INTERVAL = 0.5

_last_call: dict[str, float] = {}


def _log(m): print(f"[시세수집] {m}", file=sys.stderr)


def _throttle(host: str):
    wait = MIN_CALL_INTERVAL - (time.time() - _last_call.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.time()


def _period_to_days(period: str) -> int | None:
    m = re.fullmatch(r"(\d+)(y|mo|d)", (period or "").strip())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"y": 365, "mo": 30, "d": 1}[unit]


# ------------------------- 영구실패 목록 -------------------------
def _load_bad() -> dict:
    try:
        with open(BAD_TICKER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_bad(bad: dict):
    os.makedirs("state", exist_ok=True)
    with open(BAD_TICKER_PATH, "w", encoding="utf-8") as f:
        json.dump(bad, f, ensure_ascii=False, indent=2, sort_keys=True)


def _is_known_bad(symbol: str, bad: dict, today: pd.Timestamp) -> bool:
    e = bad.get(symbol)
    if not e:
        return False
    age = (today - pd.Timestamp(e["marked"])).days
    return age < BAD_RECHECK_DAYS


def _mark_bad(symbols: list[str], bad: dict, reason: str = "no_data"):
    if not symbols:
        return
    today = pd.Timestamp.today().date().isoformat()
    for s in symbols:
        bad[s] = {"marked": today, "reason": reason}
    _save_bad(bad)
    _log(f"영구실패 기록 {len(symbols)}종목(재확인 {BAD_RECHECK_DAYS}일 뒤): {sorted(symbols)[:10]}{'...' if len(symbols) > 10 else ''}")


# ------------------------- 로컬 가격 캐시 -------------------------
def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol.replace('/', '_')}.pkl")


def _load_cache(symbol: str) -> pd.Series | None:
    try:
        s = pd.read_pickle(_cache_path(symbol))
        return s if isinstance(s, pd.Series) and len(s) else None
    except Exception:
        return None


def _save_cache(symbol: str, series: pd.Series):
    if series is None or series.empty:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        series.to_pickle(_cache_path(symbol))
    except Exception:
        pass


def fetch_batch(symbols: list[str], period: str, yf_batch_fn) -> dict[str, pd.Series]:
    """symbols를 캐시 상태로 나눠 최소한만 야후에 요청 — 캐시가 이미 요청 기간만큼
    깊고 최근이면 최근 며칠만 다시 받아 이어붙이고, 없거나 얕으면 전체를 새로 받는다.
    야후에서 끝내 실패한 종목은 영구실패 목록에 기록해 다음부터 건너뛴다.
    yf_batch_fn(symbols, period) -> {symbol: Series} — 실제 야후 배치 호출(호출부 제공)."""
    today = pd.Timestamp.today().normalize()
    bad = _load_bad()
    todo = [s for s in symbols if not _is_known_bad(s, bad, today)]
    skipped = len(symbols) - len(todo)
    if skipped:
        _log(f"영구실패 목록으로 {skipped}종목 건너뜀")

    days = _period_to_days(period)
    need_start = today - pd.Timedelta(days=days) if days else None
    stale_cutoff = today - pd.Timedelta(days=TAIL_REFRESH_DAYS + 5)

    out: dict[str, pd.Series] = {}
    need_full, need_tail, tail_cache = [], [], {}
    for s in todo:
        c = _load_cache(s)
        deep_enough = c is not None and (need_start is None or c.index.min() <= need_start)
        fresh_enough = c is not None and c.index.max() >= stale_cutoff
        if deep_enough and fresh_enough:
            need_tail.append(s)
            tail_cache[s] = c
        else:
            need_full.append(s)

    if need_full:
        _throttle("yahoo")
        fetched = yf_batch_fn(need_full, period) or {}
        for s, ser in fetched.items():
            out[s] = ser
            _save_cache(s, ser)
    if need_tail:
        _throttle("yahoo")
        fetched = yf_batch_fn(need_tail, f"{TAIL_REFRESH_DAYS + 5}d") or {}
        for s in need_tail:
            new = fetched.get(s)
            cached = tail_cache[s]
            if new is not None and len(new):
                merged = pd.concat([cached, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                out[s] = merged
                _save_cache(s, merged)
            else:
                out[s] = cached   # 최근 갱신 실패해도 캐시는 반환(완전 누락보다 낫다)

    recovered = [s for s in todo if s in out and s in bad]
    if recovered:
        for s in recovered:
            bad.pop(s, None)
        _save_bad(bad)

    still_missing = [s for s in todo if s not in out]
    if still_missing:
        _log(f"야후 최종 실패 {len(still_missing)}종목")
        _mark_bad(still_missing, bad)
    return out
