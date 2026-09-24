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
BASIS_TOL = 0.001   # 겹치는 날짜 가격차 0.1% 초과 = 소급조정(분할·배당) → 전체 재수집

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


def _basis_changed(cached: pd.Series, new: pd.Series, tol: float = BASIS_TOL) -> bool:
    """캐시와 새로 받은 꼬리가 겹치는 날짜의 가격이 tol 넘게 다르면 수정주가 기준이 바뀐 것.
    캐시의 마지막 날짜는 비교에서 뺀다 — 장중 실행(미장 메일은 개장 30~90분 후)이면 그 봉은
    확정 종가가 아닌 장중 가격이라, 넣으면 매 실행마다 '기준 변경'으로 오판해 전체 재수집한다."""
    common = cached.index.intersection(new.index)
    common = common[common < cached.index.max()]
    if not len(common):
        return False
    ratio = (new.reindex(common) / cached.reindex(common)).dropna()
    return bool(len(ratio)) and float((ratio - 1).abs().max()) > tol


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
        rebase = []
        for s in need_tail:
            new = fetched.get(s)
            cached = tail_cache[s]
            if new is not None and len(new):
                if _basis_changed(cached, new):
                    rebase.append(s)
                    continue
                merged = pd.concat([cached, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                out[s] = merged
                _save_cache(s, merged)
            else:
                out[s] = cached   # 최근 갱신 실패해도 캐시는 반환(완전 누락보다 낫다)
        if rebase:
            # 수정주가는 분할·배당 때마다 과거 구간 전체가 소급 조정된다 — 옛 기준 캐시에 새
            # 기준 꼬리를 이어붙이면 분할일에 가짜 폭락/폭등이 생긴다(2026-09-24 전략 검토 D).
            _log(f"수정주가 기준 변경(분할·배당 소급조정) 감지 {len(rebase)}종목 → 전체 재수집: {rebase[:10]}")
            _throttle("yahoo")
            fetched = yf_batch_fn(rebase, period) or {}
            for s in rebase:
                ser = fetched.get(s)
                if ser is not None and len(ser):
                    out[s] = ser
                    _save_cache(s, ser)
                else:
                    # 재수집 실패는 일시 장애일 수 있다 — 영구실패 목록에 올리지 말고 옛 캐시를
                    # 반환(기준은 한 번 어긋나 있지만 완전 누락보다 낫고, 다음 실행에 재시도된다).
                    _log(f"{s}: 기준 변경 후 재수집 실패 → 옛 캐시 사용(다음 실행 때 재시도)")
                    out[s] = tail_cache[s]

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
