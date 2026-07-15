#!/usr/bin/env python3
"""
kr_sector.py — 코스피 종목 업종 분류 (2026-07-15, 새틀라이트 섹터캡 실험용).

지호 님 질문(새틀라이트 종목이 특정 업종에 쏠리지 않게 캡을 걸 수 있나) 대응. kr_stocks.py의
"sector" 필드는 빈 문자열로 방치돼 있었으나, pykrx가 KRX 공식 업종 분류를
get_market_sector_classifications()로 이미 제공한다 — 26개 업종(화학·기타금융·전기전자 등),
날짜별 조회 가능. backtest_kr.py의 fetch_membership/fetch_fundamentals와 동일한
"날짜별 캐시" 패턴을 따른다.

실행: 이 모듈은 단독 실행하지 않고 kr_topn_ratio_sweep.py 등에서 import해서 쓴다.
결과: output/kr_sector_cache.json
"""
from __future__ import annotations
import sys, json

CACHE_PATH = "output/kr_sector_cache.json"


def _log(m): print(f"[KR섹터] {m}", file=sys.stderr)


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(c):
    import os
    os.makedirs("output", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def fetch_sectors(dates_yyyymmdd: list[str], cache=None) -> dict:
    """리밸 날짜별 {ticker6: 업종명}. 비거래일이면 최대 7일 소급 재시도.
    반환: {date: {ticker: sector}} — 실패한 날짜는 빈 dict(호출부가 최근 유효값으로 폴백)."""
    from pykrx import stock as K
    import datetime as dt
    cache = cache if cache is not None else _load_cache()
    sec = cache.setdefault("sectors", {})
    for d in dates_yyyymmdd:
        if d in sec:
            continue
        got = {}
        day = dt.datetime.strptime(d, "%Y%m%d")
        for back in range(8):
            try_d = (day - dt.timedelta(days=back)).strftime("%Y%m%d")
            try:
                df = K.get_market_sector_classifications(try_d, market="KOSPI")
                if df is not None and len(df):
                    col = df.columns[1]     # 업종명 컬럼(한글 헤더 인코딩 이슈 회피용 위치 참조)
                    got = {t: str(row[col]) for t, row in df.iterrows()}
                    break
            except Exception:
                continue
        if not got:
            _log(f"업종 조회 실패 {d} (최대 7일 소급도 실패)")
        sec[d] = got
        _save_cache(cache)
    return sec


def sector_of(sec_by_date: dict, date_yyyymmdd: str, ticker: str) -> str | None:
    """해당 날짜(없으면 그 이전 가장 가까운 캐시된 날짜)의 종목 업종. 못 찾으면 None."""
    if date_yyyymmdd in sec_by_date and ticker in sec_by_date[date_yyyymmdd]:
        return sec_by_date[date_yyyymmdd][ticker]
    for d in sorted((k for k in sec_by_date if k <= date_yyyymmdd), reverse=True):
        if ticker in sec_by_date[d]:
            return sec_by_date[d][ticker]
    return None
