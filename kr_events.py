#!/usr/bin/env python3
"""
kr_events.py — DART 공시 이벤트 라이브 플래그 (event_study_kr.py 검증 결과의 실전 반영).

event_study_kr.py 8년 이벤트 스터디에서 코스피200 기준 유일하게 HLZ 게이트(t≥3)를 통과한
신호는 '자사주취득 공시'(CAR[+5일] +1.54%p, t=4.49, 3서브기간 전부 양수)였다. 단 +60일엔
소멸하는 단기 효과이므로 6개월 보유 새틀라이트(kr_stocks) 선정에는 넣지 않고, 국장 일간
메일에 '최근 자사주 취득 공시 종목' 정보 블록으로만 표기한다(단기 이벤트 참고 — 자동매매
승격 아님, KR_STRATEGY_OPTIONS §3 신중 원칙).

동작: DART_API_KEY 있으면 최근 N일 코스피200 자사주취득/유상증자 공시를 조회, 없으면 빈
결과(리포트에 블록 미표시 — 그레이스풀). 라이브 발송 파이프라인에 새 필수 의존성을 만들지
않는다.

GitHub Actions에서 쓰려면 Secrets에 DART_API_KEY 등록 필요(없어도 에러 없이 블록만 생략).
"""
from __future__ import annotations
import os, sys, re, datetime as dt

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
BUYBACK_STAT = "과거 8년 공시 후 5거래일 평균 초과수익 +1.5%p(t=4.5, event_study_kr)"

_PAT_BUYBACK = re.compile(r"자기주식취득결정")
_PAT_RIGHTS = re.compile(r"유상증자결정")


def _log(m): print(f"[KR이벤트] {m}", file=sys.stderr)


def recent_events(stock_codes: list[str], lookback_days: int = 5) -> dict:
    """{stock6: {"buyback": rcept_dt} | {"rights": rcept_dt}} — 최근 lookback_days 영업일.
    DART_API_KEY 없거나 실패 시 {} (그레이스풀)."""
    key = os.environ.get("DART_API_KEY")
    if not key or not stock_codes:
        return {}
    try:
        import requests
        from kr_factor_ic import _corp_map
    except Exception as e:
        _log(f"의존성 없음({e}) → 생략"); return {}
    try:
        cmap = _corp_map(key)
    except Exception as e:
        _log(f"corp_code 조회 실패({e}) → 생략"); return {}
    end = dt.date.today()
    bgn = end - dt.timedelta(days=lookback_days + 5)     # 주말 여유 포함
    bgn_s, end_s = bgn.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    out = {}
    for sc in stock_codes:
        corp = cmap.get(sc)
        if not corp:
            continue
        try:
            js = requests.get(LIST_URL, timeout=15, params={
                "crtfc_key": key, "corp_code": corp, "bgn_de": bgn_s, "end_de": end_s,
                "pblntf_ty": "B", "page_count": "30"}).json()
        except Exception:
            continue
        if js.get("status") != "000":
            continue
        for it in (js.get("list") or []):
            nm = it.get("report_nm", "")
            if nm.startswith("[") :                      # 정정공시 제외
                continue
            if _PAT_BUYBACK.search(nm):
                out.setdefault(sc, {})["buyback"] = it["rcept_dt"]
            elif _PAT_RIGHTS.search(nm):
                out.setdefault(sc, {})["rights"] = it["rcept_dt"]
    if out:
        _log(f"최근 이벤트 {len(out)}종목 (자사주취득/유상증자)")
    return out


def events_block_html(events: dict, name_map: dict | None = None) -> str:
    """recent_events 결과 → 국장 메일용 정보 블록. 자사주취득만 표기(검증된 양의 신호).
    events·매치 없으면 빈 문자열(블록 자체 생략)."""
    name_map = name_map or {}
    buys = [(sc, e["buyback"]) for sc, e in events.items() if e.get("buyback")]
    if not buys:
        return ""
    def _esc(s): return (str(s or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    rows = "".join(
        f'<span style="display:inline-block;background:#ecfdf5;color:#065f46;border-radius:6px;'
        f'padding:2px 8px;margin:2px 4px 2px 0;font-size:12px">{_esc(name_map.get(sc, sc))} '
        f'<span style="color:#059669">({d[4:6]}/{d[6:8]})</span></span>'
        for sc, d in sorted(buys, key=lambda x: x[1], reverse=True))
    return (
        '<div style="border:1px solid #a7f3d0;border-radius:10px;padding:10px 13px;margin:10px 0;'
        'background:#f0fdf4"><div style="font-size:13px;font-weight:700;color:#065f46">'
        '&#128276; 최근 자사주 취득 공시 <span style="color:#9ca3af;font-size:11px;font-weight:400">'
        f'(단기 이벤트 참고 — {_esc(BUYBACK_STAT)})</span></div>'
        f'<div style="margin-top:5px">{rows}</div>'
        '<div style="font-size:11px;color:#9ca3af;margin-top:4px">※ 6개월 보유 추천과 별개인 '
        '단기(약 1~4주) 신호입니다. 자동 편입 대상이 아니며 참고용입니다.</div></div>')


if __name__ == "__main__":   # 스모크: python kr_events.py  (DART_API_KEY 필요)
    ev = recent_events(["005930", "000660", "005380"], lookback_days=10)
    print(ev)
    print(events_block_html({"005930": {"buyback": "20260318"}}, {"005930": "삼성전자"})[:120])
