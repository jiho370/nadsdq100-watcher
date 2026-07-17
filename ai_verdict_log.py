#!/usr/bin/env python3
"""
ai_verdict_log.py — AI 검증(매수유지/관찰강등/제외) verdict의 사후 추적.

목적: verify_stage()가 매긴 verdict가 실제로 이후 수익률과 상관이 있는지 라이브로
쌓아서 검증한다. 사전(과거 시점) 백테스트는 방법론적으로 어렵다 — 최신 LLM에게 과거
날짜를 주고 "그때 어떻게 판단했을지" 재현시키면 이미 그 이후에 일어난 일을 알고
있어서(데이터 누출) 신뢰할 수 없다. 대신 매일 나온 verdict를 전부 기록해두고, 시간이
지나 실제 가격이 쌓이면 그때 사후검증한다.

기록: ai_report.build_report()가 검증 후보 전원(최종 채택 여부 무관)의 verdict+당일가를
      log()로 남긴다.
집계: weekly_report.py가 forward_return_summary()를 호출 — 그룹(긍정/부정)별 표본이
      둘 다 MIN_N 이상 모여야 결과를 반환한다(score_calibration.py의 "표본 부족 시
      조용히 생략" 게이트 패턴과 동일 철학). 표본 부족이면 주간 리포트에 그 섹션 자체가
      안 뜬다 — 수동 개입 없이 표본이 쌓이면 자동으로 켜진다.
"""
from __future__ import annotations
import os, json, datetime as dt

LOG_PATH = os.environ.get("AI_VERDICT_LOG", "output/ai_verdict_log.json")
MIN_DAYS = int(os.environ.get("AI_VERDICT_MIN_DAYS", "28"))   # 최소 경과일(약 4주 forward return)
MIN_N = int(os.environ.get("AI_VERDICT_MIN_N", "10"))          # 그룹당 최소 표본

# verdict 원문을 그룹으로 정규화 — "AI가 부정적으로 봤는가"가 검증하려는 핵심 가설.
_NEGATIVE = {"관찰강등", "강등", "관찰", "제외"}


def _group(verdict: str, severity: str = "") -> str:
    """2026-07-17(지호 님 제안 — "장기 펀더멘탈 훼손일 때만 매도가 나을지도") 대응: 부정
    판정을 severity로 한 번 더 쪼갠다. '제외-일시적' 그룹의 forward return이 실제로
    '매수유지'와 다르지 않다고 나오면 "일시적 악재로는 팔지 마라" 설계가 사후 검증되고,
    반대로 나쁘게 나오면(특히 실적 쇼크류) 그 근거로 티어를 재조정한다. severity 없는
    구버전 로그(관찰강등 등, severity 필드 도입 전)는 '일시적'로 취급(과거 정책 기본값과
    동일한 보수적 처리)."""
    if (verdict or "").strip() not in _NEGATIVE:
        return "긍정(매수유지)"
    return "부정-구조적(제외)" if (severity or "").strip() == "구조적" else "부정-일시적(강등·제외)"


def load() -> list[dict]:
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f).get("entries") or []
    except Exception:
        return []


def log(entries: list[dict]):
    """entries: [{"date","market","symbol","name","verdict","reason","price"}...].
    (date,market,symbol) 중복은 건너뛴다(하루 여러 번 실행돼도 최초 1건만 남음)."""
    if not entries:
        return
    cur = load()
    seen = {(e.get("date"), e.get("market"), e.get("symbol")) for e in cur}
    added = 0
    for e in entries:
        key = (e.get("date"), e.get("market"), e.get("symbol"))
        if key in seen or not all(key) or not e.get("price"):
            continue
        cur.append(e); seen.add(key); added += 1
    if added:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({"entries": cur}, f, ensure_ascii=False, indent=2)


def history_by_symbol(market: str, before_date: str) -> dict[str, list[dict]]:
    """market의 (before_date보다 엄격히 이전인) 로그를 종목별로 최신순 정렬해 반환.
    2026-07-17(지호 님 지적 — REGN/KCC/003030 판정이 새 근거 없이 하루 만에 뒤집힘) 대응:
    ai_report.py가 이걸로 (a)전날 판정을 프롬프트에 주입 (b)제외 판정의 연속일수(점착성)를
    계산한다."""
    entries = load()
    by_sym: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("market") != market or not e.get("date") or e["date"] >= before_date:
            continue
        by_sym.setdefault(e["symbol"], []).append(e)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda e: e["date"], reverse=True)
    return by_sym


def forward_return_summary(price_lookup, min_days: int = MIN_DAYS, min_n: int = MIN_N) -> dict | None:
    """price_lookup(symbol, market)->현재가|None.
    반환: {"긍정(매수유지)":{...}, "부정-일시적(강등·제외)":{...}, "부정-구조적(제외)":{...}}
    또는 표본 부족 시 None. 2026-07-17: 긍정/부정 2그룹 게이트는 그대로(최소 2그룹 이상 +
    각 min_n 이상이어야 표시) — 구조적 그룹은 드물어서(연 0~2회 수준) min_n을 못 채워
    한동안 안 보일 수 있는데, 그래도 긍정 vs 부정-일시적 비교는 표본이 쌓이는 대로 먼저 뜬다."""
    entries = load()
    today = dt.date.today()
    by_group: dict[str, list[float]] = {}
    for e in entries:
        try:
            since = dt.date.fromisoformat(e["date"])
        except Exception:
            continue
        if (today - since).days < min_days:
            continue
        cur = price_lookup(e["symbol"], e.get("market", "us"))
        if not cur or not e.get("price"):
            continue
        ret = (cur / e["price"] - 1) * 100
        g = _group(e.get("verdict", ""), e.get("severity", ""))
        by_group.setdefault(g, []).append(ret)
    qualifying = {g: v for g, v in by_group.items() if len(v) >= min_n}
    if len(qualifying) < 2:
        return None
    return {g: {"avg_ret_pct": round(sum(v) / len(v), 2), "n": len(v)} for g, v in qualifying.items()}


if __name__ == "__main__":   # 스모크 테스트: python ai_verdict_log.py
    import tempfile
    tmp = tempfile.mktemp(suffix=".json")
    LOG_PATH = tmp
    today = dt.date.today().isoformat()
    old = (dt.date.today() - dt.timedelta(days=40)).isoformat()
    log([{"date": old, "market": "us", "symbol": "AAA", "name": "A", "verdict": "매수유지",
         "reason": "", "price": 100.0}])
    log([{"date": old, "market": "us", "symbol": "AAA", "name": "A", "verdict": "매수유지",
         "reason": "", "price": 999.0}])   # 중복 — 무시돼야 함
    assert len(load()) == 1, load()
    prices = {"AAA": 110.0}
    summary = forward_return_summary(lambda s, m: prices.get(s), min_days=28, min_n=1)
    assert summary is None, "그룹이 하나뿐인데 결과가 나옴(2그룹 미만이면 None이어야)"
    print("[ai_verdict_log] self-test 통과")
    os.remove(tmp)
