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

# verdict 원문을 두 그룹으로 정규화 — "AI가 부정적으로 봤는가"가 검증하려는 핵심 가설.
_NEGATIVE = {"관찰강등", "강등", "관찰", "제외"}


def _group(verdict: str) -> str:
    return "부정(강등·제외)" if (verdict or "").strip() in _NEGATIVE else "긍정(매수유지)"


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


def forward_return_summary(price_lookup, min_days: int = MIN_DAYS, min_n: int = MIN_N) -> dict | None:
    """price_lookup(symbol, market)->현재가|None.
    반환: {"긍정(매수유지)":{"avg_ret_pct","n"}, "부정(강등·제외)":{...}} 또는
    표본 부족 시 None(둘 중 하나라도 min_n 미만이면 아예 표시 안 함)."""
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
        by_group.setdefault(_group(e.get("verdict", "")), []).append(ret)
    if len(by_group) < 2 or any(len(v) < min_n for v in by_group.values()):
        return None
    return {g: {"avg_ret_pct": round(sum(v) / len(v), 2), "n": len(v)} for g, v in by_group.items()}


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
