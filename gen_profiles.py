#!/usr/bin/env python3
"""
gen_profiles.py — 종목 '사업 프로필' 캐시를 Batch API + haiku 로 1회 생성 (분기 1회 재실행).

왜 캐시인가:
  "무슨 회사이고 무엇으로 돈 버는지"(②펀더멘털 축)는 매일 바뀌지 않는다.
  이걸 매일 AI에게 다시 쓰게 하는 것이 기존 비용의 큰 부분이었다.
  → 전 종목을 '한 번' 생성해 두고 데일리 리포트는 재사용(ai_report._profiles).

왜 Batch API인가:
  마감이 없는 대량 작업이라 배치가 정확히 맞는 용도 — 요금 50% 할인(입·출력 모두).
  20종목씩 묶어 요청 수를 줄이고, 제출 후 폴링해 결과를 파일에 써 넣는다.

대상:
  · sp500_profiles.json  → tickers[sym].detail 이 빈 종목만 채움 (--refresh 면 전체)
  · kospi200_profiles.json → output/kospi200_cache.json 의 구성종목으로 생성(없으면 생략)

실행(로컬, ANTHROPIC_API_KEY 필요):
  python gen_profiles.py            # 빈 것만
  python gen_profiles.py --refresh  # 전체 재생성 (분기 1회 권장)
예상 비용: 700종목 × (입력 ~100 + 출력 ~150토큰) × haiku 배치 단가 ≈ $0.1 미만.
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime

import anthropic

MODEL = os.environ.get("PROFILE_MODEL", "claude-haiku-4-5")
CHUNK = 20            # 요청당 종목 수 — 출력 JSON 이 안정적으로 파싱되는 크기
POLL_SEC = 20
KR_PROFILE_PATH = "kospi200_profiles.json"
US_PROFILE_PATH = "sp500_profiles.json"

_SYSTEM = (
    "당신은 종목 사전을 만드는 애널리스트다. 각 종목에 대해 detail(한국어 2~3문장)을 쓴다:\n"
    "① 무엇으로 돈 버는 회사인지(주력 사업·수익 구조) ② 업계 내 위치(경쟁 지위·차별점) "
    "③ 투자자가 알아야 할 구조적 특징(사이클·규제·고객 집중 등).\n"
    "규칙: 이 글은 캐시로 오래 재사용된다 — 최신 뉴스·주가·시점 표현('최근','올해') 금지. "
    "수치를 지어내지 않는다. 쉬운 한국어, 한자 금지.\n"
    '출력은 JSON 하나만: {"SYM1":"detail 문장...","SYM2":"..."}'
)


def _log(m): print(f"[PROFILE] {m}", file=sys.stderr)


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _collect_us(refresh: bool) -> tuple[dict, list]:
    """sp500_profiles.json 로드 + 채울 종목 목록."""
    with open(US_PROFILE_PATH, encoding="utf-8") as f:
        prof = json.load(f)
    tickers = prof.get("tickers") or {}
    todo = [(sym, t.get("name", ""), t.get("sub_industry") or t.get("gics_sector", ""))
            for sym, t in tickers.items() if refresh or not (t.get("detail") or "").strip()]
    return prof, todo


def _collect_kr(refresh: bool) -> tuple[dict, list]:
    """kospi200_profiles.json(없으면 새로) + KRX 캐시에서 구성종목."""
    try:
        with open(KR_PROFILE_PATH, encoding="utf-8") as f:
            prof = json.load(f)
    except Exception:
        prof = {"meta": {"note": "코스피200 사업 프로필 — gen_profiles.py 생성"}, "tickers": {}}
    try:
        with open("output/kospi200_cache.json", encoding="utf-8") as f:
            uni = (json.load(f) or {}).get("data") or {}
    except Exception:
        _log("output/kospi200_cache.json 없음 → 한국 생략(데일리 1회 실행 후 다시)")
        return prof, []
    tk = prof.setdefault("tickers", {})
    todo = []
    for code, row in uni.items():
        cur = tk.setdefault(code, {"name": row.get("name", ""), "one_liner": "", "detail": ""})
        if refresh or not (cur.get("detail") or "").strip():
            todo.append((code, row.get("name", ""), ""))
    return prof, todo


def _build_requests(todo: list, prefix: str) -> list:
    """20종목씩 묶은 배치 요청 목록. custom_id 로 결과를 되찾는다."""
    reqs = []
    for i, chunk in enumerate(_chunks(todo, CHUNK)):
        lines = [f"- {sym}: {name}" + (f" ({ind})" if ind else "") for sym, name, ind in chunk]
        reqs.append({
            "custom_id": f"{prefix}-{i}",
            "params": {
                "model": MODEL, "max_tokens": 4000, "system": _SYSTEM,
                "messages": [{"role": "user",
                              "content": "다음 종목들의 detail 을 작성하라. 키는 심볼 그대로.\n"
                                         + "\n".join(lines)}]},
        })
    return reqs


def _run_batch(client, reqs: list) -> dict:
    """배치 제출 → 폴링 → {custom_id: 응답텍스트}. (마감 없는 작업 = 배치 최적)"""
    if not reqs:
        return {}
    batch = client.messages.batches.create(requests=reqs)
    _log(f"배치 제출: {batch.id} ({len(reqs)}요청)")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        _log(f"  상태 {b.processing_status} — 완료 {c.succeeded}/{len(reqs)} 오류 {c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(POLL_SEC)
    out = {}
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            msg = r.result.message
            out[r.custom_id] = "".join(bk.text for bk in msg.content
                                       if getattr(bk, "type", "") == "text")
        else:
            _log(f"  실패: {r.custom_id} ({r.result.type})")
    return out


def _merge(results: dict, prefix: str, tickers: dict) -> int:
    """배치 응답(JSON 문자열)을 profiles 파일 구조에 병합."""
    from ai_commentary import _extract_json
    n = 0
    for cid, text in results.items():
        if not cid.startswith(prefix):
            continue
        parsed = _extract_json(text or "") or {}
        for sym, detail in parsed.items():
            if sym in tickers and isinstance(detail, str) and detail.strip():
                tickers[sym]["detail"] = detail.strip()
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="빈 것만이 아니라 전체 재생성")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY 가 필요합니다 (배치는 API 전용 — 1회성·50% 할인)")
    client = anthropic.Anthropic()

    us_prof, us_todo = _collect_us(args.refresh)
    kr_prof, kr_todo = _collect_kr(args.refresh)
    _log(f"생성 대상: 미국 {len(us_todo)} · 한국 {len(kr_todo)}종목")
    if not (us_todo or kr_todo):
        _log("채울 것이 없음 — 종료 (--refresh 로 전체 재생성 가능)"); return

    results = _run_batch(client, _build_requests(us_todo, "us") + _build_requests(kr_todo, "kr"))

    today = datetime.date.today().isoformat()
    if us_todo:
        n = _merge(results, "us", us_prof["tickers"])
        us_prof.setdefault("meta", {})["profiles_updated"] = today
        with open(US_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(us_prof, f, ensure_ascii=False, indent=1)
        _log(f"{US_PROFILE_PATH}: {n}종목 기록")
    if kr_todo:
        n = _merge(results, "kr", kr_prof["tickers"])
        kr_prof.setdefault("meta", {})["profiles_updated"] = today
        with open(KR_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(kr_prof, f, ensure_ascii=False, indent=1)
        _log(f"{KR_PROFILE_PATH}: {n}종목 기록")


if __name__ == "__main__":
    main()
