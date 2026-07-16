#!/usr/bin/env python3
"""
gen_profiles.py — 종목 '사업 프로필' 캐시를 1회 생성(분기 1회 재실행). 로컬 CLI 우선, $0.

왜 캐시인가:
  "무슨 회사이고 무엇으로 돈 버는지"(②펀더멘털 축)는 매일 바뀌지 않는다.
  이걸 매일 AI에게 다시 쓰게 하는 것이 기존 비용의 큰 부분이었다.
  → 전 종목을 '한 번' 생성해 두고 데일리 리포트는 재사용(ai_report._profiles).

왜 CLI가 기본인가:
  분기 1회·~900종목(20개씩 묶어 ~45요청)짜리 일회성 로컬 작업이라 배치의 '동시 처리'
  이점이 꼭 필요하지 않다. 로컬 claude -p(Pro 구독)로 순차 호출해도 몇 분이면 끝나고
  비용은 $0 — Batch API의 50% 할인(유료)보다 구독 쪽이 항상 더 싸다.
  claude CLI가 없는 환경(로컬 미설치)에서만 기존 Batch API(ANTHROPIC_API_KEY 필요)로 폴백.

대상:
  · sp500_profiles.json  → tickers[sym].detail 이 빈 종목만 채움 (--refresh 면 전체)
  · kospi200_profiles.json → output/kospi200_cache.json 의 구성종목으로 생성(없으면 생략)

실행(로컬):
  python gen_profiles.py            # 빈 것만
  python gen_profiles.py --refresh  # 전체 재생성 (분기 1회 권장)
CLI 있으면 $0. 없고 ANTHROPIC_API_KEY만 있으면 Batch API 폴백
(700종목 × haiku 배치 단가 ≈ $0.1 미만).
"""
from __future__ import annotations
import os, sys, json, time, shutil, argparse, datetime

MODEL = os.environ.get("PROFILE_MODEL", "claude-haiku-4-5")
# 2026-07-16(지호 님 지적 — 아세아 오서술 사건): 업종 분류 '근거'만 주는 건 여전히 기억 기반
# 지어내기다 — 실제로 검색해서 확인한 정보로 쓰는 게 원래 요청이었다. 한국 종목은 검색을
# 켜고(WebSearch), 검색+판단이 필요한 작업이라 ai_report.py의 다른 검색 단계(verify_stage)와
# 동일하게 haiku가 아니라 sonnet을 쓴다.
MODEL_SEARCH = os.environ.get("PROFILE_MODEL_SEARCH", "claude-sonnet-5")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CHUNK = 20            # 요청당 종목 수 — 출력 JSON 이 안정적으로 파싱되는 크기
CHUNK_SEARCH = 5       # 검색 모드는 종목마다 실제 검색이 필요해 묶음을 작게(요청 수는 늘지만
                       # 회사당 검색 품질을 우선) — 미국(GICS 근거 있음)은 기존 방식 유지, 한국만 적용.
POLL_SEC = 20
KR_PROFILE_PATH = "kospi200_profiles.json"
US_PROFILE_PATH = "sp500_profiles.json"

_SYSTEM_BASE = (
    "당신은 종목 사전을 만드는 애널리스트다. 각 종목에 대해 두 가지를 쓴다(2026-07-15 개편 —"
    " 리포트에서 '종목 설명'과 '②사업'으로 각각 표시되며 겹치면 안 된다):\n"
    "1) one_liner(한 문장, 40~60자): 그 회사가 지금 실제로 뭘로 돈을 버는 회사인지 일반적으로"
    " 서술 — 브랜드 나열이 아니어도 됨. 숫자·수익구조 세부 설명 금지(그건 detail 몫).\n"
    "2) detail(한국어 2~3문장, 130~180자 내외): one_liner와 안 겹치게 — 무엇으로 돈 버는지"
    "(사업부문 구성·어느 쪽이 핵심인지 상대적으로만, 예: 'A가 압도적으로 크고 B가 보조'),"
    " 업계 내 위치·차별점을 다룬다.\n"
    "**중요(2026-07-15 추가 — 실사용 중 발견된 편향 수정): '회사가 내세우는 미래 전략·비전'을"
    " '현재 사업의 실체'처럼 쓰지 않는다.** 예: GM이 '전기차 전환 목표'를 발표했다고 해서"
    " one_liner를 '전기차 전환 중심 자동차 회사'라고 쓰면 안 된다 — 실제 매출·이익의 압도적"
    " 비중은 여전히 내연기관 픽업트럭·SUV에서 나오고, EV는 아직 적자에 목표도 후퇴한 상태이니"
    " '내연기관 픽업트럭·SUV가 주력이고 전동화는 진행 중(성과는 불확실)' 정도가 정확한 서술."
    " 현재 실제 매출·이익 비중이 큰 사업을 one_liner·detail의 중심으로 삼고, 미래 전략은"
    " '진행 중'·'추진 중' 같은 미완료 표현으로 부차적으로만 언급한다(전략 성공을 기정사실화"
    " 금지). 확신이 없으면 회사의 오래되고 확실한 핵심 사업만 쓴다.\n"
    "규칙: 이 글은 캐시로 오래 재사용된다 — 최신 뉴스·주가·시점 표현('최근','올해','2025년 기준'"
    " 등 특정 연도 고정 표현) 금지. **정확한 매출액·비율 등 구체적 수치는 절대 쓰지 않는다**"
    "(부정확하거나 철 지난 것일 위험이 커서 금지 — '압도적이다/절반 이상이다' 같은 정성적"
    " 비교 표현만 허용). 쉬운 한국어, 한자 금지. 두 필드 모두 반드시 채운다(분량은 목표치 —"
    " 표시 단계에서 넘치면 문장 경계로 다시 자른다)."
)

# 미국(US): 실사용 중 검증된 GICS sub_industry 근거 + 기억 기반(검색 없음, 저비용 haiku).
_SYSTEM = (
    _SYSTEM_BASE + "\n"
    "**입력 종목명 뒤 괄호 안 업종(있으면)은 거래소(GICS)의 검증된 공식 분류다 — 반드시"
    " 그 업종과 맞는 사업으로 서술할 것.**\n"
    "**도구를 쓸 수 없다 — 파일을 찾아보거나 웹을 검색하지 말고, 확인 절차를 설명하는 문장도"
    " 쓰지 마라. 주어진 심볼·이름·업종만으로 그 자리에서 바로 판단해 JSON만 출력한다.**\n"
    '출력은 JSON 하나만: {"SYM1":{"one_liner":"...","detail":"..."},"SYM2":{...}}'
)

# 한국(KR): 2026-07-16 재설계(지호 님 지적 — "업종 분류를 근거로 서술하는 게 아니라 실제
# 정보를 검색 기반으로 생성해달라 한 거야"). 아세아(002030)를 '자동차부품사'로 완전히 잘못
# 서술한 사고를 업종 태그만으로 막으려 했으나, 지주회사처럼 업종 분류 자체가 애매한 경우
# (KRX 분류 "일반서비스")엔 태그를 줘도 AI가 또 다른 틀린 추측('백화점·마트 유통업')을
# 내놓는 걸 테스트로 확인 — 근본 해법은 태그가 아니라 실제 검색이었다. WebSearch를 켜고
# 종목마다 실제로 검색해 확인하도록 강제. 검색+판단 작업이라 haiku가 아니라 sonnet 사용,
# 묶음도 20→5로 줄여 종목당 검색 품질을 우선한다(CHUNK_SEARCH).
_SYSTEM_SEARCH = (
    _SYSTEM_BASE + "\n"
    "**반드시 웹검색으로 확인한다 — 기억에 의존해 지어내지 마라.** 각 종목마다 최소 1회"
    " 이상 검색해서 그 회사가 실제로 무슨 사업을 하는지(공식 홈페이지·사업보고서·뉴스 등)"
    " 확인한 뒤에 쓴다. 이름이 비슷하거나 덜 알려진 회사를 다른 유명한 회사와 혼동해 완전히"
    " 다른 업종으로 지어내는 사고가 실제로 발생했다(예: 지주회사 '아세아'를 검색 없이"
    " '자동차부품사'로 서술). 검색 결과가 자신의 기억과 다르면 검색 결과를 따른다. 검색해도"
    " 확실한 정보를 못 찾으면 지어내지 말고 회사명·업종에서 유추 가능한 가장 안전하고 일반적인"
    " 서술로 그친다(예: '지주회사로 여러 계열사를 보유' 정도). 종목별로 따로 검색하되, 검색"
    " 횟수가 제한되니 한 번의 검색으로 여러 정보를 확인할 수 있게 검색어를 효율적으로 구성"
    " 한다(예: '{회사명} 사업보고서' 또는 '{회사명} 주요사업').\n"
    '출력은 JSON 하나만: {"SYM1":{"one_liner":"...","detail":"..."},"SYM2":{...}}'
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
    """kospi200_profiles.json(없으면 새로) + KRX 캐시에서 구성종목.

    2026-07-16 수정(지호 님 리포트 — 아세아(002030)를 AI가 '자동차 부품사'로 완전히 잘못
    서술한 사례): _collect_us는 GICS sub_industry를 세 번째 값으로 넘겨 AI가 그 업종
    맥락 안에서만 회사를 서술하게 하는데, 이 함수는 그 자리에 빈 문자열(""))만 넘기고
    있었다 — 즉 한국 쪽만 AI가 심볼·이름 두 개만 보고 '학습 데이터 기억'에 전적으로
    의존해 서술을 지어냈다(같은 이름의 다른 회사와 혼동하기 쉬움). kr_sector.py가 이미
    확보해둔 KRX 공식 업종분류(get_market_sector_classifications, 26종)를 세 번째 값으로
    넘겨 US와 동일하게 업종 맥락을 앵커로 준다 — 완전한 환각 방지는 아니지만(맥락이 있어도
    세부 내용은 여전히 기억에 의존) '전혀 다른 업종으로 착각'하는 최악의 사례는 크게 줄어든다."""
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
    sec_by_ticker = {}
    try:
        import kr_sector as KS
        today8 = datetime.date.today().strftime("%Y%m%d")
        sec_by_date = KS.fetch_sectors([today8])
        sec_by_ticker = sec_by_date.get(today8) or {}
        _log(f"KRX 업종분류 확보: {len(sec_by_ticker)}종목")
    except Exception as e:
        _log(f"KRX 업종분류 조회 실패({type(e).__name__}: {e}) → 업종 맥락 없이 진행")
    tk = prof.setdefault("tickers", {})
    todo = []
    for code, row in uni.items():
        cur = tk.setdefault(code, {"name": row.get("name", ""), "one_liner": "", "detail": ""})
        if refresh or not (cur.get("detail") or "").strip():
            todo.append((code, row.get("name", ""), sec_by_ticker.get(code, "")))
    return prof, todo


def _build_requests(todo: list, prefix: str, chunk_size=CHUNK, model=MODEL,
                    system=_SYSTEM, web=False) -> list:
    """chunk_size종목씩 묶은 배치 요청 목록. custom_id 로 결과를 되찾는다.
    2026-07-16: model/system/web을 파라미터화 — 한국은 검색모드(_SYSTEM_SEARCH·sonnet·
    web=True·CHUNK_SEARCH)를, 미국은 기존 방식(_SYSTEM·haiku·검색 없음·CHUNK)을 쓴다."""
    reqs = []
    for i, chunk in enumerate(_chunks(todo, chunk_size)):
        lines = [f"- {sym}: {name}" + (f" ({ind})" if ind else "") for sym, name, ind in chunk]
        reqs.append({
            "custom_id": f"{prefix}-{i}",
            "params": {
                "model": model, "max_tokens": 4000, "system": system, "web": web,
                "messages": [{"role": "user",
                              "content": "다음 종목들의 one_liner·detail 을 작성하라. 키는 심볼 그대로.\n"
                                         + "\n".join(lines)}]},
        })
    return reqs


def _run_cli(reqs: list) -> dict:
    """로컬 claude -p(Pro 구독, $0)로 순차 실행. 분기 1회 일회성 작업이라 배치의
    동시 처리 없이 순차로도 충분(요청당 몇 초~수십 초, 전체 몇 분).
    2026-07-15: 703종목 --refresh 시 36청크 중 일부가 이유 불명 타임아웃/부분파싱실패로
    누락되는 게 확인돼(예: 'timed out after -1186초'처럼 음수 타임아웃까지 관측 — CLI
    프로세스 자체의 일시적 이상으로 추정, 원인 미상) 1회 재시도를 추가한다.
    2026-07-16: model/system/web을 req["params"]에서 그대로 읽는다(하드코딩된 _SYSTEM/MODEL
    대신) — 검색모드 요청(한국)과 기존 요청(미국)이 섞여도 각자 맞는 설정으로 호출된다."""
    import ai_report as AR
    out = {}
    for i, req in enumerate(reqs, 1):
        cid = req["custom_id"]
        p = req["params"]
        instr = p["messages"][0]["content"]
        _log(f"  CLI 요청 {i}/{len(reqs)}: {cid}" + (" (검색)" if p.get("web") else ""))
        for attempt in (1, 2):
            try:
                out[cid] = AR._call_cli(instr, web=p.get("web", False), system=p["system"],
                                        model=p["model"])
                break
            except Exception as e:
                _log(f"  실패({attempt}/2): {cid} ({type(e).__name__}: {e})")
    return out


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
    """배치 응답(JSON 문자열)을 profiles 파일 구조에 병합.
    2026-07-15: 응답이 {"SYM":{"one_liner":..,"detail":..}} 형태(개편) — 구버전(문자열만)도
    하위호환으로 detail에 그대로 넣는다(과도기 대비, 곧 --refresh로 전부 새 형식이 됨)."""
    from ai_commentary import _extract_json
    n = 0
    for cid, text in results.items():
        if not cid.startswith(prefix):
            continue
        parsed = _extract_json(text or "") or {}
        for sym, val in parsed.items():
            if sym not in tickers:
                continue
            if isinstance(val, dict):
                ol, det = (val.get("one_liner") or "").strip(), (val.get("detail") or "").strip()
                if ol:
                    tickers[sym]["one_liner"] = ol
                if det:
                    tickers[sym]["detail"] = det
                if ol or det:
                    n += 1
            elif isinstance(val, str) and val.strip():
                tickers[sym]["detail"] = val.strip()
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="빈 것만이 아니라 전체 재생성")
    ap.add_argument("--limit", type=int, default=None,
                    help="실험용 — 미국/한국 각각 앞 N종목만 처리(전체 실행 전 CLI 동작 확인용)")
    ap.add_argument("--dry-run", action="store_true",
                    help="실험용 — 결과를 파일에 저장하지 않고 콘솔에만 출력")
    args = ap.parse_args()

    use_cli = shutil.which(CLAUDE_BIN) is not None or os.path.exists(CLAUDE_BIN)
    if use_cli:
        _log(f"로컬 claude CLI 사용({CLAUDE_BIN}, Pro 구독 — $0)")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        _log("claude CLI 없음 → API Batch로 폴백(유료 — ANTHROPIC_API_KEY 사용)")
    else:
        sys.exit("claude CLI도 ANTHROPIC_API_KEY도 없음 — 프로필 생성 불가")

    us_prof, us_todo = _collect_us(args.refresh)
    kr_prof, kr_todo = _collect_kr(args.refresh)
    if args.limit:
        us_todo, kr_todo = us_todo[:args.limit], kr_todo[:args.limit]
        _log(f"--limit {args.limit}: 미국/한국 각각 앞 {args.limit}종목만 처리")
    _log(f"생성 대상: 미국 {len(us_todo)} · 한국 {len(kr_todo)}종목")
    if not (us_todo or kr_todo):
        _log("채울 것이 없음 — 종료 (--refresh 로 전체 재생성 가능)"); return

    reqs = (_build_requests(us_todo, "us")
           + _build_requests(kr_todo, "kr", chunk_size=CHUNK_SEARCH, model=MODEL_SEARCH,
                             system=_SYSTEM_SEARCH, web=True))
    if use_cli:
        results = _run_cli(reqs)
    else:
        import anthropic
        results = _run_batch(anthropic.Anthropic(), reqs)

    if args.dry_run:
        _log("--dry-run: 저장하지 않고 결과만 출력")
        for cid, text in results.items():
            _log(f"  [{cid}] {(text or '')[:400]}")
        return

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
