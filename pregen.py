#!/usr/bin/env python3
"""
pregen.py — 로컬 PC에서 Pro 구독 CLI(claude -p)로 AI 검증을 '미리' 생성 (메일 2통 체계).

  --kr : 한국장 메일(다음날 08:00)용. 실행 창 = 저녁 16시 이후(장 마감 확정) 또는
         다음날 새벽 08시 이전(부팅 보충). 장중(08~16시)엔 데이터가 애매해 스킵.
         → output/pregen_kr.json  (for_kst = 다음 08:00 발송일)
  --us : 미국장 메일(당일 17:00)용. 실행 창 = 06시(미국장 마감 후)~16시(발송 전).
         그 외 시각엔 이미 발송됐거나 데이터 미확정이라 스킵.
         → output/pregen_us.json  (for_kst = 오늘)

아침/오후 GitHub Actions 는 pregen_{kr,us}.json 의 for_kst 가 발송일과 일치하면
검증 단계(웹검색)를 통째로 생략 → API 비용이 haiku 서술만 남는다.
PC가 꺼져 있어 파일이 없으면 Actions 가 API 검증으로 자동 폴백 — 발송엔 지장 없음.

작업 스케줄러(register_pregen_task.ps1): KR=매일 19:00, US=매일 09:30,
둘 다 StartWhenAvailable(놓치면 다음 부팅 시 실행 — 시간 창 가드가 유효성 판단).
"""
from __future__ import annotations
import os, sys, json, argparse, datetime

# 구독 CLI 강제(이 스크립트는 로컬 PC 전용 — API 과금 경로로 새지 않게)
os.environ["AI_BACKEND"] = "cli"
os.environ.setdefault("REPORT_WEB", "1")
os.environ.setdefault("AI_TIMEOUT", "1200")

import sp500_daily_report as R
import export_data as E
import ai_report as AR

POOL_BUFFER = int(os.environ.get("PREGEN_POOL_BUFFER", "3"))   # 후보 풀 여유분(순위 변동 대비)


def _log(m): print(f"[PREGEN] {m}", file=sys.stderr)


def _holding_syms(path: str) -> list[dict]:
    """보유 종목을 '악재 점검' 대상으로. 상태파일은 읽기만 한다(밤에 수정 금지)."""
    try:
        with open(path, encoding="utf-8") as f:
            h = (json.load(f) or {}).get("holdings") or {}
        return [{"symbol": s, "name": "", "reason": "보유 중 — 악재 점검"} for s in h]
    except Exception:
        return []


def _headlines(cands, suffix=""):
    try:
        from ai_commentary import fetch_news_headlines
        ysyms = {c["symbol"]: c["symbol"] + suffix for c in cands}
        heads = fetch_news_headlines(list(ysyms.values()), R.yf)
        for c in cands:
            c["headlines"] = (heads.get(ysyms[c["symbol"]]) or [])[:4]
    except Exception as e:
        _log(f"헤드라인 수집 생략({e})")


def _save(name: str, for_kst: str, ver: dict, now):
    night_notes = " / ".join(x for x in (ver.get("market_overview"), ver.get("macro"),
                                         ver.get("risks")) if x)
    os.makedirs("output", exist_ok=True)
    path = f"output/pregen_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"for_kst": for_kst, "generated": now.isoformat(timespec="minutes"),
                   "by_sym": ver["by_sym"], "night_notes": night_notes}, f,
                  ensure_ascii=False, indent=1)
    _log(f"저장: {path} (대상일 {for_kst} · 종목 {len(ver['by_sym'])})")


def run_kr():
    now = datetime.datetime.now(R.KST)
    if now.hour >= 16:                       # 저녁 실행(정상) → 내일 아침 메일용
        for_kst = (now + datetime.timedelta(days=1)).date().isoformat()
    elif now.hour < 8:                       # 새벽 보충 실행(부팅 시) → 오늘 아침 메일용
        for_kst = now.date().isoformat()
    else:
        _log("한국장 pregen 은 16시 이후 또는 08시 이전에만 유효 → 스킵"); return
    R._require_yf()
    import kr_stocks as KR
    kr = KR.select(R.yf) or {}
    if not (kr.get("buy") or kr.get("watch")):
        _log("한국 후보 없음 → 스킵"); return
    _headlines((kr.get("buy") or []) + (kr.get("watch") or []), suffix=".KS")
    groups = {"kr_buy": kr.get("buy") or [], "kr_watch": kr.get("watch") or [],
              "sells": _holding_syms("output/kr_holdings.json")}
    market = {"as_of": kr.get("as_of"), "note": "밤 시점 검증 — 한국장 마감 확정 데이터"}
    ver = AR.verify_stage(groups, market)
    if not ver.get("by_sym"):
        _log("검증 실패 — 파일 미생성(아침에 API 폴백)"); sys.exit(1)
    _save("kr", for_kst, ver, now)


def run_us():
    now = datetime.datetime.now(R.KST)
    if not (6 <= now.hour < 16):             # 미국장 마감(새벽)~발송(17시) 사이만 유효
        _log("미국장 pregen 은 06~16시(KST)에만 유효 → 스킵"); return
    for_kst = now.date().isoformat()
    R._require_yf()
    data = R.gather_universe_data(with_volume=True)
    scored, info, _m = E.select_pool(data, int(os.environ.get("REPORT_MAX_CANDIDATES", "60")))
    cands = E.build_candidates(data, info, scored, 60)
    pool_k = int(os.environ.get("REPORT_POOL", "6")) + POOL_BUFFER
    buy, watch = E.split_by_entry(cands, k=pool_k)
    _headlines(buy + watch)
    groups = {"buy_now": buy, "watch": watch,
              "sells": _holding_syms("output/ai_holdings.json")}
    market = {"as_of": R._last_data_date(data["hist"]), **E.build_market(data)}
    ver = AR.verify_stage(groups, market)
    if not ver.get("by_sym"):
        _log("검증 실패 — 파일 미생성(오후에 API 폴백)"); sys.exit(1)
    _save("us", for_kst, ver, now)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="구독 CLI 사전 검증(메일 2통 체계)")
    ap.add_argument("--kr", action="store_true", help="한국장(다음 08:00)용")
    ap.add_argument("--us", action="store_true", help="미국장(당일 17:00)용")
    a = ap.parse_args()
    if a.kr:
        run_kr()
    elif a.us:
        run_us()
    else:   # 플래그 없으면 시간 창에 맞는 쪽을 자동 선택(둘 다 가능하면 둘 다)
        h = datetime.datetime.now(R.KST).hour
        if 6 <= h < 16:
            run_us()
        if h >= 16 or h < 8:
            run_kr()
