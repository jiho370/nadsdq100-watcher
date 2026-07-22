#!/usr/bin/env python3
"""
pregen.py — 로컬 PC에서 Pro 구독 CLI(claude -p)로 AI 검증+서술을 '미리' 생성 (메일 2통 체계).

  --kr : 한국장 메일(다음날 08:00)용. 실행 창 = 저녁 16시 이후(장 마감 확정) 또는
         다음날 새벽 08시 이전(부팅 보충). 장중(08~16시)엔 데이터가 애매해 스킵.
         검증+종목별 서술+시황 총평까지 전부 미리 씀. 시황 총평은 **전일 한국장 기준만**
         (코스피·코스닥 등락·추세신호) 다루도록 범위를 좁혀서 — 19시 시점에 이미 확정된
         데이터라 미국장 마감을 기다릴 필요가 없다(2026-07-10: 이전엔 "밤사이 미국 마감까지
         포함"을 노려 발송 시점 경량 API 콜 1회가 남아있었으나, 그 정도 내용까지는 필요 없다고
         판단해 국장 데이터만으로 19시에 전부 끝내도록 단순화함).
         → output/pregen_kr.json (for_kst = 다음 08:00 발송일)
  --us : 미국장 메일(당일 17:00)용. 실행 창 = 06시(미국장 마감 후)~16시(발송 전).
         이 시점엔 해당 세션이 이미 마감 확정이라 검증+서술+시황 총평까지 전부 미리 씀.
         그 외 시각엔 이미 발송됐거나 데이터 미확정이라 스킵. → output/pregen_us.json (for_kst = 오늘)

아침/오후 GitHub Actions 는 pregen_{kr,us}.json 의 for_kst 가 발송일과 일치하면
검증 단계(웹검색)를 생략하고, written(사전서술)까지 있으면 서술 단계(haiku)도 생략한다
→ PC가 켜져 있던 날은 발송 시점 API 호출이 완전히 0회(국장·미장 둘 다 시황까지 포함).
PC가 꺼져 있어 파일이 없으면(또는 2026-07-10부터 AI_ENABLED=0 이라 API 폴백 자체가 꺼져
있으면) 그 부분은 지표 기반(deterministic_report)으로 조용히 대체 — 발송엔 지장 없음.

작업 스케줄러(register_pregen_task.ps1): KR=19:00+22:00(재시도), US=09:30+12:30(재시도),
전부 StartWhenAvailable(놓치면 다음 부팅 시 실행 — 시간 창 가드가 유효성 판단, 재시도는
이미 성공한 날엔 최신 결과로 덮어쓰기만 함 — 멱등).
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


def _save(name: str, for_kst: str, ver: dict, now, written=None, sells_written=None, market_written=None):
    """written(종목별 서술)까지 실으면 발송 시점 write_stage 호출도 생략된다(API 0회).
    market_written 은 KR·US 둘 다 이 시점(19:00/09:30)에 이미 확정된 자기 시장 데이터만
    다루므로 항상 채워진다(KR 시황은 전일 한국장 기준으로 범위를 좁혀 미국장 마감을 안 기다림)."""
    night_notes = " / ".join(x for x in (ver.get("market_overview"), ver.get("macro"),
                                         ver.get("risks")) if x)
    os.makedirs("output", exist_ok=True)
    path = f"output/pregen_{name}.json"
    payload = {"for_kst": for_kst, "generated": now.isoformat(timespec="minutes"),
               "by_sym": ver["by_sym"], "night_notes": night_notes}
    if written:
        payload["written"] = written
    if sells_written:
        payload["sells_written"] = sells_written
    if market_written:
        payload["market_written"] = market_written
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    _log(f"저장: {path} (대상일 {for_kst} · 종목 {len(ver['by_sym'])} · 서술캐시 {len(written or {})}건)")


def _write_ahead(groups: dict, market: dict, vmap: dict, n_buy: int, n_watch: int,
                kr_n_buy: int, kr_n_watch: int, need_market: bool):
    """verify_stage 성공 뒤 write_stage까지 미리 실행 — 종목별 서술을 캐시한다(구독 CLI, $0).
    실패해도 예외를 여기서 흡수해 verify 캐시(검색 생략 효과)는 그대로 저장되게 한다.
    반환: (written, sells_written, market_written) — 실패 시 모두 {}."""
    try:
        AR.attach_plans(groups)
        fb, fw, *_ = AR._apply_verdicts(groups.get("buy_now") or [], groups.get("watch") or [],
                                        vmap, n_buy, n_watch)
        kfb, kfw, *_ = AR._apply_verdicts(groups.get("kr_buy") or [], groups.get("kr_watch") or [],
                                          vmap, kr_n_buy, kr_n_watch)
        final_pairs = ([(c, "buy") for c in fb] + [(c, "watch") for c in fw]
                       + [(c, "buy") for c in kfb] + [(c, "watch") for c in kfw])
        sells = (groups.get("sells") or []) + (groups.get("kr_sells") or [])
        parsed = AR.write_stage(final_pairs, sells, market, vmap, need_market)
        written = {str(r["symbol"]): r for r in (parsed.get("stocks") or [])
                   if isinstance(r, dict) and r.get("symbol")}
        sells_written = {str(r["symbol"]): (r.get("comment") or "") for r in (parsed.get("sells") or [])
                         if isinstance(r, dict) and r.get("symbol")}
        market_written = ({k: parsed[k] for k in ("market_overview", "macro", "signal_note", "risks")
                          if parsed.get(k)} if need_market else {})
        _log(f"서술 사전생성 {len(written)}종목" + (" (시황 포함)" if market_written else ""))
        return written, sells_written, market_written
    except Exception as e:
        _log(f"서술 사전생성 실패({type(e).__name__}: {e}) → 검증만 저장(서술은 발송 시점 API)")
        return {}, {}, {}


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
    # 시황 컨텍스트는 '전일 한국장' 범위로만 좁힌다(코스피·코스닥 등락+추세신호) — 19시엔
    # 이미 확정된 데이터라 미국장 마감을 기다릴 필요가 없다. world(해외지수)는 일부러 안 준다:
    # 밤사이 미국 마감을 다루려던 옛 설계의 흔적이라, 범위를 국장으로 좁힌 지금은 불필요.
    market = {"as_of": kr.get("as_of"), "note": "전일 한국장 마감 기준(코스피·코스닥)"}
    try:
        import market_signals as MS
        signals = MS.gather(R.yf) or {}
        if signals:
            market["signals"] = MS.lean_for_ai(signals, when="kr")
    except Exception as e:
        _log(f"지수 신호 수집 생략({e})")
    ver = AR.verify_stage(groups, market)
    if not ver.get("by_sym"):
        _log("검증 실패 — 파일 미생성(아침에 API 폴백)"); sys.exit(1)
    # 시황 총평도 지금 다 쓴다(need_market=True) — 전일 국장 데이터만 다루므로 19시에 이미 완결.
    written, sells_written, market_written = _write_ahead(
        groups, market, ver["by_sym"],
        n_buy=AR.FINAL_BUY, n_watch=AR.FINAL_WATCH,   # groups에 buy_now/watch가 없어 실질 무해
        kr_n_buy=AR.KR_FINAL_BUY, kr_n_watch=AR.KR_FINAL_WATCH,
        need_market=not bool(ver.get("market_overview")))
    _save("kr", for_kst, ver, now, written=written, sells_written=sells_written,
          market_written=market_written)


def run_us():
    now = datetime.datetime.now(R.KST)
    if not (6 <= now.hour < 16):             # 미국장 마감(새벽)~발송(17시) 사이만 유효
        _log("미국장 pregen 은 06~16시(KST)에만 유효 → 스킵"); return
    for_kst = now.date().isoformat()
    R._require_yf()
    data = R.gather_universe_data(with_volume=True)
    scored, info, _m = E.select_pool(data, int(os.environ.get("REPORT_MAX_CANDIDATES", "60")))
    cands = E.build_candidates(data, info, scored, 60)
    # 관찰(watch) 섹션은 화면엔 안 보이지만(2026-07-13), AI 제외 시 백필 예비군 검증 캐시로
    # 씀(2026-07-19, daily_ai_report.run_us와 동일 수정 — pregen 캐시에도 예비군 verdict가
    # 있어야 발송 시점에 API 재호출 없이 백필 가능).
    pool_k = int(os.environ.get("REPORT_POOL", "10")) + POOL_BUFFER
    buy, watch = E.split_by_entry(cands, k=pool_k)
    _headlines(buy + watch)
    groups = {"buy_now": buy, "watch": watch,
              "sells": _holding_syms("output/ai_holdings.json")}
    market = {"as_of": R._last_data_date(data["hist"]), **E.build_market(data)}
    ver = AR.verify_stage(groups, market)
    if not ver.get("by_sym"):
        _log("검증 실패 — 파일 미생성(오후에 API 폴백)"); sys.exit(1)
    # 09:30엔 미국장이 이미 마감 확정이라 시황 총평까지 지금 다 쓸 수 있다 → need_market=True.
    # verify_stage가 이미 market_overview 등을 냈으면 write_stage가 자동으로 빈 값 처리(중복 방지).
    written, sells_written, market_written = _write_ahead(
        groups, market, ver["by_sym"],
        n_buy=AR.FINAL_BUY, n_watch=AR.FINAL_WATCH,
        kr_n_buy=AR.KR_FINAL_BUY, kr_n_watch=AR.KR_FINAL_WATCH,   # groups에 kr_buy/kr_watch 없어 무해
        need_market=not bool(ver.get("market_overview")))
    _save("us", for_kst, ver, now, written=written, sells_written=sells_written,
          market_written=market_written)


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
