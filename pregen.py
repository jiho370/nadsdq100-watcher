#!/usr/bin/env python3
"""
pregen.py — 로컬 PC에서 Pro 구독 CLI(claude -p)로 AI 검증+서술을 '미리' 생성 (메일 2통 체계).

  --kr : 한국장 메일(다음날 10:00, 2026-07-28부터 08:00→10:00 이관)용. 실행 창 = 저녁 16시
         이후(장 마감 확정) 또는 다음날 새벽 10시 이전(부팅 보충 — 발송 10:00보다 여유 둠).
         장중(10~16시)엔 데이터가 애매해 스킵.
         검증+종목별 서술+시황 총평까지 전부 미리 씀. 시황 총평은 **전일 한국장 기준만**
         (코스피·코스닥 등락·추세신호) 다루도록 범위를 좁혀서 — 19시 시점에 이미 확정된
         데이터라 미국장 마감을 기다릴 필요가 없다(2026-07-10: 이전엔 "밤사이 미국 마감까지
         포함"을 노려 발송 시점 경량 API 콜 1회가 남아있었으나, 그 정도 내용까지는 필요 없다고
         판단해 국장 데이터만으로 19시에 전부 끝내도록 단순화함). 발송 시각이 10:00으로
         밀려도 이 pregen의 AI 검증(전일 데이터 기반) 자체는 그대로 유효 — 10:00에 달라지는
         건 daily_ai_report.py가 그때 새로 조회하는 개별 종목 가격(yfinance, 개장 1시간치
         반영)뿐이고 pregen 캐시(어제 저녁 검증)와는 무관하다.
         → output/pregen_kr.json (for_kst = 다음 10:00 발송일)
  --us : 미국장 메일(그날 저녁 개장 30분~90분 후, KST 자정 넘어 화~토 00:00대 도착)용.
         2026-07-28 마감→개장 기준 이관(STRATEGY.md §7) — 발송이 그날 저녁 미국장 개장
         직후로 당겨지면서, "검증에 필요한 최근 완결 종가"와 "발송 시각"의 관계가 국장
         (§--kr)과 똑같은 모양이 됐다: 발송 몇 시간 전, 이미 확정된 지난 세션 종가로
         미리 검증해두고, 발송 시점엔 daily_ai_report.py가 그때 막 열린 장의 실시간
         시세(yfinance)만 새로 얹는다. 실행 창 = 06시(전날 세션 마감 확정 후)~21시(그날
         저녁 개장 전 — 21시 이후는 개장이 임박/이미 진행 중이라 스킵, 다음날 재시도로
         자연 대체). 이 창 안 실행분은 **for_kst = 다음날**로 찍는다(예: 월요일 낮에
         실행 → for_kst=화요일 — 월요일 저녁 개장 리포트는 자정을 넘겨 화요일 00:00대에
         발송되므로). → output/pregen_us.json

아침/오후 GitHub Actions 는 pregen_{kr,us}.json 의 for_kst 가 발송일과 일치하면
검증 단계(웹검색)를 생략하고, written(사전서술)까지 있으면 서술 단계(haiku)도 생략한다
→ PC가 켜져 있던 날은 발송 시점 API 호출이 완전히 0회(국장·미장 둘 다 시황까지 포함).
PC가 꺼져 있어 파일이 없으면(또는 2026-07-10부터 AI_ENABLED=0 이라 API 폴백 자체가 꺼져
있으면) 그 부분은 지표 기반(deterministic_report)으로 조용히 대체 — 발송엔 지장 없음.

작업 스케줄러(register_pregen_task.ps1, 2026-07-28부터 15분 간격 재시도): KR=16:00부터,
US=06:00부터, 각자 유효 창이 끝날 때까지 15분마다 반복 트리거(지호 님 요청 — "처음에
생성 안되면 15분 간격으로 계속 시도"). run_*() 자체의 시간창 가드가 무효 시각엔 즉시
반환하고, _already_done()이 이미 그날 몫이 채워졌으면 재검증(웹검색)을 건너뛰어 — 실패한
경우에만 사실상 재시도되고, 성공한 뒤엔 반복 트리거가 계속 와도 매번 즉시 스킵돼 구독
사용량이 낭비되지 않는다. 전부 StartWhenAvailable(PC가 꺼져있다 부팅되면 놓친 트리거 중
가장 최근 것부터 실행).
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


def _already_done(name: str, for_kst: str) -> bool:
    """output/pregen_{name}.json 이 이미 이 for_kst로, 내용까지(by_sym 존재) 채워져 있으면 True.
    2026-07-28(지호 님 요청 — "처음에 생성 안되면 15분 간격으로 계속 시도"): 작업 스케줄러가
    이제 15분마다 반복 트리거되므로, 이미 성공한 뒤에도 매번 웹검색을 다시 태우면 구독 사용량만
    낭비된다 — 오늘 몫이 이미 있으면 조용히 스킵(재시도는 '실패했을 때만' 의미가 있어야 함)."""
    try:
        with open(f"output/pregen_{name}.json", encoding="utf-8") as f:
            pg = json.load(f)
        return pg.get("for_kst") == for_kst and bool(pg.get("by_sym"))
    except Exception:
        return False


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


def run_kr() -> str:
    """반환값은 run_pregen.ps1의 재시도 루프가 종료 여부를 판단하는 데 쓴다(2026-07-29,
    지호 님 지적 — "pregen 한 번 생성됐으면 파워쉘이 또 안 열려도 되는거 아냐"):
    "done"(성공/이미 완료/오늘 후보 없음 — 재시도 의미 없음) · "outside_window"(시간창 밖 —
    다음 창까지 재시도 의미 없음) · "failed"(시간창 안인데 검증 실패 — 재시도 가치 있음)."""
    now = datetime.datetime.now(R.KST)
    if now.hour >= 16:                       # 저녁 실행(정상) → 내일 아침 메일용
        for_kst = (now + datetime.timedelta(days=1)).date().isoformat()
    elif now.hour < 10:                      # 새벽 보충 실행(부팅 시) → 오늘 10:00 발송용
        # 2026-07-28: 발송이 08:00→09:30→10:00(개장 1시간 후)로 이관되면서 보충 창도
        # 8시→9시→10시로 늘림(그래야 10:00 발송 전에 pregen이 끝날 여유가 생김).
        for_kst = now.date().isoformat()
    else:
        _log("한국장 pregen 은 16시 이후 또는 10시 이전에만 유효 → 스킵"); return "outside_window"
    if _already_done("kr", for_kst):
        _log(f"이미 {for_kst}치 생성 완료 — 재시도 스킵"); return "done"
    R._require_yf()
    import kr_stocks as KR
    kr = KR.select(R.yf) or {}
    if not (kr.get("buy") or kr.get("watch")):
        _log("한국 후보 없음 → 스킵"); return "done"
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
        _log("검증 실패 — 파일 미생성(아침에 API 폴백)"); return "failed"
    # 시황 총평도 지금 다 쓴다(need_market=True) — 전일 국장 데이터만 다루므로 19시에 이미 완결.
    written, sells_written, market_written = _write_ahead(
        groups, market, ver["by_sym"],
        n_buy=AR.FINAL_BUY, n_watch=AR.FINAL_WATCH,   # groups에 buy_now/watch가 없어 실질 무해
        kr_n_buy=AR.KR_FINAL_BUY, kr_n_watch=AR.KR_FINAL_WATCH,
        need_market=not bool(ver.get("market_overview")))
    _save("kr", for_kst, ver, now, written=written, sells_written=sells_written,
          market_written=market_written)
    return "done"


def run_us() -> str:
    """반환값 의미는 run_kr() 참고("done"/"outside_window"/"failed")."""
    now = datetime.datetime.now(R.KST)
    # 2026-07-28 재설계(STRATEGY.md §7) — 발송이 "다음날 종가 분석"에서 "그날 저녁 개장
    # 30분~90분 후"로 이관됨(report.yml 참고, KST 자정을 넘겨 화~토 00:00대 도착). 검증에
    # 쓸 '이미 확정된 최근 종가'는 전날 세션 마감(06시경) 이후 계속 유효하고, 그날 저녁
    # 개장(22:30~23:30경) 전까지가 실행 유효 창 — run_kr()의 "저녁 실행 → for_kst=다음날"과
    # 정확히 같은 이유로, 이 창 안 실행분은 전부 for_kst를 다음날로 찍는다(오늘 낮에 검증한
    # 내용이 쓰이는 발송은 자정을 넘겨 "내일" 날짜로 나가므로).
    if not (6 <= now.hour < 21):
        _log("미국장 pregen 은 06~21시(KST, 그날 저녁 개장 전)에만 유효 → 스킵"); return "outside_window"
    for_kst = (now + datetime.timedelta(days=1)).date().isoformat()
    if _already_done("us", for_kst):
        _log(f"이미 {for_kst}치 생성 완료 — 재시도 스킵"); return "done"
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
        _log("검증 실패 — 파일 미생성(오후에 API 폴백)"); return "failed"
    # 09:30엔 미국장이 이미 마감 확정이라 시황 총평까지 지금 다 쓸 수 있다 → need_market=True.
    # verify_stage가 이미 market_overview 등을 냈으면 write_stage가 자동으로 빈 값 처리(중복 방지).
    written, sells_written, market_written = _write_ahead(
        groups, market, ver["by_sym"],
        n_buy=AR.FINAL_BUY, n_watch=AR.FINAL_WATCH,
        kr_n_buy=AR.KR_FINAL_BUY, kr_n_watch=AR.KR_FINAL_WATCH,   # groups에 kr_buy/kr_watch 없어 무해
        need_market=not bool(ver.get("market_overview")))
    _save("us", for_kst, ver, now, written=written, sells_written=sells_written,
          market_written=market_written)
    return "done"


_EXIT_CODE = {"done": 0, "outside_window": 2, "failed": 1}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="구독 CLI 사전 검증(메일 2통 체계)")
    ap.add_argument("--kr", action="store_true", help="한국장(다음 10:00)용")
    ap.add_argument("--us", action="store_true", help="미국장(그날 저녁 개장 30분~90분 후)용")
    a = ap.parse_args()
    # 2026-07-29(지호 님 지적 — "pregen 한 번 생성됐으면 파워쉘이 또 안 열려도 되는거 아냐"):
    # run_pregen.ps1이 이제 프로세스를 반복 실행하는 대신 자기 안에서 15분 간격으로 sleep하며
    # 재시도하고, 이 exit code로 "더 재시도할 가치가 있는지"를 판단한다(0/2=끝, 1=재시도).
    if a.kr:
        raise SystemExit(_EXIT_CODE[run_kr()])
    elif a.us:
        raise SystemExit(_EXIT_CODE[run_us()])
    else:   # 플래그 없으면 시간 창에 맞는 쪽을 자동 선택(둘 다 가능하면 둘 다) — 각 run_*()가
            # 자체적으로도 창을 재검증하므로 여기선 대략만 걸러도 됨(2026-07-28: US 창이
            # 6~16시→6~21시로, KR 창이 16시/8시→16시/10시 기준으로 넓어진 것 반영).
        h = datetime.datetime.now(R.KST).hour
        if 6 <= h < 21:
            run_us()
        if h >= 16 or h < 10:
            run_kr()
