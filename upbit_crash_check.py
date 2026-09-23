#!/usr/bin/env python3
"""
upbit_crash_check.py — realtime_circuit_breaker_paper.py를 대체하는 '상시 프로세스 없는'
버전 (2026-09-06, 지호 님 요청 — 효율화). 상시 폴링(20초 간격) 대신, 윈도우 작업
스케줄러가 15~30분마다 이 스크립트를 한 번 실행 → 캔들 API로 최근 24시간을 한 번에
받아 -5% 체크만 하고 종료한다. crash_delay_loss_analysis.py에서 "1분~12시간 반응속도
차이는 노이즈, 진짜 손실은 하루 넘어가야 커진다"를 확인했으므로, 상시 20초 감시는
과잉이었다는 게 그 분석의 결론 — 이 스크립트가 그 결론을 반영한 경량 버전이다.

방식: 업비트 공개 캔들 API(인증 불필요)로 10분봉 144개(=24시간)를 한 번에 조회 →
1시간(6봉)·4시간(24봉)·24시간(144봉) 롤링 고점 대비 하락률을 계산 → 하나라도
-5% 이하면 output/crash_check_log.jsonl 에 트리거 기록. 상태는
output/crash_check_status.json 에 매 실행마다 덮어써 마지막 체크 시각을 확인 가능.

실행: python upbit_crash_check.py   (1회 실행 후 종료 — 반복은 작업 스케줄러가 담당)
등록: setup_crash_check_task.ps1 참고(관리자 권한 불필요, 현재 사용자 로그온 트리거)
"""
from __future__ import annotations
import json, sys, os, datetime as dt
import urllib.request

MARKETS = ["KRW-BTC", "KRW-ETH"]
CANDLE_UNIT = 10          # 10분봉
CANDLE_COUNT = 144        # 10분 * 144 = 1440분 = 24시간, 업비트 1회 호출 한도(200) 이내
WINDOWS = {"1시간": 6, "4시간": 24, "24시간": 144}   # 10분봉 개수 기준
THRESHOLD = -0.05
LOG_PATH = "output/crash_check_log.jsonl"
STATUS_PATH = "output/crash_check_status.json"


def _log(m): print(f"[급락체크] {m}", file=sys.stderr)


def fetch_candles(market: str, count: int, unit: int) -> list[float]:
    """반환: 오래된 순 → 최신 순으로 정렬된 종가 리스트."""
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}?market={market}&count={count}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    closes = [c["trade_price"] for c in data]
    return list(reversed(closes))   # 업비트는 최신순 반환 → 시간순으로 뒤집기


def check_market(market: str) -> dict:
    closes = fetch_candles(market, CANDLE_COUNT, CANDLE_UNIT)
    cur = closes[-1]
    result = {"market": market, "price": cur, "windows": {}}
    triggered = []
    for wname, wlen in WINDOWS.items():
        seg = closes[-wlen:] if len(closes) >= wlen else closes
        peak = max(seg)
        chg = cur / peak - 1
        result["windows"][wname] = round(chg * 100, 2)
        if chg < THRESHOLD:
            triggered.append(wname)
    result["triggered"] = triggered
    return result


def main():
    os.makedirs("output", exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    status = {"checked_at": now, "results": []}
    any_trigger = False
    for m in MARKETS:
        try:
            r = check_market(m)
        except Exception as e:
            _log(f"{m} 조회 실패({type(e).__name__}: {e})")
            status["results"].append({"market": m, "error": str(e)})
            continue
        status["results"].append(r)
        _log(f"{m} 가격={r['price']} 창별등락={r['windows']}")
        if r["triggered"]:
            any_trigger = True
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "TRIGGER(모의)", "ts": now, **r}, ensure_ascii=False) + "\n")
            _log(f"⚠ 트리거! {m} {r['triggered']}")
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    if not any_trigger:
        _log("트리거 없음 — 정상 종료")


if __name__ == "__main__":
    main()
