#!/usr/bin/env python3
"""
realtime_circuit_breaker_paper.py — 실시간 페이퍼트레이딩(모의) 서킷브레이커 로거
(2026-09-06, 지호 님 요청).

circuit_breaker_validation.py에서 검증된 "당일 -5% 하락 → 전량 현금화·10일 대기" 규칙을
실시간 가격으로 시험해본다. 업비트 공개 시세 API만 사용(인증키 불필요) — 실제 주문은
절대 넣지 않고 "지금이면 트리거됐을 것"만 로그로 남긴다. 실거래 자금 리스크 0.

백테스트와의 핵심 차이: 백테스트는 '종가 대비 종가'(하루 단위)였지만 코인은 24시간
장이라 '하루' 경계가 없다 — 대신 1시간·4시간·24시간 세 가지 롤링창을 동시에 계산해
전부 로그로 남긴다(어느 창이 실전 트리거로 더 적합한지는 이 로그를 모아서 나중에
분석·재검증한다. 지금은 세 창 다 관찰만 하는 단계).

실행: python realtime_circuit_breaker_paper.py
로그: output/realtime_paper_log.jsonl (append, 재시작해도 이어짐)
상태: output/realtime_paper_state.json (가격 이력 + 쿨다운 상태 저장)
중지: Ctrl+C (또는 프로세스 종료)
"""
from __future__ import annotations
import json, time, sys, os, datetime as dt
import urllib.request

MARKETS = ["KRW-BTC", "KRW-ETH"]
POLL_SEC = 20                    # 업비트 시세 rate limit(10초당 1회) 대비 여유
WINDOWS_SEC = {"1h": 3600, "4h": 4 * 3600, "24h": 24 * 3600}
THRESHOLD = -0.05                 # circuit_breaker_validation.py BTC 최우수값 그대로 시험
COOLDOWN_SEC = 10 * 24 * 3600     # 백테스트의 '10거래일' → 코인은 매일이 거래일이라 10일 그대로
HEARTBEAT_SEC = 300               # 상태 로그(생존 확인용) 주기
LOG_PATH = "output/realtime_paper_log.jsonl"
STATE_PATH = "output/realtime_paper_state.json"


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _log(m):
    print(f"[실시간페이퍼] {m}", file=sys.stderr, flush=True)


def fetch_prices(markets: list[str]) -> dict:
    url = "https://api.upbit.com/v1/ticker?markets=" + ",".join(markets)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    return {d["market"]: d["trade_price"] for d in data}


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {m: {"history": [], "cooldown_until": None} for m in MARKETS}


def save_state(state: dict):
    os.makedirs("output", exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def log_event(ev: dict):
    os.makedirs("output", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def rolling_return(history: list, now_ts: float, window_sec: float, cur_price: float):
    """history=[[ts,price],...] 중 (now-window) 시점에 가장 가까운(그 이후 첫) 값을 과거가로."""
    cutoff = now_ts - window_sec
    past = None
    for ts, p in history:
        if ts >= cutoff:
            past = p
            break
    if past is None or past == 0:
        return None
    return cur_price / past - 1


def main():
    state = load_state()
    _log(f"시작 — 대상 {MARKETS} · 폴링 {POLL_SEC}s · 임계치 {THRESHOLD:+.0%} · "
        f"쿨다운 {COOLDOWN_SEC / 86400:.0f}일 · 창 {list(WINDOWS_SEC)}")
    log_event({"type": "start", "ts": _now().isoformat(), "markets": MARKETS,
              "threshold": THRESHOLD, "cooldown_days": COOLDOWN_SEC / 86400,
              "windows": list(WINDOWS_SEC)})
    last_heartbeat = 0.0
    while True:
        try:
            prices = fetch_prices(MARKETS)
            now = _now()
            now_ts = now.timestamp()
            heartbeat = (now_ts - last_heartbeat) >= HEARTBEAT_SEC
            for m in MARKETS:
                p = prices.get(m)
                if p is None:
                    continue
                st = state[m]
                st["history"].append([now_ts, p])
                cutoff_keep = now_ts - 25 * 3600   # 24h+여유만 보관(메모리 무한증가 방지)
                st["history"] = [h for h in st["history"] if h[0] >= cutoff_keep]

                rets = {w: rolling_return(st["history"], now_ts, sec, p) for w, sec in WINDOWS_SEC.items()}
                cooling = bool(st["cooldown_until"]) and now_ts < st["cooldown_until"]
                triggered = [w for w, r in rets.items() if r is not None and r < THRESHOLD]

                if triggered:
                    new_cd = now_ts + COOLDOWN_SEC
                    if not cooling:
                        ev = {"type": "TRIGGER(모의매도)", "market": m, "ts": now.isoformat(),
                             "price": p, "windows": triggered, "returns": rets,
                             "cooldown_until": dt.datetime.fromtimestamp(new_cd, dt.timezone.utc).isoformat()}
                        log_event(ev)
                        _log(f"트리거! {m} price={p} windows={triggered} rets={rets}")
                    elif new_cd > st["cooldown_until"]:
                        log_event({"type": "쿨다운연장", "market": m, "ts": now.isoformat(),
                                  "price": p, "windows": triggered})
                    st["cooldown_until"] = new_cd
                elif heartbeat:
                    log_event({"type": "status", "market": m, "ts": now.isoformat(), "price": p,
                              "returns": rets, "cooling": cooling})
            if heartbeat:
                last_heartbeat = now_ts
                _log(f"생존확인 · " + " · ".join(f"{m}={prices.get(m)}" for m in MARKETS))
            save_state(state)
        except Exception as e:
            _log(f"경고: 폴링 실패({type(e).__name__}: {e}) — {POLL_SEC}s 후 재시도")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
