#!/usr/bin/env python3
"""
kr_stocks.py — 코스피200 개별 종목 선별 (STRATEGY.md §3).

규칙:
  · 유니버스: 코스피200 (한국 모멘텀은 이 유니버스에서만 통계적으로 유의 — universe shrinkage 연구)
  · 펀더멘탈 필터(KRX 공식, pykrx): EPS>0(흑자) · ROE(EPS/BPS)≥8% · 0<PER≤40
  · 추세: 종가>200일선 필수, 점수 = z(12-1 모멘텀)×0.6 + z(52주 고점 근접도)×0.4
  · 매수 후보 풀 6(순위순) — 관찰 폐지(2026-07-13), AI 검증 후 최종 5 확정. 보유 상한 6
  · 매도: 트레일링 -20% 또는 200일선 -3% 이탈 (output/kr_holdings.json 자동 추적)

데이터: 구성종목·펀더멘탈 = pykrx(KRX), 시세 = yfinance(.KS 배치 1회).
pykrx 실패 시 output/kospi200_cache.json 캐시로 폴백(성공 시마다 갱신).
"""
from __future__ import annotations
import os, sys, json, datetime as dt

import market_signals as MS

CACHE = "output/kospi200_cache.json"
KR_HOLDINGS = "output/kr_holdings.json"
ROE_MIN = float(os.environ.get("KR_ROE_MIN", "0.08"))
PER_MAX = float(os.environ.get("KR_PER_MAX", "40"))
# 후보 '풀' 크기 — AI 검증(강등/제외) 후 최종 채택은 ai_report가 KR_FINAL_BUY(5)로 확정.
# 관찰 폐지(2026-07-13): 풀 전체가 매수 후보, N_WATCH 기본 0.
N_BUY = int(os.environ.get("KR_POOL_BUY", "6"))
N_WATCH = int(os.environ.get("KR_POOL_WATCH", "0"))
MAX_HOLD = int(os.environ.get("KR_MAX_HOLD", "6"))   # 보유 상한(팔아야 산다)


def _log(m): print(f"[KR] {m}", file=sys.stderr)


# ------------------------- 유니버스 + 펀더멘탈 (pykrx) -------------------------
def _krx_universe_funda() -> dict | None:
    """{ticker6: {"name":..,"per":..,"eps":..,"bps":..,"roe":..}} 또는 None."""
    try:
        from pykrx import stock as K
    except Exception as e:
        _log(f"pykrx 없음({e}) → 캐시 폴백"); return None
    try:
        day = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
        # 최근 영업일 탐색(주말·휴장 대비 최대 7일 소급). 구성종목과 재무데이터는 반드시
        # '같은 날짜'에서 함께 유효해야 한다 — 장 시작 전(예: KST 08시 국장 메일) 조회 시
        # 그날 날짜로는 구성종목은 나오지만 재무데이터(PER/EPS/BPS)가 아직 미발행이라
        # 전부 NaN으로 채워지는 경우가 실사용 중 확인됨(2026-07-13). float(nan)은 예외를
        # 던지지 않아 이걸 '성공'으로 오인하면 캐시 폴백(_cached_universe)이 아예 발동하지
        # 않고 0/200 통과라는 빈 결과가 그대로 나간다 — 반드시 NaN을 걸러야 한다.
        out, d = {}, None
        for back in range(8):
            d = (day - dt.timedelta(days=back)).strftime("%Y%m%d")
            try:
                members = list(K.get_index_portfolio_deposit_file("1028", d))
            except Exception:
                members = []
            if not members:
                continue
            try:
                df = K.get_market_fundamental(d, market="KOSPI")
            except Exception:
                continue
            cand = {}
            for t in members:
                try:
                    row = df.loc[t]
                    per, eps, bps = float(row["PER"]), float(row["EPS"]), float(row["BPS"])
                except Exception:
                    continue
                if per != per or eps != eps or bps != bps:   # NaN — 그날 재무데이터 미발행
                    continue
                roe = (eps / bps) if bps else 0.0
                name = ""
                try:
                    name = K.get_market_ticker_name(t)
                except Exception:
                    pass
                cand[t] = {"name": name, "per": per, "eps": eps, "bps": bps, "roe": round(roe, 4)}
            if len(cand) >= len(members) * 0.5:   # 절반 이상 유효해야 그 날짜를 채택
                out = cand
                break
            _log(f"{d} 재무데이터 대부분 미발행({len(cand)}/{len(members)}) → 하루 더 소급")
        if not out:
            _log("코스피200 구성종목/재무데이터 조회 실패"); return None
        os.makedirs("output", exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump({"as_of": d, "data": out}, f, ensure_ascii=False)
        return out
    except Exception as e:
        _log(f"KRX 조회 실패({type(e).__name__}: {e}) → 캐시 폴백"); return None


def _cached_universe() -> dict | None:
    try:
        with open(CACHE, encoding="utf-8") as f:
            d = json.load(f)
        _log(f"캐시 사용(기준일 {d.get('as_of')})")
        return d.get("data") or None
    except Exception:
        _log(f"캐시 없음({CACHE}) — pykrx 미설치 시 대체 데이터가 없음")
        return None


# ------------------------- 선별 -------------------------
def _z(values: dict) -> dict:
    xs = [v for v in values.values() if v is not None]
    if len(xs) < 3:
        return {k: 0.0 for k in values}
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
    if not sd:
        return {k: 0.0 for k in values}
    return {k: max(-3.0, min(3.0, (v - m) / sd)) if v is not None else 0.0 for k, v in values.items()}


def _hot(c) -> bool:
    rsi = c.get("rsi"); price, ma50 = c.get("price"), c.get("ma50")
    gap50 = ((price / ma50 - 1) * 100) if (price and ma50) else 0
    return (rsi is not None and rsi >= 72) or gap50 >= 15


def select(yf) -> dict:
    """반환: {"as_of":..,"buy":[cand..],"watch":[cand..],"ind_map":{sym:{price,ma200}}}  실패 시 {}"""
    funda = _krx_universe_funda() or _cached_universe()
    if not funda:
        _log("한국 종목 데이터 없음 → 이번 실행은 한국 섹션이 빈 채로 발송됨. "
             "가장 흔한 원인(2025-12-27부터): KRX 정보데이터시스템이 로그인 필수로 바뀜 — "
             "환경변수 KRX_ID/KRX_PW 필요(data.krx.co.kr 무료가입, GITHUB_SETUP.md 참고). "
             "pykrx 자체가 없다면 `pip install pykrx`. 한 번 성공하면 캐시가 생겨 이후엔 안전망이 됨.")
        return {}
    passed = {t: f for t, f in funda.items()
              if f["eps"] > 0 and f["roe"] >= ROE_MIN and 0 < f["per"] <= PER_MAX}
    _log(f"펀더멘탈 통과 {len(passed)}/{len(funda)}종목 (EPS>0·ROE≥{ROE_MIN:.0%}·PER≤{PER_MAX:.0f})")
    if not passed:
        return {}
    raw = MS.fetch_closes(yf, [f"{t}.KS" for t in passed])
    cands, mom_v, prox_v, ind_map = {}, {}, {}, {}
    as_of = None
    for t, f in passed.items():
        d = raw.get(f"{t}.KS")
        if not d or len(d["closes"]) < 260:
            continue
        c = d["closes"]
        price = c[-1]
        ma200 = MS._sma(c, 200); ma50 = MS._sma(c, 50); ma20 = MS._sma(c, 20)
        ind_map[t] = {"price": price, "ma200": ma200,
                      "closes": c[-252:], "dates": d["dates"][-252:]}
        if not ma200 or price <= ma200:      # 200일선 위 필수
            continue
        mom = MS._mom_12_1(c)
        hi52 = max(c[-252:])
        prox = (price / hi52 - 1) * 100 if hi52 else None
        mom_v[t], prox_v[t] = mom, prox
        as_of = as_of or d["dates"][-1]
        cands[t] = {
            "symbol": t, "yf_symbol": f"{t}.KS", "name": f.get("name") or t,
            "sector": "", "price": round(price, 0),
            "pe": round(f["per"], 1), "roe": f["roe"],
            "rsi": MS._rsi(c), "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "high_52w": hi52, "prox52": round(prox, 1) if prox is not None else None,
            "above_ma200": True,
            "ret": {"1w": MS._ret(c, 5), "1m": MS._ret(c, 21), "3m": MS._ret(c, 63),
                    "6m": MS._ret(c, 126), "1y": MS._ret(c, 252)},
            "mom12_1": round(mom, 1) if mom is not None else None,
            "closes": [round(v, 0) for v in c[-252:]],
        }
    if not cands:
        _log("추세 조건(200일선 위) 통과 종목 없음")
        return {"as_of": as_of, "buy": [], "watch": [], "ind_map": ind_map}
    zm, zp = _z(mom_v), _z(prox_v)
    for t, c in cands.items():
        c["score"] = round(0.6 * zm.get(t, 0.0) + 0.4 * zp.get(t, 0.0), 3)
        c["score_reason"] = (f"12-1 모멘텀 {c['mom12_1']:+.1f}%"
                             + (f" · 52주고점 {c['prox52']:+.1f}%" if c.get("prox52") is not None else ""))
        c["hot"] = _hot(c)
    ranked = sorted(cands.values(), key=lambda x: x["score"], reverse=True)
    # 관찰 폐지(2026-07-13): 눌림/상승지속 구분 없이 팩터 순위 그대로 매수 후보
    # (미국 backtest_entry_gate와 동일 취지 — 기술 게이트가 성과를 깎음. hot 태그는 분할계획용 유지)
    buy = ranked[:N_BUY]
    watch = ranked[N_BUY:N_BUY + N_WATCH]
    _log(f"선정: 매수 {len(buy)} · 관찰 {len(watch)} (후보 {len(ranked)})")
    return {"as_of": as_of, "buy": buy, "watch": watch, "ind_map": ind_map,
            "pool": [c["symbol"] for c in ranked]}   # 6개월 재평가용 후보풀(필터+추세 통과 전체)


# ------------------------- 보유 추적(매도 시그널) -------------------------
def update_holdings(buy_syms: list, ind_map: dict, today: str, pool_syms=None) -> list:
    """holdings.py 와 동일 규칙(6개월 재평가/200일선 -3%)을 한국 종목에 적용.
    2026-07-14 수정: pool_syms를 안 넘겨서 6개월 정기 재평가가 한국에서는 한 번도 발동하지
    않고 있었다(STRATEGY.md '미국과 동일' 명시와 불일치) — select()의 pool을 받도록 확장."""
    import holdings as H
    state = H.load(KR_HOLDINGS)
    sells = H.update(state, buy_syms, ind_map, today,
                     pool_syms=set(pool_syms) if pool_syms else None)
    H.save(state, KR_HOLDINGS)
    return sells


def add_holdings(buy_syms: list, ind_map: dict, today: str):
    """AI 검증 후 '최종 매수'로 확정된 종목만 보유목록에 편입(상한 MAX_HOLD — 팔아야 산다)."""
    import holdings as H
    state = H.load(KR_HOLDINGS)
    H.add(state, buy_syms, ind_map, today, max_n=MAX_HOLD)
    H.save(state, KR_HOLDINGS)
