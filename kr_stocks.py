#!/usr/bin/env python3
"""
kr_stocks.py — 코스피200 개별 종목 선별 (STRATEGY.md §3).

규칙:
  · 유니버스: 코스피200 (한국 모멘텀은 이 유니버스에서만 통계적으로 유의 — universe shrinkage 연구)
  · 펀더멘탈 필터(KRX 공식, pykrx): EPS>0(흑자) · ROE(EPS/BPS)≥8% · 0<PER≤40
  · 추세: 종가>200일선 필수, 점수 = z(12-1 모멘텀)×0.6 + z(52주 고점 근접도)×0.4
  · 지금매수 3 + 관찰(눌림) 2 — 눌림 = 20일선 -2% 아래 또는 1주 -2%
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
# 후보 '풀' 크기 — AI 검증(강등/제외) 후 최종 채택은 ai_report가 3/2로 확정
N_BUY = int(os.environ.get("KR_POOL_BUY", "4"))
N_WATCH = int(os.environ.get("KR_POOL_WATCH", "3"))


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
        # 최근 영업일 탐색(주말·휴장 대비 최대 7일 소급)
        for back in range(8):
            d = (day - dt.timedelta(days=back)).strftime("%Y%m%d")
            try:
                members = list(K.get_index_portfolio_deposit_file("1028", d))
            except Exception:
                members = []
            if members:
                break
        if not members:
            _log("코스피200 구성종목 조회 실패"); return None
        df = K.get_market_fundamental(d, market="KOSPI")
        out = {}
        for t in members:
            try:
                row = df.loc[t]
                per, eps, bps = float(row["PER"]), float(row["EPS"]), float(row["BPS"])
            except Exception:
                continue
            roe = (eps / bps) if bps else 0.0
            name = ""
            try:
                name = K.get_market_ticker_name(t)
            except Exception:
                pass
            out[t] = {"name": name, "per": per, "eps": eps, "bps": bps, "roe": round(roe, 4)}
        if out:
            os.makedirs("output", exist_ok=True)
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump({"as_of": d, "data": out}, f, ensure_ascii=False)
        return out or None
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


def _pulling_back(c) -> bool:
    w = (c.get("ret") or {}).get("1w")
    price, ma20 = c.get("price"), c.get("ma20")
    below20 = (price is not None and ma20 and price < ma20 * 0.98)
    return bool(below20 or (w is not None and w < -2))


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
        ind_map[t] = {"price": price, "ma200": ma200, "closes": c[-252:]}
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
    now = [c for c in ranked if not _pulling_back(c)]
    back = [c for c in ranked if _pulling_back(c)]
    buy = now[:N_BUY]
    if len(buy) < N_BUY:
        buy += [c for c in back if c not in buy][:N_BUY - len(buy)]
    watch = [c for c in back if c not in buy][:N_WATCH]
    if len(watch) < N_WATCH:
        watch += [c for c in now if c not in buy and c not in watch][:N_WATCH - len(watch)]
    _log(f"선정: 매수 {len(buy)} · 관찰 {len(watch)} (후보 {len(ranked)})")
    return {"as_of": as_of, "buy": buy, "watch": watch, "ind_map": ind_map}


# ------------------------- 보유 추적(매도 시그널) -------------------------
def update_holdings(buy_syms: list, ind_map: dict, today: str) -> list:
    """holdings.py 와 동일 규칙(-20% 트레일링/200일선 -3%)을 한국 종목에 적용."""
    import holdings as H
    state = H.load(KR_HOLDINGS)
    sells = H.update(state, buy_syms, ind_map, today)
    H.save(state, KR_HOLDINGS)
    return sells


def add_holdings(buy_syms: list, ind_map: dict, today: str):
    """AI 검증 후 '최종 매수'로 확정된 종목만 보유목록에 편입."""
    import holdings as H
    state = H.load(KR_HOLDINGS)
    holdings = state.setdefault("holdings", {})
    for sym in buy_syms:
        if sym not in holdings:
            p = (ind_map.get(sym) or {}).get("price")
            holdings[sym] = {"since": today, "entry_price": p, "peak": p}
    H.save(state, KR_HOLDINGS)
