#!/usr/bin/env python3
"""
kr_stocks.py — 코스피200 개별 종목 선별 (STRATEGY.md §3, 2026-07-14 valuediv로 교체).

규칙(KR_STRATEGY_OPTIONS.md §8 검증 반영 — backtest_kr_strategies.py Phase 3):
  · 유니버스: 코스피200
  · 펀더멘탈 필터(KRX 공식, pykrx): EPS>0(흑자) · ROE(EPS/BPS)>0 — 예전 ROE≥8%·PER≤40·
    200일선 위 필터는 밸류 전략의 성과를 깎는 것으로 확인돼 폐기(추세필터 강제 시 진짜
    저평가 구간을 걸러버림 — 미국 진입게이트 폐지와 동일 취지).
  · 점수(교체, 옛 mom12_1×0.6+hi52×0.4 폐기) = z(1/PER) + z(1/PBR) + z(배당수익률)
    — "밸류×주주환원" 계열. 코어-새틀라이트 구조(§2-F, STRATEGY.md §3)의 새틀라이트 역할.
  · 매수 후보 풀 5(순위순, 2026-07-16 6→5 변경 — topn 정밀검증 Stage 3.1·3.2 결과 반영,
    STRATEGY.md §3 참고). 보유 상한 5
  · 매도: 6개월 정기 재평가(후보풀 이탈)만 활성 — 200일선 백업은 2026-07-15부로 기본
    비활성(holdings.py SELL_MA200_BACKUP, 근거는 STRATEGY.md §3 Stage 6)
    (output/kr_holdings.json 자동 추적)

데이터: 구성종목·펀더멘탈(PER/PBR/EPS/BPS/DIV) = pykrx(KRX), 시세 = yfinance(.KS 배치 1회).
pykrx 실패 시 output/kospi200_cache.json 캐시로 폴백(성공 시마다 갱신).
"""
from __future__ import annotations
import os, sys, json, datetime as dt

import market_signals as MS

CACHE = "output/kospi200_cache.json"
KR_HOLDINGS = "output/kr_holdings.json"
# 후보 '풀' 크기 — AI 검증(강등/제외) 후 최종 채택은 ai_report가 KR_FINAL_BUY(5)로 확정.
# 관찰 폐지(2026-07-13): 풀 전체가 매수 후보, N_WATCH 기본 0.
# 2026-07-16: topn 6→5 변경(STRATEGY.md §3 Stage 3.1·3.2 — 13년 재검증에서 5가 CAGR·샤프·
# MDD·Calmar·다운캡처 전부 근소 우위. 통계적으로 유의하진 않으나(부트스트랩 CI 전부 겹침)
# 6을 유지할 데이터 근거도 마찬가지로 없어 point-estimate 승자로 결정).
N_BUY = int(os.environ.get("KR_POOL_BUY", "5"))
N_WATCH = int(os.environ.get("KR_POOL_WATCH", "0"))
MAX_HOLD = int(os.environ.get("KR_MAX_HOLD", "5"))   # 보유 상한(팔아야 산다)


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
        # 그날 날짜로는 구성종목은 나오지만 재무데이터(PER/EPS/BPS)가 아직 미발행인 경우가
        # 실사용 중 확인됨. 처음엔 NaN으로 채워지는 줄 알았는데(2026-07-13 수정), 실제로는
        # pykrx가 미발행 구간을 '0.0으로 채운 자리표시자' 행으로 돌려주는 경우도 있어(2026-
        # 07-14 재확인 — 이날은 PER·EPS·BPS가 전부 0.0, NaN 검사를 그냥 통과함) 여전히 뚫렸다.
        # 0.0 자체는 적자기업의 정상적인 PER 값이기도 해서(코스피200 정상 거래일 기준 약
        # 15%는 PER=0) 개별 종목 단위로는 못 거른다 — 그 날짜 전체가 자리표시자인지는 '그날
        # PER>0인 종목 비율'로 판정한다(정상 거래일 실측 85% 대비, 미발행일은 0%).
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
                    pbr, div = float(row["PBR"]), float(row["DIV"])
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
                cand[t] = {"name": name, "per": per, "eps": eps, "bps": bps, "roe": round(roe, 4),
                          "pbr": pbr, "div_yield": div if div == div else 0.0}
            nonzero_ratio = (sum(1 for c in cand.values() if c["per"] > 0) / len(cand)) if cand else 0.0
            if len(cand) >= len(members) * 0.5 and nonzero_ratio >= 0.3:
                out = cand
                break
            _log(f"{d} 재무데이터 미발행/자리표시자 의심(유효 {len(cand)}/{len(members)}, "
                 f"PER>0 비율 {nonzero_ratio:.0%}) → 하루 더 소급")
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
    # 2026-07-14: 옛 EPS>0·ROE≥8%·PER≤40 필터를 EPS>0·ROE>0로 완화(backtest_kr_strategies.py
    # Phase 3 검증값 그대로 재현 — 밸류 전략에 ROE 8% 문턱·PER 상한을 추가로 걸면 저평가
    # 구간을 스스로 걸러내는 모순이 생김. value(1/PER)·pbr_inv(1/PBR) 자체가 극단 고평가를
    # 이미 낮은 점수로 벌점 처리한다).
    passed = {t: f for t, f in funda.items() if f["eps"] > 0 and f["roe"] > 0}
    _log(f"펀더멘탈 통과 {len(passed)}/{len(funda)}종목 (EPS>0·ROE>0)")
    if not passed:
        return {}
    raw = MS.fetch_closes(yf, [f"{t}.KS" for t in passed])
    cands, value_v, pbrinv_v, div_v, ind_map = {}, {}, {}, {}, {}
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
        # 2026-07-14: 200일선 진입 필터 폐지(§8 검증 — 추세필터 강제 시 밸류 전략 성과 훼손,
        # 미국 진입게이트 폐지와 동일 결론). 매도 측 200일선 -3% 백업은 holdings.py에 그대로.
        pbr, div = f.get("pbr"), f.get("div_yield")
        value = (1.0 / f["per"]) if f["per"] > 0 else None
        pbr_inv = (1.0 / pbr) if pbr and pbr > 0 else None
        value_v[t], pbrinv_v[t], div_v[t] = value, pbr_inv, div
        as_of = as_of or d["dates"][-1]
        cands[t] = {
            "symbol": t, "yf_symbol": f"{t}.KS", "name": f.get("name") or t,
            "sector": "", "price": round(price, 0),
            "pe": round(f["per"], 1), "roe": f["roe"], "pbr": pbr, "div_yield": div,
            "rsi": MS._rsi(c), "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "high_52w": max(c[-252:]), "above_ma200": bool(ma200 and price > ma200),
            "ret": {"1w": MS._ret(c, 5), "1m": MS._ret(c, 21), "3m": MS._ret(c, 63),
                    "6m": MS._ret(c, 126), "1y": MS._ret(c, 252)},
            "closes": [round(v, 0) for v in c[-252:]],
        }
    if not cands:
        _log("펀더멘탈 통과 종목 중 시세 확보된 종목 없음")
        return {"as_of": as_of, "buy": [], "watch": [], "ind_map": ind_map}
    zv, zp, zd = _z(value_v), _z(pbrinv_v), _z(div_v)
    for t, c in cands.items():
        c["score"] = round(zv.get(t, 0.0) + zp.get(t, 0.0) + zd.get(t, 0.0), 3)
        c["score_reason"] = (f"PER {c['pe']:.1f}"
                             + (f" · PBR {c['pbr']:.2f}" if c.get("pbr") else "")
                             + (f" · 배당수익률 {c['div_yield']:.1f}%" if c.get("div_yield") else ""))
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
