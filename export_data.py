#!/usr/bin/env python3
"""
export_data.py — GitHub Actions(장 마감 직후)용 데이터 export.

역할: sp500_daily_report 의 계산 엔진을 그대로 재사용해 500종목 지표를 산출하고,
Claude(Cowork 스케줄 태스크)가 web_fetch 로 가져가 '해석·종목선별·보고서'를 만들 수 있도록
컴팩트한 파일 3개로 저장한다. (네트워크가 되는 GitHub 러너에서 실행)

출력(기본 out 디렉터리):
  candidates.json  — 기술 사전필터 통과 후보 풀(전 지표+펀더멘탈+차트용 시세). Claude 가 여기서 선별.
  market.json      — 시황 집계(SPY·탐욕지수·시장폭·섹터 1주 등락·당일 상하위).
  snapshot.csv     — 500종목 최신 기술지표 전체(기록/확대용).

사용:  python export_data.py            # ./data 에 저장
       python export_data.py --out data --max-candidates 60
"""
from __future__ import annotations
import os, sys, json, csv, argparse
from datetime import datetime

import sp500_daily_report as R   # 계산 엔진 재사용(gather/score/지표/섹터맵 등)


def _round(x, nd=2):
    return None if R._isnan(x) else round(float(x), nd)


def _closes_tail(hist, sym, n=252):
    """차트용 최근 종가 리스트(소수 2자리). 없으면 []"""
    s = hist.get(sym)
    if s is None or len(s) == 0:
        return []
    tail = s.dropna().tail(n)
    return [round(float(v), 2) for v in tail.tolist()]


def build_candidates(data, info, scored, max_candidates):
    """scored=[(sym,score,reason)] 상위 max_candidates 를 전 지표와 함께 직렬화."""
    ind_map, sector_map, hist = data["ind_map"], data["sector_map"], data["hist"]
    profiles = data.get("profiles") or {}
    pool = scored[:max_candidates]
    out = []
    for rank, (sym, score, reason) in enumerate(pool, 1):
        ind = ind_map.get(sym, {}); meta = info.get(sym, {})
        out.append({
            "symbol": sym,
            "name": R._company_name(sym, meta),
            "sector": R._kr_sector(sector_map.get(sym, "") or meta.get("sector_en", "")),
            "industry": meta.get("industry", ""),
            "score": round(float(score), 2),
            "score_reason": reason,
            "rank": rank, "pool_size": len(pool),
            "entry_label": R.entry_label(ind),
            "price": _round(ind.get("price")),
            "pe": _round(meta.get("pe")),
            "rsi": _round(ind.get("rsi"), 0),
            "above_ma200": bool(ind.get("above_ma200")),
            "ma20": _round(ind.get("ma20")), "ma50": _round(ind.get("ma50")),
            "ma200": _round(ind.get("ma200")), "high_52w": _round(ind.get("high_52w")),
            "macd_up": bool(ind.get("macd_up")), "cross": ind.get("cross"),
            "entry_streak": ind.get("entry_streak"),
            "vol_ann_pct": _round((ind.get("vol_ann") or float("nan")) * 100, 0),
            "ret": {"1w": _round(ind.get("chg_1w"), 1), "1m": _round(ind.get("chg_1m"), 1),
                    "3m": _round(ind.get("chg_3m"), 1), "6m": _round(ind.get("chg_6m"), 1),
                    "1y": _round(ind.get("chg_1y"), 1), "3y": _round(ind.get("chg_3y"), 1)},
            "roe": _round(meta.get("roe"), 3), "fcf": meta.get("fcf"),
            "de": _round(meta.get("de"), 1), "rev_growth": _round(meta.get("rev_growth"), 3),
            "profit_margin": _round(meta.get("profit_margin"), 3),
            "market_cap": meta.get("marketCap"),
            "closes": _closes_tail(hist, sym),   # 차트용
        })
    return out


def build_market(data):
    """시황 집계: SPY 레짐 + 탐욕지수 + 시장폭 + 섹터 1주 등락 + 당일 상하위."""
    ind_map, sector_map, regime = data["ind_map"], data["sector_map"], data["regime"]
    spy = data.get("spy")
    # 시장폭: 200일선 위 비율
    above = [1 for i in ind_map.values() if i.get("above_ma200")]
    n = len(ind_map)
    # 섹터 1주 등락 중앙값
    buckets = {}
    for s, ind in ind_map.items():
        sec = sector_map.get(s, "")
        v = ind.get("chg_1w")
        if sec and not R._isnan(v):
            buckets.setdefault(sec, []).append(v)
    import statistics as st
    sectors_1w = sorted(
        [{"sector": R.GICS_KR.get(sec, sec), "median_1w_pct": round(st.median(v), 2), "n": len(v)}
         for sec, v in buckets.items() if len(v) >= 3],
        key=lambda x: x["median_1w_pct"], reverse=True)
    # 당일 상하위(1일 등락)
    moves = [(s, ind.get("chg_1d")) for s, ind in ind_map.items() if not R._isnan(ind.get("chg_1d"))]
    moves.sort(key=lambda x: x[1], reverse=True)
    top_g = [{"symbol": s, "pct": round(v, 2)} for s, v in moves[:10]]
    top_l = [{"symbol": s, "pct": round(v, 2)} for s, v in moves[-10:]][::-1]
    return {
        "spy": {"price": _round(regime.get("spy")), "ma200": _round(regime.get("ma200")),
                "gap_pct": _round(regime.get("gap_pct"), 1), "risk_on": bool(regime.get("risk_on"))},
        "fear_greed": {"score": _round(regime.get("fng_score"), 0), "rating": regime.get("fng_rating")},
        "breadth": {"pct_above_ma200": round(100 * len(above) / n, 1) if n else None,
                    "universe": n},
        "sectors_1w": sectors_1w,
        "top_gainers_1d": top_g, "top_losers_1d": top_l,
        "spy_closes": _closes_tail(data["hist"], "SPY") if "SPY" in data["hist"] else
                      ([round(float(v), 2) for v in spy.dropna().tail(252).tolist()] if spy is not None else []),
    }


def write_snapshot_csv(path, data):
    """500종목 최신 기술지표 전체(펀더멘탈은 후보 풀에만 있음)."""
    ind_map, sector_map = data["ind_map"], data["sector_map"]
    cols = ["symbol", "sector", "price", "rsi", "above_ma200", "macd_up", "cross",
            "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "vol_ann_pct", "high_52w"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for s in sorted(ind_map):
            i = ind_map[s]
            w.writerow([s, R._kr_sector(sector_map.get(s, "")),
                        _round(i.get("price")), _round(i.get("rsi"), 0),
                        int(bool(i.get("above_ma200"))), int(bool(i.get("macd_up"))),
                        i.get("cross") or "",
                        _round(i.get("chg_1w"), 1), _round(i.get("chg_1m"), 1),
                        _round(i.get("chg_3m"), 1), _round(i.get("chg_6m"), 1),
                        _round(i.get("chg_1y"), 1),
                        _round((i.get("vol_ann") or float("nan")) * 100, 0),
                        _round(i.get("high_52w"))])


def _score_pool(data):
    """기술 사전필터 통과 종목을 스코어링(build_recommendations 로직, 캡 없음)."""
    ind_map, sector_map = data["ind_map"], data["sector_map"]
    tech_pass = [s for s, ind in ind_map.items() if R._tech_ok(ind)]
    info = R.get_info_for(tech_pass)
    pe_by = {}
    for s, m in info.items():
        try: p = float(m.get("pe"))
        except (TypeError, ValueError): continue
        if 0 < p <= 200: pe_by[s] = p
    pe_ref = R._peer_median_resolver(pe_by, sector_map)
    scored = []
    for s in tech_pass:
        meta = info.get(s, {}); pe = meta.get("pe"); ref = pe_ref(s)
        try: rel = float(pe) / ref if (pe and ref and not R._isnan(ref) and ref > 0) else 1.0
        except (TypeError, ValueError): rel = 1.0
        r = R.score_reco(ind_map[s], meta, pe, rel)
        if r: scored.append((s, r[0], r[1]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, info


def load_best_model(default="momentum_12_1") -> str:
    """백테스트가 저장한 최우수 모델명을 읽는다(없으면 기본값). self-test 결과면 무시."""
    try:
        with open("output/best_model.json", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("self_test"):
            return default
        return d.get("model", default)
    except Exception:
        return default


def select_by_model(model: str, ind_map: dict, n: int) -> list[tuple]:
    """백테스트와 '같은 가격기반 랭킹'으로 현재 시점 상위 n 종목 선정.
       반환: [(sym, rankscore, 사유)]. 백테스트 승자 모델을 실추천에 그대로 적용하기 위한 함수."""
    import numpy as _np
    def ok(v): return v is not None and not (isinstance(v, float) and _np.isnan(v))
    rows = []
    for s, ind in ind_map.items():
        rows.append((s, ind.get("chg_6m"), ind.get("chg_12_1"), ind.get("vol_ann"),
                     bool(ind.get("above_ma200"))))
    if model == "momentum":
        c = sorted([(s, m6) for s, m6, m121, v, ab in rows if ok(m6)], key=lambda x: x[1], reverse=True)
        lbl = "6개월 모멘텀 상위"
    elif model == "lowvol":
        c = sorted([(s, v) for s, m6, m121, v, ab in rows if ok(v)], key=lambda x: x[1])
        lbl = "저변동성 상위"
    elif model in ("trend", "mom+trend"):
        c = sorted([(s, m6) for s, m6, m121, v, ab in rows if ab and ok(m6) and m6 > 0],
                   key=lambda x: x[1], reverse=True)
        lbl = "추세(정배열)+모멘텀 상위"
    elif model == "mom+lowvol":
        m6r = {s: r for r, (s, _v) in enumerate(sorted([(s, m6) for s, m6, m121, v, ab in rows if ok(m6)],
               key=lambda x: x[1], reverse=True))}
        vlr = {s: r for r, (s, _v) in enumerate(sorted([(s, v) for s, m6, m121, v, ab in rows if ok(v)],
               key=lambda x: x[1]))}
        common = set(m6r) & set(vlr)
        c = sorted([(s, m6r[s] + vlr[s]) for s in common], key=lambda x: x[1])  # 순위합 작을수록 우수
        lbl = "모멘텀+저변동성 상위"
    else:  # momentum_12_1 (기본/최우수)
        c = sorted([(s, m121) for s, m6, m121, v, ab in rows if ok(m121)], key=lambda x: x[1], reverse=True)
        lbl = "12-1 모멘텀 상위"
    return [(s, float(val), lbl) for s, val in c[:n]]


def pool_by_best_model(data: dict, n: int):
    """백테스트 최우수 모델로 후보 선정 + 그 종목들의 펀더멘탈 조회.
       반환: (scored[(sym,score,사유)], info, model_used)."""
    model = load_best_model()
    scored = select_by_model(model, data["ind_map"], n)
    info = R.get_info_for([s for s, _, _ in scored])
    return scored, info, model


# ===== 가중치 기반 선정(backtest_weights.py 가 찾은 최적 가중치 사용) =====
def load_best_weights():
    """backtest_weights 가 저장한 최적 가중치. 없거나 self-test면 None."""
    try:
        with open("output/best_weights.json", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("self_test"):
            return None
        w = d.get("weights") or {}
        return w if any(w.values()) else None
    except Exception:
        return None


def _load_funds():
    p = "output/fundamentals_cache.json"
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def select_by_weights(weights: dict, ind_map: dict, n: int, funds: dict | None = None,
                      cross: dict | None = None, sector_map: dict | None = None,
                      sector_cap: int | None = 2) -> list[tuple]:
    """지표 z-score 가중합성점수로 상위 n 선정 — 백테스트(backtest_weights)와 '동일 지표·정의' 사용.
       모멘텀=ind_map, 펀더멘탈=fundamentals_edgar.factor_values, 크로스오버=tech_factors(cross).
    sector_cap(기본 2, 2026-07-17 지호 님 결정 — us_sector_cap_sweep.py 백테스트 반영):
       rd_mktcap(R&D/시가총액) 팩터가 구조적으로 바이오/제약을 편애(REGN 0.539로 전체 1위,
       2위의 2배)해 실제로 상위 8종목 중 4종목이 Health Care로 쏠리는 걸 실측 확인 — 임상
       실패 등 섹터 공통 리스크가 상관돼 8종목 '분산'의 실효성이 떨어짐. 백테스트(10년,
       topn=8 고정) 결과 캡=2가 CAGR -5%p(40.4%→35.4%) 대신 MDD -35.8%→-32.8% 개선 —
       한국(Stage 4, 캡이 사실상 공짜)과 달리 여기선 실질적 트레이드오프이고 PBO도 93.5%로
       높아(3후보 중 뭐가 진짜 나은지 이 백테스트로 확정 못 함) 순수 수익 극대화 관점에선
       무제한이 유리하나, 분산을 우선한 지호 님 선택 — CAGR 저하는 감수. sector_cap=None이면
       기존 동작(무제한)."""
    import numpy as _np, pandas as _pd, datetime as _dt
    today = _dt.date.today().isoformat()
    try:
        import fundamentals_edgar as _F
    except Exception:
        _F = None
    rows = {}
    for s, ind in ind_map.items():
        price = ind.get("price")
        r = {"mom6": ind.get("chg_6m"), "mom12_1": ind.get("chg_12_1")}
        if funds and _F and price:
            r.update(_F.factor_values(funds.get(s) or {}, today, float(price)))
        if cross and s in cross:
            r.update({k: v for k, v in cross[s].items() if v is not None})
        rows[s] = r
    df = _pd.DataFrame(rows).T.astype(float)

    def z(col):
        sd = df[col].std()
        zz = (df[col] - df[col].mean()) / sd if sd and not _np.isnan(sd) else df[col] * 0.0
        # shareholder_yield는 ±3 대신 ±5(2026-07-18 클립 완화 검증: 극단치 종목단위 재검정
        # t=2.83 유의, topn8+cap2 라이브조건 재테스트 +0.68%p 개선 — 지호 님 반영 결정)
        clip = (-5, 5) if col == "shareholder_yield" else (-3, 3)
        return zz.clip(*clip).fillna(0.0)
    active = [f for f in weights if weights.get(f) and f in df.columns]
    comp = sum(float(weights[f]) * z(f) for f in active) if active else _pd.Series(0.0, index=df.index)
    valid = df["mom6"].notna() | df["mom12_1"].notna()   # 모멘텀 결측 종목 제외
    comp = comp[valid].sort_values(ascending=False)
    lbl = "가중합성(" + "·".join(f"{k}{v}" for k, v in weights.items() if v) + ")"
    scored_all = [(s, float(comp[s]), lbl) for s in comp.index]
    if sector_cap is not None and sector_map:
        return R.pick_with_sector_cap(scored_all, sector_map, n, sector_cap)
    return scored_all[:n]


def split_by_entry(candidates: list, k: int = 5):
    """후보(점수순)를 분할 (STRATEGY.md §2 진입 필터 반영):
       지금매수 = 200일선 위 & 52주 고점 -25% 이내 & 상승 지속 — 과열은 hot=True 표시.
       관찰     = 점수 상위지만 지금 하락·조정 중(눌림목) 또는 진입 필터 미달.
    조정 판정: 1주 수익률 -2% 이하 이거나 종가가 20일선 아래(-2% 넘게)."""
    def hot(c):   # 과열(지금 사되 분할 권고 대상)
        rsi = c.get("rsi"); price = c.get("price"); ma50 = c.get("ma50")
        gap50 = ((price / ma50 - 1) * 100) if (price and ma50) else 0
        return (rsi is not None and rsi >= 72) or (gap50 >= 15) or (c.get("entry_label") == "C")

    def entry_ok(c):   # 진입 필터: 200일선 위 + 52주 고점 근접(-25% 이내, George-Hwang)
        if not c.get("above_ma200"):
            return False
        price, hi = c.get("price"), c.get("high_52w")
        if price and hi:
            return (price / hi - 1) * 100 >= -25
        return True   # 고점 데이터 없으면 통과(보수적 제외보다 기존 동작 유지)

    def pulling_back(c):   # 지금 조정/하락 중 → 관찰(눌림목)
        w = (c.get("ret") or {}).get("1w")
        price, ma20 = c.get("price"), c.get("ma20")
        below20 = (price is not None and ma20 and price < ma20 * 0.98)
        weak1w = (w is not None and w < -2)
        return bool(below20 or weak1w)

    for c in candidates:
        c["hot"] = hot(c)
    # 2026-07 재검증(backtest_entry_gate.py): entry_ok(200일선·52주고점)·조정중 분류를 포함한
    # 기술 게이트 후보 6종 전부가 검증된 팩터 바스켓의 성과를 깎는 것으로 확인
    # (현행 게이트 6M -3.85%p t=-2.1 유의, 스윕 6종 전부 diff<0) → 게이트 폐지.
    # 매수 = 팩터 순위 상위 k, 관찰 = 다음 k. hot(과열)은 분할매수 계획 표기용으로만 유지.
    # entry_ok/pulling_back은 정보 표시용으로 남긴다(카드에 근거 표기 가능).
    return candidates[:k], candidates[k:2 * k]


def select_pool(data: dict, n: int):
    """일일 후보 선정 진입점 — 우선순위: 최적가중치(모멘텀+펀더멘탈) > 최우수모델 > 하이브리드.
       반환: (scored, info, method_label)."""
    w = load_best_weights()
    if w:
        funds = _load_funds()
        cross = None
        try:
            import tech_factors as _T
            cross = _T.latest_by_sym(data.get("hist") or {})
        except Exception:
            cross = None
        scored = select_by_weights(w, data["ind_map"], n, funds=funds, cross=cross,
                                   sector_map=data.get("sector_map"))
        label = ("weights " + "·".join(f"{k}{v}" for k, v in w.items() if v)
                 + ("" if funds else " [펀더멘탈캐시 없음:모멘텀만]"))
    else:
        model = load_best_model()
        scored = select_by_model(model, data["ind_map"], n)
        label = f"model {model}"
    if not scored:                       # 최후 폴백: 퀄리티+모멘텀 하이브리드
        scored, info = _score_pool(data)
        return scored, info, "hybrid(score_reco)"
    info = R.get_info_for([s for s, _, _ in scored])
    return scored, info, label


def run(out_dir="data", max_candidates=60):
    R._require_yf()
    os.makedirs(out_dir, exist_ok=True)
    data = R.gather_universe_data(with_volume=True)
    as_of = R._last_data_date(data["hist"])
    gen = datetime.now(R.KST).date().isoformat()
    scored, info = _score_pool(data)
    meta = {"as_of": as_of, "generated_kst": gen}

    candidates = {**meta, "count": min(len(scored), max_candidates),
                  "candidates": build_candidates(data, info, scored, max_candidates)}
    market = {**meta, **build_market(data)}

    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "market.json"), "w", encoding="utf-8") as f:
        json.dump(market, f, ensure_ascii=False, indent=1)
    write_snapshot_csv(os.path.join(out_dir, "snapshot.csv"), data)
    print(f"[export] as_of={as_of} 후보={candidates['count']}종목 전체={market['breadth']['universe']}종목 "
          f"-> {out_dir}/ (candidates.json, market.json, snapshot.csv)", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S&P500 데이터 export(GitHub Actions용)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-candidates", type=int, default=60)
    args = ap.parse_args()
    run(args.out, args.max_candidates)
