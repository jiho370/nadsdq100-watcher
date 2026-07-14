#!/usr/bin/env python3
"""
backtest_etf_control.py — 팩터 ETF 대조군 백테스트 (2026-07-14 작업지시).

배경: 코어-새틀라이트 검증(STRATEGY.md §3 Stage 1~3)에서 새틀라이트의 '구체적 구성'
(topn·가중 변형)은 통계적으로 안 갈렸다. 그렇다면 커스텀 종목선정(개별종목 선정·AI
검증·상폐 리스크·증권거래세가 딸린) 대신 같은 스타일의 상장 ETF로 새틀라이트를 통째로
대체해도 위험조정 성과가 구분되지 않는지(H1), 나아가 코리아밸류업 지수가 코어 슬롯
자체를 대체할 후보가 되는지(H2)를 검증한다. 기존 backtest_portfolio.py(NAV 프레임·비용
모델)·core_satellite_kr.py(레짐·혼합·서브기간)·overfit_stats.py(PBO/DSR)를 재사용하며
중복 구현하지 않는다.

프레임(§2):
  A 현행        코어65(KODEX200+레짐) + 새틀35(커스텀 valuediv_flow/미국 topn10)
  B ETF새틀라이트 코어(레짐) + 팩터ETF, 비중그리드 — KR {100,90,80,65,50,35,20,10,0} ·
                US {100,65,35,0}(축약)
  C 순수지수     KODEX200(레짐) 100% / SPY(레짐) 100%
  D 코어스왑(H2)  코어 슬롯을 코리아밸류업 지수로 교체 — {밸류업100},{밸류업90+KODEX200 10},
                {밸류업65+커스텀새틀35}

분류 규칙(§4): 후보의 일별수익률-KODEX200 상관 ≥ 0.95면 '새틀라이트'가 아니라 '코어
대체재'로 분류 — 프레임 B가 아니라 D로만 해석(밸류업 지수가 여기 해당할 것으로 예상).

데이터 함정 3개(§3) 반영:
  1) 분배금 재투자 — KR ETF는 TR 지수 자동매칭 우선. 못 찾으면 가격 시계열로 '평가'하지
     않는다(§6 금지) — status=blocked_need_tr 로 남기고 건너뜀. 미국은 yfinance Adj Close.
  2) 밸류업 지수 look-ahead — 지수 발표일(2024-09) 이전 구간을 별도 컬럼으로 분리 표기.
  3) 비용 비대칭 — ETF: 총보수(연, 근사치)+스프레드를 연환산 drag로 반영, 증권거래세 없음.
     개별주(커스텀 새틀라이트): 기존 backtest_kr_strategies 산출(kr_strategy_navs.json)이
     이미 CostModel("kospi") net이므로 재적용하지 않음.

실행(PC, 네트워크 필요 — pykrx/yfinance):
  python backtest_etf_control.py --market kr [--valueup-csv data/valueup_index.csv]
  python backtest_etf_control.py --market us
  python backtest_etf_control.py --self-test
결과: output/etf_control_kr.json · output/etf_control_us.json · output/pbo_report_etf_control.json
      output/etf_control_price_cache.json (원시 가격 캐시 — 재실행 시 네트워크 최소화)
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP
import core_satellite_kr as CS

PRICE_CACHE = "output/etf_control_price_cache.json"
OUT_KR = "output/etf_control_kr.json"
OUT_US = "output/etf_control_us.json"
PBO_OUT = "output/pbo_report_etf_control.json"
CORR_CORE_REPLACEMENT = 0.95     # §4: 이 이상이면 새틀라이트가 아니라 코어대체재로 분류
VALUEUP_ANNOUNCE = "2024-09-24"  # 코리아밸류업 지수 발표일(§3-2 look-ahead 분리 기준)

CORE_WEIGHTS_KR = [1.00, 0.90, 0.80, 0.65, 0.50, 0.35, 0.20, 0.10, 0.00]
CORE_WEIGHTS_US = [1.00, 0.65, 0.35, 0.00]

# fee_pct는 근사치(공시 총보수 기준, 2026-07 시점 확인 필요 — 로컬 실행 시 실제 값으로 교체 권장).
# index_hint: TR(총수익) 지수 자동탐색용 이름 키워드.
KR_CANDIDATES = {
    "plus_highdiv_161510": {"ticker": "161510", "name": "PLUS 고배당주",
                             "fee_pct": 0.30, "index_hint": ["고배당50", "고배당 50", "고배당지수"]},
    "kodex_divgrowth_211900": {"ticker": "211900", "name": "KODEX 배당성장",
                                "fee_pct": 0.25, "index_hint": ["배당성장", "배당 성장"]},
    "tiger_divgrowth_211560": {"ticker": "211560", "name": "TIGER 배당성장",
                                "fee_pct": 0.29, "index_hint": ["배당성장", "배당 성장"]},
    "kodex_highdiv_279530": {"ticker": "279530", "name": "KODEX 고배당",
                              "fee_pct": 0.30, "index_hint": ["고배당"]},
}
# 동일 지수(코스피 배당성장50)를 추종하는 두 후보 중 데이터 긴 쪽만 채택(§1 표) — 런타임에 결정.
DUPLICATE_GROUPS = [{"kodex_divgrowth_211900", "tiger_divgrowth_211560"}]

US_CANDIDATES = {
    "syld": {"tickers": ["SYLD"], "name": "Cambria Shareholder Yield", "fee_pct": 0.59},
    "pkw": {"tickers": ["PKW"], "name": "Invesco Buyback Achievers", "fee_pct": 0.62},
    "qual": {"tickers": ["QUAL"], "name": "iShares MSCI USA Quality Factor", "fee_pct": 0.15},
    "cowz": {"tickers": ["COWZ"], "name": "Pacer US Cash Cows 100", "fee_pct": 0.49},
    "schd": {"tickers": ["SCHD"], "name": "Schwab US Dividend Equity", "fee_pct": 0.06},
    "qual50_syld50": {"tickers": ["QUAL", "SYLD"], "name": "QUAL 50% + SYLD 50% 합성",
                       "fee_pct": (0.15 + 0.59) / 2},
}
US_CORE_TICKER = "SPY"
US_CORE_FEE_PCT = 0.09
KR_CORE_FEE_PCT = 0.15   # KODEX200 총보수 근사(참고 — B1 지수 자체엔 미반영, note로만 기록)


def _log(m): print(f"[ETF대조군] {m}", file=sys.stderr)


# ------------------------- 캐시 -------------------------
def _load_cache():
    try:
        with open(PRICE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(c):
    os.makedirs("output", exist_ok=True)
    with open(PRICE_CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


# ------------------------- 비용 -------------------------
def apply_fee_drag(nav: pd.Series, annual_fee_pct: float) -> pd.Series:
    """연 비용(%, 총보수+스프레드 근사)을 일별 산술 차감으로 반영. 증권거래세 없음(ETF)."""
    r = nav.pct_change().fillna(0.0) - annual_fee_pct / 100.0 / 252.0
    return float(nav.iloc[0]) * (1 + r).cumprod()


def corr_vs_core(nav_a: pd.Series, nav_b: pd.Series) -> float | None:
    ra = nav_a.pct_change().dropna()
    rb = nav_b.reindex(nav_a.index).pct_change().dropna()
    idx = ra.index.intersection(rb.index)
    if len(idx) < 60:
        return None
    return round(float(ra.loc[idx].corr(rb.loc[idx])), 3)


# ------------------------- KR ETF 로더 -------------------------
def _find_tr_index(hints: list[str]):
    """이름에 hints 중 하나 + ('TR'|'총수익')을 포함하는 지수 티커 탐색. 못 찾으면 None."""
    from pykrx import stock as K
    seen = {}
    for market in ("KOSPI", "KRX", "테마"):
        try:
            for t in K.get_index_ticker_list(market=market):
                if t in seen:
                    continue
                try:
                    name = K.get_index_ticker_name(t)
                except Exception:
                    continue
                seen[t] = name or ""
        except Exception as e:
            _log(f"  지수 티커 목록 조회 실패({market}): {e}")
    for t, name in seen.items():
        if any(h in name for h in hints) and ("TR" in name or "총수익" in name):
            return t, name
    return None


def fetch_kr_etf_series(cand_id: str, cfg: dict, start="20120101", cache=None) -> dict:
    """반환: {nav: Series, tr_used: bool, source_name: str|None, status: str}.
    §6 금지: TR(또는 분배금 재투자) 확보 못하면 가격 시계열로 '평가'하지 않고 status만 기록."""
    cache = cache if cache is not None else {}
    key = f"kr_{cand_id}"
    end = pd.Timestamp.today().strftime("%Y%m%d")
    if key in cache and cache[key].get("nav"):
        d = cache[key]
        nav = pd.Series(d["nav"]); nav.index = pd.to_datetime(nav.index); nav = nav.sort_index()
        return {"nav": nav, "tr_used": d.get("tr_used", False),
                "source_name": d.get("source_name"), "status": d.get("status", "ok")}
    from pykrx import stock as K
    found = None
    try:
        found = _find_tr_index(cfg["index_hint"])
    except Exception as e:
        _log(f"  {cand_id}: TR 지수 탐색 실패: {e}")
    if found:
        tr_ticker, tr_name = found
        try:
            df = K.get_index_ohlcv_by_date(start, end, tr_ticker)
            col = "종가" if "종가" in df.columns else df.columns[-1]
            nav = df[col].astype(float)
            nav.index = pd.to_datetime(nav.index)
            nav = nav[nav > 0].sort_index()
            rec = {"nav": {d.date().isoformat(): float(v) for d, v in nav.items()},
                   "tr_used": True, "source_name": tr_name, "status": "ok"}
            cache[key] = rec; _save_cache(cache)
            _log(f"  {cand_id}: TR 지수 '{tr_name}'({tr_ticker}) 확보 {len(nav)}일")
            return {"nav": nav, "tr_used": True, "source_name": tr_name, "status": "ok"}
        except Exception as e:
            _log(f"  {cand_id}: TR 지수 {tr_ticker} 다운로드 실패: {e}")
    _log(f"  {cand_id}: TR(총수익) 지수를 못 찾음 — 가격 시계열로는 평가 금지(§6). "
         f"분배금 이력 또는 TR 지수 CSV를 지호 님께 요청할 것.")
    rec = {"nav": None, "tr_used": False, "source_name": None, "status": "blocked_need_tr"}
    cache[key] = rec; _save_cache(cache)
    return {"nav": None, "tr_used": False, "source_name": None, "status": "blocked_need_tr"}


def load_valueup_csv(path: str) -> pd.Series | None:
    """코리아 밸류업 지수 백데이터(수동 CSV, columns: date,value). 없으면 None."""
    if not path or not os.path.exists(path):
        _log(f"밸류업 지수 CSV 없음({path}) — KRX 정보데이터시스템에서 수동 다운로드 후 "
             f"--valueup-csv 로 지정 필요(자동 수집 불가, §5).")
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or df.columns[0]
    vcol = cols.get("value") or cols.get("close") or df.columns[1]
    s = pd.Series(df[vcol].astype(float).values, index=pd.to_datetime(df[dcol]))
    return s.sort_index()


# ------------------------- US ETF 로더 -------------------------
def fetch_us_series(tickers: list[str], years: float, cache=None) -> dict:
    """yfinance Adj Close(배당 반영) — 다중 티커면 동일가중 일별수익 합성 NAV."""
    cache = cache if cache is not None else {}
    key = "us_" + "_".join(tickers)
    if key in cache and cache[key].get("nav"):
        d = cache[key]
        nav = pd.Series(d["nav"]); nav.index = pd.to_datetime(nav.index)
        return {"nav": nav.sort_index(), "status": "ok"}
    try:
        import yfinance as yf
        navs = {}
        for t in tickers:
            df = yf.download(t, period=f"{int(years)+1}y", auto_adjust=False, progress=False)
            if df is None or df.empty or "Adj Close" not in df:
                continue
            navs[t] = df["Adj Close"].dropna()
        if not navs:
            raise RuntimeError("데이터 없음")
        rets = pd.DataFrame({t: s.pct_change() for t, s in navs.items()}).dropna(how="all")
        combo = rets.mean(axis=1).fillna(0.0)
        nav = (1 + combo).cumprod()
        rec = {"nav": {d.date().isoformat(): float(v) for d, v in nav.items()}, "status": "ok"}
        cache[key] = rec; _save_cache(cache)
        _log(f"  {'+'.join(tickers)}: 확보 {len(nav)}일")
        return {"nav": nav, "status": "ok"}
    except Exception as e:
        _log(f"  {'+'.join(tickers)}: 다운로드 실패({e}) — 네트워크 차단 가능성(원격 세션). "
             f"로컬 PC에서 재실행 필요.")
        cache[key] = {"nav": None, "status": "blocked_network"}; _save_cache(cache)
        return {"nav": None, "status": "blocked_network"}


# ------------------------- 가중치 그리드 스윕(kr_topn_ratio_sweep.run_ratio_stage 패턴 재사용) ---
def weight_grid_sweep(core: pd.Series, sat: pd.Series, weights: list, bench: pd.Series,
                       label: str) -> tuple[list, dict, dict]:
    """반환: (rows, subperiods_by_weight, trial_data(dates·매트릭스 미확정 n_ev)).
    core/sat: 이미 시작 1.0로 정규화된 NAV. bench: 월간초과수익 판정 기준(B1/SPY)."""
    rows, matrix, dates0, subs_out = [], [], None, {}
    for w in weights:
        mixed = CS.mix_nav(core, sat, w) if 0 < w < 1 else (core if w == 1 else sat)
        subs = {tag: CS.stats(mixed, a, b) for tag, a, b in CS.SUBS}
        subs_out[w] = subs
        f = subs["full"]
        if f is None:
            continue
        rows.append({"core_weight": w, **f})
        d, r = BP.monthly_excess(mixed, bench.reindex(mixed.index).ffill())
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        _log(f"  [{label}] core={w:.2f}: CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} "
             f"MDD {f['mdd_pct']:6.1f}%")
    if len(rows) < 2:
        return rows, subs_out, None
    n_ev = min(len(m) for m in matrix)
    trial_data = {"dates": dates0[:n_ev], "trials": [f"{label}_core{r['core_weight']:.2f}" for r in rows],
                  "matrix": [m[:n_ev] for m in matrix]}
    return rows, subs_out, trial_data


def _merge_trials(parts: list[dict]) -> dict | None:
    """서로 다른 스윕(candidate별)의 trial_data를 하나의 PBO 입력 행렬로 합침(같은 market/bench 기준
    이벤트 수만 맞추면 됨 — CSCV는 조합 간 이벤트 정렬만 요구, 날짜 실제 일치는 불필요)."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    n_ev = min(len(p["matrix"][0]) for p in parts)
    trials, matrix, dates0 = [], [], None
    for p in parts:
        if dates0 is None:
            dates0 = p["dates"][:n_ev]
        trials += p["trials"]
        matrix += [m[:n_ev] for m in p["matrix"]]
    return {"dates": dates0, "trials": trials, "excess_returns": matrix}


# ------------------------- KR 실행 -------------------------
def run_kr(valueup_csv: str | None, save=True):
    from benchmarks_kr import load_benchmarks
    navs_bm = load_benchmarks()
    b1 = navs_bm["B1_kospi200"].dropna()
    reg = CS.regime_series(b1)
    core_kospi = CS.timed_nav(b1, reg)                       # KODEX200(레짐) — 프레임 B/C의 코어

    with open("output/kr_strategy_navs.json", encoding="utf-8") as f:
        navs = json.load(f)
    custom_sat = pd.Series(navs["valuediv_flow"]); custom_sat.index = pd.to_datetime(custom_sat.index)
    custom_sat = (custom_sat.sort_index() / custom_sat.sort_index().iloc[0])

    cache = _load_cache()
    n_trials_registered = 0
    trial_parts = []

    # ---- 프레임 A(현행, 대조 기준) 재현 ----
    frame_a_rows, frame_a_subs, _ = weight_grid_sweep(core_kospi, custom_sat, [0.65], b1, "A_current")
    _log(f"[프레임A] 현행 코어65:새틀35(커스텀) 재현: {frame_a_rows}")

    # ---- 프레임 C(순수 지수) ----
    frame_c = {tag: CS.stats(core_kospi, a, b) for tag, a, b in CS.SUBS}
    _log(f"[프레임C] KODEX200(레짐) 100%: {frame_c['full']}")

    # ---- 후보 데이터 확보(각 후보 1회만 조회) + 중복 지수 그룹은 데이터 긴 쪽만 유지 ----
    fetched = {cid: fetch_kr_etf_series(cid, cfg, cache=cache) for cid, cfg in KR_CANDIDATES.items()}
    active = dict(KR_CANDIDATES)
    for group in DUPLICATE_GROUPS:
        avail = {cid: fetched[cid] for cid in group if fetched[cid]["nav"] is not None}
        if len(avail) > 1:
            keep = max(avail, key=lambda c: len(avail[c]["nav"]))
            for cid in group:
                if cid != keep:
                    _log(f"  {cid}: 동일 지수 추종 중복 — 데이터 더 긴 {keep}만 채택, 제외")
                    del active[cid]

    # ---- 프레임 B: ETF 새틀라이트, 코어비중 그리드 ----
    candidates_out = []
    frame_b = {}
    frame_d_candidates = {}   # 상관 0.95↑ → 코어대체재로 재분류된 후보
    for cid, cfg in active.items():
        r = fetched[cid]
        info = {"id": cid, "name": cfg["name"], "ticker": cfg["ticker"],
                "fee_pct_approx": cfg["fee_pct"], "status": r["status"], "tr_used": r["tr_used"],
                "source_index": r.get("source_name")}
        if r["nav"] is None:
            candidates_out.append(info)
            continue
        sat_nav = apply_fee_drag(r["nav"] / r["nav"].iloc[0], cfg["fee_pct"])
        corr = corr_vs_core(sat_nav, b1)
        info["corr_daily_vs_kospi200"] = corr
        info["classification"] = ("core_replacement" if (corr is not None and corr >= CORR_CORE_REPLACEMENT)
                                   else "satellite")
        candidates_out.append(info)
        if info["classification"] == "core_replacement":
            frame_d_candidates[cid] = sat_nav
            _log(f"  {cid}: 상관 {corr} ≥ {CORR_CORE_REPLACEMENT} → 코어대체재로 재분류(프레임 D로만 해석)")
            continue
        rows, subs, td = weight_grid_sweep(core_kospi, sat_nav, CORE_WEIGHTS_KR, b1, cid)
        frame_b[cid] = {"rows": rows, "subperiods": {str(w): s for w, s in subs.items()}}
        if td:
            trial_parts.append(td)
            n_trials_registered += len(td["trials"])

    # ---- 프레임 D: 코어 스왑(H2) — 밸류업 지수 ----
    frame_d = None
    valueup = load_valueup_csv(valueup_csv) if valueup_csv else None
    if valueup is not None and len(valueup) > 260:
        vu = valueup / valueup.iloc[0]
        vu_reg = CS.regime_series(vu)
        vu_core = CS.timed_nav(vu, vu_reg)
        corr_vu = corr_vs_core(vu_core, b1)
        combos = {
            "valueup100": vu_core,
            "valueup90_kospi10": CS.mix_nav(vu_core, core_kospi, 0.90),
            "valueup65_customsat35": CS.mix_nav(vu_core, custom_sat, 0.65),
        }
        rows_d = []
        for name, nav in combos.items():
            pre = nav.loc[:pd.Timestamp(VALUEUP_ANNOUNCE) - pd.Timedelta(days=1)]
            post = nav.loc[pd.Timestamp(VALUEUP_ANNOUNCE):]
            rows_d.append({"combo": name, **{tag: CS.stats(nav, a, b) for tag, a, b in CS.SUBS},
                           "pre_announcement_lookahead_caution": CS.stats(pre) if len(pre) > 260 else None,
                           "post_announcement": CS.stats(post) if len(post) > 60 else None})
        d, r = BP.monthly_excess(combos["valueup65_customsat35"], b1.reindex(combos["valueup65_customsat35"].index).ffill())
        trial_parts.append({"dates": d, "trials": ["D_" + n for n in combos], "matrix": [r] * len(combos)})
        frame_d = {"corr_valueup_vs_kospi200": corr_vu, "rows": rows_d,
                   "announce_date": VALUEUP_ANNOUNCE,
                   "caution": "발표일 이전 구간은 지수 설계 자체의 사후 적합(look-ahead) 가능성 —"
                              " pre_announcement_lookahead_caution 컬럼은 참고용, 채택 근거로 쓰지 않음. "
                              "24+ 단독 성과만으로도 채택 결론 금지(§3-2·§6)."}
        for cid, nav in frame_d_candidates.items():
            corr = corr_vs_core(nav, b1)
            _log(f"  (참고) {cid} 도 코어대체재 후보 — 프레임 D 해석 대상(별도 조합 미실행, 밸류업과 역할 중복)")
    elif frame_d_candidates:
        frame_d = {"note": "밸류업 CSV 미제공 — 코어대체재로 재분류된 후보만 존재, 조합 실행 보류",
                   "reclassified_candidates": list(frame_d_candidates)}

    payload = {"as_of": b1.index[-1].date().isoformat(), "market": "kr",
               "frame_A_current": {"rows": frame_a_rows, "subperiods": frame_a_subs,
                                   "note": "커스텀 새틀라이트(valuediv_flow) NAV는 이미 CostModel(kospi) net "
                                           "— 재비용화 안 함"},
               "frame_C_pure_index": frame_c,
               "frame_B_etf_satellite": frame_b,
               "frame_D_core_swap": frame_d,
               "candidates": candidates_out,
               "cost_note": {"etf": "총보수(연,근사)+스프레드를 연환산 산술 drag로 반영, 증권거래세 없음",
                             "custom_satellite": "backtest_kr_strategies.py 산출 시 이미 CostModel(kospi) "
                                                 "net(거래세 0.20% 포함) — 이중 반영 안 함",
                             "core_kospi200_etf_fee_pct_approx": KR_CORE_FEE_PCT,
                             "core_note": "B1(코스피200 지수) 자체엔 KODEX200 보수 미반영(기존 코드 관행과 동일 — "
                                          "benchmarks_kr.py B1과 동일 계열)"},
               "corr_classification_threshold": CORR_CORE_REPLACEMENT,
               "n_trials_registered_this_market": n_trials_registered,
               "data_gaps": [c["id"] for c in candidates_out if c["status"] != "ok"]
                            + (["valueup_index"] if valueup is None else [])}
    merged = _merge_trials(trial_parts)
    rpt = None
    if merged:
        trial_data = {"horizon": "etf_control_kr", "universe": "kr_etf", "cost": "asymmetric(see cost_note)",
                      "rebal_days": BP.MONTH, "hold_days": BP.MONTH, **merged}
        rpt = OS.analyze(trial_data, save=False)
        with open("output/trial_returns_etf_control_kr.json", "w", encoding="utf-8") as f:
            json.dump({**trial_data, "excess_returns": trial_data["excess_returns"]}, f, ensure_ascii=False)
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_KR, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_KR} (등록 시행수 {n_trials_registered} · 데이터 공백 {payload['data_gaps']})")
    return payload, rpt


# ------------------------- US 실행 -------------------------
def run_us(years=10.0, save=True):
    cache = _load_cache()
    spy = fetch_us_series([US_CORE_TICKER], years, cache=cache)
    if spy["nav"] is None:
        payload = {"as_of": pd.Timestamp.today().date().isoformat(), "market": "us",
                   "status": "blocked_network", "note": "SPY 코어 시계열 확보 실패(원격 세션 네트워크 차단 가능성) "
                   "— 로컬 PC에서 python backtest_etf_control.py --market us 재실행 필요."}
        if save:
            os.makedirs("output", exist_ok=True)
            with open(OUT_US, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            _log(f"저장: {OUT_US} (SPY 확보 실패 — 데이터 없이 종료)")
        return payload, None

    spy_nav = spy["nav"] / spy["nav"].iloc[0]
    reg = CS.regime_series(spy_nav)
    core_spy = CS.timed_nav(spy_nav, reg)
    core_spy = apply_fee_drag(core_spy, US_CORE_FEE_PCT)

    frame_c = {tag: CS.stats(core_spy, a, b) for tag, a, b in CS.SUBS}

    n_trials_registered, trial_parts = 0, []
    candidates_out, frame_b = [], {}
    for cid, cfg in US_CANDIDATES.items():
        r = fetch_us_series(cfg["tickers"], years, cache=cache)
        info = {"id": cid, "name": cfg["name"], "tickers": cfg["tickers"],
                "fee_pct_approx": round(cfg["fee_pct"], 3), "status": r["status"]}
        if r["nav"] is None:
            candidates_out.append(info)
            continue
        sat_nav = apply_fee_drag(r["nav"] / r["nav"].iloc[0], cfg["fee_pct"])
        corr = corr_vs_core(sat_nav, spy_nav)
        info["corr_daily_vs_spy"] = corr
        candidates_out.append(info)
        rows, subs, td = weight_grid_sweep(core_spy, sat_nav, CORE_WEIGHTS_US, spy_nav, cid)
        frame_b[cid] = {"rows": rows, "subperiods": {str(w): s for w, s in subs.items()}}
        if td:
            trial_parts.append(td)
            n_trials_registered += len(td["trials"])

    payload = {"as_of": spy_nav.index[-1].date().isoformat(), "market": "us",
               "frame_C_pure_index": frame_c,
               "frame_B_etf_satellite": frame_b,
               "candidates": candidates_out,
               "cost_note": {"etf": "총보수(연,근사)를 연환산 drag로 반영",
                             "core_spy_fee_pct_approx": US_CORE_FEE_PCT},
               "unreplicable_factors": ["rd_mktcap — R&D/시총 대응 ETF 없음, 복제 불가로 결과 해석 시 명시"],
               "interpretation_rule": "커스텀 백테스트(topn10, +11~14%p 초과) 대비 ETF 혼합이 크게 못 미치는 "
                                      "것이 정상 — 그 격차 자체를 백테스트 인플레이션 탐지 지표로 기록",
               "n_trials_registered_this_market": n_trials_registered,
               "data_gaps": [c["id"] for c in candidates_out if c["status"] != "ok"]}
    merged = _merge_trials(trial_parts)
    rpt = None
    if merged:
        trial_data = {"horizon": "etf_control_us", "universe": "us_etf", "cost": "asymmetric(see cost_note)",
                      "rebal_days": BP.MONTH, "hold_days": BP.MONTH, **merged}
        rpt = OS.analyze(trial_data, save=False)
        with open("output/trial_returns_etf_control_us.json", "w", encoding="utf-8") as f:
            json.dump({**trial_data, "excess_returns": trial_data["excess_returns"]}, f, ensure_ascii=False)
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_US, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_US} (등록 시행수 {n_trials_registered} · 데이터 공백 {payload['data_gaps']})")
    return payload, rpt


def run_all(valueup_csv, years, save=True):
    kr_payload, kr_rpt = run_kr(valueup_csv, save=save)
    us_payload, us_rpt = run_us(years, save=save)
    total = (kr_payload.get("n_trials_registered_this_market", 0)
             + us_payload.get("n_trials_registered_this_market", 0))
    combined = {"kr": kr_rpt, "us": us_rpt, "total_trials_registered": total,
               "note": "PBO/DSR은 시장별 CSCV로 각각 계산(서로 다른 캘린더·유니버스 — 행렬 합산 불가). "
                       "total_trials_registered는 이번 세션 전체 시행 수(다중검정 예산 등록용)."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(PBO_OUT, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        _log(f"저장: {PBO_OUT} (총 등록 시행수 {total})")
    return combined


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 가중치그리드·비용drag·상관분류·PBO배선 검증(네트워크 없이)")
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2016-01-01", periods=2600)
    core_raw = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, len(idx)))), index=idx)
    reg = CS.regime_series(core_raw)
    core = CS.timed_nav(core_raw, reg)

    # (1) 비용 drag: 연 5% 비용이면 10년 후 NAV가 대략 exp(-0.05*10) 배 축소되어야 함
    dragged = apply_fee_drag(core, 5.0)
    ratio = float(dragged.iloc[-1] / core.iloc[-1])
    yrs = len(idx) / 252
    expected = np.exp(-0.05 * yrs)
    assert abs(ratio - expected) / expected < 0.05, f"비용drag 부정확: {ratio} vs {expected}"

    # (2) 상관 분류: 코어(원지수)를 그대로 복제한 시리즈는 상관 ~1.0 → core_replacement 분류돼야 함
    # (레짐타이밍된 core가 아니라 core_raw로 비교 — 레짐OFF 구간엔 core가 평탄해져 승수노이즈가
    #  인위적으로 상관을 낮추므로 원지수 대 원지수 복제로 검증)
    near_dup = core_raw * (1 + rng.normal(0, 0.0005, len(core_raw)))
    c = corr_vs_core(near_dup, core_raw)
    assert c is not None and c >= CORR_CORE_REPLACEMENT, f"상관 분류 실패: {c}"
    indep = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, len(idx)))), index=idx)
    c2 = corr_vs_core(indep, core_raw)
    assert c2 is None or c2 < CORR_CORE_REPLACEMENT, f"독립 시리즈가 오분류됨: {c2}"

    # (3) 가중치그리드: core=0(위성만)/1(코어만) 경계값이 순수 시리즈와 같아야 함
    sat = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.018, len(idx)))), index=idx)
    sat = sat / sat.iloc[0]
    bench = core_raw / core_raw.iloc[0]
    rows, subs, td = weight_grid_sweep(core, sat, CORE_WEIGHTS_KR, bench, "selftest")
    assert len(rows) == len(CORE_WEIGHTS_KR)
    w1 = next(r for r in rows if r["core_weight"] == 1.0)
    w0 = next(r for r in rows if r["core_weight"] == 0.0)
    assert abs(w1["cagr_pct"] - CS.stats(core)["cagr_pct"]) < 1e-6
    assert abs(w0["cagr_pct"] - CS.stats(sat)["cagr_pct"]) < 1e-6
    # 코어(변동성 낮음)와 새틀(변동성 높음)이 섞이면 극단보다 MDD가 완만해지는 지점이 존재해야 함
    mdds = [r["mdd_pct"] for r in rows]
    assert max(mdds) > w0["mdd_pct"] or max(mdds) > w1["mdd_pct"], mdds
    assert td is not None and len(td["trials"]) == len(rows)

    # (4) PBO/DSR 배선: merge 후 analyze가 죽지 않고 유효 판정을 내는지
    merged = _merge_trials([td, td])
    assert merged is not None and len(merged["trials"]) == 2 * len(rows)
    trial_data = {"horizon": "selftest", "universe": "synthetic", "cost": "n/a",
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH, **merged}
    rpt = OS.analyze(trial_data, save=False)
    assert "passed" in rpt and rpt["n_trials"] == len(merged["trials"])

    # (5) 밸류업 CSV 로더: 없는 경로는 None, 있으면 정렬된 Series
    assert load_valueup_csv("/nonexistent/path.csv") is None
    tmp_path = "/tmp/_etf_control_selftest_valueup.csv"
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "value": [1000.0, 1005.0]}).to_csv(tmp_path, index=False)
    vu = load_valueup_csv(tmp_path)
    assert vu is not None and len(vu) == 2 and vu.iloc[1] == 1005.0
    os.remove(tmp_path)

    _log(f"[self-test] 통과: 비용drag 비율 {ratio:.4f}(기대 {expected:.4f}) · 상관분류 {c}/{c2} · "
         f"그리드 {len(rows)}종 · PBO n_trials={rpt['n_trials']}")


def main():
    ap = argparse.ArgumentParser(description="팩터 ETF 대조군 백테스트(H1/H2)")
    ap.add_argument("--market", choices=["kr", "us", "all"], default="all")
    ap.add_argument("--valueup-csv", default="data/valueup_index.csv",
                    help="코리아밸류업 지수 백데이터 CSV(date,value) — 자동 수집 불가")
    ap.add_argument("--years", type=float, default=10, help="미국 조회 기간(년)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.market == "kr":
        run_kr(args.valueup_csv)
    elif args.market == "us":
        run_us(args.years)
    else:
        run_all(args.valueup_csv, args.years)


if __name__ == "__main__":
    main()
