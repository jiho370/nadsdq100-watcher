#!/usr/bin/env python3
"""
backtest_kr.py — 한국(코스피200) 팩터 백테스트: 미국 파이프라인과 동일 규율 적용 (2026-07).

배경(지호 님 가설): 개인 비중이 높고 차익거래 제약이 큰 한국 시장에서 팩터 효과가 더 클 수
있다. 판정은 이 백테스트가 한다 — DART 키 없이 pykrx(무료)만으로 가능한 팩터부터 검증.

데이터(전부 무료·키 불필요):
  · 시세: yfinance .KS 배치(상장폐지 종목 누락 → 커버리지 %로 정직 보고, 미국과 동일)
  · 유니버스: pykrx 코스피200 구성종목을 리밸 시점별 조회(PIT), 실패 시 현재 구성 폴백+기록
  · 펀더멘탈: pykrx get_market_fundamental (PER/PBR/EPS/BPS/DIV — 날짜별 제공 = look-ahead 없음)
  · 벤치마크: 코스피200 지수(pykrx)
  · 전부 output/kr_bt_cache.json 에 캐시(재실행 시 네트워크 최소화)

팩터 11종(전부 '높을수록 좋다' 방향):
  가격: mom12_1, mom6, hi52_prox, low_vol(60d 변동성 역수)
  펀더멘탈: value(1/PER), pbr_inv(1/PBR), div_yield, roe(EPS/BPS)
  한국 고유(2026-07 피드백 반영): frgn_flow(외국인 60d 순매수/시총),
    inst_flow(기관 60d 순매수/시총), size(-log 시총 — KOSPI200 내 소형 틸트)

진단 옵션(2026-07 외부 피드백 — 반도체 쏠림·좁은 리더십 장 가설):
  --exclude-top N : 각 시점 시총 상위 N종목 제외(KOSPI50 초대형주 문제 검정)
  기간분리 IC     : 2024-07 전/후 IC를 자동 분리 보고(이상 급등 구간 스트레스 분리)

시나리오 비교:
  [live]     현행 kr_stocks 규칙 재현 — 필터(EPS>0·ROE≥8%·0<PER≤40·종가>200MA) 후
             z(mom12_1)×0.6 + z(hi52)×0.4 상위 N. 탐색 없음 → 현행 시스템의 정직한 기대값.
  [searched] 훈련구간 IC>0 상위 K개 그리드 탐색(+워크포워드 OOS) → PBO/DSR 게이트 판정.
비용: CostModel("kospi") — 매도 거래세 0.20% 포함(미국 0.05%의 4배 — 한국 회전비용이 큼).

실행: python backtest_kr.py --years 10 --oos 0.4
      python backtest_kr.py --self-test
결과: output/backtest_kr_compare.json · output/trial_returns_kr.json · output/pbo_report_kr.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS

CACHE_PATH = "output/kr_bt_cache.json"
COMPARE_PATH = "output/backtest_kr_compare.json"
TRIAL_PATH = "output/trial_returns_kr.json"
REPORT_PATH = "output/pbo_report_kr.json"

PRICE_FACTORS = ["mom12_1", "mom6", "hi52_prox", "low_vol"]
FUND_FACTORS = ["value", "pbr_inv", "div_yield", "roe"]
FLOW_FACTORS = ["frgn_flow", "inst_flow", "size"]
ALL_FACTORS = PRICE_FACTORS + FUND_FACTORS + FLOW_FACTORS
ANOMALY_CUTOFF = "2024-07-01"   # 이후는 반도체 쏠림 급등장(외부 피드백) — IC 분리 보고
KEEP = 5
LEVELS = (0, 1, 2)
LOOKBACK = 260


def _log(m): print(f"[KR백테스트] {m}", file=sys.stderr)


# ------------------------- 데이터 수집(캐시) -------------------------
def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"membership": {}, "fundamentals": {}, "bench": {}}


def _save_cache(c):
    os.makedirs("output", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def fetch_membership(dates_yyyymmdd: list[str], cache) -> dict:
    """리밸 날짜별 코스피200 구성종목(PIT). 실패 날짜는 None(호출부가 폴백+기록)."""
    from kr_factor_ic import kospi200_members
    mem = cache.setdefault("membership", {})
    for d in dates_yyyymmdd:
        if d not in mem:
            mem[d] = kospi200_members(d)
    _save_cache(cache)
    return mem


def fetch_fundamentals(dates_yyyymmdd: list[str], cache) -> dict:
    """리밸 날짜별 전 종목 PER/PBR/EPS/BPS/DIV (pykrx, 날짜당 1콜)."""
    from pykrx import stock as K
    fnd = cache.setdefault("fundamentals", {})
    for d in dates_yyyymmdd:
        if d in fnd:
            continue
        try:
            df = K.get_market_fundamental(d, market="KOSPI")
            fnd[d] = {t: {c: float(row[c]) for c in ("PER", "PBR", "EPS", "BPS", "DIV")
                          if c in row and pd.notna(row[c])}
                      for t, row in df.iterrows()} if df is not None and len(df) else {}
        except Exception as e:
            _log(f"펀더멘탈 조회 실패 {d}: {e}")
            fnd[d] = {}
        _save_cache(cache)
    return fnd


def fetch_flows(dates_yyyymmdd: list[str], cache) -> dict:
    """리밸 날짜별 외국인·기관 순매수거래대금(직전 ~60거래일 창, 날짜당 2콜)."""
    from pykrx import stock as K
    fl = cache.setdefault("flows", {})
    for d in dates_yyyymmdd:
        if d in fl:
            continue
        start = (pd.Timestamp(d) - pd.Timedelta(days=90)).strftime("%Y%m%d")
        rec = {}
        for tag, inv in (("frgn", "외국인"), ("inst", "기관합계")):
            try:
                df = K.get_market_net_purchases_of_equities(start, d, "KOSPI", inv)
                rec[tag] = {t: float(v) for t, v in df["순매수거래대금"].items()}
            except Exception as e:
                _log(f"수급 조회 실패 {d}/{inv}: {e}")
                rec[tag] = {}
        fl[d] = rec
        _save_cache(cache)
    return fl


def fetch_mktcap(dates_yyyymmdd: list[str], cache) -> dict:
    """리밸 날짜별 시가총액(날짜당 1콜)."""
    from pykrx import stock as K
    mc = cache.setdefault("mktcap", {})
    for d in dates_yyyymmdd:
        if d in mc:
            continue
        try:
            df = K.get_market_cap_by_ticker(d, market="KOSPI")
            mc[d] = {t: float(v) for t, v in df["시가총액"].items()}
        except Exception as e:
            _log(f"시총 조회 실패 {d}: {e}")
            mc[d] = {}
        _save_cache(cache)
    return mc


def fetch_bench(start_yyyymmdd, end_yyyymmdd, cache) -> pd.Series:
    """코스피200 지수 종가(pykrx). 캐시."""
    key = f"{start_yyyymmdd}_{end_yyyymmdd}"
    b = cache.setdefault("bench", {})
    if key not in b:
        from pykrx import stock as K
        df = K.get_index_ohlcv(start_yyyymmdd, end_yyyymmdd, "1028")
        col = "종가" if "종가" in df.columns else df.columns[3]
        b[key] = {d.strftime("%Y-%m-%d"): float(v) for d, v in df[col].items()}
        _save_cache(cache)
    s = pd.Series(b[key])
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# ------------------------- 팩터·스냅샷 -------------------------
def _price_factors(panel: pd.DataFrame, p: int) -> pd.DataFrame:
    px = panel.iloc[: p + 1]
    cur = px.iloc[-1]
    out = pd.DataFrame(index=panel.columns)
    out["mom12_1"] = px.iloc[-21] / px.iloc[-252] - 1 if len(px) > 252 else np.nan
    out["mom6"] = cur / px.iloc[-126] - 1 if len(px) > 126 else np.nan
    hi52 = px.iloc[-252:].max()
    out["hi52_prox"] = cur / hi52 - 1
    vol = px.iloc[-60:].pct_change().std()
    out["low_vol"] = -vol
    return out


def _fund_factors(fund_by_t: dict, tickers) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        f = fund_by_t.get(t) or {}
        per, pbr, eps, bps, div = (f.get("PER"), f.get("PBR"), f.get("EPS"),
                                   f.get("BPS"), f.get("DIV"))
        rows[t] = {
            "value": (1.0 / per) if per and per > 0 else np.nan,
            "pbr_inv": (1.0 / pbr) if pbr and pbr > 0 else np.nan,
            "div_yield": div if div is not None else np.nan,
            "roe": (eps / bps) if (eps is not None and bps) else np.nan,
            "_eps": eps, "_per": per,   # live 필터 재현용(팩터 아님)
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _flow_factors(flow_by_tag: dict, mktcap_by_t: dict, tickers) -> pd.DataFrame:
    rows = {}
    frgn, inst = (flow_by_tag or {}).get("frgn") or {}, (flow_by_tag or {}).get("inst") or {}
    for t in tickers:
        mc = mktcap_by_t.get(t)
        rows[t] = {
            "frgn_flow": (frgn[t] / mc) if (t in frgn and mc) else np.nan,
            "inst_flow": (inst[t] / mc) if (t in inst and mc) else np.nan,
            "size": -np.log(mc) if mc else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def build_kr_snaps(panel, bench, membership, fundamentals, rebal_days=63,
                   flows=None, mktcaps=None, exclude_top=0):
    """backtest_costs.eval_config 가 그대로 쓸 수 있는 snaps 구조를 만든다.
    exclude_top>0 이면 각 시점 시총 상위 N종목 제외(초대형주 쏠림 진단용)."""
    bench = bench.reindex(panel.index).ffill()
    ma200 = panel.rolling(200, min_periods=200).mean()
    n = len(panel); max_h = max(BW.TD.values())
    ps = list(range(LOOKBACK, n - max_h - 1, rebal_days))
    snaps, n_pit_ok = [], 0
    for p in ps:
        d8 = panel.index[p].strftime("%Y%m%d")
        members = membership.get(d8)
        if members:
            n_pit_ok += 1
        else:
            members = membership.get("_current") or list(panel.columns)
        idx = panel.columns.intersection(members)
        idx = [s for s in idx if pd.notna(panel.iloc[p][s]) and pd.notna(panel.iloc[p - LOOKBACK][s])]
        mc_by_t = (mktcaps or {}).get(d8) or {}
        if exclude_top > 0 and mc_by_t:
            top = set(sorted((t for t in idx if t in mc_by_t),
                             key=lambda t: mc_by_t[t], reverse=True)[:exclude_top])
            idx = [t for t in idx if t not in top]
        if len(idx) < 20:
            continue
        pf = _price_factors(panel, p).loc[idx]
        ff = _fund_factors(fundamentals.get(d8) or {}, idx)
        lf = _flow_factors((flows or {}).get(d8), mc_by_t, idx)
        raw = pd.concat([pf, ff[FUND_FACTORS], lf], axis=1).loc[idx]
        e = p + 1
        fwd = {h: (panel.iloc[e + hd][idx] / panel.iloc[e][idx] - 1) for h, hd in BW.TD.items()}
        bnc = {h: float(bench.iloc[e + hd] / bench.iloc[e] - 1) for h, hd in BW.TD.items()}
        # live 필터 재현 재료: EPS>0 · ROE≥8% · 0<PER≤40 · 종가>200MA
        eps = ff["_eps"].reindex(idx); per = ff["_per"].reindex(idx); roe = ff["roe"].reindex(idx)
        above = panel.iloc[p][idx] > ma200.iloc[p][idx]
        live_ok = ((eps > 0) & (roe >= 0.08) & (per > 0) & (per <= 40) & above).fillna(False)
        snaps.append({"date": panel.index[p].date().isoformat(), "raw": raw,
                      "z": raw.apply(BW._z).fillna(0.0), "fwd": fwd, "bench": bnc,
                      "live_ok": live_ok})
    return snaps, n_pit_ok, len(ps)


# ------------------------- live 규칙 재현 평가 -------------------------
def eval_live(snaps, cost, topn):
    """현행 kr_stocks 규칙: 필터 통과 종목 중 z(mom12_1)×0.6 + z(hi52)×0.4 상위 N."""
    gv = {h: [] for h in BW.TD}; nv = {h: [] for h in BW.TD}; ex = {h: [] for h in BW.TD}
    sels = []
    for s in snaps:
        ok = s["live_ok"]
        pool = ok[ok].index
        if len(pool) < max(topn, 5):
            continue
        z = s["z"].loc[pool]
        score = z["mom12_1"] * 0.6 + z["hi52_prox"] * 0.4
        top = score.sort_values(ascending=False).index[:topn]
        sels.append(set(top))
        for h in BW.TD:
            r = s["fwd"][h].reindex(top).dropna()
            if len(r):
                gross = float(r.mean())
                net = float(np.mean([cost.net(x) for x in r]))
                gv[h].append(gross); nv[h].append(net); ex[h].append(net - s["bench"][h])
    row = {"weights": {"mom12_1": 0.6, "hi52_prox": 0.4},
           "filter": "EPS>0 & ROE>=8% & 0<PER<=40 & 종가>200MA"}
    turns = [1 - len(sels[j] & sels[j-1]) / max(len(sels[j]), 1) for j in range(1, len(sels))]
    row["turnover"] = round(100 * float(np.mean(turns)), 1) if turns else None
    row["n_events"] = len(nv["6m"])
    for h in BW.TD:
        if nv[h]:
            a, e2 = np.array(nv[h]), np.array(ex[h])
            row[f"ret_{h}_gross"] = round(100 * float(np.mean(gv[h])), 2)
            row[f"ret_{h}"] = round(100 * a.mean(), 2)
            row[f"excess_{h}"] = round(100 * e2.mean(), 2)
            row[f"win_{h}"] = round(100 * float((a > 0).mean()), 1)
            row[f"worst_{h}"] = round(100 * float(a.min()), 1)
            if e2.std() > 0:
                row[f"sharpe_{h}"] = round(float(e2.mean() / e2.std()) * (252.0 / BW.TD[h]) ** 0.5, 2)
    return row


def _pick_broad(ic_sorted, keep=KEEP):
    sel = [f for f, ic in ic_sorted if ic > 0][:keep]
    return sel if len(sel) >= 2 else [f for f, _ in ic_sorted[:2]]


def run(panel, bench, membership, fundamentals, cost, topn=10, rebal_days=63, oos_frac=0.4,
        flows=None, mktcaps=None, exclude_top=0):
    snaps, n_pit_ok, n_ps = build_kr_snaps(panel, bench, membership, fundamentals, rebal_days,
                                           flows=flows, mktcaps=mktcaps, exclude_top=exclude_top)
    if len(snaps) < 8:
        raise RuntimeError(f"이벤트 부족({len(snaps)}) — 기간·데이터를 확인하세요.")
    _log(f"이벤트 {len(snaps)}개 (PIT 멤버십 확보 {n_pit_ok}/{n_ps}) · 비용 {cost.describe()}"
         + (f" · 시총상위 {exclude_top}종목 제외" if exclude_top else ""))

    # 기간분리 IC(외부 피드백: 2024-07 이후 반도체 쏠림 급등장 분리 진단)
    pre_idx = [i for i, s in enumerate(snaps) if s["date"] < ANOMALY_CUTOFF]
    rec_idx = [i for i, s in enumerate(snaps) if s["date"] >= ANOMALY_CUTOFF]
    ic_pre = dict(BC._agg_ic(snaps, pre_idx)) if len(pre_idx) >= 4 else None
    ic_recent = dict(BC._agg_ic(snaps, rec_idx)) if len(rec_idx) >= 4 else None

    live = eval_live(snaps, cost, topn)

    allidx = list(range(len(snaps)))
    if oos_frac and 0 < oos_frac < 0.9:
        cut = int(len(snaps) * (1 - oos_frac))
        train, test = allidx[:cut], allidx[cut:]
    else:
        train, test = allidx, None
    ic_sorted = BC._agg_ic(snaps, train)
    selected = _pick_broad(ic_sorted, KEEP)
    _log(f"IC 순위(훈련구간): {ic_sorted}")
    _log(f"후보(IC>0 상위 {KEEP}, 강제포함 없음): {selected}")

    grid = [BC.eval_config(w, snaps, train, cost, topn) for w in BW._weight_grid(selected, LEVELS)]
    best = max(grid, key=BW.score_config)
    oos = None
    if test:
        o = BC.eval_config(best["weights"], snaps, test, cost, topn)
        oos = {"train": {k: best.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
               "test": {k: o.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
               "n_train": len(train), "n_test": len(test)}
        best = BC.eval_config(best["weights"], snaps, allidx, cost, topn)

    trials, matrix = [], []
    for w in BW._weight_grid(selected, LEVELS):
        row, ev6 = BC.eval_config(w, snaps, allidx, cost, topn, collect_6m=True)
        trials.append(BW._wstr(w)); matrix.append(ev6)
    n_ev = min(len(m) for m in matrix) if matrix else 0
    trial_data = {"horizon": "6m", "universe": "kospi200_pit", "cost": cost.describe(),
                  "rebal_days": rebal_days, "hold_days": BW.TD["6m"],
                  "dates": [snaps[i]["date"] for i in range(n_ev)],
                  "trials": trials, "excess_returns": [m[:n_ev] for m in matrix]}
    sfx = f"_ex{exclude_top}" if exclude_top else ""
    trial_path = TRIAL_PATH.replace(".json", f"{sfx}.json")
    report_path = REPORT_PATH.replace(".json", f"{sfx}.json")
    compare_path = COMPARE_PATH.replace(".json", f"{sfx}.json")
    os.makedirs("output", exist_ok=True)
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    try:
        rpt = OS.analyze(trial_data, save=False)
    except RuntimeError as e:
        rpt = {"error": str(e), "passed": False,
               "pbo": {"pbo": None}, "pbo_verdict": "계산 불가", "dsr": {}, "dsr_verdict": "계산 불가"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2)

    payload = {"as_of": panel.index[-1].date().isoformat(),
               "universe": "KOSPI200 (PIT, pykrx)", "n_events": len(snaps),
               "pit_membership_ok": f"{n_pit_ok}/{n_ps}",
               "coverage_note": "yfinance .KS 시세 — 상장폐지 종목 누락분은 잔존 생존편향",
               "cost_model": cost.describe(), "topn": topn, "rebal_days": rebal_days,
               "exclude_top": exclude_top,
               "live_rule": live,
               "ic_train": dict(ic_sorted), "candidates": selected,
               "ic_pre_anomaly": ic_pre, "ic_recent_anomaly": ic_recent,
               "anomaly_cutoff": ANOMALY_CUTOFF,
               "searched_best": best, "oos": oos,
               "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
               "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
               "passed": rpt.get("passed", False),
               "note": ("[live]=현행 kr_stocks 규칙 재현(탐색 없음 — 그대로 신뢰 가능한 기대값). "
                        "[searched]=IC 그리드 탐색 — passed=true일 때만 채택 논의. "
                        "DART 재무팩터(gross_margin·accruals 등)는 DART_API_KEY 발급 후 "
                        "kr_factor_ic.py로 별도 검증(이 결과와 합산 판단).")}
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {compare_path}")
    _log(f"  [live] 6M초과 {live.get('excess_6m')}%p · 12M초과 {live.get('excess_12m')}%p · "
         f"승률6M {live.get('win_6m')}% (이벤트 {live.get('n_events')})")
    _log(f"  [searched] {BW._wstr(best['weights'])} 6M초과 {best.get('excess_6m')}%p · "
         f"PBO {payload['pbo']} · DSR {payload['dsr']} · passed={payload['passed']}")
    return payload


# ------------------------- self-test (네트워크 불필요) -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 KR 스냅샷·live 필터·탐색 파이프라인 점검")
    rng = np.random.default_rng(11)
    n_days, n_syms = 2100, 60
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    quality = rng.normal(0, 1, n_syms)
    drift = 0.0002 + 0.0004 * quality
    panel = pd.DataFrame({f"{i:06d}": 100 * np.exp(np.cumsum(rng.normal(drift[i], 0.02, n_days)))
                          for i in range(n_syms)}, index=dates)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days))), index=dates)
    membership = {"_current": list(panel.columns)}
    fundamentals = {}
    ma_dates = [dates[p].strftime("%Y%m%d") for p in range(LOOKBACK, n_days, 63)]
    for d in ma_dates:
        fundamentals[d] = {c: {"PER": float(np.clip(20 - 5 * quality[i] + rng.normal(0, 2), 3, 60)),
                               "PBR": float(np.clip(1.5 + rng.normal(0, 0.5), 0.3, 8)),
                               "EPS": float(1000 * (1 + 0.5 * quality[i])),
                               "BPS": 10000.0,
                               "DIV": float(np.clip(2 + quality[i], 0, 8))}
                           for i, c in enumerate(panel.columns)}
    # 수급·시총 합성: frgn_flow에도 quality 신호를 심음(외국인이 좋은 종목을 삼)
    flows, mktcaps = {}, {}
    for d in ma_dates:
        mktcaps[d] = {c: float(1e12 * np.exp(0.5 * quality[i]))
                      for i, c in enumerate(panel.columns)}
        flows[d] = {"frgn": {c: float(1e10 * (quality[i] + rng.normal(0, 0.5)))
                             for i, c in enumerate(panel.columns)},
                    "inst": {c: float(rng.normal(0, 1e10)) for c in panel.columns}}
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    payload = run(panel, bench, membership, fundamentals, cost, topn=8, rebal_days=63,
                  oos_frac=0.4, flows=flows, mktcaps=mktcaps)
    assert payload["n_events"] >= 8
    assert payload["live_rule"].get("excess_6m") is not None, "live 규칙 평가 실패"
    assert payload["candidates"], "IC 후보 선정 실패"
    ic = payload["ic_train"]
    # 심은 신호: quality가 drift·EPS·저PER·외국인수급을 모두 결정 → 해당 팩터 IC>0이어야 함
    assert any(ic.get(f, -1) > 0 for f in ("value", "roe", "div_yield")), f"심은 펀더멘탈 신호 미탐지: {ic}"
    assert "frgn_flow" in ic and ic["frgn_flow"] > 0, f"심은 수급 신호 미탐지: {ic.get('frgn_flow')}"
    # exclude_top 진단 경로: 시총 상위 5 제외 시에도 파이프라인이 정상 동작해야 함
    p2 = run(panel, bench, membership, fundamentals, cost, topn=8, rebal_days=63,
             oos_frac=0.4, flows=flows, mktcaps=mktcaps, exclude_top=5)
    assert p2["exclude_top"] == 5 and p2["n_events"] >= 8
    assert os.path.exists(COMPARE_PATH.replace(".json", "_ex5.json")), "제외 버전 별도 파일 저장 실패"
    _log("[self-test] 통과: 스냅샷 · live 필터 · 펀더멘탈/수급 IC 탐지 · exclude_top · 기간분리 OK")


def prepare_kr_data(years: int, rebal_days: int):
    """실데이터 준비(다른 스크립트도 재사용 — backtest_exec/entry_gate의 KR 모드).
    반환: (panel, membership, fundamentals, flows, mktcaps, bench)"""
    from kr_factor_ic import build_kr_panel, kospi200_members
    cache = _load_cache()
    # 최근 영업일 탐색(최대 7일 소급) — 기존 '오늘/3일 전' 2회 시도는 새벽 실행(당일 데이터
    # 미발행)+주말이 겹치면 실패했다(kr_stocks._krx_universe_funda와 동일 관행으로 통일)
    cur = None
    for back in range(8):
        cur = kospi200_members((pd.Timestamp.today() - pd.Timedelta(days=back)).strftime("%Y%m%d"))
        if cur:
            break
    if not cur:
        raise RuntimeError("코스피200 구성종목 조회 실패 — 네트워크 확인")

    # 리밸 날짜 격자 확정을 위해 임시로 현재 구성 시세부터 받고, PIT 멤버십 합집합으로 보강
    _log(f"코스피200 현재 구성 {len(cur)}종목 — 시세 다운로드(yfinance .KS)…")
    panel = build_kr_panel(cur, years)
    n = len(panel)
    ps = list(range(LOOKBACK, n - max(BW.TD.values()) - 1, rebal_days))
    rebal_dates = [panel.index[p].strftime("%Y%m%d") for p in ps]
    _log(f"리밸 시점 {len(rebal_dates)}개 — PIT 멤버십·펀더멘탈 수집(pykrx, 캐시)…")
    membership = fetch_membership(rebal_dates, cache)
    membership["_current"] = cur
    union = set(cur)
    for d in rebal_dates:
        if membership.get(d):
            union |= set(membership[d])
    extra = sorted(union - set(panel.columns))
    if extra:
        _log(f"과거 편입 이력 {len(extra)}종목 시세 보강…")
        panel2 = build_kr_panel(extra, years)
        panel = pd.concat([panel, panel2], axis=1).sort_index()
        cov = 100 * len(panel.columns) / len(union)
        _log(f"시세 확보 {len(panel.columns)}/{len(union)}종목 (커버리지 {cov:.1f}%)")
        # concat으로 패널 거래일 인덱스가 바뀌면 리밸 날짜도 바뀐다 → 최종 패널 기준 재계산·재수집
        n = len(panel)
        ps = list(range(LOOKBACK, n - max(BW.TD.values()) - 1, rebal_days))
        rebal_dates = [panel.index[p].strftime("%Y%m%d") for p in ps]
        _log(f"최종 패널 기준 리밸 시점 {len(rebal_dates)}개 — 멤버십·펀더멘탈 보강 수집…")
        membership = fetch_membership(rebal_dates, cache)
        membership["_current"] = cur
    fundamentals = fetch_fundamentals(rebal_dates, cache)
    _log("수급(외국인·기관)·시총 수집(pykrx, 캐시)…")
    flows = fetch_flows(rebal_dates, cache)
    mktcaps = fetch_mktcap(rebal_dates, cache)
    bench = fetch_bench(panel.index[0].strftime("%Y%m%d"),
                        panel.index[-1].strftime("%Y%m%d"), cache)
    return panel, membership, fundamentals, flows, mktcaps, bench


def run_sr_ic(panel, membership, rebal_days=63, hold="6m"):
    """S/R 신호 7종(backtest_exec.SR_CANDIDATES)의 국장 횡단면 IC — 스냅샷 단위,
    중첩 보정 t. 결과: output/kr_sr_ic.json (미장은 score_calibration --candidates sr 담당)."""
    import backtest_exec as BE
    from statistics import NormalDist
    hd = BW.TD[hold]
    _log(f"S/R 신호 패널 계산({panel.shape[1]}종목 — 수 분 소요)…")
    sig = BE.sr_signal_panels(panel)
    n = len(panel)
    ics = {f: [] for f in BE.SR_CANDIDATES}
    n_ev = 0
    for p in range(LOOKBACK, n - hd - 1, rebal_days):
        d8 = panel.index[p].strftime("%Y%m%d")
        members = membership.get(d8) or membership.get("_current") or []
        idx = [s for s in panel.columns.intersection(members)
               if pd.notna(panel.iloc[p][s]) and pd.notna(panel.iloc[p + hd][s])]
        if len(idx) < 20:
            continue
        fwd_rank = (panel.iloc[p + 1 + hd][idx] / panel.iloc[p + 1][idx] - 1).rank()
        for f in BE.SR_CANDIDATES:
            v = sig[f].iloc[p][idx]
            if v.notna().sum() >= 15:
                r = v.rank().corr(fwd_rank)
                if pd.notna(r):
                    ics[f].append(float(r))
        n_ev += 1
    n_eff = max(int(round(n_ev * min(rebal_days / hd, 1.0))), 3)
    rows = []
    for f, arr in ics.items():
        if len(arr) < 4:
            continue
        a = np.array(arr)
        t = float(a.mean() / (a.std(ddof=1) / np.sqrt(n_eff))) if a.std(ddof=1) > 0 else 0.0
        p_val = 2 * (1 - NormalDist().cdf(abs(t)))
        rows.append({"factor": f, "ic_mean": round(float(a.mean()), 4),
                     "t_stat_eff": round(t, 2), "p_value": round(p_val, 4),
                     "n_snapshots": len(arr), "n_eff": n_eff})
    rows.sort(key=lambda r: r["ic_mean"], reverse=True)
    payload = {"as_of": panel.index[-1].date().isoformat(), "horizon": hold,
               "universe": "KOSPI200(PIT)", "rows": rows,
               "note": "S/R 신호 국장 횡단면 IC — 채택 기준은 미장과 동일(스냅샷 t≥2, 다중검정 예산 기록)"}
    os.makedirs("output", exist_ok=True)
    with open("output/kr_sr_ic.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log("저장: output/kr_sr_ic.json")
    for r in rows:
        _log(f"  {r['factor']:16s} IC {r['ic_mean']:+.4f}  t={r['t_stat_eff']}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="한국(코스피200) 팩터 백테스트 — 미국과 동일 규율")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--oos", type=float, default=0.4)
    ap.add_argument("--commission-bps", type=float, default=1.5)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--exclude-top", type=int, default=0,
                    help="각 시점 시총 상위 N종목 제외(초대형주·반도체 쏠림 진단)")
    ap.add_argument("--sr-ic", action="store_true",
                    help="S/R 신호 7종의 국장 횡단면 IC만 계산(팩터 백테스트 대신)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    panel, membership, fundamentals, flows, mktcaps, bench = prepare_kr_data(args.years, args.rebal_days)
    if args.sr_ic:
        run_sr_ic(panel, membership, rebal_days=args.rebal_days); return
    cost = BC.CostModel("kospi", args.commission_bps, args.slippage_bps)
    run(panel, bench, membership, fundamentals, cost,
        topn=args.topn, rebal_days=args.rebal_days, oos_frac=args.oos,
        flows=flows, mktcaps=mktcaps, exclude_top=args.exclude_top)


if __name__ == "__main__":
    main()
