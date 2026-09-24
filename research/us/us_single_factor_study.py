#!/usr/bin/env python3
"""
us_single_factor_study.py — 지호 님 요청(2026-09-24): "변수를 훨씬 더 많이 더해서, 가중치는 무시하고
개별 팩터가 얼마나 영향력 있는지 확인."

가중합성·종목선정 없이 **팩터 하나씩** 예측력을 잰다. 두 부분:

A. S&P500 PIT 10년(분할 보정·종료종목 포함 데이터), 월초 스냅샷, 약 60개 팩터
   - 가격 기반(모멘텀·반전·추세·변동성·베타·꼬리·계절성 등), 기존 EDGAR 팩터 24개,
     EDGAR 추가 팩터(규모·B/M·회전율·투자·순발행·이익변화·F-score 등)
   - 지표: 순위IC(1·3·6·12개월 순방향, Newey-West t), 섹터중립 IC(6개월, 현재 GICS — PIT 아님),
     5분위 동일비중·시총비중 수익(6개월, 유니버스 평균 대비)과 Q5−Q1 스프레드, 단조성,
     Q5 vs SPY, 연도별 IC·양수 연도 비율, 학습(결과 확정 <2022)·검증(2022~) IC,
     다중검정: 6개월 IC의 BH-FDR q값, Harvey-Liu-Zhu(2016) 기준 |t|>3 표시
   - 팩터 간 평균 단면 순위상관(중복 팩터 묶음 확인용)
B. 켄 프렌치 장기(1963~, 미국 전체 상장주): 같은 계열 팩터의 10분위 Hi−Lo(시총·동일비중)
   월수익 — 전체·1963-99·2000-09·2010~·2016-10~ 구간. A의 10년 결과가 우연인지 대조.

해석 주의: 팩터 수가 많아 우연히 |t|>2가 몇 개 나오는 건 정상(60개면 약 3개). "영향력 있음"은
FDR q<0.10 이면서 장기(B)에서도 같은 방향일 때만 말한다. 방향은 원값 그대로(부호가 음수면
'낮을수록 좋음').

실행: python -m research.us.us_single_factor_study [--years 10]
결과: output/us_single_factor_study.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import fundamentals_edgar as F
import research.us.us_capweight_momentum_validation as V

HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MIN_NAMES = 100
TEST_START = "2022-01-01"
FF_FILES = {   # 우리 팩터 ↔ 켄 프렌치 단변량 정렬 파일(10분위)
    "book_to_market": "Portfolios_Formed_on_BE-ME_CSV",
    "op_profitability(≈op_margin·gp_assets)": "Portfolios_Formed_on_OP_CSV",
    "investment(≈asset_growth, 부호반대)": "Portfolios_Formed_on_INV_CSV",
    "size(≈log_mktcap)": "Portfolios_Formed_on_ME_CSV",
    "momentum_12_2(≈mom12_1)": "10_Portfolios_Prior_12_2_CSV",
    "reversal_1m(≈mom1)": "10_Portfolios_Prior_1_0_CSV",
    "long_reversal_60_13(≈mom36_12)": "10_Portfolios_Prior_60_13_CSV",
    "earnings_yield(≈value)": "Portfolios_Formed_on_E-P_CSV",
    "cashflow_yield(≈fcf_yield)": "Portfolios_Formed_on_CF-P_CSV",
    "dividend_yield(≈div_yield)": "Portfolios_Formed_on_D-P_CSV",
    "accruals(≈accruals, 부호반대)": "Portfolios_Formed_on_AC_CSV",
    "net_share_issues(≈net_issuance)": "Portfolios_Formed_on_NI_CSV",
    "beta(≈beta_1y)": "Portfolios_Formed_on_BETA_CSV",
    "variance(≈vol_1m)": "Portfolios_Formed_on_VAR_CSV",
    "residual_variance(≈idio_vol_1y)": "Portfolios_Formed_on_RESVAR_CSV",
}


def _log(m): print(f"[단일팩터]  {m}", file=sys.stderr)


# ------------------------- 팩터 계산 -------------------------
def price_panels(panel: pd.DataFrame, spy: pd.Series) -> dict:
    """가격만으로 계산하는 팩터 패널(날짜×종목). 전부 t일 종가까지의 정보만 사용."""
    import tech_factors as T
    r = panel.pct_change()
    m = spy.reindex(panel.index).ffill().pct_change()
    out = {
        "mom1": panel / panel.shift(21) - 1,
        "mom3": panel / panel.shift(63) - 1,
        "mom6": panel / panel.shift(126) - 1,
        "mom12_1": panel.shift(21) / panel.shift(252) - 1,
        "mom12": panel / panel.shift(252) - 1,
        "mom36_12": panel.shift(252) / panel.shift(756) - 1,
        "mom_accel": panel / panel.shift(126) - panel.shift(126) / panel.shift(252),
        "hi52": panel / panel.rolling(252, min_periods=200).max() - 1,
        "lo52": panel / panel.rolling(252, min_periods=200).min() - 1,
        "ma50_gap": panel / panel.rolling(50).mean() - 1,
        "ma200_gap": panel / panel.rolling(200).mean() - 1,
        "vol_1m": r.rolling(21).std(),
        "vol_12m": r.rolling(252, min_periods=200).std(),
        "downside_vol_12m": r.where(r < 0, 0.0).rolling(252, min_periods=200).std(),
        "max_ret_1m": r.rolling(21).max(),
        "skew_12m": r.rolling(252, min_periods=200).skew(),
        "pct_up_12m": (r > 0).astype(float).where(r.notna()).rolling(252, min_periods=200).mean(),
    }
    out["risk_adj_mom"] = out["mom12_1"] / out["vol_12m"]
    # 정보 이산성(Da-Gurun-Warachka 2014): sign(12-1개월) × (하락일% − 상승일%) — 낮을수록 '연속적' 모멘텀
    up = (r > 0).astype(float).where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    dn = (r < 0).astype(float).where(r.notna()).shift(21).rolling(231, min_periods=180).mean()
    out["info_discreteness"] = np.sign(out["mom12_1"]) * (dn - up)
    var_m = m.rolling(252, min_periods=200).var()
    beta = r.rolling(252, min_periods=200).cov(m).div(var_m, axis=0)
    out["beta_1y"] = beta
    resid = r.sub(beta.mul(m, axis=0))
    out["idio_vol_1y"] = resid.rolling(252, min_periods=200).std()
    # 계절성(Heston-Sadka 2008): 과거 1~5년 같은 달(=다음 달) 수익 평균 — 월초 스냅샷에서 '이번 달'
    mon = panel.resample("ME").last().pct_change()
    seas = sum(mon.shift(12 * k) for k in range(1, 6)) / 5.0
    # 스냅샷 날짜 t(월초)의 값 = t가 속한 달의 과거 같은 달 평균 → 월말 인덱스를 다음 달 초로 밀기
    seas.index = seas.index + pd.offsets.MonthBegin(0)
    out["seasonality_same_month"] = seas.shift(-1).reindex(panel.index, method="ffill")
    cross = T.build_panels(panel)
    for k in ("gc60_200", "squeeze2060", "ma_align", "resid_mom", "ma100_gap"):
        out[k] = cross[k]
    return out


def _pairs(rec, key, d):
    """filed<=d 인 마지막 두 값 (now, prev)."""
    return F.asof_pair(rec.get(key), d)


def _shares_now_prev(rec: dict, d: str):
    ni = {x["end"]: x["val"] for x in rec.get("ni") or [] if x["filed"] <= d}
    sh = []
    for e in rec.get("eps") or []:
        if e["filed"] > d:
            break
        n = ni.get(e["end"])
        if n is None or not e["val"]:
            continue
        s = n / (e["val"] / F.split_factor_after(rec, e["filed"]))
        if s > 0:
            sh.append(s)
    return (sh[-1] if sh else None), (sh[-2] if len(sh) > 1 else None)


def extra_fund(rec: dict, d: str, price: float) -> dict:
    """fundamentals_edgar.FUND_FACTOR_NAMES에 없는 추가 EDGAR 팩터(연구 전용)."""
    rec = rec or {}
    g = lambda k: F.asof(rec.get(k), d)
    out = {}
    sh, sh_prev = _shares_now_prev(rec, d)
    mc = sh * price if sh and price else None
    assets, eq, rev, cash = g("assets"), g("equity"), g("revenue"), g("cash")
    ocf, ni, capex, rnd, sga, liab = g("ocf"), g("ni"), g("capex"), g("rnd"), g("sga"), g("liab")
    if mc:
        out["log_mktcap"] = math.log(mc)
        if eq is not None:
            out["book_to_market"] = eq / mc
    if assets:
        if rev is not None: out["asset_turnover"] = rev / assets
        if capex is not None: out["capex_assets"] = abs(capex) / assets
        if cash is not None: out["cash_assets"] = cash / assets
        if ocf is not None: out["ocf_assets"] = ocf / assets
        if liab is not None: out["liab_assets"] = liab / assets
    if rev:
        if rnd is not None: out["rd_sales"] = rnd / rev
        if sga is not None: out["sga_sales"] = sga / rev
    if sh and sh_prev:
        out["net_issuance"] = sh / sh_prev - 1
    if ocf is not None and ni not in (None, 0):
        out["ocf_to_ni"] = ocf / abs(ni)
    eps_pts = [e for e in rec.get("eps") or [] if e["filed"] <= d]
    if len(eps_pts) >= 2 and price:                  # 각 EPS를 자기 공시일 기준 분할배수로 현재 기준화
        e_now, e_prev = [e["val"] / F.split_factor_after(rec, e["filed"]) for e in eps_pts[-2:]][::-1]
        out["eps_chg_price"] = (e_now - e_prev) / price
    gp_now, gp_prev = _pairs(rec, "gross", d)
    if gp_now is not None and gp_prev not in (None, 0):
        out["gp_growth"] = gp_now / abs(gp_prev) - 1
    # Piotroski F-score 근사(유동비율 신호는 데이터 없어 제외, 8점 만점)
    ni_n, ni_p = _pairs(rec, "ni", d)
    a_n, a_p = _pairs(rec, "assets", d)
    dbt_n, dbt_p = _pairs(rec, "debt", d)
    rv_n, rv_p = _pairs(rec, "revenue", d)
    if None not in (ni_n, a_n) and a_n:
        s = 0
        roa_n = ni_n / a_n
        s += roa_n > 0
        s += (ocf or 0) > 0
        s += (ocf or 0) > ni_n
        if ni_p is not None and a_p:
            s += roa_n > ni_p / a_p
        if dbt_n is not None and dbt_p is not None and a_p:
            s += dbt_n / a_n < dbt_p / a_p
        if sh and sh_prev:
            s += sh <= sh_prev * 1.001
        if gp_now is not None and gp_prev is not None and rv_n and rv_p:
            s += gp_now / rv_n > gp_prev / rv_p
        if rv_n and rv_p and a_p:
            s += rv_n / a_n > rv_p / a_p
        out["fscore"] = float(s)
    return out


# ------------------------- 통계 -------------------------
def nw_t(x, lag: int) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def p_two(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2))) if np.isfinite(t) else 1.0


def bh_q(pvals: dict) -> dict:
    ks = sorted(pvals, key=lambda k: pvals[k])
    m = len(ks)
    q, prev = {}, 1.0
    for i, k in reversed(list(enumerate(ks, 1))):
        prev = min(prev, pvals[k] * m / i)
        q[k] = prev
    return q


def _srank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


# ------------------------- B. 켄 프렌치 -------------------------
def _ff_first_tables(name: str) -> dict:
    import io, os, re, zipfile
    os.makedirs(V.FF_CACHE, exist_ok=True)
    path = os.path.join(V.FF_CACHE, name + ".zip")
    if not os.path.exists(path):
        import requests
        r = requests.get(V.FF_URL.format(name), timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    with zipfile.ZipFile(path) as z:
        lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    out = {}
    for kind, pat in (("vw", r"Value Weight"), ("ew", r"Equal Weight")):
        i = next((k for k, l in enumerate(lines) if re.search(pat, l) and "Monthly" in l), None)
        if i is None:
            continue
        rows = []
        for l in lines[i + 2:]:
            if not re.match(r"^\s*\d{6}\s*,", l):
                break
            rows.append(l)
        try:
            df = pd.read_csv(io.StringIO(lines[i + 1] + "\n" + "\n".join(rows)), index_col=0)
        except pd.errors.ParserError:
            # 원본 결함: 6_Portfolios_ME_Prior_* 의 동일비중 표는 헤더 6열·데이터 8열 — 그 표만 건너뜀
            continue
        df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m")
        df.columns = [c.strip() for c in df.columns]
        df = df.astype(float).replace([-99.99, -999], np.nan) / 100
        out[kind] = df
    return out


def long_run_ff() -> dict:
    eras = [("full", None, None), ("1963-1999", "1963", "1999"), ("2000-2009", "2000", "2009"),
            ("2010-", "2010", None), ("2016-10-", "2016-10", None)]
    res = {}
    for label, fname in FF_FILES.items():
        try:
            tabs = _ff_first_tables(fname)
        except Exception as e:
            res[label] = {"error": str(e)}
            continue
        row = {}
        for kind, df in tabs.items():
            lo = next(c for c in df.columns if c in ("Lo 10", "Lo PRIOR"))
            hi = next(c for c in df.columns if c in ("Hi 10", "Hi PRIOR"))
            spread = (df[hi] - df[lo]).loc["1963-07":]
            for era, a, b in eras:
                s = spread.loc[a:b].dropna()
                if len(s) < 24:
                    continue
                row[f"{kind}_{era}"] = {"hi_minus_lo_ann_pct": round(1200 * s.mean(), 2),
                                        "t": round(float(s.mean() / s.std(ddof=1) * math.sqrt(len(s))), 2),
                                        "months": int(len(s))}
        res[label] = row
    return res


# ------------------------- A. 실행 -------------------------
def run(years: float = 10) -> dict:
    import sp500_daily_report as R
    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "BIL"], period=f"{int(years)}y", drop_stale=False)) \
        .reindex(panel.index).ffill()
    spy = etf["SPY"]
    try:
        sector_map = R.fetch_wikipedia_sectors()
    except Exception:
        sector_map = {}
    _log(f"섹터맵 {len(sector_map)}종목(현재 GICS, 과거 종료 종목은 'NA')")
    P = price_panels(panel, spy)
    idx = panel.index
    months = V.month_starts(idx, BW.LOOKBACK)
    t_test = pd.Timestamp(TEST_START)

    snaps = []
    for p in months:
        d = idx[p].date().isoformat()
        mem = BC.membership_asof(pit, d)
        cols = [c for c in panel.columns if c in mem and c not in dupes]
        px = panel.iloc[p][cols]
        alive = panel.iloc[p - 10:p + 1][cols].nunique() > 1
        univ = px[px.notna() & alive & panel.iloc[p - 252][cols].notna()].index
        if len(univ) < MIN_NAMES:
            continue
        X = {k: v.iloc[p].reindex(univ) for k, v in P.items()}
        fv, ev = {}, {}
        for s in univ:
            price = float(px[s])
            fv[s] = F.factor_values(funds.get(s) or {}, d, price)
            ev[s] = extra_fund(funds.get(s) or {}, d, price)
        fdf = pd.DataFrame.from_dict(fv, orient="index")
        edf = pd.DataFrame.from_dict(ev, orient="index")
        X.update({c: fdf[c].reindex(univ) for c in fdf.columns})
        X.update({c: edf[c].reindex(univ) for c in edf.columns})
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        fwd = {}
        for h, hd in HORIZONS.items():
            e = p + 1
            if e + hd < len(idx):
                fwd[h] = panel.iloc[e + hd][univ] / panel.iloc[e][univ] - 1
                fwd[h + "_spy"] = float(spy.iloc[e + hd] / spy.iloc[e] - 1)
        mcap = np.exp(X["log_mktcap"]) if "log_mktcap" in X else pd.Series(np.nan, index=univ)
        sect = pd.Series({s: sector_map.get(s, "NA") for s in univ})
        snaps.append({"p": p, "date": idx[p], "X": X, "fwd": fwd, "mcap": mcap, "sect": sect})
    _log(f"스냅샷 {len(snaps)}개, 팩터 후보 {len(snaps[-1]['X'].columns)}개")
    factors = sorted(set().union(*[set(s["X"].columns) for s in snaps]))

    res = {}
    for f in factors:
        ic = {h: [] for h in HORIZONS}
        ic_dates = {h: [] for h in HORIZONS}
        sn_ic, q_ew, q_cw, q5_spy, cov = [], [], [], [], []
        for s in snaps:
            x = s["X"][f].dropna() if f in s["X"] else pd.Series(dtype=float)
            if len(x) < MIN_NAMES or x.nunique() < 5:
                continue
            cov.append(len(x))
            for h in HORIZONS:
                if h in s["fwd"]:
                    y = s["fwd"][h].reindex(x.index)
                    ok = y.notna()
                    if ok.sum() >= MIN_NAMES:
                        ic[h].append(_srank(x[ok]).corr(_srank(y[ok])))
                        ic_dates[h].append(s["date"])
            if "6m" not in s["fwd"]:
                continue
            y = s["fwd"]["6m"].reindex(x.index)
            ok = y.notna()
            x6, y6 = x[ok], y[ok]
            sec = s["sect"].reindex(x6.index)
            rx = _srank(x6) - _srank(x6).groupby(sec).transform("mean")
            ry = _srank(y6) - _srank(y6).groupby(sec).transform("mean")
            sn_ic.append(rx.corr(ry))
            q = pd.qcut(x6.rank(method="first"), 5, labels=False) + 1
            mean_all = y6.mean()
            q_ew.append([y6[q == k].mean() - mean_all for k in range(1, 6)])
            mc = s["mcap"].reindex(x6.index)
            cw = []
            for k in range(1, 6):
                w = mc[q == k].dropna()
                cw.append(float((y6[w.index] * w).sum() / w.sum()) if w.sum() > 0 else np.nan)
            q_cw.append(cw)
            q5_spy.append(V.COST.net(float(y6[q == 5].mean())) - s["fwd"]["6m_spy"])
        if not ic["6m"]:
            continue
        row = {"coverage_avg": int(np.mean(cov))}
        for h, hd in HORIZONS.items():
            a = np.array(ic[h])
            row[f"ic_{h}"] = round(float(np.nanmean(a)), 4) if len(a) else None
            row[f"t_{h}"] = round(nw_t(a, max(hd // 21 - 1, 1)), 2) if len(a) > 4 else None
        s6 = pd.Series(ic["6m"], index=ic_dates["6m"])
        by_year = s6.groupby(s6.index.year).mean()
        row["ic_6m_by_year"] = {str(y): round(float(v), 3) for y, v in by_year.items()}
        row["pct_years_positive"] = round(float((by_year > 0).mean()), 2)
        settle = s6.index + pd.offsets.BDay(HORIZONS["6m"] + 1)
        tr, te = s6[settle < t_test], s6[s6.index >= t_test]
        row["ic_6m_train"] = round(float(tr.mean()), 4) if len(tr) else None
        row["ic_6m_test"] = round(float(te.mean()), 4) if len(te) else None
        row["sector_neutral_ic_6m"] = round(float(np.nanmean(sn_ic)), 4)
        row["sector_neutral_t_6m"] = round(nw_t(sn_ic, 5), 2)
        qe = np.nanmean(np.array(q_ew), axis=0)
        qc = np.nanmean(np.array(q_cw), axis=0)
        spread = np.array(q_ew)[:, 4] - np.array(q_ew)[:, 0]
        cspread = np.array(q_cw)[:, 4] - np.array(q_cw)[:, 0]
        row["quintile_ew_6m_pct"] = [round(100 * float(v), 2) for v in qe]
        row["quintile_cw_6m_pct"] = [round(100 * float(v), 2) for v in qc]
        row["q5_minus_q1_ew_6m_pct"] = round(100 * float(np.nanmean(spread)), 2)
        row["q5_minus_q1_ew_t"] = round(nw_t(spread, 5), 2)
        row["q5_minus_q1_cw_6m_pct"] = round(100 * float(np.nanmean(cspread)), 2)
        row["q5_minus_q1_cw_t"] = round(nw_t(cspread, 5), 2)
        row["monotonicity"] = round(float(pd.Series(qe).corr(pd.Series(range(5)), method="spearman")), 2)
        row["q5_vs_spy_6m_pct"] = round(100 * float(np.mean(q5_spy)), 2)
        row["q5_vs_spy_t"] = round(nw_t(q5_spy, 5), 2)
        res[f] = row

    q = bh_q({f: p_two(r["t_6m"]) for f, r in res.items()})
    for f in res:
        res[f]["fdr_q_6m"] = round(q[f], 3)
        res[f]["hlz_t_gt_3"] = bool(abs(res[f]["t_6m"] or 0) > 3)

    # 팩터 간 평균 단면 순위상관(분기 간격 스냅샷)
    fl = list(res)
    acc, n = np.zeros((len(fl), len(fl))), 0
    for s in snaps[::3]:
        c = s["X"].reindex(columns=fl).rank(pct=True).corr(min_periods=MIN_NAMES).fillna(0.0).values
        acc += c
        n += 1
    corr = pd.DataFrame(acc / max(n, 1), index=fl, columns=fl)
    clusters = {f: [g for g in fl if g != f and abs(corr.loc[f, g]) >= 0.7] for f in fl}

    for f, r in sorted(res.items(), key=lambda kv: -abs(kv[1]["t_6m"] or 0)):
        _log(f"{f:24s} IC6m {r['ic_6m']:+.4f} t{r['t_6m']:+.2f} q{r['fdr_q_6m']:.2f} | 섹터중립 t{r['sector_neutral_t_6m']:+.2f} "
             f"| Q5-Q1 ew {r['q5_minus_q1_ew_6m_pct']:+.2f}%(t{r['q5_minus_q1_ew_t']:+.2f}) cw {r['q5_minus_q1_cw_6m_pct']:+.2f}% "
             f"| 학습 {r['ic_6m_train']} 검증 {r['ic_6m_test']} | 양수연도 {r['pct_years_positive']}")
    out = {"as_of": idx[-1].date().isoformat(), "n_snapshots": len(snaps), "n_factors": len(res),
           "universe": "S&P500 PIT members (share-class dupes removed), monthly snapshots",
           "test_start": TEST_START, "factors": res,
           "corr_clusters_abs_ge_0.7": {k: v for k, v in clusters.items() if v},
           "long_run_ff": long_run_ff(),
           "notes": ["방향은 원값 기준(IC 음수=낮을수록 좋음)", "섹터는 현재 GICS(PIT 아님)",
                     "6개월 IC t는 Newey-West(lag 5)", "5분위 수익은 유니버스 평균 대비 6개월 초과(비용 없음), "
                     "q5_vs_spy는 편도비용 반영"]}
    with open("output/us_single_factor_study.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_single_factor_study.json")
    return out


def self_test():
    q = bh_q({"a": 0.001, "b": 0.02, "c": 0.5})
    assert q["a"] <= q["b"] <= q["c"] and abs(q["a"] - 0.003) < 1e-9
    rec = {"eps": [{"end": "2019-12-31", "filed": "2020-02-01", "val": 2.0},
                   {"end": "2020-12-31", "filed": "2021-02-01", "val": 4.0}],
           "ni": [{"end": "2019-12-31", "filed": "2020-02-01", "val": 200.0},
                  {"end": "2020-12-31", "filed": "2021-02-01", "val": 440.0}],
           "splits": [["2022-01-01", 2.0]]}
    sh, shp = _shares_now_prev(rec, "2021-06-01")
    assert abs(sh - 220.0) < 1e-9 and abs(shp - 200.0) < 1e-9          # 분할 보정 후 110×2, 100×2
    e = extra_fund(rec, "2021-06-01", 50.0)
    assert abs(e["net_issuance"] - 0.10) < 1e-9
    assert abs(e["eps_chg_price"] - (4.0 - 2.0) / 2 / 50.0) < 1e-9
    _log("self-test 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    run(a.years)


if __name__ == "__main__":
    main()
