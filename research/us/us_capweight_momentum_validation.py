#!/usr/bin/env python3
"""
us_capweight_momentum_validation.py — 지호 님 요청(2026-09-24): "수정 후 재검증에서 현행 전략이
지수(S&P500·나스닥100)보다 나빴다. 지수보다 기대수익률·샤프가 높은 전략을 세워라."

진단(이 스크립트의 A절이 재현): 현행 1:2:2 팩터 top10은 **동일비중** 소수 종목이라, 지난 10년
시총 상위 소수가 지수 수익을 끌고 간 장에서 구조적으로 불리했다(동일비중 S&P500인 RSP도 SPY에
연 -3.6%p). 팩터 자체(가치·R&D·주주환원)도 이 기간 성장주 장세와 반대 방향이었다.

후보(사전등록 — 이 파일을 쓰기 전에 고정, 결과를 보고 바꾸지 않음):
  capmom30      S&P500 PIT 유니버스, S&P 모멘텀지수(SPMO) 방법론 그대로의 점수
                (12-1개월 수익 / 12개월 일별 변동성 → z(±3 클립) → z>0이면 1+z, 아니면 1/(1-z)),
                상위 30종목, 비중 ∝ 시가총액×점수, 종목당 상한 min(9%, 지수 내 시총비중×3).
                리밸런싱 시점 운(timing luck)을 없애려고 6개 슬리브가 매달 1개씩 교대로
                반기 리밸런싱(= 매달 자본의 1/6만 교체) — 특정 달 선택이 결과를 좌우하지 않게.
  capmom30_eq   위와 같은 종목·같은 슬리브, 비중만 동일비중(대조군 — "가중방식 효과" 분리)
  capmom100     종목 수 100 = SPMO 복제(엔진 검증용, 실제 SPMO와 상관·성과 비교)
  (비교) SPY·QQQ·SPMO 매수후보유, 현행 라이브 전략 계좌 NAV(--with-live)

장기 근거(B절): 켄 프렌치 데이터(1927~) 대형주·고모멘텀(BIG HiPRIOR, 시총가중) vs 시장 —
10년 표본 하나로는 기대수익을 말할 수 없어서, 같은 아이디어의 99년 성과를 시대별로 본다.

통계(C절): 월별 초과수익(vs SPY) 행렬로 PBO/DSR(overfit_stats), SPY·QQQ 대비 샤프·CAGR 차이의
짝지은 블록부트스트랩(6개월 블록), 슬리브 시작월 6가지별 성과(시점 운의 크기).

한계: 시가총액 = 수정주가 × (순이익/EPS, 분할 보정) 추정치라 실제 유동시총과 다르다. EDGAR
재무가 없는 종목(~13%)은 후보에서 빠진다. 상장폐지 종목 중 야후에 시세가 없는 것은 여전히
빠져 있다(생존편향 잔여). 세금·환율은 반영 안 함.

실행: python -m research.us.us_capweight_momentum_validation [--years 10] [--with-live]
결과: output/us_capweight_momentum_validation.json
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import fundamentals_edgar as F
import overfit_stats as OS

N_SLEEVES = 6                 # 반기 리밸런싱 × 매달 교대 = 6개 슬리브
REBAL_MONTHS = 6
WEIGHT_CAP = 0.09             # SPMO 방법론: 종목당 min(9%, 지수 내 시총비중×3)
PARENT_MULT = 3.0
COST = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}.zip"
FF_CACHE = "output/ff_cache"


def _log(m): print(f"[시총모멘텀]  {m}", file=sys.stderr)


# ------------------------- 시가총액·점수 -------------------------
def _share_class_dupes(funds: dict) -> set:
    """같은 회사의 복수 주식 클래스(GOOG/GOOGL 등)는 EDGAR 기록이 동일 — 시총이 두 번 잡히지
    않게 하나만 남긴다(사전순 첫 티커 유지)."""
    seen, drop = {}, set()
    for t in sorted(funds):
        rec = funds[t]
        if not isinstance(rec, dict) or not rec.get("ni"):
            continue
        key = json.dumps([rec.get("ni"), rec.get("eps")], sort_keys=True)
        if key in seen:
            drop.add(t)
        else:
            seen[key] = t
    return drop


def shares_asof(rec: dict, date_iso: str):
    """공시일 기준 가장 최근 (순이익/EPS) 주식수, EPS를 현재 분할 기준으로 환산."""
    ni_by_end = {}
    best = None
    for e in rec.get("eps") or []:
        if e["filed"] > date_iso:
            break
        if e["end"] not in ni_by_end:
            ni_by_end = {x["end"]: x["val"] for x in rec.get("ni") or [] if x["filed"] <= date_iso}
        n = ni_by_end.get(e["end"])
        if n is None or not e["val"]:
            continue
        sh = n / (e["val"] / F.split_factor_after(rec, e["filed"]))
        if sh > 0:
            best = sh
    return best


def momentum_z(panel, logr, members, p) -> pd.Series:
    """S&P 모멘텀지수 방법론: (12-1개월 수익) / (12개월 일별 변동성) → z, ±3 클립."""
    cols = [c for c in panel.columns if c in members]
    s, s252, s21 = panel.iloc[p][cols], panel.iloc[p - 252][cols], panel.iloc[p - 21][cols]
    alive = panel.iloc[p - 10:p + 1][cols].nunique() > 1        # 거래 종료(ffill 고정) 종목 제외
    ok = s.notna() & s252.notna() & alive
    mom = (s21 / s252 - 1)[ok]
    vol = logr.iloc[p - 251:p + 1][mom.index].std() * np.sqrt(252)
    ra = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna()
    return ((ra - ra.mean()) / ra.std()).clip(-3, 3)


def target_weights(panel, logr, funds, dupes, members, p, topn, weighting) -> dict:
    z = momentum_z(panel, logr, members, p)
    score = pd.Series(np.where(z > 0, 1 + z, 1 / (1 - z)), index=z.index)
    date_iso = panel.index[p].date().isoformat()
    mc = {}
    for s in z.index:
        if s in dupes:
            continue
        sh = shares_asof(funds.get(s) or {}, date_iso)
        if sh:
            mc[s] = sh * float(panel.iloc[p][s])
    mc = pd.Series(mc, dtype=float)
    if mc.empty:
        return {}
    top = score.reindex(mc.index).dropna().sort_values(ascending=False).index[:topn]
    if weighting == "equal":
        return {s: 1.0 / len(top) for s in top}
    w = mc[top] * score[top]
    w = w / w.sum()
    lim = np.minimum(WEIGHT_CAP, PARENT_MULT * mc[top] / mc.sum())
    if lim.sum() < 1:                      # 종목 수가 적어 두 상한을 동시에 못 맞추면 9%만 적용
        lim = pd.Series(WEIGHT_CAP, index=top)
    for _ in range(100):
        over = w > lim + 1e-12
        if not over.any():
            break
        excess = float((w[over] - lim[over]).sum())
        w[over] = lim[over]
        room = ~over & (w < lim)
        w[room] += excess * w[room] / w[room].sum()
    return dict(w / w.sum())


# ------------------------- NAV 엔진 -------------------------
def sleeve_nav(panel, rf, sched) -> pd.Series:
    """sched: [(체결일 인덱스, {종목: 비중})]. 비중 합<1이면 나머지는 현금(rf). 체결일 사이엔
    비중이 가격 따라 표류. 리밸런싱 비용 = 편도 비용 × 매매대금."""
    rets = panel.pct_change().fillna(0.0)
    rfv = rf.reindex(panel.index).fillna(0.0).values
    col = {c: i for i, c in enumerate(panel.columns)}
    R = rets.values
    nav = np.full(len(panel), np.nan)
    v, w, k = 1.0, {}, 0
    t0 = sched[0][0]
    for t in range(t0, len(panel)):
        if t > t0:
            grown = {s: x * (1 + R[t, col[s]]) for s, x in w.items()}
            cash = 1 - sum(w.values())
            g = sum(grown.values()) + cash * (1 + rfv[t])
            v *= g
            w = {s: x / g for s, x in grown.items()}
        if k < len(sched) and sched[k][0] == t:
            tgt = sched[k][1]
            buys = sum(max(tgt.get(s, 0) - w.get(s, 0), 0) for s in set(tgt) | set(w))
            sells = sum(max(w.get(s, 0) - tgt.get(s, 0), 0) for s in set(tgt) | set(w))
            v *= 1 - buys * COST.buy - sells * COST.sell
            w, k = dict(tgt), k + 1
        nav[t] = v
    return pd.Series(nav, index=panel.index).dropna()


def month_starts(idx: pd.DatetimeIndex, first: int) -> list:
    """각 달의 첫 거래일 인덱스(first 이후)."""
    out, prev = [], None
    for i in range(first, len(idx) - 1):
        m = (idx[i].year, idx[i].month)
        if m != prev:
            out.append(i)
            prev = m
    return out


def staggered(panel, rf, pit, logr, funds, dupes, topn, weighting, cache):
    """6개 슬리브: 슬리브 j는 (j번째 달부터) 6개월마다 리밸런싱. 신호는 월초 종가(p), 체결은 p+1.
    반환 (합성 NAV — 슬리브를 1/6씩 들고 교차 리밸런싱 없음, 슬리브별 NAV dict)."""
    starts = month_starts(panel.index, BW.LOOKBACK)
    sleeves = {}
    for j in range(N_SLEEVES):
        sched = []
        for p in starts[j::REBAL_MONTHS]:
            key = (p, topn, weighting)
            if key not in cache:
                cache[key] = target_weights(panel, logr, funds, dupes,
                                            BC.membership_asof(pit, panel.index[p].date().isoformat()),
                                            p, topn, weighting)
            sched.append((p + 1, cache[key]))
        sleeves[j] = sleeve_nav(panel, rf, sched)
    t0 = max(s.index[0] for s in sleeves.values())       # 모든 슬리브가 투자된 뒤부터 합성
    combo = sum(s.loc[t0:] / s.loc[t0] for s in sleeves.values()) / N_SLEEVES
    return combo, sleeves


# ------------------------- 지표 -------------------------
def metrics(nav: pd.Series, rf: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = len(r) / 252
    ex = r - rf.reindex(r.index).fillna(0.0)
    return {"cagr_pct": round(100 * float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1), 2),
            "vol_pct": round(100 * float(r.std() * np.sqrt(252)), 1),
            "sharpe": round(float(ex.mean() / r.std() * np.sqrt(252)), 3),
            "mdd_pct": round(100 * float((nav / nav.cummax() - 1).min()), 1),
            "start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat()}


def monthly(nav: pd.Series) -> pd.Series:
    return nav.resample("ME").last().pct_change().dropna()


def blend(a: pd.Series, b: pd.Series, wa: float = 0.5) -> pd.Series:
    """두 NAV를 wa:(1-wa)로 매월 첫 거래일 리밸런싱한 합성 NAV(리밸런싱 비용 무시 — 월 교체분이 작음)."""
    ix = a.index.intersection(b.index)
    r = pd.concat([a.reindex(ix).pct_change(), b.reindex(ix).pct_change()], axis=1).fillna(0.0).values
    mon = ix.to_period("M")
    va, vb, out = wa, 1 - wa, []
    for i, (ra, rb) in enumerate(r):
        if i > 0 and mon[i] != mon[i - 1]:
            tot = va + vb
            va, vb = tot * wa, tot * (1 - wa)
        va *= 1 + ra
        vb *= 1 + rb
        out.append(va + vb)
    return pd.Series(out, index=ix)


def paired_bootstrap(a: pd.Series, b: pd.Series, rf_m: pd.Series, n=10000, block=6, seed=20260924) -> dict:
    """월수익 a(전략)·b(비교) 짝지은 이동블록 부트스트랩 — 샤프·연환산수익 차이 분포."""
    df = pd.concat([a, b, rf_m], axis=1, join="inner").dropna().values
    T = len(df)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    d_sh, d_ret = np.empty(n), np.empty(n)

    def sh(x, f):
        e = x - f
        return e.mean() / x.std(ddof=1) * np.sqrt(12)

    for i in range(n):
        starts = rng.integers(0, T - block + 1, nb)
        ix = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        s = df[ix]
        d_sh[i] = sh(s[:, 0], s[:, 2]) - sh(s[:, 1], s[:, 2])
        d_ret[i] = (np.prod(1 + s[:, 0]) ** (12 / T) - np.prod(1 + s[:, 1]) ** (12 / T))
    return {"sharpe_diff_mean": round(float(d_sh.mean()), 3),
            "sharpe_diff_ci90": [round(float(np.percentile(d_sh, 5)), 3), round(float(np.percentile(d_sh, 95)), 3)],
            "p_sharpe_better": round(float((d_sh > 0).mean()), 3),
            "cagr_diff_mean_pct": round(100 * float(d_ret.mean()), 2),
            "cagr_diff_ci90_pct": [round(100 * float(np.percentile(d_ret, 5)), 2),
                                   round(100 * float(np.percentile(d_ret, 95)), 2)],
            "p_cagr_better": round(float((d_ret > 0).mean()), 3), "n_months": T, "block_months": block}


# ------------------------- B. 켄 프렌치 장기 -------------------------
def _ff_table(name: str, header_token: str) -> pd.DataFrame:
    os.makedirs(FF_CACHE, exist_ok=True)
    path = os.path.join(FF_CACHE, name + ".zip")
    if not os.path.exists(path):
        import requests
        r = requests.get(FF_URL.format(name), timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin1")
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if header_token in l)
    rows = []
    for l in lines[i + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", l):
            break
        rows.append(l)
    df = pd.read_csv(io.StringIO(lines[i] + "\n" + "\n".join(rows)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m")
    return df.astype(float) / 100


def long_run_ff() -> dict:
    ff = _ff_table("F-F_Research_Data_Factors_CSV", "Mkt-RF")
    p6 = _ff_table("6_Portfolios_ME_Prior_12_2_CSV", "SMALL LoPRIOR")
    rf = ff["RF"]
    series = {"market": ff["Mkt-RF"] + rf, "big_high_momentum": p6["BIG HiPRIOR"]}

    def st(r):
        r = r.dropna()
        nav = (1 + r).cumprod()
        yrs = len(r) / 12
        return {"cagr_pct": round(100 * float(nav.iloc[-1] ** (1 / yrs) - 1), 2),
                "sharpe": round(float((r - rf.reindex(r.index)).mean() / r.std() * np.sqrt(12)), 3),
                "mdd_pct": round(100 * float((nav / nav.cummax() - 1).min()), 1)}

    eras = [("full", None, None), ("1927-1949", None, "1949"), ("1950-1979", "1950", "1979"),
            ("1980-1999", "1980", "1999"), ("2000-2009", "2000", "2009"), ("2010-2019", "2010", "2019"),
            ("2020-", "2020", None)]
    out = {"source": "Kenneth R. French Data Library — 6 Portfolios Formed on Size and Prior (12-2) "
                     "Returns (value-weighted), F-F Research Factors",
           "last_month": ff.index[-1].strftime("%Y-%m"), "eras": {}}
    for name, a, b in eras:
        out["eras"][name] = {k: st(v.loc[a:b]) for k, v in series.items()}
    return out


# ------------------------- A. 현행 라이브 계좌 NAV -------------------------
def live_nav(panel, funds, pit, rf) -> pd.Series:
    """현행 라이브 규칙(1:2:2·floor·100일선·top10·180일+후보풀60 재평가·트레일링 없음)의 계좌 NAV.
    10거래일마다 판정(라이브는 매일 — 근사), 전량 1회 진입(분할매수 미체결 현금은 더 불리하므로
    이 근사는 라이브에 유리한 쪽). 현금은 0% 가정(backtest_portfolio.simulate 그대로)."""
    import tech_factors as T
    import backtest_exec as BE
    from research.us import backtest_portfolio as BP
    from research.us.us_full_stack_exec_validation import _live_ranked
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    dec = []
    for p in range(BW.LOOKBACK, len(panel) - 1, 10):
        r = _live_ranked(panel, p, funds, cross, pit, weights)
        if r:
            dec.append((p, r[:60]))
    ma200 = panel.rolling(200, min_periods=200).mean()
    return BP.simulate(panel, ma200, dec, 10, COST, ma200_backup=False)


# ------------------------- 실행 -------------------------
def run(years: float = 10, with_live: bool = False) -> dict:
    import sp500_daily_report as R
    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = _share_class_dupes(funds)
    _log(f"주식 클래스 중복 제외: {sorted(dupes)}")
    etf = R.download_histories(["SPY", "QQQ", "SPMO", "BIL"], period=f"{int(years)}y", drop_stale=False)
    etf = pd.DataFrame(etf).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    logr = np.log(panel).diff()

    cache, navs, sleeve_stats = {}, {}, {}
    for name, topn, wt in [("capmom30", 30, "capscore"), ("capmom30_eq", 30, "equal"),
                           ("capmom100", 100, "capscore")]:
        combo, sleeves = staggered(panel, rf, pit, logr, funds, dupes, topn, wt, cache)
        navs[name] = combo
        if name == "capmom30":
            sleeves30 = sleeves
        sleeve_stats[name] = {f"sleeve{j}": metrics(s.loc[combo.index[0]:], rf) for j, s in sleeves.items()}
        _log(f"{name}: {metrics(combo, rf)}")
    t0 = max(v.index[0] for v in navs.values())
    for k in ("SPY", "QQQ", "SPMO"):
        navs[k] = etf[k]
    if with_live:
        ln = live_nav(panel, funds, pit, rf)
        if ln is not None:
            navs["live_122_top10"] = ln
            t0 = max(t0, ln.index[0])
    navs = {k: v.loc[t0:] / v.loc[t0] for k, v in navs.items()}
    if "live_122_top10" in navs:
        # 사후 추가 후보(결과를 본 뒤 추가 — 사전등록 아님): 현행 밸류·퀄리티 바스켓의 초과수익이
        # 모멘텀·QQQ 초과수익과 음의 상관(가치×모멘텀 분산, Asness-Moskowitz-Pedersen 2013)이라
        # 비율 탐색 없이 50:50만 본다. 과거 §6-H(70:30·비율 스윕)와 같은 계열이므로 시행 수에 포함.
        navs["live50_spmo50"] = blend(navs["live_122_top10"], navs["SPMO"])
        navs["live50_capmom50"] = blend(navs["live_122_top10"], navs["capmom30"])

    res = {k: metrics(v, rf) for k, v in navs.items()}
    for k, v in res.items():
        _log(f"{k:15s} CAGR {v['cagr_pct']:6.2f}% 샤프 {v['sharpe']:.2f} MDD {v['mdd_pct']:.1f}% 변동성 {v['vol_pct']}%")

    m = {k: monthly(v) for k, v in navs.items()}
    rf_m = (1 + rf).resample("ME").prod().sub(1).reindex(m["SPY"].index)
    cands = [k for k in navs if k not in ("SPY", "QQQ")]
    boot = {f"{c}_vs_{b}": paired_bootstrap(m[c], m[b], rf_m) for c in cands for b in ("SPY", "QQQ")}
    # 끝나는 시점 민감도: 2026년 급등(현행 바스켓 MRNA 등)을 빼도 결론이 유지되는지
    windows = {"full": (None, None), "to_2025-09": (None, "2025-09-30"),
               "to_2022": (None, "2022-12-31"), "2023_on": ("2023-01-01", None)}
    window_tbl = {w: {k: metrics(v.loc[a:b] / v.loc[a:b].iloc[0], rf) for k, v in navs.items()}
                  for w, (a, b) in windows.items()}
    corr = (pd.DataFrame({k: v.pct_change() for k, v in navs.items() if k not in ("SPY",)})
            .sub(navs["SPY"].pct_change(), axis=0).dropna().corr().round(3))
    years_tbl = {k: {str(d.year): round(100 * float(x), 1)
                     for d, x in v.resample("YE").last().pct_change().dropna().items()}
                 for k, v in navs.items()}

    # 다중검정 N에 슬리브 단독(=시점 운이 그대로 남은 반기 리밸런싱 6가지)까지 넣어 보수적으로
    series = {k: m[k] for k in cands}
    for j, s in sleeves30.items():
        series[f"capmom30_sleeve{j}"] = monthly(s.loc[t0:])
    ex = pd.DataFrame({k: v - m["SPY"] for k, v in series.items()}).dropna()
    trial_data = {"horizon": "capmom_us_monthly", "universe": "pit", "cost": COST.describe(),
                  "rebal_days": 21, "hold_days": 21, "dates": [d.date().isoformat() for d in ex.index],
                  "trials": list(ex.columns), "excess_returns": ex.T.values.tolist()}
    report = OS.analyze(trial_data, save=False)

    out = {"as_of": panel.index[-1].date().isoformat(), "window_start": t0.date().isoformat(),
           "preregistered_candidates": ["capmom30", "capmom30_eq", "capmom100"],
           "method": {"score": "SPMO: (12-1m return / 12m daily vol) z±3 → 1+z | 1/(1-z)",
                      "weight": "mktcap×score, cap min(9%, 3×parent weight); _eq = equal",
                      "rebalance": f"{N_SLEEVES} staggered sleeves, each every {REBAL_MONTHS} months",
                      "mktcap": "adj price × NI/EPS (split-normalized), share-class dupes removed",
                      "cost": COST.describe(), "cash": "BIL total return"},
           "posthoc_candidates": ["live50_spmo50", "live50_capmom50"] if "live50_spmo50" in navs else [],
           "metrics": res, "windows": window_tbl, "excess_vs_spy_daily_corr": corr.to_dict(),
           "sleeves_capmom30": sleeve_stats["capmom30"],
           "calendar_year_pct": years_tbl, "paired_bootstrap_monthly": boot,
           "pbo_dsr_vs_spy": {"trials": list(ex.columns), "pbo": report["pbo"]["pbo"],
                              "dsr": report["dsr"].get("dsr"), "best": report["dsr"]["best_trial"],
                              "passed": report["passed"]},
           "long_run_ff": long_run_ff()}
    with open("output/us_capweight_momentum_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_capweight_momentum_validation.json")
    return out


def self_test():
    """합성 데이터로 엔진 무결성: (1) 비중 상한 준수 (2) 현금 비중은 rf만큼 자람 (3) 비용 차감."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    panel = pd.DataFrame({"A": np.linspace(100, 200, 300), "B": np.full(300, 50.0)}, index=idx)
    rf = pd.Series(0.0001, index=idx)
    nav = sleeve_nav(panel, rf, [(10, {"A": 0.5})])       # 50% A, 50% 현금
    exp_a = 0.5 * panel["A"].iloc[-1] / panel["A"].iloc[10]
    exp_c = 0.5 * (1.0001 ** (len(idx) - 11))
    exp = (exp_a + exp_c) * (1 - 0.5 * COST.buy)
    assert abs(nav.iloc[-1] - exp) < 1e-6, (nav.iloc[-1], exp)
    rec = {"eps": [{"end": "2020-12-31", "filed": "2021-02-01", "val": 4.0}],
           "ni": [{"end": "2020-12-31", "filed": "2021-02-01", "val": 400.0}],
           "splits": [["2022-06-01", 4.0]]}
    assert abs(shares_asof(rec, "2021-06-01") - 400.0) < 1e-9     # 100주 × 4:1 분할 = 400주(현재 기준)
    assert shares_asof(rec, "2021-01-15") is None                  # 공시 전엔 없음
    a = panel["A"] / panel["A"].iloc[0]
    bl = blend(a, a)
    assert bl.index[0] == a.index[0] and np.allclose(bl.values, a.values)   # 시작일 보존·같은 자산이면 그대로
    _log("self-test 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--with-live", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    run(a.years, a.with_live)


if __name__ == "__main__":
    main()
