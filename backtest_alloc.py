#!/usr/bin/env python3
"""
backtest_alloc.py — 주간 자산배분(weekly_report.py) 규칙의 백테스트·그리드서치.

리서치 명세(weekly_asset_allocation_research*.md) 축소 구현:
  · 후보 신호는 가격 기반만: 추세필터(SMA N) × 절대 모멘텀(M) × 컷 강도 × 도피처 × 기준배분
  · 도피처(컷한 비중이 가는 곳): 달러현금(SHY) / 미국채(IEF) / 스마트(한국주식→원화현금, 나머지→미국채)
  · 매크로/상관관계는 매매 신호에서 제외(리포트 설명용) — look-ahead 편향 방지
  · 신호는 월말 확정 → '다음 달'에 적용 (같은 종가 체결 오류 방지)
  · 리밸런싱은 월간 (주간 리포트 ≠ 매매 주기 분리 원칙)
  · 거래비용 편도 0.25% 차감, USD·KRW 두 기준 평가
  · 워크포워드: 앞 60% IS 로 상위권 선별 → 뒤 40% OOS 생존 확인
  · 생존 게이트는 원화 기준 포함(환율 자연헤지를 이겨야 동적 규칙 채택). 미달 시 정적 폴백.

실행:  python backtest_alloc.py                # 실데이터(야후) 다운로드 후 전체 그리드
       python backtest_alloc.py --synthetic    # 합성 데이터로 파이프라인 점검
출력:  output/alloc_backtest.json (전체), output/best_alloc.json (승자 설정 — weekly_report 가 읽음),
       output/ALLOC_RESULT.md (사람용 요약)
"""
from __future__ import annotations
import os, sys, json, math, argparse, itertools, datetime as dt

import numpy as np
import pandas as pd

# ------------------------- 유니버스 -------------------------
# 장기 이력 ETF/지수 (전부 야후 무료). 공통 시작 ≈ 2004-11 (GLD 상장) → 2008·2020·2022 포함.
TICKERS = {
    "US_STOCK": "SPY",    # 미국 주식 (1993~)
    "KR_STOCK": "^KS11",  # 코스피 (가격지수 — 배당 미포함, 보수적)
    "GLOBAL":   "EFA",    # 미국외 선진국 (2001~, VXUS 프록시)
    "BOND":     "IEF",    # 미국채 7-10년 (2002~)
    "GOLD":     "GLD",    # 금 (2004~)
    "REIT":     "VNQ",    # 리츠 (2004~)
}
CASH_TICKER = "SHY"       # 달러 단기채(현금 프록시, 2002~)
FX_TICKER = "KRW=X"       # USD/KRW
RISK_KEYS = ["US_STOCK", "KR_STOCK", "GLOBAL", "GOLD", "REIT"]
KRW_CASH_RATE = 0.025     # 원화 현금 연 이자 근사(콜금리 장기 평균 수준, 보수적)

# 기준 배분 후보 (합계 100)
BASE_ALLOCS = {
    "growth":   {"US_STOCK": 35, "KR_STOCK": 10, "GLOBAL": 15, "BOND": 25, "GOLD": 10, "REIT": 5},  # 현행
    "sixty40":  {"US_STOCK": 40, "KR_STOCK": 10, "GLOBAL": 10, "BOND": 40, "GOLD": 0,  "REIT": 0},
    "equal":    {"US_STOCK": 17, "KR_STOCK": 17, "GLOBAL": 17, "BOND": 17, "GOLD": 16, "REIT": 16},
    "allweather": {"US_STOCK": 25, "KR_STOCK": 5, "GLOBAL": 10, "BOND": 40, "GOLD": 15, "REIT": 5},
}

# 그리드
TREND_WINDOWS = [100, 150, 200, 250]          # SMA N일
MOM_MODES = ["none", "m6", "m12", "multi"]    # 절대 모멘텀: 없음/6M/12M/1·3·6·12 평균
CUT_SCHEMES = {                               # (둘 다 악화 시 컷, 하나만 악화 시 컷)
    "full":    (1.00, 0.50),
    "half":    (0.50, 0.25),   # 초기 weekly_report 규칙
    "binary":  (1.00, 0.00),
}
DEST_OPTS = ["BOND", "CASH", "KRW_CASH"]      # 도피처 후보: 미국채 / 달러현금 / 원화현금
# 한국주식 컷과 달러자산 컷의 도피처를 독립 탐색(매도 후 환전·KRX 상장 미국채 ETF 매수 등 자유로우므로)
DEST_SCHEMES = {f"kr>{a}|us>{b}": (a, b) for a in DEST_OPTS for b in DEST_OPTS}
COST_ONEWAY = 0.0025                           # 편도 0.25% (수수료+슬리피지 보수적)
OOS_FRAC = 0.4                                 # 뒤 40% 표본외

ALL_KEYS = list(TICKERS) + ["CASH", "KRW_CASH"]


def _log(m): print(f"[백테스트] {m}", file=sys.stderr)


def _dest_for(key: str, mode: tuple) -> str:
    """자산 key 를 컷했을 때 비중이 가는 곳. mode=(한국주식 도피처, 그 외 자산 도피처)."""
    return mode[0] if key == "KR_STOCK" else mode[1]


# ------------------------- 데이터 -------------------------
def _add_synthetic_cash(df: pd.DataFrame) -> pd.DataFrame:
    """KRW_CASH: 원화 현금의 'USD 표시 가격' = (1/환율) × 이자 누적. USD 기준 백테스트에 편입."""
    n = np.arange(len(df))
    accr = (1 + KRW_CASH_RATE) ** (n / 252)
    df["KRW_CASH"] = (1.0 / df["FX"]) * accr * 1000.0
    return df


def fetch_prices() -> pd.DataFrame:
    import yfinance as yf
    tickers = list(TICKERS.values()) + [CASH_TICKER, FX_TICKER]
    df = yf.download(tickers, period="25y", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    df = df.rename(columns={v: k for k, v in TICKERS.items()})
    df = df.rename(columns={CASH_TICKER: "CASH", FX_TICKER: "FX"})
    # 코스피는 원화 → 달러 환산(모든 자산 USD 기준 통일)
    fx = df["FX"].ffill()
    df["KR_STOCK"] = df["KR_STOCK"] / fx
    df = df.drop(columns=["FX"]).join(fx.rename("FX"))
    df = df.dropna(how="any")                  # 공통 구간만
    df = _add_synthetic_cash(df)
    _log(f"데이터 {df.index[0].date()} ~ {df.index[-1].date()} · {len(df)}일")
    return df


def synthetic_prices(days=5200, seed=7) -> pd.DataFrame:
    """네트워크 없이 파이프라인 점검용 합성 시계열(추세+국면 전환)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=dt.date.today(), periods=days)
    cols = {}
    regime = np.sign(np.sin(np.arange(days) / 260 * math.pi)) * 0.0005
    for i, k in enumerate(list(TICKERS) + ["CASH", "FX"]):
        drift = 0.0002 if k != "CASH" else 0.00008
        vol = 0.01 if k not in ("CASH", "BOND") else 0.003
        shock = regime * (1.5 if k in RISK_KEYS else -0.3)
        r = drift + shock + rng.normal(0, vol, days)
        cols[k] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(cols, index=idx)
    df["FX"] = 1300 * df["FX"] / df["FX"].iloc[0]
    return _add_synthetic_cash(df)


# ------------------------- 신호 -------------------------
def month_ends(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.groupby(df.index.to_period("M")).tail(1).index


def momentum(px: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(1.0, index=px.index)   # 항상 양수(모멘텀 미사용)
    if mode == "m6":
        return px.pct_change(126)
    if mode == "m12":
        return px.pct_change(252)
    # multi: 1·3·6·12개월 평균
    return (px.pct_change(21) + px.pct_change(63) + px.pct_change(126) + px.pct_change(252)) / 4


def target_weights(df, me, base, trend_n, mom_mode, cut_strong, cut_weak,
                   dest_mode=("CASH", "CASH")) -> pd.DataFrame:
    """월말별 목표 비중(%). 컷된 위험자산 비중은 도피처(dest_mode)로 이동. (벡터화)"""
    w = pd.DataFrame(0.0, index=me, columns=ALL_KEYS)
    for k, v in base.items():
        w[k] = float(v)
    if trend_n > len(df):                       # 정적(틸트 발동 불가)
        return w
    for k in RISK_KEYS:
        below = (df[k] < df[k].rolling(trend_n).mean()).reindex(me).fillna(False).to_numpy()
        weak = (momentum(df[k], mom_mode) < 0).reindex(me).fillna(False).to_numpy()
        cut = np.where(below & weak, cut_strong, np.where(below | weak, cut_weak, 0.0))
        moved = base.get(k, 0) * cut
        w[k] = base.get(k, 0) - moved
        w[_dest_for(k, dest_mode)] += moved
    return w


# ------------------------- 시뮬레이션 -------------------------
def monthly_returns(df) -> pd.DataFrame:
    """자산별 월간 수익률(한 번만 계산해 재사용)."""
    mret = df[ALL_KEYS].pct_change().add(1).groupby(df.index.to_period("M")).prod().sub(1)
    mret.index = mret.index.to_timestamp("M")
    return mret


def simulate(mret, tw: pd.DataFrame) -> tuple[pd.Series, float]:
    """월간 리밸런스(벡터화). t월말 신호 → t+1개월 수익에 적용. 반환: (월간수익, 연간 회전율)."""
    idx = pd.DatetimeIndex([t.to_period("M").to_timestamp("M") for t in tw.index])
    R = np.nan_to_num(mret[ALL_KEYS].reindex(idx).to_numpy())   # R[i] = i번째 월의 수익률
    W = tw[ALL_KEYS].to_numpy() / 100.0
    Wn, Rn = W[:-1], R[1:]                       # i월말 비중 → i+1월 수익
    gross = (Wn * Rn).sum(axis=1)
    grown = Wn[:-1] * (1 + Rn[:-1])              # 전월 비중의 드리프트
    prev = grown / grown.sum(axis=1, keepdims=True)
    turn = np.abs(Wn[1:] - prev).sum(axis=1)
    net = gross.copy()
    net[1:] -= turn * COST_ONEWAY
    s = pd.Series(net, index=idx[1:])
    return s, float(turn.mean() * 12) if len(turn) else 0.0


def metrics(mr: pd.Series, fx: pd.Series | None = None) -> dict:
    if len(mr) < 12:
        return {}
    curve = (1 + mr).cumprod()
    yrs = len(mr) / 12
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = mr.std() * math.sqrt(12)
    sharpe = (mr.mean() * 12) / vol if vol > 0 else 0.0
    mdd = float((curve / curve.cummax() - 1).min())
    out = {"cagr": round(100 * cagr, 2), "vol": round(100 * vol, 2),
           "sharpe": round(sharpe, 2), "mdd": round(100 * mdd, 2),
           "calmar": round(cagr / abs(mdd), 2) if mdd else None,
           "worst_12m": round(100 * float((curve.pct_change(12)).min()), 2) if len(mr) > 12 else None}
    if fx is not None:                          # 원화 기준
        f = fx.reindex(mr.index, method="ffill")
        krw = (1 + mr) * (f / f.shift(1)).fillna(1.0) - 1
        kc = (1 + krw).cumprod()
        out["krw_cagr"] = round(100 * (kc.iloc[-1] ** (1 / yrs) - 1), 2)
        out["krw_mdd"] = round(100 * float((kc / kc.cummax() - 1).min()), 2)
    return out


CRISES = [("2008 금융위기", "2007-10-31", "2009-03-31"),
          ("2020 코로나", "2020-01-31", "2020-04-30"),
          ("2022 인플레 발작", "2022-01-31", "2022-10-31")]


def crisis_returns(mr: pd.Series) -> dict:
    out = {}
    for name, a, b in CRISES:
        seg = mr.loc[a:b]
        if len(seg):
            out[name] = round(100 * float((1 + seg).prod() - 1), 1)
    return out


# ------------------------- 그리드서치 + 워크포워드 -------------------------
def score(m: dict) -> float:
    """비용 차감 후 위험조정성과: 샤프 + MDD 통제(칼마 절반 가중)."""
    if not m:
        return -9.9
    return m["sharpe"] + 0.5 * (m["calmar"] or 0)


def run_grid(df: pd.DataFrame) -> dict:
    me = month_ends(df)
    warm = max(TREND_WINDOWS) + 5
    me = me[me >= df.index[warm]]
    split = int(len(me) * (1 - OOS_FRAC))
    fx_m = df["FX"].groupby(df.index.to_period("M")).last()
    fx_m.index = fx_m.index.to_timestamp("M")

    mret = monthly_returns(df)
    results = []
    combos = list(itertools.product(BASE_ALLOCS, TREND_WINDOWS, MOM_MODES, CUT_SCHEMES, DEST_SCHEMES))
    _log(f"조합 {len(combos)}개 · 월말 {len(me)}개 (IS {split} / OOS {len(me)-split})")
    for base_name, tn, mm, cs, dm in combos:
        cut_s, cut_w = CUT_SCHEMES[cs]
        tw = target_weights(df, me, BASE_ALLOCS[base_name], tn, mm, cut_s, cut_w, DEST_SCHEMES[dm])
        mr, turn = simulate(mret, tw)
        is_end = me[split - 1]
        m_is, m_oos = metrics(mr[mr.index <= is_end]), metrics(mr[mr.index > is_end], fx_m)
        results.append({"base": base_name, "trend_n": tn, "mom": mm, "cut": cs, "dest": dm,
                        "turnover_yr": round(turn, 2),
                        "is": m_is, "oos": m_oos, "full": metrics(mr, fx_m),
                        "crisis": crisis_returns(mr),
                        "is_score": round(score(m_is), 3), "oos_score": round(score(m_oos), 3)})

    # 정적 벤치마크(틸트 없음)
    bench = {}
    for base_name in BASE_ALLOCS:
        tw = target_weights(df, me, BASE_ALLOCS[base_name], 10**9, "none", 0, 0)  # 컷 발동 불가
        mr, _ = simulate(mret, tw)
        is_end = me[split - 1]
        bench[base_name] = {"is": metrics(mr[mr.index <= is_end]),
                            "oos": metrics(mr[mr.index > is_end], fx_m),
                            "full": metrics(mr, fx_m), "crisis": crisis_returns(mr)}

    # 워크포워드 선별: IS 상위 10% → OOS 점수 최고. 생존 게이트(원화 기준 포함) 미달 시 정적 폴백.
    results.sort(key=lambda r: r["is_score"], reverse=True)
    top = results[:max(10, len(results) // 10)]
    top.sort(key=lambda r: r["oos_score"], reverse=True)
    winner, reason = None, ""
    for c in top:
        b = bench[c["base"]]["oos"]
        o = c["oos"]
        survived = (
            c["oos_score"] >= 0.5 * c["is_score"]                          # OOS 완전 붕괴 없음
            and c["oos_score"] >= score(b) - 0.05                          # 정적 대비 점수 열위 아님
            and o.get("mdd", -99) >= b.get("mdd", 0) + 3                   # USD MDD 3%p 이상 개선
            # 원화 기준(한국 투자자 체감): 환율 자연헤지 효과를 이겨야만 동적 규칙 채택
            and o.get("krw_mdd", -99) >= b.get("krw_mdd", 0) - 2           # 원화 MDD 크게 악화 금지
            and o.get("krw_cagr", -99) >= b.get("krw_cagr", 99) - 1.5)     # 원화 수익 크게 열위 금지
        if survived:
            winner, reason = c, "워크포워드 통과(정적 대비 USD MDD 개선 + 원화 기준 열위 아님)"
            break
    if winner is None:
        best_static = max(bench, key=lambda k: score(bench[k]["oos"]))
        winner = {"base": best_static, "trend_n": None, "mom": "none", "cut": "none", "dest": "none",
                  "is": bench[best_static]["is"], "oos": bench[best_static]["oos"],
                  "full": bench[best_static]["full"], "crisis": bench[best_static]["crisis"],
                  "turnover_yr": 0.0, "is_score": None, "oos_score": round(score(bench[best_static]["oos"]), 3)}
        reason = "동적 조합 전부 탈락 → 정적 기준배분 폴백 (리서치 원칙)"

    # 파라미터 평탄성: 승자의 이웃(추세창만 변경) 성과 분포
    plateau = []
    if winner.get("trend_n"):
        for r in results:
            if (r["base"], r["mom"], r["cut"], r["dest"]) == (winner["base"], winner["mom"], winner["cut"], winner["dest"]):
                plateau.append({"trend_n": r["trend_n"], "oos_score": r["oos_score"]})
    return {"as_of": dt.date.today().isoformat(),
            "period": [str(df.index[0].date()), str(df.index[-1].date())],
            "cost_oneway": COST_ONEWAY, "oos_frac": OOS_FRAC,
            "winner": winner, "reason": reason, "plateau": plateau,
            "benchmarks": bench, "top20": results[:20],
            "top_oos20": sorted(results, key=lambda r: r["oos_score"], reverse=True)[:20]}


# ------------------------- 출력 -------------------------
def export(out: dict):
    os.makedirs("output", exist_ok=True)
    with open("output/alloc_backtest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    w = out["winner"]
    best = {
        "as_of": out["as_of"], "source": "backtest_alloc.py", "reason": out["reason"],
        "base_name": w["base"], "base_weights": BASE_ALLOCS[w["base"]],
        "trend_window": w.get("trend_n"),            # None 이면 정적(틸트 없음)
        "mom_mode": w.get("mom", "none"),
        "cut_strong": CUT_SCHEMES.get(w.get("cut"), (0, 0))[0],
        "cut_weak": CUT_SCHEMES.get(w.get("cut"), (0, 0))[1],
        # 도피처: {"kr": 한국주식 컷 행선지, "us": 그 외 컷 행선지} — BOND|CASH|KRW_CASH
        "dest": ({"kr": DEST_SCHEMES[w["dest"]][0], "us": DEST_SCHEMES[w["dest"]][1]}
                 if w.get("dest") in DEST_SCHEMES else "none"),
        "dest_name": w.get("dest", "none"),
        "oos": w["oos"], "crisis": w["crisis"],
    }
    with open("output/best_alloc.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1)

    def _row(r):
        return (f"| {r['base']} | {r['trend_n']} | {r['mom']} | {r['cut']} | {r.get('dest')} | {r['is_score']} | "
                f"{r['oos_score']} | {r['oos'].get('mdd')}% | {r['oos'].get('krw_cagr')}% | "
                f"{r['oos'].get('krw_mdd')}% | {r['turnover_yr']} |")
    hdr = ["| base | 추세 | 모멘텀 | 컷 | 도피처 | IS점수 | OOS점수 | OOS MDD | 원화CAGR | 원화MDD | 회전율 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines = [f"# 자산배분 백테스트 결과 ({out['as_of']})", "",
             f"- 기간: {out['period'][0]} ~ {out['period'][1]} · 비용 편도 {100*COST_ONEWAY:.2f}% · OOS 뒤 {int(100*OOS_FRAC)}%",
             f"- **승자**: 기준배분 `{w['base']}` · 추세 {w.get('trend_n')}일 · 모멘텀 {w.get('mom')} · "
             f"컷 {w.get('cut')} · 도피처 {w.get('dest')}",
             f"- 선정 사유: {out['reason']}",
             f"- OOS: CAGR {w['oos'].get('cagr')}% · 샤프 {w['oos'].get('sharpe')} · MDD {w['oos'].get('mdd')}% · "
             f"원화 CAGR {w['oos'].get('krw_cagr')}% · 원화 MDD {w['oos'].get('krw_mdd')}% · 회전율 {w.get('turnover_yr')}x/년",
             f"- 위기 방어: {w['crisis']}", ""]
    lines.append("## 정적 벤치마크 (OOS)")
    for k, b in out["benchmarks"].items():
        o = b["oos"]
        lines.append(f"- {k}: CAGR {o.get('cagr')}% · 샤프 {o.get('sharpe')} · MDD {o.get('mdd')}% · "
                     f"원화 {o.get('krw_cagr')}%/{o.get('krw_mdd')}% · 위기 {b['crisis']}")
    if out["plateau"]:
        lines += ["", "## 파라미터 평탄성(추세창별 OOS 점수)",
                  " · ".join(f"{p['trend_n']}일={p['oos_score']}" for p in sorted(out["plateau"], key=lambda x: x["trend_n"]))]
    lines += ["", "## IS 상위 20 (참고)"] + hdr + [_row(r) for r in out["top20"]]
    lines += ["", "## OOS 상위 20 (참고 — 사후선택 주의, 채택 근거로 쓰지 말 것)"] + hdr + [_row(r) for r in out["top_oos20"]]
    with open("output/ALLOC_RESULT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _log("저장: output/best_alloc.json · alloc_backtest.json · ALLOC_RESULT.md")
    _log(f"승자: {w['base']} / 추세 {w.get('trend_n')} / 모멘텀 {w.get('mom')} / 컷 {w.get('cut')} / "
         f"도피처 {w.get('dest')} — {out['reason']}")


def main():
    ap = argparse.ArgumentParser(description="주간 자산배분 백테스트")
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터로 파이프라인 점검")
    args = ap.parse_args()
    df = synthetic_prices() if args.synthetic else fetch_prices()
    out = run_grid(df)
    export(out)


if __name__ == "__main__":
    main()
