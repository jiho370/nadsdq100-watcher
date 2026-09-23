#!/usr/bin/env python3
"""
us_insider_buying_factor.py — 내부자 매수(SEC Form 4) 신호 IC 파일럿 검증 (2026-09-22)

배경: "애널리스트 목표가" 논의에서 파생된 질문 — "퀀트펀드가 실제 쓰는데 이 프로젝트가
안 쓰는 팩터 후보"로 내부자 매수를 1순위 제안했다(Lakonishok-Lee 2001, Jeng-Metrick-
Zeckhauser 2003 등 — 특히 공개시장 매수(P코드)가 후속 초과수익력 보고됨). 정식 팩터로
등록하기 전에 데이터 확보가 되는지 + 방향성 있는 IC가 나오는지만 빠르게 찍어보는
파일럿(지호 님 "ㄱㄱ" 승인) — 이 프로젝트의 정식 채택기준(PBO/DSR 게이트)은 아니다.

데이터: data.sec.gov/submissions/CIK{cik}.json 이 회사 CIK 기준으로도 Form 4 제출이력을
포함함(실측 확인 — AAPL 최근 1000건 중 591건이 Form4). 각 제출의 raw XML
(.../{accession}/{primaryDocument 마지막 파일명})에 거래코드(P=공개시장매수·S=공개시장매도·
A=부여·F=세금원천징수·M=옵션행사 등)·주식수·단가가 구조화돼 있음(실측 확인, 2개 필자
서로 다른 회사에서 패턴 재현). fundamentals_edgar.py의 CIK맵·레이트리미터·User-Agent
패턴을 그대로 재사용(신규 EDGAR 접근 로직 재구현 안 함).

방법(파일럿 스코프 — 명시적으로 정식 검증 아님):
  1. 기존 PIT 유니버스(output/fundamentals_cache.json, 919종목)에서 스트라이드 샘플
     (표본 자체가 이번 파일럿의 한계 — 전체 스캔 아님, 방향성만 먼저 확인)
  2. 종목별 최근 Form4 최대 N건에서 공개시장 매수(P)·매도(S)만 추출(A/F/M/G 등은 제외
     — 재량적 시장타이밍 결정이 아니므로 Lakonishok-Lee 방법론과 동일 선별)
  3. 월말마다 "최근 90일 순매수 대금"(매수액-매도액, $ 기준) 계산
  4. 그 시점 이후 63거래일(약 3개월) 수익률과 횡단면 스피어만 순위상관(IC)을 월별로
     구해 평균 + "순매수>0 vs 순매도<0 vs 무활동" 3그룹 평균 순방향수익률 비교(이벤트스터디형)

한계(정직히 명시): (a) 표본 40~60종목, 전체 유니버스 아님 (b) 가격은 yfinance 현재
조회라 상장폐지 종목 생존편향 있음 (c) SEC "recent" 제출이력이 보통 몇 년 치라 장기
검증 아님 (d) 이건 1차 스크리닝이며 이 프로젝트 정식 게이트(PBO/DSR)를 통과한 적 없음.

실행: python research/us/us_insider_buying_factor.py [--n-tickers 50] [--max-filings 60]
결과: output/us_insider_buying_pilot.json
"""
from __future__ import annotations
import os, sys, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fundamentals_edgar as FE   # CIK맵·레이트리미터·User-Agent 재사용

CACHE_DIR = "output"
RAW_CACHE = f"{CACHE_DIR}/us_insider_form4_raw.json"
OUT_PATH = f"{CACHE_DIR}/us_insider_buying_pilot.json"
OPEN_MARKET_CODES = {"P": 1, "S": -1}   # 공개시장 매수/매도만(부여·세금원천징수·옵션행사 제외)
FWD_DAYS = 63       # 약 3개월 순방향수익률
LOOKBACK_DAYS = 90  # 순매수 신호 집계창


def _log(m): print(f"[내부자매수파일럿] {m}", file=sys.stderr)


def _sample_tickers(n: int) -> list[str]:
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        universe = sorted(json.load(f).keys())
    stride = max(1, len(universe) // n)
    return universe[::stride][:n]


def _fetch_form4_for_ticker(ticker: str, cik: str, max_filings: int) -> list[dict]:
    FE._LIMITER.wait()
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=FE.HDRS, timeout=20)
    if r.status_code != 200:
        return []
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    idxs = [i for i, f in enumerate(forms) if f == "4"][:max_filings]
    trades = []
    for i in idxs:
        acc = recent["accessionNumber"][i].replace("-", "")
        base = recent["primaryDocument"][i].rsplit("/", 1)[-1]
        cik_nolead = str(int(cik))
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc}/{base}"
        FE._LIMITER.wait()
        try:
            xr = requests.get(url, headers=FE.HDRS, timeout=20)
            if xr.status_code != 200 or not xr.text.strip().startswith("<?xml"):
                continue
            root = ET.fromstring(xr.text)
        except Exception:
            continue
        for tx in root.findall(".//nonDerivativeTransaction"):
            code_el = tx.find("./transactionCoding/transactionCode")
            date_el = tx.find("./transactionDate/value")
            shares_el = tx.find("./transactionAmounts/transactionShares/value")
            price_el = tx.find("./transactionAmounts/transactionPricePerShare/value")
            if code_el is None or code_el.text not in OPEN_MARKET_CODES:
                continue
            if date_el is None or shares_el is None or price_el is None:
                continue
            try:
                shares = float(shares_el.text); price = float(price_el.text)
            except (TypeError, ValueError):
                continue
            trades.append({"ticker": ticker, "date": date_el.text, "code": code_el.text,
                           "dollar": OPEN_MARKET_CODES[code_el.text] * shares * price})
    return trades


def collect_trades(tickers: list[str], max_filings: int, use_cache=True) -> list[dict]:
    if use_cache and os.path.exists(RAW_CACHE):
        _log(f"원자료 캐시 재사용: {RAW_CACHE}")
        with open(RAW_CACHE, encoding="utf-8") as f:
            return json.load(f)
    cik_map = FE.ticker_cik_map()
    all_trades = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for t in tickers:
            cik = cik_map.get(t)
            if not cik:
                _log(f"{t}: CIK 매핑 실패, 건너뜀")
                continue
            futs[ex.submit(_fetch_form4_for_ticker, t, cik, max_filings)] = t
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                trades = fut.result()
                all_trades.extend(trades)
                _log(f"{t}: 공개시장 매수/매도 {len(trades)}건")
            except Exception as e:
                _log(f"{t}: 실패({type(e).__name__}: {e})")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RAW_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_trades, f, ensure_ascii=False, indent=2)
    _log(f"원자료 저장: {RAW_CACHE} (총 {len(all_trades)}건)")
    return all_trades


def _monthly_signal(trades: pd.DataFrame, tickers: list[str], buy_only: bool = False) -> pd.DataFrame:
    """종목×월말 패널: 그 월말 기준 최근 LOOKBACK_DAYS일 순매수 대금(P-S), 또는
    buy_only=True면 매도(S) 완전 배제하고 매수(P) 대금만(문헌 원래 정의에 더 근접 —
    매도는 세금·옵션행사 처분 등 정보무관 사유가 많아 노이즈원이라는 게 통설)."""
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    if buy_only:
        trades = trades[trades["code"] == "P"]
        if trades.empty:
            return pd.DataFrame()
    trades["date"] = pd.to_datetime(trades["date"])
    start, end = trades["date"].min(), trades["date"].max()
    month_ends = pd.date_range(start + pd.Timedelta(days=LOOKBACK_DAYS), end, freq="ME")
    rows = []
    for t in tickers:
        td = trades[trades["ticker"] == t]
        if td.empty:
            continue
        for me in month_ends:
            win = td[(td["date"] > me - pd.Timedelta(days=LOOKBACK_DAYS)) & (td["date"] <= me)]
            if win.empty:
                continue
            rows.append({"ticker": t, "month_end": me, "net_dollar": win["dollar"].sum(),
                        "n_trades": len(win)})
    return pd.DataFrame(rows)


def _fwd_returns(tickers: list[str], signal_df: pd.DataFrame) -> pd.DataFrame:
    import yfinance as yf
    _log(f"가격 다운로드: {len(tickers)}종목")
    px = yf.download(tickers, period="5y", auto_adjust=True, interval="1d",
                     progress=False, group_by="ticker", threads=True)
    out = []
    for _, row in signal_df.iterrows():
        t = row["ticker"]
        try:
            s = px[t]["Close"].dropna()
        except Exception:
            continue
        if s.empty:
            continue
        after = s[s.index > row["month_end"]]
        before = s[s.index <= row["month_end"]]
        if len(after) < FWD_DAYS or before.empty:
            continue
        p0 = before.iloc[-1]
        p1 = after.iloc[FWD_DAYS - 1]
        out.append({**row, "fwd_ret": float(p1 / p0 - 1)})
    return pd.DataFrame(out)


def analyze(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {"error": "패널이 비어있음(데이터 수집 실패)"}
    ics, n_per_month = [], []
    for me, grp in panel.groupby("month_end"):
        if len(grp) < 5:
            continue
        ic = grp["net_dollar"].corr(grp["fwd_ret"], method="spearman")
        if not np.isnan(ic):
            ics.append(ic); n_per_month.append(len(grp))
    mean_ic = float(np.mean(ics)) if ics else None
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else None
    t_stat = (mean_ic / (ic_std / np.sqrt(len(ics)))) if ic_std else None

    buy = panel[panel["net_dollar"] > 0]["fwd_ret"]
    sell = panel[panel["net_dollar"] < 0]["fwd_ret"]
    return {
        "n_observations": len(panel), "n_months": len(ics),
        "mean_monthly_ic": round(mean_ic, 4) if mean_ic is not None else None,
        "ic_std": round(ic_std, 4) if ic_std is not None else None,
        "ic_tstat": round(t_stat, 2) if t_stat is not None else None,
        "group_avg_fwd_return": {
            "net_buy_group": {"n": int(len(buy)), "mean_fwd_ret_pct": round(float(buy.mean()) * 100, 2) if len(buy) else None},
            "net_sell_group": {"n": int(len(sell)), "mean_fwd_ret_pct": round(float(sell.mean()) * 100, 2) if len(sell) else None},
        },
        "note": "파일럿 스코프 — PBO/DSR 등 이 프로젝트 정식 게이트는 미실시. 표본·생존편향 한계는 docstring 참고.",
    }


def main():
    ap = argparse.ArgumentParser(description="내부자 매수(Form4) IC 파일럿")
    ap.add_argument("--n-tickers", type=int, default=50)
    ap.add_argument("--max-filings", type=int, default=60)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--buy-only", action="store_true",
                    help="매도 배제, 매수(P)만으로 신호 구성 — 문헌 원래 정의에 더 근접")
    args = ap.parse_args()

    tickers = _sample_tickers(args.n_tickers)
    _log(f"표본 {len(tickers)}종목: {tickers[:10]}...")

    trades = collect_trades(tickers, args.max_filings, use_cache=not args.no_cache)
    trades_df = pd.DataFrame(trades)
    _log(f"공개시장 매수/매도 총 {len(trades_df)}건 "
        f"(매수 {len(trades_df[trades_df['dollar']>0]) if not trades_df.empty else 0}건 · "
        f"매도 {len(trades_df[trades_df['dollar']<0]) if not trades_df.empty else 0}건)")

    signal_df = _monthly_signal(trades_df, tickers, buy_only=args.buy_only)
    _log(f"월별 신호 관측치: {len(signal_df)}" + (" (매수전용)" if args.buy_only else " (순매수)"))

    panel = _fwd_returns(tickers, signal_df)
    _log(f"순방향수익률 결합 후: {len(panel)}")

    result = analyze(panel)
    result["signal_mode"] = "buy_only" if args.buy_only else "net_buy_minus_sell"
    out_path = OUT_PATH.replace(".json", "_buyonly.json") if args.buy_only else OUT_PATH
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    _log(f"저장: {out_path}")
    _log(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
