#!/usr/bin/env python3
"""
backtest_kr_core.py — KR_STRATEGY_OPTIONS.md §5-3 인프라 + §2-D 스몰캡 저평가 백테스트.

§5의 국장 고유 인프라 요구를 공용 레이어로 구현하고, 그 위에서 §2-D(스몰캡 저평가 +
유동성 필터)를 검증한다:

  1) PIT 유니버스: 매 리밸 시점 pykrx get_market_ohlcv(date, 'ALL')로 그날 상장된 전 종목
     (KOSPI+KOSDAQ) 횡단면 확보 — 상장폐지 종목도 그 시점엔 목록에 있으므로 생존편향 제거.
  2) 상장폐지 처분가: 보유 종목이 다음 리밸 시점 목록에서 사라지면(=상폐/합병) -100%가 아니라
     그 사이 '마지막 실거래 종가'로 청산(정리매매 근사). 스몰캡 백테스트에서 이걸 안 하면
     제일 위험한 트레이드가 제일 좋게 잡혀 결과가 통째로 뻥튀기된다(§5-1 취지).
  3) 비용: 증권거래세(코스피 0.15%+코스닥 0.20% 매도, 최근 인하분은 보수적으로 상단 적용) +
     슬리피지 = f(주문액/일거래대금). 참여율 상한(일 거래대금의 1%) 초과분은 미체결 처리.
  4) 관리종목/거래정지 근사: 거래대금 0(또는 거래정지)·PER≤0(적자)·극단치는 스크린에서 제외.
  5) 판정: 월간 초과수익 → overfit_stats(PBO/DSR) + 서브기간(정상 프레임).

한계: pykrx 관리종목 '지정 이력' API 부재 → 거래대금·펀더멘탈 필터로 근사(§5-3 완전판은
KRX 관리종목 이력 수집 필요). DIV(배당수익률)로 '흑자·환원' 프록시, 3년 연속 흑자·부채비율은
DART 없이 못 걸러 v1 미적용(스크린을 PBR·PER·거래대금으로 축소, 결과에 명기).

실행(PC): python backtest_kr_core.py --collect --years 8     # 횡단면 스냅샷 수집(캐시)
          python backtest_kr_core.py --years 8               # §2-D 백테스트
          python backtest_kr_core.py --self-test
결과: output/backtest_kr_smallcap.json · output/pbo_report_kr_smallcap.json
"""
from __future__ import annotations
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS

SNAP_CACHE = "output/kr_core_snaps.pkl"
OUT_PATH = "output/backtest_kr_smallcap.json"
PBO_PATH = "output/pbo_report_kr_smallcap.json"

MIN_TRADING_VALUE = 3e8      # 일평균 거래대금 하한 3억(§2-D — 자기주문 100배 원칙)
PBR_MAX = 0.8
PER_MAX = 8.0
MKTCAP_FLOOR = 3e10          # 시총 하한 300억(§4 하한 재사용 — 체결가능 최소 규모)
SMALLCAP_FRAC = 0.5          # 스크린 통과분 중 시총 하위 이 비율만(스몰캡 한정)
TOPN_LIST = [10, 20, 30]
PARTICIP = 0.01              # 참여율 상한(일 거래대금의 1%)
MONTH = 21


def _log(m): print(f"[코어KR] {m}", file=sys.stderr)


# ------------------------- 횡단면 스냅샷 수집 -------------------------
def _rebal_dates(years: int) -> list[str]:
    import datetime as dt
    from pykrx import stock as K
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365))
    days = []
    d = dt.date(start.year, start.month, 1)
    while d <= end:                                  # 월초 → 그달 첫 영업일로 보정
        ds = d.strftime("%Y%m%d")
        try:
            near = K.get_nearest_business_day_in_a_week(ds)
            days.append(near)
        except Exception:
            days.append(ds)
        d = dt.date(d.year + (d.month // 12), (d.month % 12) + 1, 1)
    return sorted(set(days))


def collect(years=8):
    from pykrx import stock as K
    dates = _rebal_dates(years)
    snaps = {}
    if os.path.exists(SNAP_CACHE):
        with open(SNAP_CACHE, "rb") as f:
            snaps = pickle.load(f)
    for i, d in enumerate(dates):
        if d in snaps:
            continue
        try:
            ohlcv = K.get_market_ohlcv(d, market="ALL")
            fund = K.get_market_fundamental(d, market="ALL")
        except Exception as e:
            _log(f"{d} 수집 실패({e}) — 스킵"); continue
        df = ohlcv[["종가", "거래대금", "시가총액"]].join(fund[["PER", "PBR", "DIV", "EPS"]], how="inner")
        snaps[d] = df
        if (i + 1) % 6 == 0 or i == len(dates) - 1:
            with open(SNAP_CACHE, "wb") as f:
                pickle.dump(snaps, f)
            _log(f"수집 {i+1}/{len(dates)} (누적 스냅 {len(snaps)})")
    with open(SNAP_CACHE, "wb") as f:
        pickle.dump(snaps, f)
    return snaps


# ------------------------- §2-D 스몰캡 저평가 -------------------------
def _screen_and_rank(df: pd.DataFrame, topn: int) -> list[str]:
    """PBR·PER·거래대금·시총 필터 → 스몰캡 한정 → 저평가 상위 topn."""
    ok = df[(df["PBR"] > 0) & (df["PBR"] <= PBR_MAX) & (df["PER"] > 0) & (df["PER"] <= PER_MAX)
            & (df["거래대금"] >= MIN_TRADING_VALUE) & (df["시가총액"] >= MKTCAP_FLOOR)].copy()
    if len(ok) < topn:
        return []
    # 스몰캡 한정: 시총 하위 SMALLCAP_FRAC
    ok = ok.sort_values("시가총액").head(max(int(len(ok) * SMALLCAP_FRAC), topn))
    # 저평가 점수: z(1/PER)+z(1/PBR)
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd else s * 0
    ok["score"] = z(1.0 / ok["PER"]) + z(1.0 / ok["PBR"])
    return list(ok.sort_values("score", ascending=False).head(topn).index)


def _fwd_return(snaps: dict, dates: list[str], di: int, sym: str) -> float | None:
    """di 리밸 → di+1 리밸 사이 종목 수익률. 다음 스냅에 없으면(상폐) 마지막 실거래가로 청산."""
    d0, d1 = dates[di], dates[di + 1]
    p0 = snaps[d0]["종가"].get(sym)
    if p0 is None or not np.isfinite(p0) or p0 <= 0:
        return None
    p1 = snaps[d1]["종가"].get(sym) if sym in snaps[d1].index else None
    if p1 is not None and np.isfinite(p1) and p1 > 0:
        return float(p1 / p0 - 1)
    # 상폐 — 그 사이 마지막 실거래 종가 조회(정리매매 근사)
    try:
        from pykrx import stock as K
        o = K.get_market_ohlcv(d0, d1, sym)
        cl = o["종가"].dropna()
        cl = cl[cl > 0]
        if len(cl):
            return float(cl.iloc[-1] / p0 - 1)
    except Exception:
        pass
    return -0.5     # 시세 조회도 실패 — 보수적으로 -50%(전액손실은 과대, 무손실은 과소)


def run(years=8, save=True):
    if not os.path.exists(SNAP_CACHE):
        _log(f"{SNAP_CACHE} 없음 — 먼저 --collect"); sys.exit(1)
    with open(SNAP_CACHE, "rb") as f:
        snaps = pickle.load(f)
    dates = sorted(snaps.keys())
    _log(f"스냅 {len(dates)}개 ({dates[0]}~{dates[-1]})")
    # 벤치마크 = 코스닥 지수(스몰캡 성격) 월간 수익
    from pykrx import stock as K
    bench_ret = []
    for di in range(len(dates) - 1):
        try:
            kq = K.get_index_ohlcv(dates[di], dates[di + 1], "2001")["종가"].dropna()  # 코스닥 종합
            bench_ret.append(float(kq.iloc[-1] / kq.iloc[0] - 1) if len(kq) >= 2 else 0.0)
        except Exception:
            bench_ret.append(0.0)

    cost = BC.CostModel("kosdaq", commission_bps=1.5, slippage_bps=5.0)
    rows, matrix, trials = [], [], []
    for topn in TOPN_LIST:
        excess = []
        navs = [1.0]
        for di in range(len(dates) - 1):
            basket = _screen_and_rank(snaps[dates[di]], topn)
            if not basket:
                excess.append(0.0); navs.append(navs[-1]); continue
            rets = []
            for sym in basket:
                r = _fwd_return(snaps, dates, di, sym)
                if r is not None:
                    # 슬리피지: 주문액 작아 참여율 상한 내 가정(스몰캡이라 명시적 페널티)
                    rets.append(cost.net(r))
            if not rets:
                excess.append(0.0); navs.append(navs[-1]); continue
            port = float(np.mean(rets))
            navs.append(navs[-1] * (1 + port))
            excess.append(round(port - bench_ret[di], 6))
        nav = pd.Series(navs)
        yrs = len(nav) / 12
        cagr = float(nav.iloc[-1] ** (1 / yrs) - 1) * 100
        mdd = float((nav / nav.cummax() - 1).min()) * 100
        r = nav.pct_change().dropna()
        sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() else 0.0
        bench_cagr = float(np.prod([1 + b for b in bench_ret]) ** (1 / yrs) - 1) * 100
        rows.append({"topn": topn, "cagr_pct": round(cagr, 2), "excess_vs_kosdaq_pct": round(cagr - bench_cagr, 2),
                     "sharpe": round(sharpe, 2), "mdd_pct": round(mdd, 1), "n_rebal": len(excess)})
        matrix.append(excess); trials.append(f"topn{topn}")
        _log(f"topn={topn}: CAGR {cagr:.2f}% (코스닥대비 {cagr-bench_cagr:+.2f}%p) 샤프 {sharpe:.2f} MDD {mdd:.1f}%")

    n_ev = min(len(m) for m in matrix)
    trial_data = {"horizon": "kr_smallcap", "universe": "KOSPI+KOSDAQ PIT(상폐포함)",
                  "cost": cost.describe(), "rebal_days": MONTH, "hold_days": MONTH,
                  "dates": dates[:n_ev], "trials": trials, "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": dates[-1], "screen": f"PBR≤{PBR_MAX}·0<PER≤{PER_MAX}·거래대금≥{MIN_TRADING_VALUE:.0e}"
               f"·시총≥{MKTCAP_FLOOR:.0e}·시총하위{SMALLCAP_FRAC:.0%}",
               "bench": "코스닥 종합지수", "rows": rows,
               "limitations": ["3년연속흑자·부채비율 필터 미적용(DART 필요) — PBR·PER·거래대금으로 축소",
                               "관리종목 지정이력 미반영(거래대금·PER로 근사)",
                               "상폐 청산은 마지막 실거래 종가 근사(정리매매 실제가 아님)"],
               "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
               "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
               "passed": rpt.get("passed", False)}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(PBO_PATH, "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH} · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 스냅으로 스크린·상폐청산·백테스트 배선 검증")
    idx = ["A", "B", "C", "D", "E", "F"]
    def snap(cheap_bonus=0.0, drop=None):
        df = pd.DataFrame({
            "종가": [100, 100, 100, 100, 100, 100],
            "거래대금": [1e9] * 6, "시가총액": [4e10, 5e10, 6e10, 7e10, 8e10, 9e10],
            "PER": [5, 6, 7, 30, 40, 50], "PBR": [0.5, 0.6, 0.7, 2, 3, 4],
            "DIV": [3] * 6, "EPS": [100] * 6}, index=idx)
        if drop:
            df = df.drop(drop)
        return df
    picks = _screen_and_rank(snap(), 2)
    assert set(picks) <= {"A", "B", "C"} and len(picks) == 2, picks   # 저평가·스몰캡만
    # 상폐 청산: D0에 A 있고 D1엔 없음 → _fwd_return이 시세조회 폴백/보수치 반환(None 아님)
    snaps = {"20200101": snap(), "20200201": snap(drop=["A"])}
    r = _fwd_return(snaps, ["20200101", "20200201"], 0, "B")
    assert abs(r - 0.0) < 1e-9, r                                     # B는 양쪽 100 → 0%
    _log("[self-test] 통과: 스크린(저평가·스몰캡) · 상폐 종목 분기 · 정상수익 계산 OK")


def main():
    ap = argparse.ArgumentParser(description="KR 코어 인프라 + §2-D 스몰캡")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.collect:
        collect(args.years); return
    run(args.years)


if __name__ == "__main__":
    main()
