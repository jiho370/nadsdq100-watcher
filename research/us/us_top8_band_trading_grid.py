#!/usr/bin/env python3
"""
us_top8_band_trading_grid.py — 현재 라이브 8종목(state/ai_holdings.json)에 가격 밴드
매매(상승 X%마다 매수/매도, 하락 Y%마다 매수/매도)를 적용했을 때 성과가 어땠는지
대규모 그리드 서치로 백테스트한다. (2026-09-08, 지호 님 요청 — "여러 조합 동시에 돌려서
시간을 줄여달라"는 지시대로 ProcessPoolExecutor로 그리드 조합을 병렬 실행한다.)

질문 확인 결과(AskUserQuestion):
  - 방향 조합: 4가지 전부 스윕 — meanrev(상승매도·하락매수, 역추세 밴드매매) /
    trend(상승매수·하락매도, 추세추종·피라미딩) / accumulate(상승·하락 모두 매수) /
    derisk(상승·하락 모두 매도)
  - 매매 방식: 전량 스위치(full)와 분할매매(tranche, 트랜치 비율 그리드)를 둘 다 스윕
  - 범위: 8종목을 하나의 통합 포트폴리오(현금 공유)로 운용

설계 가정(단순화를 위해 명시적으로 고정한 것들 — 결과 JSON에도 그대로 기록):
  1. 시작 시점에 8종목을 각각 CAPITAL_PER_STOCK(=$10,000)씩 동일비중 매수, 이후 현금은
     8종목이 공유하는 단일 풀(포트폴리오 통합) — 한 종목을 팔아 번 현금으로 다른 종목을
     추가매수할 수 있다.
  2. 밴드 기준가(ref price)는 종목별로 따로 관리하며, 그날 종가가 기준가 대비
     +rise_pct% 이상이면 rise_action, -fall_pct% 이하이면 fall_action을 실행하고,
     실제 체결 여부와 무관하게 기준가를 그날 종가로 재설정한다(밴드 재중심화 —
     그리드매매 표준 동작. 안 그러면 한 번 밴드를 벗어난 뒤 계속 같은 신호가 남는다).
  3. 분할매매(tranche) 1회 매매 수량은 "최초 매수 수량 × tranche_pct%"로 고정한다
     (현재 보유수량의 %로 하면 반복될수록 크기가 기하급수로 줄어/늘어나 트랜치 크기
     비교가 무의미해짐). 전량 스위치(full)는 보유 0↔최초수량 전체를 오간다.
  4. 매수는 공유 현금 잔고 한도 내에서만 체결(레버리지 없음) — accumulate처럼 매수만
     반복되는 조합은 현금이 바닥나면 자연히 멈춘다. accumulate+full, derisk+full은
     구조상 최초 1회 이후 무력화되거나(accumulate: 이미 최초수량 보유 중이라 살 것이
     없음 = 매수&보유와 동일) 완전 청산 후 유지(derisk: 최초 트리거에서 전량매도 후
     현금 보유)되는 실질적으로 단순한 결과를 낸다 — 이는 설계 한계가 아니라 "전량
     스위치 방식이 같은 방향 반복 조합과는 안 맞는다"는 실제 특성이라 그대로 결과에
     남겨 비교한다(tranche 방식이 이 두 방향 조합의 진짜 비교 기준이 된다).
  5. 같은 날 두 번째 이상의 밴드 이탈(예: 하루에 +30% 갭)은 반영하지 않는다 — 그날은
     한 번만 체크해 기준가를 그날 종가로 재설정하고 다음 날부터 새 기준으로 다시 잰다.
  6. 비용은 기존 라이브/타 백테스트와 동일하게 BC.CostModel("us", 0bp수수료, 5bp
     슬리피지) + 미국 매도 시 SEC수수료 — 회전율이 큰 조합(트랜치 작을수록·밴드
     좁을수록)일수록 비용 영향이 커진다.
  7. 임계값(rise_pct·fall_pct)은 8종목에 공통 적용(종목별 최적화 아님) — "이 전략을
     8종목에 동일하게 적용했을 때"라는 질문 그대로.

실행: python us_top8_band_trading_grid.py [--years 10] [--workers N]
결과: output/us_top8_band_trading_grid.json
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

import backtest_costs as BC
import sp500_daily_report as R

CAPITAL_PER_STOCK = 10_000.0
RISE_GRID = [3, 5, 7, 10, 15, 20, 25]
FALL_GRID = [3, 5, 7, 10, 15, 20, 25]
TRANCHE_GRID = [10, 20, 25, 33, 50]
DIRECTIONS = {
    "meanrev":    ("sell", "buy"),   # 상승→익절매도, 하락→물타기매수 (역추세 밴드매매)
    "trend":      ("buy", "sell"),   # 상승→추가매수(불타기), 하락→매도(손절/추세이탈)
    "accumulate": ("buy", "buy"),    # 상승·하락 모두 매수 (피라미딩, 현금 소진 시 정지)
    "derisk":     ("sell", "sell"),  # 상승·하락 모두 매도 (위험회피, 현금화 후 유지)
}
DIRECTION_DESC_KO = {
    "meanrev": "상승 시 매도(익절) / 하락 시 매수(물타기) — 역추세 밴드매매",
    "trend": "상승 시 매수(불타기) / 하락 시 매도(손절) — 추세추종·피라미딩",
    "accumulate": "상승·하락 모두 매수 — 계속 물타기·불타기(현금 소진 시 정지)",
    "derisk": "상승·하락 모두 매도 — 둘 다 위험회피(청산 후 현금 보유)",
}


def _log(m): print(f"[밴드매매그리드] {m}", file=sys.stderr)


def load_holdings() -> list[str]:
    with open("state/ai_holdings.json", encoding="utf-8") as f:
        d = json.load(f)
    return sorted(d["holdings"].keys())


def load_prices(tickers: list[str], years: float):
    hist = R.download_histories(tickers, period=f"{int(years)}y")
    missing = [t for t in tickers if hist.get(t) is None or len(hist[t]) == 0]
    if missing:
        raise RuntimeError(f"시세 다운로드 실패 종목: {missing}")
    panel = pd.DataFrame({t: hist[t] for t in tickers}).sort_index().dropna(how="any")
    spy = R.download_histories(["SPY"], period=f"{int(years)}y").get("SPY")
    if spy is None:
        raise RuntimeError("SPY 시세 다운로드 실패")
    idx = panel.index.intersection(spy.dropna().index)
    panel = panel.loc[idx]
    spy = spy.reindex(idx).ffill()
    return panel, spy


def simulate(prices: np.ndarray, rise_pct: float, fall_pct: float, rise_action: str,
             fall_action: str, sizing: str, tranche_pct: float | None,
             cost_buy: float, cost_sell: float, capital_per_stock: float):
    """prices: (n_days, n_stocks), NaN 없음. 반환: (nav ndarray, 매수건수, 매도건수)."""
    n, k = prices.shape
    init_price = prices[0].copy()
    init_shares = capital_per_stock / init_price
    shares = init_shares.copy()
    ref = init_price.copy()
    cash = 0.0
    nav = np.empty(n)
    nav[0] = capital_per_stock * k
    buys = sells = 0
    rise_th = rise_pct / 100.0
    fall_th = fall_pct / 100.0
    lot = (tranche_pct / 100.0) * init_shares if sizing == "tranche" else None

    for i in range(1, n):
        p = prices[i]
        chg = p / ref - 1.0
        for j in range(k):
            if chg[j] >= rise_th:
                action = rise_action
            elif chg[j] <= -fall_th:
                action = fall_action
            else:
                continue
            if action == "sell" and shares[j] > 1e-9:
                qty = shares[j] if sizing == "full" else min(shares[j], lot[j])
                if qty > 0:
                    cash += qty * p[j] * (1.0 - cost_sell)
                    shares[j] -= qty
                    sells += 1
            elif action == "buy" and cash > 1e-9:
                need = max(init_shares[j] - shares[j], 0.0) if sizing == "full" else lot[j]
                if need > 0:
                    cost_amt = need * p[j] * (1.0 + cost_buy)
                    spend = min(cash, cost_amt)
                    qty = spend / (p[j] * (1.0 + cost_buy))
                    if qty > 0:
                        shares[j] += qty
                        cash -= spend
                        buys += 1
            ref[j] = p[j]
        nav[i] = cash + float(np.dot(shares, p))
    return nav, buys, sells


def buy_and_hold_nav(prices: np.ndarray, capital_per_stock: float) -> np.ndarray:
    """매매 없는 순수 매수&보유 베이스라인. 밴드 임계값을 극단값(예:999%)으로 잡아
    simulate()를 재사용하는 방식은 위험하다 — MRNA가 코로나 백신 랠리로 IPO가 대비
    30배 넘게 뛴 적이 있어 999% 임계값도 실제로 뚫려 매도 1건이 섞여 들어갔었다
    (버그로 발견, 2026-09-08). 그래서 밴드 로직을 아예 거치지 않는 별도 함수로 분리."""
    init_shares = capital_per_stock / prices[0]
    return prices @ init_shares


def _metrics(nav: np.ndarray, bench: np.ndarray) -> dict:
    final_value = float(nav[-1])
    navn = nav / nav[0]
    bench = bench / bench[0]
    yrs = len(navn) / 252
    ret = np.diff(navn) / navn[:-1]
    cagr = float(navn[-1] ** (1 / yrs) - 1) * 100
    bench_cagr = float(bench[-1] ** (1 / yrs) - 1) * 100
    vol = float(np.std(ret) * np.sqrt(252)) * 100
    sharpe = float(np.mean(ret) / np.std(ret) * np.sqrt(252)) if np.std(ret) > 0 else 0.0
    cummax = np.maximum.accumulate(navn)
    mdd = float(np.min(navn / cummax - 1)) * 100
    return {"cagr_pct": round(cagr, 2), "excess_cagr_pct": round(cagr - bench_cagr, 2),
            "vol_pct": round(vol, 1), "sharpe": round(sharpe, 2), "mdd_pct": round(mdd, 1),
            "final_value_usd": round(final_value, 0), "years": round(yrs, 2)}


_PRICES = None
_BENCH = None


def _init_worker(prices, bench):
    global _PRICES, _BENCH
    _PRICES = prices
    _BENCH = bench


def _run_one(task):
    direction, rise_pct, fall_pct, sizing, tranche_pct, cost_buy, cost_sell, capital = task
    rise_action, fall_action = DIRECTIONS[direction]
    nav, buys, sells = simulate(_PRICES, rise_pct, fall_pct, rise_action, fall_action,
                                 sizing, tranche_pct, cost_buy, cost_sell, capital)
    m = _metrics(nav, _BENCH)
    yrs = m["years"]
    return {"direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
            "sizing": sizing, "tranche_pct": tranche_pct, **m,
            "buys_per_year": round(buys / yrs, 1), "sells_per_year": round(sells / yrs, 1),
            "trades_per_year": round((buys + sells) / yrs, 1)}


_PRICES_BY_TICKER = None
_BENCH2 = None


def _init_worker_single(prices_by_ticker, bench):
    global _PRICES_BY_TICKER, _BENCH2
    _PRICES_BY_TICKER = prices_by_ticker
    _BENCH2 = bench


def _run_one_single(task):
    """종목 1개짜리 독립 백테스트 — simulate()를 그대로 재사용하되 가격행렬 폭이 1."""
    ticker, direction, rise_pct, fall_pct, sizing, tranche_pct, cost_buy, cost_sell, capital = task
    prices_col = _PRICES_BY_TICKER[ticker].reshape(-1, 1)
    rise_action, fall_action = DIRECTIONS[direction]
    nav, buys, sells = simulate(prices_col, rise_pct, fall_pct, rise_action, fall_action,
                                 sizing, tranche_pct, cost_buy, cost_sell, capital)
    m = _metrics(nav, _BENCH2)
    yrs = m["years"]
    return {"ticker": ticker, "direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
            "sizing": sizing, "tranche_pct": tranche_pct, **m,
            "buys_per_year": round(buys / yrs, 1), "sells_per_year": round(sells / yrs, 1),
            "trades_per_year": round((buys + sells) / yrs, 1)}


def build_grid() -> list[tuple]:
    sizings = [("full", None)] + [("tranche", t) for t in TRANCHE_GRID]
    combos = []
    for direction in DIRECTIONS:
        for rise_pct, fall_pct in itertools.product(RISE_GRID, FALL_GRID):
            for sizing, tranche_pct in sizings:
                combos.append((direction, rise_pct, fall_pct, sizing, tranche_pct))
    return combos


def run(years: float = 10, workers: int | None = None, save: bool = True) -> dict:
    tickers = load_holdings()
    _log(f"대상 8종목(state/ai_holdings.json): {tickers}")
    panel, spy = load_prices(tickers, years)
    _log(f"데이터 {len(panel)}거래일 확보 ({panel.index[0].date()} ~ {panel.index[-1].date()})"
         f" — 상장이 짧은 종목(예: MRNA 2018-12 IPO) 때문에 요청한 {years}년보다 짧을 수 있음")

    prices = panel[tickers].values.astype(float)
    bench = spy.values.astype(float)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)

    combos = build_grid()
    workers = workers or os.cpu_count() or 4
    _log(f"그리드 조합 {len(combos)}개를 워커 {workers}개로 병렬 실행 "
         f"(방향{len(DIRECTIONS)}×상승{len(RISE_GRID)}×하락{len(FALL_GRID)}"
         f"×사이징{1+len(TRANCHE_GRID)})")

    tasks = [(d, r, f, s, t, cost.buy, cost.sell, CAPITAL_PER_STOCK) for d, r, f, s, t in combos]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(prices, bench)) as ex:
        for res in ex.map(_run_one, tasks, chunksize=8):
            results.append(res)
    _log(f"완료: {len(results)}개 조합, {time.time() - t0:.1f}초")

    # 베이스라인: 8종목 동일비중 매수 후 매매 없음
    bh_nav = buy_and_hold_nav(prices, CAPITAL_PER_STOCK)
    bh_m = _metrics(bh_nav, bench)
    total_capital = CAPITAL_PER_STOCK * len(tickers)
    spy_nav = bench / bench[0] * total_capital   # final_value_usd 비교 가능하게 동일 원금 스케일
    spy_m = _metrics(spy_nav, bench)

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    best_per_direction = {d: max((r for r in results if r["direction"] == d),
                                 key=lambda r: r["sharpe"]) for d in DIRECTIONS}

    payload = {
        "as_of": panel.index[-1].date().isoformat(),
        "start_date": panel.index[0].date().isoformat(),
        "years": round(len(panel) / 252, 2),
        "tickers": tickers,
        "n_combos": len(results),
        "capital_per_stock_usd": CAPITAL_PER_STOCK,
        "cost_assumption": cost.describe(),
        "direction_types": DIRECTION_DESC_KO,
        "baseline_buy_and_hold_8stocks": bh_m,
        "benchmark_spy_buy_and_hold": spy_m,
        "top20_by_sharpe": results[:20],
        "best_per_direction": best_per_direction,
        "all_results": results,
        "note": ("전량스위치(full)는 accumulate/derisk처럼 상승·하락에 같은 매매방향을 쓰는 "
                "조합에서 최초 1회 이후 사실상 무력화되거나(accumulate: 이미 최초수량 보유 "
                "중이라 buy가 no-op = 매수&보유와 동일) 전량청산 후 유지(derisk)되므로, 이 "
                "두 방향의 진짜 비교 기준은 tranche(분할매매) 결과를 봐야 한다. 단일 그리드 "
                "탐색 결과이며 PBO/DSR 등 다중검정 게이트를 통과한 것은 아니다 — 상위 조합이 "
                "과최적화(다중비교로 우연히 좋아 보임)일 위험을 감안해서 해석할 것."),
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_top8_band_trading_grid.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_per_stock(years: float = 10, workers: int | None = None, save: bool = True) -> dict:
    """8종목을 독립적으로(각자 CAPITAL_PER_STOCK, 현금도 종목별 분리) 같은 그리드로 백테스트.
    run()의 통합 포트폴리오(현금 공유)와 달리, 종목별로 어떤 %가 잘 맞는지 개별 비교용."""
    tickers = load_holdings()
    _log(f"대상 8종목(개별): {tickers}")
    panel, spy = load_prices(tickers, years)
    _log(f"데이터 {len(panel)}거래일 확보 ({panel.index[0].date()} ~ {panel.index[-1].date()})")

    prices_by_ticker = {t: panel[t].values.astype(float) for t in tickers}
    bench = spy.values.astype(float)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)

    combos = build_grid()
    workers = workers or os.cpu_count() or 4
    tasks = [(t, d, r, f, s, tr, cost.buy, cost.sell, CAPITAL_PER_STOCK)
             for t in tickers for d, r, f, s, tr in combos]
    _log(f"종목 8개 × 조합 {len(combos)}개 = {len(tasks)}개 작업을 워커 {workers}개로 병렬 실행")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_single,
                             initargs=(prices_by_ticker, bench)) as ex:
        for res in ex.map(_run_one_single, tasks, chunksize=16):
            results.append(res)
    _log(f"완료: {len(results)}개 작업, {time.time() - t0:.1f}초")

    baseline_by_ticker, best_by_ticker, top5_by_ticker = {}, {}, {}
    for t in tickers:
        bh_nav = buy_and_hold_nav(prices_by_ticker[t].reshape(-1, 1), CAPITAL_PER_STOCK)
        baseline_by_ticker[t] = _metrics(bh_nav, bench)
        cands = sorted((r for r in results if r["ticker"] == t),
                       key=lambda r: r["sharpe"], reverse=True)
        best_by_ticker[t] = cands[0]
        top5_by_ticker[t] = cands[:5]

    payload = {
        "as_of": panel.index[-1].date().isoformat(),
        "start_date": panel.index[0].date().isoformat(),
        "years": round(len(panel) / 252, 2),
        "tickers": tickers,
        "n_combos_per_ticker": len(combos),
        "capital_per_stock_usd": CAPITAL_PER_STOCK,
        "cost_assumption": cost.describe(),
        "direction_types": DIRECTION_DESC_KO,
        "baseline_buy_and_hold_by_ticker": baseline_by_ticker,
        "best_by_ticker": best_by_ticker,
        "top5_by_ticker": top5_by_ticker,
        "note": ("종목별 독립 백테스트(현금도 종목별로 분리 — run()의 통합 포트폴리오와 달리 "
                "한 종목을 팔아 다른 종목을 사는 효과 없음). 그 외 가정은 통합 포트폴리오와 "
                "동일(us_top8_band_trading_grid.json 참고). 단일 그리드 탐색이라 종목별 "
                "'최고' 조합은 그 종목 표본에 대한 과최적화일 위험이 있음 — 종목마다 8년치 "
                "표본 하나로 1,176개 조합을 비교했으니 최고값은 우연히 잘 맞은 것일 수 있다."),
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_top8_band_trading_grid_per_stock.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mode", choices=["portfolio", "per_stock", "both"], default="portfolio")
    args = ap.parse_args()

    if args.mode in ("per_stock", "both"):
        ps = run_per_stock(years=args.years, workers=args.workers)
        print(f"\n=== 종목별 개별 밴드매매 그리드 백테스트 ({ps['start_date']} ~ {ps['as_of']}, "
              f"{ps['years']}년, 종목당 {ps['n_combos_per_ticker']}개 조합) ===")
        for t in ps["tickers"]:
            bh = ps["baseline_buy_and_hold_by_ticker"][t]
            best = ps["best_by_ticker"][t]
            tranche_str = "" if best["tranche_pct"] is None else f"({best['tranche_pct']}%)"
            print(f"{t:>5s} 매수&보유 CAGR {bh['cagr_pct']:>7.2f}% 샤프 {bh['sharpe']:>5.2f} "
                  f"MDD {bh['mdd_pct']:>6.1f}%  |  최고: {best['direction']} "
                  f"상승{best['rise_pct']}%/하락{best['fall_pct']}% {best['sizing']}{tranche_str} "
                  f"→ CAGR {best['cagr_pct']:>7.2f}% 샤프 {best['sharpe']:>5.2f} "
                  f"MDD {best['mdd_pct']:>6.1f}%")
    if args.mode in ("portfolio", "both"):
        payload = run(years=args.years, workers=args.workers)
    print(f"\n=== 8종목({', '.join(payload['tickers'])}) 밴드매매 그리드 백테스트 ===")
    print(f"기간: {payload['start_date']} ~ {payload['as_of']} ({payload['years']}년), "
          f"조합수: {payload['n_combos']}")
    bh, spy = payload["baseline_buy_and_hold_8stocks"], payload["benchmark_spy_buy_and_hold"]
    print(f"베이스라인(매매없음, 8종목 매수&보유): CAGR {bh['cagr_pct']}% 샤프 {bh['sharpe']} "
          f"MDD {bh['mdd_pct']}%")
    print(f"SPY 매수&보유: CAGR {spy['cagr_pct']}% 샤프 {spy['sharpe']} MDD {spy['mdd_pct']}%")
    print("\n--- 샤프 기준 상위 10개 조합 ---")
    for r in payload["top20_by_sharpe"][:10]:
        tranche_str = "" if r["tranche_pct"] is None else f"({r['tranche_pct']}%)"
        print(f"{r['direction']:>10s} 상승{r['rise_pct']:>2}%/하락{r['fall_pct']:>2}% "
              f"{r['sizing']:>7s}{tranche_str} "
              f"→ CAGR {r['cagr_pct']:>6.2f}% 샤프 {r['sharpe']:>5.2f} MDD {r['mdd_pct']:>6.1f}% "
              f"연매매 {r['trades_per_year']:>5.1f}건")
    print("\n--- 방향별 최고(샤프 기준) ---")
    for d, r in payload["best_per_direction"].items():
        print(f"{d:>10s}({DIRECTION_DESC_KO[d]}): 상승{r['rise_pct']}%/하락{r['fall_pct']}% "
              f"{r['sizing']} → CAGR {r['cagr_pct']}% 샤프 {r['sharpe']} MDD {r['mdd_pct']}%")
