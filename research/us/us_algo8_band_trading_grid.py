#!/usr/bin/env python3
"""
us_algo8_band_trading_grid.py — 현재 라이브와 동일한 방식(팩터랭킹 상위 8종목, 6개월(180일)
정기 재평가로 종목 교체)으로 과거 전체 기간에 걸쳐 실제로 뽑혔을 회전하는 8종목 바스켓에,
가격 밴드매매(상승 X%마다 매수/매도, 하락 Y%마다 매수/매도)를 얹었을 때 성과가 어땠는지
대규모 그리드 서치로 백테스트한다. (2026-09-08, 지호 님 요청 정정 — "지금 라이브 8종목
고정이 아니라, 역사적으로 이 방식으로 뽑혔던 8종목 회전 + 6개월 리밸런싱에 밴드매매를
추가해서 백테스트해달라")

베이스(밴드매매 없는 대조군) = us_daily_top8_vs_baseline.py의 "A(기존, 현행 라이브)"와 동일
설정: BP.us_decisions(step=21거래일, 풀60) → BP.simulate(topn=8, reeval_days=180,
ma200_backup=False, sector_cap=2, 팩터가중치 1:2:2·output/best_weights.json 그대로). 이
대조군은 BP.simulate()를 그대로 호출해서 만든다(기존에 검증된 라이브 재현 함수를 그대로
재사용 — 별도로 재구현하면 미묘하게 달라질 위험이 있어서 피함).

밴드매매를 얹은 버전(simulate_band)은 BP.simulate()의 ②(6개월 재평가+빈슬롯 충원) 골격을
그대로 옮기고, 그 위에 "보유 중인 종목에 매일 가격 밴드를 체크해서 추가매수/매도"를 추가한
것이다 — 어떤 종목을 언제 새로 사고 완전히 파는지(구조)는 원래 알고리즘 그대로 두고,
그 안에서 물타기/불타기/익절/손절만 얹는다는 뜻. ma200 백업·진입스톱·연말청산·
full_rebalance 같은 다른 실행규칙은 지원하지 않는다(비교 대상인 A(기존)과 조건을 맞추기
위해 어차피 꺼두는 것들).

밴드매매 규칙(us_top8_band_trading_grid.py의 정적 8종목 버전과 동일 설계):
  - 방향 4종 스윕: meanrev(상승매도·하락매수)/trend(상승매수·하락매도)/accumulate(둘다매수)/
    derisk(둘다매도)
  - 매매방식 6종 스윕: full(전량스위치) + tranche(분할매매, 트랜치% 5단계)
  - 기준가(ref)는 그 종목이 구조적으로 매수된 날의 매수가로 시작하고, 밴드 트리거마다
    (체결 여부와 무관하게) 그날 종가로 재설정한다.
  - 밴드매도로 보유수량이 0이 되어도 포지션 슬롯 자체는 유지한다(구조적 매도가 아니므로
    — topn 슬롯 계산과 180일 재평가 판정은 구조적 매도/매수만 건드린다). 이후 밴드매수로
    다시 채워질 수 있다.
  - 8종목이 현금을 공유(통합 포트폴리오) — 한 종목을 밴드매도해서 번 현금을 다른 종목
    밴드매수에 쓸 수 있다.
  - 비용은 기존과 동일 BC.CostModel("us", 0bp수수료, 5bp슬리피지) — 구조적 매매·밴드매매
    모두 동일하게 적용.

실행: python us_algo8_band_trading_grid.py [--years 10] [--workers N]
결과: output/us_algo8_band_trading_grid.json
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import research.us.backtest_portfolio as BP
import sp500_daily_report as R

TOPN = 8
REEVAL_DAYS = 180
SECTOR_CAP = 2

RISE_GRID = [3, 5, 7, 10, 15, 20, 25]
FALL_GRID = [3, 5, 7, 10, 15, 20, 25]
TRANCHE_GRID = [10, 20, 25, 33, 50]
DIRECTIONS = {
    "meanrev":    ("sell", "buy"),
    "trend":      ("buy", "sell"),
    "accumulate": ("buy", "buy"),
    "derisk":     ("sell", "sell"),
}
DIRECTION_DESC_KO = {
    "meanrev": "상승 시 매도(익절) / 하락 시 매수(물타기) — 역추세 밴드매매",
    "trend": "상승 시 매수(불타기) / 하락 시 매도(손절) — 추세추종·피라미딩",
    "accumulate": "상승·하락 모두 매수 — 계속 물타기·불타기(현금 소진 시 정지)",
    "derisk": "상승·하락 모두 매도 — 둘 다 위험회피(청산 후 현금 보유)",
}


def _log(m): print(f"[알고8밴드그리드] {m}", file=sys.stderr)


def build_context(years: float):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    _log("팩터 랭킹 결정 시점(월간) 계산 중...")
    dec = BP.us_decisions(panel, funds, pit)
    sector_map = R.fetch_wikipedia_sectors()
    _log(f"위키 섹터맵 {len(sector_map)}종목 확보")
    return panel, spy, dec, sector_map


def simulate_band(panel: pd.DataFrame, decisions: list, topn: int, cost: BC.CostModel,
                   reeval_days: int, sector_map: dict, sector_cap: int,
                   rise_pct: float, fall_pct: float, rise_action: str, fall_action: str,
                   sizing: str, tranche_pct: float | None, trade_log=None):
    """반환: (nav Series|None, buys, sells)."""
    if not decisions:
        return None, 0, 0
    dec_by_p = {p: syms for p, syms in decisions}
    p0 = decisions[0][0]
    px = panel.ffill()
    dates = panel.index
    cash, pos = 1.0, {}
    nav_out = np.full(len(dates), np.nan)
    rise_th, fall_th = rise_pct / 100.0, fall_pct / 100.0
    buys = sells = 0

    for i in range(p0, len(dates)):
        today = dates[i]
        today_s = today.strftime("%Y%m%d")
        prices = px.iloc[i]

        # ── 매일: 보유종목 가격 밴드매매
        for sym in list(pos):
            p_now = prices.get(sym)
            if not np.isfinite(p_now) or p_now <= 0:
                continue
            info = pos[sym]
            chg = p_now / info["ref"] - 1.0
            if chg >= rise_th:
                action = rise_action
            elif chg <= -fall_th:
                action = fall_action
            else:
                continue
            if action == "sell" and info["sh"] > 1e-9:
                qty = (info["sh"] if sizing == "full"
                       else min(info["sh"], (tranche_pct / 100.0) * info["init_sh"]))
                if qty > 0:
                    cash += qty * p_now * (1.0 - cost.sell)
                    info["sh"] -= qty
                    sells += 1
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell", "reason": "band"})
            elif action == "buy" and cash > 1e-9:
                need = (max(info["init_sh"] - info["sh"], 0.0) if sizing == "full"
                        else (tranche_pct / 100.0) * info["init_sh"])
                if need > 0:
                    cost_amt = need * p_now * (1.0 + cost.buy)
                    spend = min(cash, cost_amt)
                    qty = spend / (p_now * (1.0 + cost.buy))
                    if qty > 0:
                        info["sh"] += qty
                        cash -= spend
                        buys += 1
                        if trade_log is not None:
                            trade_log.append({"date": today_s, "sym": sym, "action": "buy", "reason": "band"})
            info["ref"] = p_now

        # ── 결정일: 6개월 재평가 매도 + 빈 슬롯 충원 (BP.simulate ②와 동일 구조)
        ranked = dec_by_p.get(i)
        if ranked:
            pool_set = set(ranked)
            for sym in list(pos):
                held = (today - pos[sym]["entry_date"]).days
                p_now = prices.get(sym)
                if not np.isfinite(p_now):
                    continue
                if held >= reeval_days and sym not in pool_set:
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "reeval", "held_days": held})
                    del pos[sym]
            nav_now = cash + sum(v["sh"] * prices.get(s, np.nan) for s, v in pos.items()
                                 if np.isfinite(prices.get(s, np.nan)))
            sec_count = {}
            for sym in pos:
                sc = sector_map.get(sym)
                if sc:
                    sec_count[sc] = sec_count.get(sc, 0) + 1
            deferred = []
            for sym in ranked:
                if len(pos) >= topn or cash <= 1e-9:
                    break
                p_now = panel.iloc[i].get(sym)
                if sym in pos or not np.isfinite(p_now) or p_now <= 0:
                    continue
                sc = sector_map.get(sym)
                if sc is not None and sec_count.get(sc, 0) >= sector_cap:
                    deferred.append(sym)
                    continue
                alloc = min(nav_now / topn, cash)
                sh = alloc * (1 - cost.buy) / p_now
                pos[sym] = {"sh": sh, "init_sh": sh, "ref": p_now, "entry_date": today}
                cash -= alloc
                if sc:
                    sec_count[sc] = sec_count.get(sc, 0) + 1
                if trade_log is not None:
                    trade_log.append({"date": today_s, "sym": sym, "action": "buy", "reason": "structural"})
            if deferred and len(pos) < topn and cash > 1e-9:
                for sym in deferred:
                    if len(pos) >= topn or cash <= 1e-9:
                        break
                    p_now = panel.iloc[i].get(sym)
                    if sym in pos or not np.isfinite(p_now) or p_now <= 0:
                        continue
                    alloc = min(nav_now / topn, cash)
                    sh = alloc * (1 - cost.buy) / p_now
                    pos[sym] = {"sh": sh, "init_sh": sh, "ref": p_now, "entry_date": today}
                    cash -= alloc
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "buy",
                                          "reason": "structural_sector_relaxed"})

        nav_out[i] = cash + sum(v["sh"] * prices.get(s, np.nan) for s, v in pos.items()
                                if np.isfinite(prices.get(s, np.nan)))

    s = pd.Series(nav_out, index=dates).dropna()
    return (s if len(s) > BP.MONTH else None), buys, sells


_PANEL = None
_DEC = None
_SECTOR_MAP = None
_BENCH = None


def _init_worker(panel, dec, sector_map, bench):
    global _PANEL, _DEC, _SECTOR_MAP, _BENCH
    _PANEL, _DEC, _SECTOR_MAP, _BENCH = panel, dec, sector_map, bench


def _run_one(task):
    direction, rise_pct, fall_pct, sizing, tranche_pct, cost_buy, cost_sell = task
    rise_action, fall_action = DIRECTIONS[direction]
    cost = BC.CostModel.__new__(BC.CostModel)
    cost.market, cost.buy, cost.sell = "us", cost_buy, cost_sell
    nav, buys, sells = simulate_band(_PANEL, _DEC, TOPN, cost, REEVAL_DAYS, _SECTOR_MAP,
                                     SECTOR_CAP, rise_pct, fall_pct, rise_action, fall_action,
                                     sizing, tranche_pct)
    if nav is None:
        return {"direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
                "sizing": sizing, "tranche_pct": tranche_pct, "failed": True}
    idx = nav.index.intersection(_BENCH.index)
    m = BP.metrics(nav.reindex(idx) / nav.reindex(idx).iloc[0], _BENCH.reindex(idx))
    yrs = m["years"]
    return {"direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
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
    t_setup = time.time()
    panel, spy, dec, sector_map = build_context(years)
    _log(f"패널 확보 {panel.shape[1]}종목 × {len(panel)}거래일, 결정시점 {len(dec)}개 "
         f"({time.time() - t_setup:.0f}초)")

    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = lambda date_s, sym: sector_map.get(sym)
    ma200 = panel.rolling(200, min_periods=200).mean()

    _log("대조군(A, 밴드매매 없음 — 현행 라이브 방식) NAV 계산 중...")
    trade_a = []
    nav_a = BP.simulate(panel, ma200, dec, TOPN, cost, reeval_days=REEVAL_DAYS,
                        ma200_backup=False, sector_of=sector_of, sector_cap=SECTOR_CAP,
                        trade_log=trade_a)
    if nav_a is None:
        raise RuntimeError("대조군 NAV 산출 실패")
    bench = spy.reindex(nav_a.index).ffill()
    a_m = BP.metrics(nav_a / nav_a.iloc[0], bench)
    buys_a = sum(1 for e in trade_a if e.get("action") == "buy")
    sells_a = sum(1 for e in trade_a if e.get("action") == "sell")
    a_m["trades_per_year"] = round((buys_a + sells_a) / a_m["years"], 1)
    _log(f"대조군: CAGR {a_m['cagr_pct']}% 샤프 {a_m['sharpe']} MDD {a_m['mdd_pct']}%")

    combos = build_grid()
    workers = workers or os.cpu_count() or 4
    _log(f"그리드 조합 {len(combos)}개를 워커 {workers}개로 병렬 실행 "
         f"(방향{len(DIRECTIONS)}×상승{len(RISE_GRID)}×하락{len(FALL_GRID)}"
         f"×사이징{1 + len(TRANCHE_GRID)})")

    tasks = [(d, r, f, s, t, cost.buy, cost.sell) for d, r, f, s, t in combos]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(panel, dec, sector_map, bench)) as ex:
        for res in ex.map(_run_one, tasks, chunksize=4):
            results.append(res)
    n_failed = sum(1 for r in results if r.get("failed"))
    results = [r for r in results if not r.get("failed")]
    _log(f"완료: {len(results)}개 조합 성공(실패 {n_failed}개), {time.time() - t0:.1f}초")

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    best_per_direction = {d: max((r for r in results if r["direction"] == d),
                                 key=lambda r: r["sharpe"], default=None) for d in DIRECTIONS}

    payload = {
        "as_of": panel.index[-1].date().isoformat(),
        "start_date": panel.index[0].date().isoformat(),
        "years": round(len(panel) / 252, 2),
        "n_combos": len(results),
        "topn": TOPN, "reeval_days": REEVAL_DAYS, "sector_cap": SECTOR_CAP,
        "cost_assumption": cost.describe(),
        "direction_types": DIRECTION_DESC_KO,
        "baseline_no_band_trading": a_m,
        "top20_by_sharpe": results[:20],
        "best_per_direction": best_per_direction,
        "all_results": results,
        "note": ("대조군(A)은 us_daily_top8_vs_baseline.py의 'A(기존, 현행 라이브)'와 완전히 "
                "동일 설정(step21·풀60·topn8·reeval180일·ma200_backup=False·sector_cap2·"
                "가중치1:2:2). 밴드매매 버전은 그 구조(언제 새로 사고 완전히 파는지) 위에 "
                "보유종목 일별 가격 밴드매매만 얹은 것 — 재구현이라 BP.simulate()와 완전히 "
                "같지는 않을 수 있으니 밴드 임계값을 매우 크게(도달 불가능) 주면 A와 거의 "
                "같아야 정합성 확인이 된다. 단일 그리드 탐색이며 PBO/DSR 등 다중검정 게이트를 "
                "통과한 것은 아니다 — 상위 조합은 과최적화 위험을 감안해서 해석할 것."),
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_algo8_band_trading_grid.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    payload = run(years=args.years, workers=args.workers)
    print(f"\n=== 알고리즘 회전 8종목(6개월 재평가) + 밴드매매 그리드 백테스트 ===")
    print(f"기간: {payload['start_date']} ~ {payload['as_of']} ({payload['years']}년), "
          f"조합수: {payload['n_combos']}")
    a = payload["baseline_no_band_trading"]
    print(f"대조군(밴드매매 없음, 현행 라이브 방식): CAGR {a['cagr_pct']}% 샤프 {a['sharpe']} "
          f"MDD {a['mdd_pct']}% 연매매 {a['trades_per_year']}건")
    print("\n--- 샤프 기준 상위 10개 조합 ---")
    for r in payload["top20_by_sharpe"][:10]:
        tranche_str = "" if r["tranche_pct"] is None else f"({r['tranche_pct']}%)"
        print(f"{r['direction']:>10s} 상승{r['rise_pct']:>2}%/하락{r['fall_pct']:>2}% "
              f"{r['sizing']:>7s}{tranche_str} "
              f"→ CAGR {r['cagr_pct']:>6.2f}% 샤프 {r['sharpe']:>5.2f} MDD {r['mdd_pct']:>6.1f}% "
              f"연매매 {r['trades_per_year']:>5.1f}건")
    print("\n--- 방향별 최고(샤프 기준) ---")
    for d, r in payload["best_per_direction"].items():
        if r is None:
            continue
        print(f"{d:>10s}({DIRECTION_DESC_KO[d]}): 상승{r['rise_pct']}%/하락{r['fall_pct']}% "
              f"{r['sizing']} → CAGR {r['cagr_pct']}% 샤프 {r['sharpe']} MDD {r['mdd_pct']}%")
