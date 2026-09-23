#!/usr/bin/env python3
"""
kr_band_trading_extreme.py — 한국(valuediv topn5) 버전, us_band_trading_extreme.py와
완전히 동일한 그리드로 재검증 (2026-09-22, 지호 님 요청: "한국 쪽도 같은 그리드로 돌려줘").

미국 엔진(research.us.us_algo8_band_trading_grid.simulate_band/_init_worker)을 그대로
재사용한다 — 재구현 금지(MAINTENANCE.md §1). 단, 그 모듈의 `_run_one`은 TOPN·REEVAL_DAYS·
SECTOR_CAP을 "함수 인자가 아니라 모듈 전역"으로 읽는데, ProcessPoolExecutor는 Windows에서
spawn 방식이라 각 워커가 모듈을 처음부터 다시 임포트한다 — 즉 메인 프로세스에서
`G.TOPN = 5`처럼 패치해도 워커에는 반영 안 되고 원래값(미국 TOPN=8 등)으로 되돌아가는
함정이 있다(그리드 상수 RISE_GRID 등은 메인 프로세스에서만 읽히는 build_grid() 안에서만
쓰여서 안전했지만, TOPN 계열은 워커 안에서 읽혀서 위험함). 그래서 `_run_one`만 이 파일에
한국 전용 상수(TOPN=5·REEVAL_DAYS=180·SECTOR_CAP=999=무제한)를 참조하도록 얇게 새로
작성했다 — simulate_band 자체(핵심 로직)는 여전히 100% 재사용.

라이브 설정: valuediv topn=5(HISTORY.md §3 Stage 6) · 6개월(180일) 재평가 ·
ma200_backup=False · CostModel("kospi", commission1.5bp+slippage5bp) · 섹터캡 없음(한국
라이브 파이프라인엔 sector_of/sector_cap 인자 자체가 없음).

그리드: 상승·하락 [15,20,25,30,40,50]% × 트랜치 [5,10,15,20,25,33,50]%+전량 × 방향4종
= 1,152조합 — us_band_trading_extreme.py와 숫자까지 동일(직접 비교 목적).

실행: python -m research.kr.kr_band_trading_extreme [--workers N] [--quick]
결과: output/kr_band_trading_extreme.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import backtest_costs as BC
import research.us.backtest_portfolio as BP
import research.us.us_algo8_band_trading_grid as G

RISE_GRID = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
FALL_GRID = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
TRANCHE_GRID = [5, 10, 15, 20, 25, 33, 50]
TOPN = 5
REEVAL_DAYS = 180
SECTOR_CAP = 999  # 사실상 무제한 — 한국 라이브는 섹터캡 미사용
OUT_PATH = "output/kr_band_trading_extreme.json"


def _log(m): print(f"[한국극단밴드그리드] {m}", file=sys.stderr)


def _run_one_kr(task):
    """G._run_one과 동일 로직이되 TOPN/REEVAL_DAYS/SECTOR_CAP을 이 파일(워커에서도 신선하게
    재임포트되는 한국 전용 상수)에서 읽는다. simulate_band 자체는 G의 것을 그대로 호출."""
    direction, rise_pct, fall_pct, sizing, tranche_pct, cost_buy, cost_sell = task
    rise_action, fall_action = G.DIRECTIONS[direction]
    cost = BC.CostModel.__new__(BC.CostModel)
    cost.market, cost.buy, cost.sell = "kospi", cost_buy, cost_sell
    nav, buys, sells = G.simulate_band(G._PANEL, G._DEC, TOPN, cost, REEVAL_DAYS, G._SECTOR_MAP,
                                       SECTOR_CAP, rise_pct, fall_pct, rise_action, fall_action,
                                       sizing, tranche_pct)
    if nav is None:
        return {"direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
                "sizing": sizing, "tranche_pct": tranche_pct, "failed": True}
    idx = nav.index.intersection(G._BENCH.index)
    m = BP.metrics(nav.reindex(idx) / nav.reindex(idx).iloc[0], G._BENCH.reindex(idx))
    yrs = m["years"]
    return {"direction": direction, "rise_pct": rise_pct, "fall_pct": fall_pct,
            "sizing": sizing, "tranche_pct": tranche_pct, **m,
            "buys_per_year": round(buys / yrs, 1), "sells_per_year": round(sells / yrs, 1),
            "trades_per_year": round((buys + sells) / yrs, 1)}


def build_context():
    from research.kr.benchmarks_kr import load_research_data, build_benchmarks
    import backtest_kr as BK
    import research.kr.backtest_kr_strategies as KS

    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    navs = build_benchmarks(panel, membership, mktcaps, bench)
    b1 = navs["B1_kospi200"].dropna()
    return panel, decisions, b1


def build_grid(rise_grid, fall_grid, tranche_grid) -> list[tuple]:
    sizings = [("full", None)] + [("tranche", t) for t in tranche_grid]
    combos = []
    for direction in G.DIRECTIONS:
        for rise_pct in rise_grid:
            for fall_pct in fall_grid:
                for sizing, tranche_pct in sizings:
                    combos.append((direction, rise_pct, fall_pct, sizing, tranche_pct))
    return combos


def run(workers: int | None = None, quick: bool = False, save: bool = True) -> dict:
    rise_grid = [20] if quick else RISE_GRID
    fall_grid = [20] if quick else FALL_GRID
    tranche_grid = [20] if quick else TRANCHE_GRID

    t0 = time.time()
    panel, decisions, bench = build_context()
    _log(f"패널 확보 {panel.shape[1]}종목 x {len(panel)}거래일, 결정시점 {len(decisions)}개 "
        f"({time.time() - t0:.0f}초)")

    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    sector_map = {}

    _log("대조군(밴드매매 없음, 현행 라이브 valuediv topn5) NAV 계산 중...")
    trade_a = []
    ma200 = panel.rolling(200, min_periods=200).mean()
    nav_a = BP.simulate(panel, ma200, decisions, TOPN, cost, reeval_days=REEVAL_DAYS,
                        ma200_backup=False, trade_log=trade_a)
    if nav_a is None:
        raise RuntimeError("한국 대조군 NAV 산출 실패")
    bench_aligned = bench.reindex(nav_a.index).ffill()
    a_m = BP.metrics(nav_a / nav_a.iloc[0], bench_aligned)
    buys_a = sum(1 for e in trade_a if e.get("action") == "buy")
    sells_a = sum(1 for e in trade_a if e.get("action") == "sell")
    a_m["trades_per_year"] = round((buys_a + sells_a) / a_m["years"], 1)
    _log(f"대조군: CAGR {a_m['cagr_pct']}% 샤프 {a_m['sharpe']} MDD {a_m['mdd_pct']}% "
        f"연매매 {a_m['trades_per_year']}건")

    combos = build_grid(rise_grid, fall_grid, tranche_grid)
    workers = workers or os.cpu_count() or 4
    _log(f"그리드 {len(combos)}개(quick={quick})를 워커 {workers}개로 병렬 실행")
    tasks = [(d, r, f, s, t, cost.buy, cost.sell) for d, r, f, s, t in combos]

    t1 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=G._init_worker,
                             initargs=(panel, decisions, sector_map, bench_aligned)) as ex:
        for res in ex.map(_run_one_kr, tasks, chunksize=4):
            results.append(res)
    n_failed = sum(1 for r in results if r.get("failed"))
    results = [r for r in results if not r.get("failed")]
    _log(f"완료: {len(results)}개 성공(실패 {n_failed}), {time.time() - t1:.1f}초")

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    best_per_direction = {d: max((r for r in results if r["direction"] == d),
                                 key=lambda r: r["sharpe"], default=None) for d in G.DIRECTIONS}

    payload = {
        "as_of": panel.index[-1].date().isoformat(),
        "start_date": panel.index[0].date().isoformat(),
        "years": round(len(panel) / 252, 2),
        "n_combos": len(results),
        "topn": TOPN, "reeval_days": REEVAL_DAYS, "sector_cap": None,
        "cost_assumption": cost.describe(),
        "direction_types": G.DIRECTION_DESC_KO,
        "baseline_no_band_trading": a_m,
        "top20_by_sharpe": results[:20],
        "best_per_direction": best_per_direction,
        "all_results": results,
        "note": ("us_band_trading_extreme.json과 동일 그리드·동일 엔진(simulate_band 재사용). "
                "한국은 valuediv topn5·CostModel('kospi', 1.5bp+5bp)·섹터캡 없음만 다름. "
                "단일 그리드 탐색이며 PBO/DSR 다중검정 게이트는 미적용(참고용, 미국판과 "
                "동일한 해석상 유의 필요)."),
    }
    if save and not quick:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="한국(valuediv topn5) 극단 밴드매매 그리드")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="배선 확인용 초소형 그리드(4조합)")
    args = ap.parse_args()
    payload = run(workers=args.workers, quick=args.quick)
    a = payload["baseline_no_band_trading"]
    print(f"\n대조군: CAGR {a['cagr_pct']}% 샤프 {a['sharpe']} MDD {a['mdd_pct']}% "
         f"연매매 {a['trades_per_year']}건 ({payload['n_combos']}조합 완료)")
