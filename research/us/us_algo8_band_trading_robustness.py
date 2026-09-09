#!/usr/bin/env python3
"""
us_algo8_band_trading_robustness.py — us_algo8_band_trading_grid.py 결과 강건성 검증
(2026-09-08, 지호 님 요청 2건):
  1) "상위 조합들이 진짜 고원을 이루는지" — 그리드 이웃(상승·하락 임계값이 한 칸 옆인) 조합도
     같이 좋은지 확인(STRATEGY.md에서 반복적으로 쓰는 방식: 1등만 좋고 이웃이 나쁘면 과최적화
     의심). output/us_algo8_band_trading_grid.json을 그대로 읽어서 계산(재실행 불필요, 빠름).
  2) PBO(Probability of Backtest Overfitting)·DSR(Deflated Sharpe Ratio) — overfit_stats.py의
     기존 방법론(Bailey et al. 2017/2014, 이 저장소의 표준 다중검정 게이트)을 그대로 재사용해
     1,176개 조합 전체를 "시행 모집단"으로 놓고 "샤프 1등이 과최적화가 아닌지"를 통계적으로
     판정. 조합별 비중첩 21거래일 초과수익(BP.monthly_excess, 기존 관례와 동일)을 다시
     계산해야 해서 패널을 재구축(~4분)하고 1,176개 조합을 재시뮬레이션(~4분)한다.
     주의: output/pbo_report.json(라이브 가중치 탐색용 기존 게이트)을 덮어쓰지 않도록
     별도 경로(output/us_algo8_band_trading_pbo.json)에 저장.
  3) (추가 요청) "우리 방식"(대조군 — 밴드매매 없는 현행 라이브 방식)의 달력월별 수익률
     계절성 — 특정 달(예: 9월)이 평균적으로 나쁜지. 2)와 같은 패널 재구축을 공유해서
     한 번의 실행으로 같이 계산한다.

실행: python us_algo8_band_trading_robustness.py [--years 10] [--workers N]
결과: output/us_algo8_band_trading_plateau.json
      output/us_algo8_band_trading_pbo.json
      output/us_algo8_band_trading_seasonality.json
"""
from __future__ import annotations
import os, sys, json, math, argparse, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

import backtest_costs as BC
import research.us.backtest_portfolio as BP
import overfit_stats as OS
import research.us.us_algo8_band_trading_grid as G

MONTH = BP.MONTH  # 21거래일


def _log(m): print(f"[강건성검증] {m}", file=sys.stderr)


# ═══════════════════════════ 1) 고원(plateau) 체크 — 재실행 불필요 ═══════════════════════════
def check_plateau(results: list, top_k: int = 5) -> dict:
    out = {}
    for direction in G.DIRECTIONS:
        dir_rows = [r for r in results if r["direction"] == direction]
        by_sizing = {}
        for r in dir_rows:
            by_sizing.setdefault((r["sizing"], r["tranche_pct"]), {})[
                (r["rise_pct"], r["fall_pct"])] = r
        dir_sorted = sorted(dir_rows, key=lambda r: r["sharpe"], reverse=True)
        rows = []
        for r in dir_sorted[:top_k]:
            key = (r["sizing"], r["tranche_pct"])
            grid = by_sizing[key]
            same_sizing_sharpes = np.array([v["sharpe"] for v in grid.values()])
            median_sharpe = float(np.median(same_sizing_sharpes))
            ri, fi = G.RISE_GRID.index(r["rise_pct"]), G.FALL_GRID.index(r["fall_pct"])
            neigh_sharpes = []
            for dri in (-1, 0, 1):
                for dfi in (-1, 0, 1):
                    if dri == 0 and dfi == 0:
                        continue
                    nri, nfi = ri + dri, fi + dfi
                    if 0 <= nri < len(G.RISE_GRID) and 0 <= nfi < len(G.FALL_GRID):
                        nb = grid.get((G.RISE_GRID[nri], G.FALL_GRID[nfi]))
                        if nb:
                            neigh_sharpes.append(nb["sharpe"])
            neigh_mean = float(np.mean(neigh_sharpes)) if neigh_sharpes else None
            gap = round(r["sharpe"] - neigh_mean, 3) if neigh_mean is not None else None
            if neigh_mean is None:
                verdict = "이웃 데이터 부족"
            elif neigh_mean >= median_sharpe and (gap is None or gap < 0.25):
                verdict = "고원(이웃도 같이 좋음)"
            elif gap is not None and gap >= 0.25:
                verdict = "고립된 스파이크(과최적화 의심)"
            else:
                verdict = "애매(이웃이 중간 정도)"
            rows.append({"rise_pct": r["rise_pct"], "fall_pct": r["fall_pct"],
                        "sizing": r["sizing"], "tranche_pct": r["tranche_pct"],
                        "sharpe": r["sharpe"], "cagr_pct": r["cagr_pct"],
                        "neighbor_mean_sharpe": round(neigh_mean, 3) if neigh_mean is not None else None,
                        "neighbor_n": len(neigh_sharpes),
                        "gap_vs_neighbor": gap,
                        "same_sizing_median_sharpe": round(median_sharpe, 3),
                        "verdict": verdict})
        out[direction] = rows
    return out


# ═══════════════════════════ 2) PBO/DSR — 패널 재구축 필요 ═══════════════════════════
_PANEL = None
_DEC = None
_SECTOR_MAP = None
_BENCH = None


def _init_worker(panel, dec, sector_map, bench):
    global _PANEL, _DEC, _SECTOR_MAP, _BENCH
    _PANEL, _DEC, _SECTOR_MAP, _BENCH = panel, dec, sector_map, bench


def _run_one_events(task):
    direction, rise_pct, fall_pct, sizing, tranche_pct, cost_buy, cost_sell = task
    rise_action, fall_action = G.DIRECTIONS[direction]
    cost = BC.CostModel.__new__(BC.CostModel)
    cost.market, cost.buy, cost.sell = "us", cost_buy, cost_sell
    nav, buys, sells = G.simulate_band(_PANEL, _DEC, G.TOPN, cost, G.REEVAL_DAYS, _SECTOR_MAP,
                                       G.SECTOR_CAP, rise_pct, fall_pct, rise_action, fall_action,
                                       sizing, tranche_pct)
    label = f"{direction}_r{rise_pct}_f{fall_pct}_{sizing}{tranche_pct or ''}"
    if nav is None:
        return label, None
    _, rets = BP.monthly_excess(nav, _BENCH)
    return label, rets


def run_pbo_dsr(panel, dec, sector_map, bench, cost, workers, years) -> dict:
    combos = G.build_grid()
    tasks = [(d, r, f, s, t, cost.buy, cost.sell) for d, r, f, s, t in combos]
    _log(f"조합 {len(tasks)}개 재시뮬레이션 → 비중첩 21거래일 초과수익 이벤트 행렬 구성 중...")
    t0 = time.time()
    labels, rows = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(panel, dec, sector_map, bench)) as ex:
        for label, rets in ex.map(_run_one_events, tasks, chunksize=4):
            if rets is not None:
                labels.append(label)
                rows.append(rets)
    n_events = min(len(r) for r in rows)
    M = np.array([r[:n_events] for r in rows], dtype=float)
    _log(f"완료: {M.shape[0]}개 조합 × {M.shape[1]}이벤트 행렬, {time.time() - t0:.1f}초")

    data = {"trials": labels, "excess_returns": M.tolist(),
            "rebal_days": MONTH, "hold_days": MONTH,   # 비중첩 월간 수익률 — 중첩 보정 불필요
            "horizon": f"{years}y", "universe": "us_algo8_band_grid",
            "cost": cost.describe()}
    report = OS.analyze(data, n_blocks=12, save=False)
    path = "output/us_algo8_band_trading_pbo.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: {path} (기존 output/pbo_report.json은 건드리지 않음)")
    return report


# ═══════════════════════════ 3) 월별(달력월) 계절성 ═══════════════════════════
def _ttest_1samp(x: np.ndarray):
    n = len(x)
    if n < 2:
        return None, None
    se = x.std(ddof=1) / math.sqrt(n)
    t = float(x.mean() / se) if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def _welch_ttest(a: np.ndarray, b: np.ndarray):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    t = float((a.mean() - b.mean()) / se) if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))  # 정규근사(us_daily_top8_vs_baseline.py와 동일 관례)
    return t, p


def monthly_seasonality(nav: pd.Series, bench: pd.Series) -> dict:
    m_nav = nav.resample("ME").last()
    m_bench = bench.reindex(nav.index).ffill().resample("ME").last()
    ret = m_nav.pct_change().dropna()
    bret = m_bench.pct_change().dropna()
    idx = ret.index.intersection(bret.index)
    ret, bret = ret.loc[idx], bret.loc[idx]

    by_month = []
    for mo in range(1, 13):
        sub = ret[ret.index.month == mo].values
        subb = bret[bret.index.month == mo].values
        rest = ret[ret.index.month != mo].values
        if len(sub) == 0:
            continue
        t1, p1 = _ttest_1samp(sub)
        tw, pw = _welch_ttest(sub, rest)
        by_month.append({
            "month": mo, "n": len(sub),
            "mean_pct": round(float(sub.mean()) * 100, 2),
            "median_pct": round(float(np.median(sub)) * 100, 2),
            "std_pct": round(float(sub.std(ddof=1)) * 100, 2) if len(sub) > 1 else None,
            "win_rate_pct": round(float((sub > 0).mean()) * 100, 1),
            "bench_mean_pct": round(float(subb.mean()) * 100, 2) if len(subb) else None,
            "vs_zero_t": round(t1, 2) if t1 is not None else None,
            "vs_zero_p": round(p1, 4) if p1 is not None else None,
            "vs_other_months_t": round(tw, 2) if tw is not None else None,
            "vs_other_months_p": round(pw, 4) if pw is not None else None,
        })
    worst = min(by_month, key=lambda r: r["mean_pct"])
    best = max(by_month, key=lambda r: r["mean_pct"])
    return {
        "n_years": round(len(ret) / 12, 1),
        "by_month": by_month,
        "worst_month": worst, "best_month": best,
        "note": ("달마다 표본이 ~9~10개뿐이라(연도 수만큼) 통계적 검정력이 매우 낮음 — "
                "vs_other_months_p가 낮게 나와도 12개월을 동시에 비교(다중검정)한 것이라 "
                "우연히 하나쯤 낮게 나올 수 있음(대략 p<0.004 정도는 되어야 12회 비교를 "
                "감안해도 유의). 정규근사 t검정(자유도 보정 없음, 표본이 작을수록 부정확) "
                "이라는 점도 감안할 것."),
    }


def run(years: float = 10, workers: int | None = None) -> dict:
    workers = workers or os.cpu_count() or 4

    # ── 1) 고원 체크 (기존 그리드 결과 재사용, 빠름) ──
    grid_path = "output/us_algo8_band_trading_grid.json"
    with open(grid_path, encoding="utf-8") as f:
        grid_payload = json.load(f)
    plateau = check_plateau(grid_payload["all_results"], top_k=5)
    with open("output/us_algo8_band_trading_plateau.json", "w", encoding="utf-8") as f:
        json.dump(plateau, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_algo8_band_trading_plateau.json")

    # ── 2)+3) 패널 재구축(공유) → PBO/DSR + 월별 계절성 ──
    panel, spy, dec, sector_map = G.build_context(years)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = lambda date_s, sym: sector_map.get(sym)
    ma200 = panel.rolling(200, min_periods=200).mean()

    _log("대조군(밴드매매 없음) NAV 계산 중 — 계절성 분석용...")
    nav_a = BP.simulate(panel, ma200, dec, G.TOPN, cost, reeval_days=G.REEVAL_DAYS,
                        ma200_backup=False, sector_of=sector_of, sector_cap=G.SECTOR_CAP)
    bench = spy.reindex(nav_a.index).ffill()
    seasonality = monthly_seasonality(nav_a, bench)
    with open("output/us_algo8_band_trading_seasonality.json", "w", encoding="utf-8") as f:
        json.dump(seasonality, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_algo8_band_trading_seasonality.json")

    pbo_report = run_pbo_dsr(panel, dec, sector_map, bench, cost, workers, years)

    return {"plateau": plateau, "seasonality": seasonality, "pbo_dsr": pbo_report}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = run(years=args.years, workers=args.workers)

    print("\n=== 1) 고원(plateau) 체크 — 방향별 샤프 상위 5개 ===")
    for d, rows in out["plateau"].items():
        print(f"\n[{d}]")
        for r in rows:
            tranche_str = "" if r["tranche_pct"] is None else f"({r['tranche_pct']}%)"
            print(f"  상승{r['rise_pct']}%/하락{r['fall_pct']}% {r['sizing']}{tranche_str}"
                  f" 샤프{r['sharpe']} | 이웃평균샤프{r['neighbor_mean_sharpe']}"
                  f"(n={r['neighbor_n']}) 격차{r['gap_vs_neighbor']} → {r['verdict']}")

    print("\n=== 2) PBO/DSR ===")
    pbo, dsr = out["pbo_dsr"]["pbo"], out["pbo_dsr"]["dsr"]
    print(f"PBO = {pbo['pbo']:.1%} → {out['pbo_dsr']['pbo_verdict']}")
    print(f"DSR = {dsr.get('dsr')} → {out['pbo_dsr']['dsr_verdict']}")
    print(f"채택 최고조합: {out['pbo_dsr']['dsr']['best_trial']}")

    print("\n=== 3) 월별(달력월) 계절성 — 대조군(밴드매매 없음) ===")
    print(f"표본: 약 {out['seasonality']['n_years']}년")
    for r in out["seasonality"]["by_month"]:
        print(f"  {r['month']:>2d}월(n={r['n']:>2d}): 평균 {r['mean_pct']:>6.2f}% "
              f"중앙값 {r['median_pct']:>6.2f}% 승률 {r['win_rate_pct']:>5.1f}% "
              f"SPY평균 {r['bench_mean_pct']:>6.2f}% | vs0 p={r['vs_zero_p']} "
              f"vs다른달 p={r['vs_other_months_p']}")
    w, b = out["seasonality"]["worst_month"], out["seasonality"]["best_month"]
    print(f"\n최악: {w['month']}월(평균 {w['mean_pct']}%, vs다른달 p={w['vs_other_months_p']}) "
          f"/ 최고: {b['month']}월(평균 {b['mean_pct']}%)")
