#!/usr/bin/env python3
"""
us_daily_top8_vs_baseline.py — "매일 top8 갱신" 알고리즘 vs 기존(현행 라이브) 알고리즘 비교
(2026-07-23, 지호 님 요청)

배경: 현행 라이브는 결정을 월 1회(21거래일)만 다시 계산하고, 보유종목은 후보풀(60) 안에
있는 한 계속 들고 가다가 180일(달력) 지나고 풀 밖일 때만 교체하는 "끈적한 보유" 방식이다
(STRATEGY.md §2 매도규칙, Stage 6 계열에서 검증됨). 이 스크립트는 그 반대 극단을 본다:
**매일** 팩터 랭킹을 다시 계산해서 top8이 바뀌면 그날 바로 매도/매수로 교체하는 알고리즘.

지호 님 지시대로 수수료·슬리피지를 0으로 가정한다(비용 자체의 효과를 "신호를 얼마나
자주 갱신하느냐" 효과와 분리하기 위함 — 실제로는 일별 교체가 회전율을 훨씬 크게 늘려
비용이 그만큼 커진다는 걸 별도로 알아두라고 아래 note와 연간 매수건수로 같이 기록한다).

비교:
  A(기존, 현행 라이브) = us_decisions(step=21, 풀 60) → simulate(reeval_days=180,
    full_rebalance=False) — 끈적한 보유, 후보풀(60) 밖으로 나가고 180일 지나야 교체.
  B(매일 top8 교체)    = 매일(step=1) 그날의 top8만 산출 → simulate(reeval_days=1,
    full_rebalance=False) — ranked 자체가 top8뿐이라 pool_set=top8, 보유종목이 오늘
    top8에서 밀려나면 다음 결정일(=내일)에 즉시 매도, 빈 슬롯은 오늘 top8 신규 진입
    종목으로 채운다.
  둘 다 동일: 팩터 가중치 1:2:2(best_weights.json, 라이브) · 섹터캡=2 · ma200_backup=False
  (현행) · 비용 0bp.

실행: python us_daily_top8_vs_baseline.py [--years 10]
결과: output/us_daily_top8_vs_baseline.json
"""
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import backtest_exec as BE
import tech_factors as T
import sp500_daily_report as R

TOPN = 8


def _log(m): print(f"[일일top8비교] {m}", file=sys.stderr)


def _sector_of_factory():
    sector_map = R.fetch_wikipedia_sectors()
    _log(f"위키 섹터맵 {len(sector_map)}종목 확보")
    return lambda date_s, sym: sector_map.get(sym)


def _daily_top8_decisions(panel, funds, pit):
    """매일(step=1) 그날의 top8만 산출 — us_decisions(POOL_SIZE=60)와 달리 ranked 자체를
    top8로 좁혀서, simulate()의 'held>=reeval_days and not in pool_set' 조건이 '오늘
    top8 밖으로 밀려나면'을 의미하게 만든다(60위 안에만 있으면 안 팔리는 기존 끈적한
    보유와 대비)."""
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    out = []
    n = len(panel)
    for p in range(BW.LOOKBACK, n - 1):
        ranked = BE._select_basket(panel, p, funds, cross, pit, weights, TOPN)
        if ranked:
            out.append((p, ranked))
    _log(f"매일top8 결정 시점 {len(out)}개")
    return out


def _monthly_returns(nav: pd.Series) -> np.ndarray:
    return np.array([nav.iloc[t + BP.MONTH] / nav.iloc[t] - 1
                     for t in range(0, len(nav) - BP.MONTH, BP.MONTH)])


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    n = len(d)
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = float(d.mean()) / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def run(years=10, save=True):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=0.0)   # 지호 님 지시: 수수료 0
    sector_of = _sector_of_factory()

    _log("A(기존, 월간결정+6개월재평가) NAV 계산 중...")
    dec_month = BP.us_decisions(panel, funds, pit)
    trade_a = []
    nav_a = BP.simulate(panel, ma200, dec_month, TOPN, cost, reeval_days=180,
                        ma200_backup=False, sector_of=sector_of, sector_cap=2,
                        trade_log=trade_a)
    if nav_a is None:
        raise RuntimeError("A(기존) NAV 산출 실패")

    _log("B(매일top8 즉시교체) 결정 산출 중... (매일 팩터 재계산 — 시간이 걸림)")
    dec_daily = _daily_top8_decisions(panel, funds, pit)
    _log("B(매일top8) NAV 계산 중...")
    trade_b = []
    nav_b = BP.simulate(panel, ma200, dec_daily, TOPN, cost, reeval_days=1,
                        ma200_backup=False, sector_of=sector_of, sector_cap=2,
                        trade_log=trade_b)
    if nav_b is None:
        raise RuntimeError("B(매일top8) NAV 산출 실패")

    idx = nav_a.index.intersection(nav_b.index)
    nav_a = (nav_a.reindex(idx)); nav_a = nav_a / nav_a.iloc[0]
    nav_b = (nav_b.reindex(idx)); nav_b = nav_b / nav_b.iloc[0]
    bench = spy.reindex(idx).ffill()

    m_a = BP.metrics(nav_a, bench)
    m_b = BP.metrics(nav_b, bench)
    _log(f"A(기존): CAGR {m_a['cagr_pct']}% 샤프 {m_a['sharpe']} MDD {m_a['mdd_pct']}%")
    _log(f"B(매일top8): CAGR {m_b['cagr_pct']}% 샤프 {m_b['sharpe']} MDD {m_b['mdd_pct']}%")

    r_a, r_b = _monthly_returns(nav_a), _monthly_returns(nav_b)
    n = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:n], r_b[:n]
    tstat, pval = _paired_ttest(r_b, r_a)
    _log(f"페어드 t검정(B-A, 월간수익률, n={n}): t={tstat:+.2f} p={pval:.3f}")

    yrs = len(idx) / 252
    buys_a = sum(1 for e in trade_a if e.get("action") == "buy")
    buys_b = sum(1 for e in trade_b if e.get("action") == "buy")
    sells_a = sum(1 for e in trade_a if e.get("action") == "sell")
    sells_b = sum(1 for e in trade_b if e.get("action") == "sell")
    _log(f"연간 매매건수: A(기존) 매수{buys_a/yrs:.1f}/매도{sells_a/yrs:.1f}건 · "
         f"B(매일top8) 매수{buys_b/yrs:.1f}/매도{sells_b/yrs:.1f}건")

    direction = ("B가 A보다 유의하게 낫다" if tstat >= 1.96 else
                ("A가 B보다 유의하게 낫다" if tstat <= -1.96 else "유의한 차이 없음"))

    payload = {
        "as_of": idx[-1].date().isoformat(), "years": round(yrs, 1), "n_months": n,
        "cost_assumption": "0bp(수수료·슬리피지 전부 0 — 지호 님 지시. 신호갱신빈도 효과를 "
                           "거래비용과 분리하기 위함)",
        "config_common": {"topn": TOPN, "weights_source": "best_weights.json(1:2:2, 라이브)",
                          "sector_cap": 2, "ma200_backup": False},
        "A_existing": {**m_a, "decision_freq": "월간(21거래일)",
                      "sell_rule": "후보풀(60) 이탈 + 180일 경과(끈적한 보유, 현행 라이브)",
                      "trades_per_year": {"buy": round(buys_a / yrs, 1), "sell": round(sells_a / yrs, 1)}},
        "B_daily_top8": {**m_b, "decision_freq": "매일",
                        "sell_rule": "당일 top8 이탈 시 즉시 교체",
                        "trades_per_year": {"buy": round(buys_b / yrs, 1), "sell": round(sells_b / yrs, 1)}},
        "paired_ttest_monthly_excess": {"t": round(float(tstat), 3), "p": round(float(pval), 4),
                                        "n": n, "verdict": direction},
        "note": "단일 비교(그리드 탐색 아님, PBO/DSR 다중검정 게이트 대상 아님) — 회전율이 "
                "극단적으로 높아지는 B는 수수료=0 가정 하의 상한선(best-case) 추정치다. "
                "실제 왕복비용(현재 US CostModel 기준 편도 5bp·왕복 약 10bp)을 반영하면 "
                "B의 연간 매매건수만큼 그 비용이 그대로 수익을 깎아먹는다 — 위 trades_per_year "
                "비율이 그 영향의 대략적인 크기를 보여준다.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_daily_top8_vs_baseline.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    run(years=args.years)
