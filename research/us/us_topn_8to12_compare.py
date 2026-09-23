#!/usr/bin/env python3
"""
us_topn_8to12_compare.py — 현행 라이브 방식(팩터랭킹, 6개월 재평가) 그대로 두고 보유종목
수(topn)만 8~12개로 바꿨을 때 성과 비교. (2026-09-10, 지호 님 질문 — "지금 8종목 운영 중인데
9·10·11·12종목이었으면 어땠을지")

설정은 us_algo8_band_trading_grid.py의 대조군(A)과 동일: BP.us_decisions(step21·풀60) →
BP.simulate(reeval_days=180, ma200_backup=False, sector_cap=2, 가중치1:2:2). topn만 바꿔서
비교(패널·결정시점은 topn과 무관하게 동일하므로 한 번만 구축해 재사용).

실행: python us_topn_8to12_compare.py [--years 10]
결과: output/us_topn_8to12_compare.json
"""
from __future__ import annotations
import os, sys, json, argparse
import backtest_costs as BC
import research.us.backtest_portfolio as BP
import research.us.us_algo8_band_trading_grid as G

TOPN_LIST = [8, 9, 10, 11, 12]


def _log(m): print(f"[topn비교] {m}", file=sys.stderr)


def run(years: float = 10, save: bool = True) -> dict:
    panel, spy, dec, sector_map = G.build_context(years)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = lambda date_s, sym: sector_map.get(sym)
    ma200 = panel.rolling(200, min_periods=200).mean()

    rows = []
    for topn in TOPN_LIST:
        trade_log = []
        nav = BP.simulate(panel, ma200, dec, topn, cost, reeval_days=G.REEVAL_DAYS,
                          ma200_backup=False, sector_of=sector_of, sector_cap=G.SECTOR_CAP,
                          trade_log=trade_log)
        if nav is None:
            _log(f"topn={topn}: NAV 산출 실패")
            continue
        bench = spy.reindex(nav.index).ffill()
        m = BP.metrics(nav / nav.iloc[0], bench)
        buys = sum(1 for e in trade_log if e.get("action") == "buy")
        sells = sum(1 for e in trade_log if e.get("action") == "sell")
        m["trades_per_year"] = round((buys + sells) / m["years"], 1)
        rows.append({"topn": topn, **m})
        _log(f"topn={topn}: CAGR {m['cagr_pct']}% 샤프 {m['sharpe']} MDD {m['mdd_pct']}% "
             f"연매매 {m['trades_per_year']}건")

    payload = {"as_of": panel.index[-1].date().isoformat(),
              "start_date": panel.index[0].date().isoformat(),
              "years": round(len(panel) / 252, 2),
              "cost_assumption": cost.describe(),
              "config": {"reeval_days": G.REEVAL_DAYS, "sector_cap": G.SECTOR_CAP,
                        "ma200_backup": False, "weights": "1:2:2(output/best_weights.json)"},
              "rows": rows,
              "note": "단일 실행 비교(그리드 탐색 아님) — topn=8이 현재 라이브 설정."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_topn_8to12_compare.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_topn_8to12_compare.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    payload = run(years=args.years)
    print(f"\n=== topn 8~12 비교 ({payload['start_date']} ~ {payload['as_of']}, "
          f"{payload['years']}년) ===")
    for r in payload["rows"]:
        print(f"topn={r['topn']:>2d}: CAGR {r['cagr_pct']:>6.2f}% 초과CAGR {r['excess_cagr_pct']:>6.2f}%p "
              f"샤프 {r['sharpe']:>5.2f} MDD {r['mdd_pct']:>6.1f}% 변동성 {r['vol_pct']:>5.1f}% "
              f"연매매 {r['trades_per_year']:>4.1f}건")
