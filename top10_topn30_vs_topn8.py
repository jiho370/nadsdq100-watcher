#!/usr/bin/env python3
"""
top10_topn30_vs_topn8.py — topn=30 팩터평가 풀에서 샤프 상위 10개 가중치 조합을 뽑아
전부 실제 라이브 조건(topn=8·섹터캡2)으로 재평가 (2026-08-23, 지호 님 요청: "30종목 상위
10개를 8종목으로 다 돌려봐").

배경: §9-K-7에서 roa 후보 1건이 topn=30→topn=8 이관 시 순위가 뒤집히는 걸 확인했다 —
1건만으론 "우연히 이 조합만 그런 건지, topn=30 순위 자체가 topn=8을 예측 못 하는 건지"를
못 가른다. 상위 10개 전부를 topn=8로 돌려 패턴을 확인한다.

실행: python top10_topn30_vs_topn8.py
결과: output/top10_topn30_vs_topn8.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP
import tech_factors as T
import core_satellite_kr as CS

YEARS = 10          # 원 961조합 탐색과 동일 창(§9-K-4)
KEEP, LEVELS = 5, (0, 1, 2, 3)
TOPN30, TOPN8 = 30, 8


def _log(m): print(f"[top10비교] {m}", file=sys.stderr)


def avg_sharpe(row):
    vals = [row.get(f"sharpe_{h}") for h in ("3m", "6m", "12m")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def main():
    pit = BC.load_pit()
    panel, spy, opens = BC.build_panel_pit(YEARS, pit)
    funds = BW.load_funds()
    cost30 = BC.CostModel("us", 0.0, 5.0)
    snaps = BC.build_snaps(panel, spy, funds, opens, 63)
    pit_snaps, cov = BC._filter_snaps(snaps, pit, "pit")
    _log(f"PIT 이벤트 {len(pit_snaps)}회 · 커버리지 {cov['mean']}%")

    ic_sorted = BC._agg_ic(pit_snaps, list(range(len(pit_snaps))))
    selected = BC._pick(ic_sorted, KEEP)
    _log(f"후보 팩터(keep={KEEP}): {selected}")

    allidx = list(range(len(pit_snaps)))
    rows = []
    for w in BW._weight_grid(selected, LEVELS):
        row = BC.eval_config(w, pit_snaps, allidx, cost30, TOPN30)
        rows.append(row)
    rows.sort(key=avg_sharpe, reverse=True)
    top10 = rows[:10]
    _log(f"topn=30 샤프평균 상위 10개(조합 총 {len(rows)}개 중):")
    for i, r in enumerate(top10, 1):
        _log(f"  #{i} {BW._wstr(r['weights'])} 평균샤프={avg_sharpe(r):.3f} "
            f"6M초과={r.get('excess_6m')}%p 12M초과={r.get('excess_12m')}%p")

    # ---- topn=8 실제 라이브 조건으로 전부 재평가(같은 panel/pit/funds 재사용, 판넬 재다운로드 없음) ----
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost8 = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    sector_of = SP._sector_of_factory()
    cross = T.build_panels(panel)

    results = []
    for i, r in enumerate(top10, 1):
        w = r["weights"]
        decisions = []
        for p in range(BW.LOOKBACK, len(panel) - 1, BP.MONTH):
            ranked = SP._select_basket_live_clip(panel, p, funds, cross, pit, w, BP.POOL_SIZE)
            if ranked:
                decisions.append((p, ranked))
        nav = BP.simulate(panel, ma200, decisions, TOPN8, cost8, ma200_backup=False,
                          sector_of=sector_of, sector_cap=2)
        if nav is None:
            _log(f"  #{i} {BW._wstr(w)}: topn=8 NAV 산출 실패 — 건너뜀")
            continue
        idx = nav.index.intersection(spy.reindex(nav.index).ffill().dropna().index)
        nav_a = nav.reindex(idx); nav_a = nav_a / nav_a.iloc[0]
        spy_a = spy.reindex(idx).ffill(); spy_a = spy_a / spy_a.iloc[0]
        s8 = CS.stats(nav_a)
        results.append({"rank_topn30_by_sharpe": i, "weights": w,
                        "topn30": {"avg_sharpe": round(avg_sharpe(r), 3),
                                  "excess_6m": r.get("excess_6m"), "excess_12m": r.get("excess_12m"),
                                  "win_6m": r.get("win_6m"), "win_12m": r.get("win_12m"),
                                  "sharpe_6m": r.get("sharpe_6m"), "sharpe_12m": r.get("sharpe_12m")},
                        "topn8_live_config": s8})
        _log(f"  #{i} {BW._wstr(w)} → topn8: CAGR {s8['cagr_pct']}% 샤프 {s8['sharpe']} "
            f"MDD {s8['mdd_pct']}%")

    s_spy8 = CS.stats(spy_a)
    _log(f"[SPY 동일구간] CAGR {s_spy8['cagr_pct']}% 샤프 {s_spy8['sharpe']} MDD {s_spy8['mdd_pct']}%")

    results_by_topn8 = sorted(results, key=lambda x: x["topn8_live_config"]["sharpe"], reverse=True)
    _log("\ntopn=8 샤프 기준 재정렬:")
    for j, r in enumerate(results_by_topn8, 1):
        _log(f"  topn8 #{j}(topn30 원순위 #{r['rank_topn30_by_sharpe']}) "
            f"{BW._wstr(r['weights'])}: CAGR {r['topn8_live_config']['cagr_pct']}% "
            f"샤프 {r['topn8_live_config']['sharpe']} MDD {r['topn8_live_config']['mdd_pct']}%")

    payload = {"years": YEARS, "keep": KEEP, "levels": LEVELS, "selected_factors": selected,
              "n_grid_combos": len(rows), "spy_topn8_window": s_spy8,
              "top10_topn30_reevaluated_at_topn8": results}
    os.makedirs("output", exist_ok=True)
    with open("output/top10_topn30_vs_topn8.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/top10_topn30_vs_topn8.json")


if __name__ == "__main__":
    main()
