#!/usr/bin/env python3
"""
kr_top10_topn30_vs_live.py — 한국판 top10_topn30_vs_topn8.py. topn=30(POOL_KR) 넓은
평가풀에서 IC기반 가중치 그리드 탐색 → 샤프평균 상위 10개 → 실제 라이브 조건(topn=5·
섹터캡 없음·ma200_backup=False)으로 재검증 (2026-08-23, 지호 님 요청).

라이브 스코어: z(1/PER)+z(1/PBR)+z(배당수익률) = value:1+pbr_inv:1+div_yield:1(동일가중,
kr_stocks.py). backtest_kr.build_kr_snaps()가 만드는 snaps는 backtest_costs.eval_config가
그대로 쓸 수 있는 구조(문서화됨)라 미국과 동일한 _agg_ic/_pick/_weight_grid/eval_config를
그대로 재사용한다. 가용 팩터: mom12_1·mom6·hi52_prox·low_vol(가격) + value·pbr_inv·
div_yield·roe(펀더멘탈) + frgn_flow·inst_flow·size(수급).

실행: python kr_top10_topn30_vs_live.py
결과: output/kr_top10_topn30_vs_live.json
"""
from __future__ import annotations
import os, sys, json

import backtest_costs as BC
import backtest_weights as BW
import backtest_kr as BK
import backtest_portfolio as BP
import core_satellite_kr as CS

KEEP, LEVELS = 5, (0, 1, 2, 3)
POOL_KR = 30
TOPN_LIVE = 5          # kr_stocks.py MAX_HOLD(2026-07-16부로 5)
LIVE_WEIGHTS = {"value": 1, "pbr_inv": 1, "div_yield": 1}


def _log(m): print(f"[KR top10]{m}", file=sys.stderr)


def avg_sharpe(row):
    vals = [row.get(f"sharpe_{h}") for h in ("3m", "6m", "12m")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def decisions_for_weights(panel, snaps, weights, topn):
    """backtest_kr_strategies.build_decisions와 동일 로직(roe>0 필터만), 임의 가중치 인자화."""
    pos_by_date = {d.date().isoformat(): i for i, d in enumerate(panel.index)}
    out = []
    for s in snaps:
        p = pos_by_date.get(s["date"])
        if p is None:
            continue
        roe = s["raw"]["roe"]
        pool = roe[roe > 0].index
        if len(pool) < topn + 2:
            continue
        z = s["z"].loc[pool]
        score = sum(z[f] * wt for f, wt in weights.items() if f in z.columns)
        out.append((p, list(score.sort_values(ascending=False).index[:POOL_KR])))
    return out


def main():
    from benchmarks_kr import load_research_data, build_benchmarks
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, n_pit_ok, n_total = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                                 rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개(PIT 확인 {n_pit_ok}/{n_total})")
    cost30 = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    allidx = list(range(len(snaps)))
    ic_sorted = BC._agg_ic(snaps, allidx)
    selected = BC._pick(ic_sorted, KEEP)
    _log(f"IC 순위: {ic_sorted} → 채택: {selected}")

    live_row = BC.eval_config(LIVE_WEIGHTS, snaps, allidx, cost30, POOL_KR)
    _log(f"[라이브 1:1:1 @topn=30] 평균샤프={avg_sharpe(live_row):.3f} "
        f"6M초과={live_row.get('excess_6m')}%p 12M초과={live_row.get('excess_12m')}%p")

    rows = []
    for w in BW._weight_grid(selected, LEVELS):
        rows.append(BC.eval_config(w, snaps, allidx, cost30, POOL_KR))
    rows.sort(key=avg_sharpe, reverse=True)
    top10 = rows[:10]
    _log(f"topn=30 샤프평균 상위10(전체 {len(rows)}조합 중):")
    for i, r in enumerate(top10, 1):
        _log(f"  #{i} {BW._wstr(r['weights'])} 평균샤프={avg_sharpe(r):.3f} "
            f"6M초과={r.get('excess_6m')}%p 12M초과={r.get('excess_12m')}%p")

    # ---- 실제 라이브 조건(topn=5·섹터캡 없음·ma200_backup=False)으로 재검증 ----
    navs_bm = build_benchmarks(panel, membership, mktcaps, bench)
    b1 = navs_bm["B1_kospi200"].dropna()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost_live = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    def live_stats(weights, label):
        decisions = decisions_for_weights(panel, snaps, weights, TOPN_LIVE)
        nav = BP.simulate(panel, ma200, decisions, TOPN_LIVE, cost_live, ma200_backup=False)
        if nav is None:
            _log(f"  {label}: NAV 실패"); return None
        idx = nav.index.intersection(b1.index)
        nav_a = nav.reindex(idx); nav_a = nav_a / nav_a.iloc[0]
        s = CS.stats(nav_a)
        _log(f"  {label} → topn=5(라이브): CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}%")
        return s, idx

    live_s, idx0 = live_stats(LIVE_WEIGHTS, "라이브1:1:1")
    b1_a = b1.reindex(idx0); b1_a = b1_a / b1_a.iloc[0]
    s_b1 = CS.stats(b1_a)
    _log(f"[B1 코스피200 동일구간] CAGR {s_b1['cagr_pct']}% 샤프 {s_b1['sharpe']} MDD {s_b1['mdd_pct']}%")

    results = []
    for i, r in enumerate(top10, 1):
        out = live_stats(r["weights"], f"#{i}(topn30랭크{i})")
        if out:
            s8, _ = out
            results.append({"rank_topn30": i, "weights": r["weights"],
                            "topn30": {"avg_sharpe": round(avg_sharpe(r), 3),
                                      "excess_6m": r.get("excess_6m"), "excess_12m": r.get("excess_12m")},
                            "topn5_live_config": s8})

    payload = {"keep": KEEP, "levels": LEVELS, "selected_factors": selected,
              "n_grid_combos": len(rows), "b1_kospi200": s_b1,
              "live_1_1_1_topn30": live_row, "live_1_1_1_topn5": live_s,
              "top10_topn30_reevaluated_at_topn5": results}
    os.makedirs("output", exist_ok=True)
    with open("output/kr_top10_topn30_vs_live.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    _log("저장: output/kr_top10_topn30_vs_live.json")


if __name__ == "__main__":
    main()
