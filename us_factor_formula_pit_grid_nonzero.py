#!/usr/bin/env python3
"""
us_factor_formula_pit_grid_nonzero.py — 지호 님 요청(2026-07-17): "백필하고, 더 세부
수치를 파고들어보자. 기존에 샤프값이 높게 나왔던 조합을 바탕으로. 팩터 세 개 다 0은
아니어야함."

us_factor_formula_pit_grid_top8.py(topn8·무캡, 레벨 0~3)와 같은 PIT+백필 인프라를 그대로
쓰되:
  - 레벨을 1~5로 확장(0 배제 — 세 팩터 전부 반드시 nonzero, 동시에 더 세밀한 비율 탐색)
  - 백필 재확인: 잔여 102종목(전체 697 중 85.4% 커버리지)은 SEC 현재 활성 티커 목록
    (company_tickers.json·company_tickers_exchange.json 둘 다) 자체에 없어 티커 기반
    조회로는 추가 복구 불가 확인(인수합병·상장폐지로 사라진 종목) — 이번 스윕은 기존
    백필 상태 그대로 사용

실행: python us_factor_formula_pit_grid_nonzero.py
결과: output/us_factor_formula_pit_grid_nonzero.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import us_factor_formula_pit_sweep as PS
import us_factor_formula_pit_grid_top8 as G8

TOPN = 8
LEVELS = (1, 2, 3, 4, 5)   # 0 배제 — 세 팩터 전부 nonzero
FACTORS = PS.FACTORS


def _log(m): print(f"[PIT세밀nonzero] {m}", file=sys.stderr)


def run(years=10, save=True):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = PS.build_snaps(panel, spy, funds, pit)
    G8._SECTOR_MAP = PS.sector_of_map()

    grid = BW._weight_grid(FACTORS, LEVELS)
    _log(f"가중치 조합 {len(grid)}개(전부 3팩터 nonzero) x formulation 3종 = {len(grid)*3}개 시행")

    trials = []
    for w in grid:
        for rd_mode in ("raw", "rank", "qgate"):
            excess, max_sec = G8.eval_trial_nocap(snaps, w, rd_mode)
            if sum(1 for v in excess if v == v) < 10:
                continue
            label = "·".join(f"{k}{v}" for k, v in w.items() if v) + f"[{rd_mode}]"
            trials.append({"label": label, "weights": w, "rd_mode": rd_mode,
                          "excess": excess, "max_sec": max_sec})

    if len(trials) < 2:
        raise RuntimeError("시행 부족")

    clean_matrix = [[v if v == v else 0.0 for v in t["excess"]] for t in trials]
    n_ev_full = min(len(m) for m in clean_matrix)
    trial_data = {"horizon": "us_factor_pit_nonzero", "universe": "sp500_pit",
                 "cost": {"approx_bps": PS.BW_COST*10000},
                 "rebal_days": PS.REBAL_DAYS, "hold_days": PS.TD_DAYS,
                 "dates": [s["date"] for s in snaps[:n_ev_full]],
                 "trials": [t["label"] for t in trials],
                 "excess_returns": [m[:n_ev_full] for m in clean_matrix]}
    rpt = OS.analyze(trial_data, save=False)

    rows = []
    for t in trials:
        ex = np.array([v for v in t["excess"] if v == v])
        ms = np.array(t["max_sec"])
        rows.append({"label": t["label"], "weights": t["weights"], "rd_mode": t["rd_mode"],
                    "n": len(ex), "excess_6m_mean_pct": round(100*float(ex.mean()), 2),
                    "excess_6m_sharpe": round(float(ex.mean()/ex.std()*math.sqrt(252/PS.TD_DAYS)), 2) if ex.std() > 0 else 0.0,
                    "avg_max_sector_in_top8": round(float(ms.mean()), 2)})
    rows.sort(key=lambda r: r["excess_6m_sharpe"], reverse=True)

    live_label = "int_gp_assets1·rd_mktcap2·shareholder_yield2[raw]"
    live_row = next((r for r in rows if r["label"] == live_label), None)
    live_rank = next((i+1 for i, r in enumerate(rows) if r["label"] == live_label), None)

    _log("상위 20개 시행(6M 초과수익 샤프 기준, 세 팩터 전부 nonzero):")
    for r in rows[:20]:
        _log(f"  {r['label']:50s} 초과6M {r['excess_6m_mean_pct']:+6.2f}%p · 샤프 {r['excess_6m_sharpe']:5.2f} · "
             f"쏠림 {r['avg_max_sector_in_top8']:.2f}")
    if live_row:
        _log(f"현행({live_label}): 초과6M {live_row['excess_6m_mean_pct']:+.2f}%p · "
             f"샤프 {live_row['excess_6m_sharpe']:.2f} · 쏠림 {live_row['avg_max_sector_in_top8']:.2f} · "
             f"순위 {live_rank}/{len(rows)}")

    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "sector_cap": None,
              "n_trials": len(trials), "grid_levels": LEVELS, "rows": rows,
              "live_config_label": live_label, "live_rank": live_rank,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "note": "레벨 1~5(0 배제) — 세 팩터 전부 nonzero인 조합만. 백필 잔여 102종목은 "
                      "SEC 활성 티커목록 자체에 없어 추가 복구 불가 확인."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_factor_formula_pit_grid_nonzero.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/us_factor_formula_pit_grid_nonzero.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


if __name__ == "__main__":
    run()
