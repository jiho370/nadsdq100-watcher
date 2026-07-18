#!/usr/bin/env python3
"""
us_factor_formula_pit_grid_top8.py — 지호 님 요청(2026-07-17): "topn8 무캡으로 여러 조건
세밀하고 다양하게 백테스트". 지금까지 본 조건 2개(topn8+섹터캡2=쏠림이 캡으로 고정돼
공식 비교 무의미, topn30+무캡=원 인증 잣대지만 라이브 풀사이즈 8과 다름)와 달리 이건
"라이브 그대로의 풀사이즈(8)에서, 캡 없이 순수하게 공식 자체의 쏠림·성과 특성"을 보는
새 조건 — 지금까지 테스트 안 했음.

us_factor_formula_pit_sweep.py의 PIT 인프라(펀더멘탈 백필 완료된 패널·시점별 실제
멤버십 필터)를 그대로 재사용하되:
  - us_factor_formula_sweep.py(첫 시도, 생존편향 패널이라 폐기)와 같은 넓은 그리드
    (3팩터 × 레벨 0~3 × rd_mktcap formulation 3종 = 최대 147시행)를 이번엔 올바른
    PIT+백필 패널로 재실행
  - topn=8, sector_cap=None 고정
  - 전 시행을 하나로 등록해 PBO/DSR 정직 처리, 섹터쏠림(평균 최대섹터쏠림)도 병기

실행: python us_factor_formula_pit_grid_top8.py
결과: output/us_factor_formula_pit_grid_top8.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import us_factor_formula_pit_sweep as PS   # build_snaps/sector_of_map/_rd_variant 재사용

TOPN = 8
LEVELS = (0, 1, 2, 3)
FACTORS = PS.FACTORS   # ["int_gp_assets", "rd_mktcap", "shareholder_yield"]


def _log(m): print(f"[PIT그리드top8] {m}", file=sys.stderr)


def eval_trial_nocap(snaps, weights: dict, rd_mode: str):
    excess, max_sec = [], []
    from collections import Counter
    sector_map = _SECTOR_MAP
    for snap in snaps:
        raw = snap["raw"]
        z_gp = ((raw["int_gp_assets"] - raw["int_gp_assets"].mean()) / raw["int_gp_assets"].std()).clip(-3, 3).fillna(0.0)
        z_sy = ((raw["shareholder_yield"] - raw["shareholder_yield"].mean()) / raw["shareholder_yield"].std()).clip(-3, 3).fillna(0.0)
        z_rd = PS._rd_variant(raw["rd_mktcap"], z_gp, z_sy, rd_mode)
        comp = weights.get("int_gp_assets", 0) * z_gp + weights.get("rd_mktcap", 0) * z_rd \
             + weights.get("shareholder_yield", 0) * z_sy
        top = comp.sort_values(ascending=False).index[:TOPN]
        r = snap["fwd"].reindex(top).dropna()
        if len(r):
            excess.append(float(r.mean()) - PS.BW_COST - snap["bench"])
        else:
            excess.append(np.nan)
        c = Counter(sector_map.get(s, "(미상)") for s in top)
        max_sec.append(c.most_common(1)[0][1] if c else 0)
    return excess, max_sec


_SECTOR_MAP = {}


def run(years=10, save=True):
    global _SECTOR_MAP
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _SECTOR_MAP = PS.sector_of_map()

    grid = BW._weight_grid(FACTORS, LEVELS)
    _log(f"가중치 조합 {len(grid)}개 x formulation 3종 = {len(grid)*3}개 시행 (topn={TOPN}, 무캡)")

    trials = []
    for w in grid:
        for rd_mode in ("raw", "rank", "qgate"):
            excess, max_sec = eval_trial_nocap(snaps, w, rd_mode)
            if sum(1 for v in excess if v == v) < 10:
                continue
            label = "·".join(f"{k}{v}" for k, v in w.items() if v) + f"[{rd_mode}]"
            trials.append({"label": label, "weights": w, "rd_mode": rd_mode,
                          "excess": excess, "max_sec": max_sec})

    if len(trials) < 2:
        raise RuntimeError("시행 부족")

    n_ev = min(sum(1 for v in t["excess"] if v == v) for t in trials)
    # NaN 없는 공통 길이로 정렬(간단화를 위해 NaN은 0으로 대체 — overfit_stats 입력용)
    clean_matrix = [[v if v == v else 0.0 for v in t["excess"]] for t in trials]
    n_ev_full = min(len(m) for m in clean_matrix)
    trial_data = {"horizon": "us_factor_pit_grid_top8", "universe": "sp500_pit",
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

    _log("상위 15개 시행(6M 초과수익 샤프 기준):")
    for r in rows[:15]:
        _log(f"  {r['label']:45s} 초과6M {r['excess_6m_mean_pct']:+6.2f}%p · 샤프 {r['excess_6m_sharpe']:5.2f} · "
             f"평균최대섹터쏠림(topn8) {r['avg_max_sector_in_top8']:.2f}")
    if live_row:
        _log(f"현행 라이브({live_label}): 초과6M {live_row['excess_6m_mean_pct']:+.2f}%p · "
             f"샤프 {live_row['excess_6m_sharpe']:.2f} · 쏠림 {live_row['avg_max_sector_in_top8']:.2f} · "
             f"순위 {live_rank}/{len(rows)}")

    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "sector_cap": None,
              "n_trials": len(trials), "grid_levels": LEVELS, "rows": rows,
              "live_config_label": live_label, "live_rank": live_rank,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "note": "PIT 패널+펀더멘탈 백필 완료 상태에서 topn=8·무캡 조건 전체 그리드. "
                      "topn8+섹터캡2(쏠림 강제고정)·topn30+무캡(원 인증 잣대)과 다른 세 번째 "
                      "조건 — 라이브 그대로의 풀사이즈에서 순수 공식 효과 확인용."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_factor_formula_pit_grid_top8.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/us_factor_formula_pit_grid_top8.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


if __name__ == "__main__":
    run()
