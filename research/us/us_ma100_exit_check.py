#!/usr/bin/env python3
"""
us_ma100_exit_check.py — 지호 님 질문(2026-09-23): "100일선 이탈하면 파는 게 낫나?"
현재 라이브 매도 규칙은 6개월 정기재평가뿐(가격 개입 없음 — holdings.py 참고, 한국은
이미 2026-07-15에 가격개입형 매도규칙이 전부 성과를 깎는다는 걸 검증했지만 미국은
100일선 필터를 막 추가한 김에 "진입 필터"가 아니라 "이탈 시 매도"로도 검증 안 해봤음).

방법: floor+100일선(진입 시점) 통과한 topn8을, 매 종목마다 진입일부터 126거래일(6개월)
동안 일별로 따라가며:
  a) 기준: 무조건 126거래일 보유(현행 라이브와 동일)
  b) 100일선 이탈 즉시 매도: 보유 중 처음으로 종가<100일선인 날 매도, 그 이후 잔여
     기간은 현금(0%) 취급 — "판 게 나은지"를 같은 126일 창 안에서 공정 비교.
이벤트(스냅샷×종목) 단위로 페어드 비교 + 몇 % 종목이 실제로 이탈을 겪는지도 같이 보고.

실행: python -m research.us.us_ma100_exit_check [--years 10]
결과: output/us_ma100_exit_check.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import research.us.us_factor_formula_pit_sweep as PS
from research.us.us_factor_value_vs_rank import _composite
from research.us.us_momentum_overlay import _mom_features

TOPN = 8
FLOOR = 3.25
HOLD_DAYS = 126
OUT_PATH = "output/us_ma100_exit_check.json"


def _log(m): print(f"[US100일선매도]  {m}", file=sys.stderr)


def run(years: float = 10, save: bool = True) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    snaps = PS.build_snaps(panel, spy, funds, pit)
    _log(f"스냅샷 {len(snaps)}개")

    ma100_panel = panel.rolling(100, min_periods=100).mean()

    hold_ex, exit_ex, breached_flags = [], [], []
    for snap in snaps:
        raw, fwd, bench = snap["raw"], snap["fwd"], snap["bench"]
        score = _composite(raw).reindex(fwd.index).dropna()
        score = score[score >= FLOOR]
        if len(score) < TOPN:
            continue
        mom = _mom_features(panel, snap["date"], score.index)
        above = mom["ma100_gap"] > 0 if "ma100_gap" in mom.columns else pd.Series(True, index=score.index)
        score = score[above.reindex(score.index).fillna(False)]
        if len(score) < TOPN:
            continue
        top = score.sort_values(ascending=False).index[:TOPN]

        p = panel.index.get_indexer([pd.Timestamp(snap["date"])], method="pad")[0]
        e = p + 1
        if e + HOLD_DAYS >= len(panel):
            continue

        hold_rets, exit_rets, breached = [], [], []
        for sym in top:
            if sym not in panel.columns:
                continue
            path = panel.iloc[e:e + HOLD_DAYS + 1][sym]
            ma_path = ma100_panel.iloc[e:e + HOLD_DAYS + 1][sym]
            if path.isna().any() or len(path) < HOLD_DAYS + 1:
                continue
            entry = float(path.iloc[0])
            hold_rets.append(float(path.iloc[-1]) / entry - 1.0)
            below = (path < ma_path) & ma_path.notna()
            first_breach = below[below].index.min() if below.any() else None
            if first_breach is not None:
                exit_price = float(path.loc[first_breach])
                exit_rets.append(exit_price / entry - 1.0)
                breached.append(1)
            else:
                exit_rets.append(hold_rets[-1])
                breached.append(0)
        if not hold_rets:
            continue
        hold_ex.append(float(np.mean(hold_rets)) - bench)
        exit_ex.append(float(np.mean(exit_rets)) - bench)
        breached_flags.append(float(np.mean(breached)))

    h, x = np.array(hold_ex), np.array(exit_ex)
    diff = x - h
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else None
    t = float(diff.mean() / se) if se else None
    payload = {"n_events": len(h), "hold_days": HOLD_DAYS,
              "always_hold_mean_excess_pct": round(100 * float(h.mean()), 3),
              "always_hold_win_rate_pct": round(100 * float((h > 0).mean()), 1),
              "exit_on_ma100_break_mean_excess_pct": round(100 * float(x.mean()), 3),
              "exit_on_ma100_break_win_rate_pct": round(100 * float((x > 0).mean()), 1),
              "paired_diff_mean_pct": round(100 * float(diff.mean()), 3),
              "paired_t_stat": round(t, 3) if t is not None else None,
              "pct_positions_that_breached_ma100_during_hold": round(100 * float(np.mean(breached_flags)), 1),
              "method": ("진입 시점 floor+100일선 통과 topn8을 126거래일 보유 vs 보유 중 "
                        "100일선 이탈 시 그 시점 가격으로 매도 후 잔여기간 현금(0%) 취급 — "
                        "같은 126일 창 안에서 페어드 비교. 종목별 일별 추적.")}
    _log(f"[결과] n={payload['n_events']} 무조건보유 {payload['always_hold_mean_excess_pct']}%p"
        f"(승률{payload['always_hold_win_rate_pct']}%) vs 이탈시매도 "
        f"{payload['exit_on_ma100_break_mean_excess_pct']}%p(승률"
        f"{payload['exit_on_ma100_break_win_rate_pct']}%) 페어드diff={payload['paired_diff_mean_pct']}%p "
        f"t={payload['paired_t_stat']} 보유중이탈비율={payload['pct_positions_that_breached_ma100_during_hold']}%")

    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    args = ap.parse_args()
    run(years=args.years)


if __name__ == "__main__":
    main()
