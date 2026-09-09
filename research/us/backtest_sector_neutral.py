#!/usr/bin/env python3
"""
backtest_sector_neutral.py — 섹터중립 z-score + 넓은 팩터 후보군 재검증 (지호 님 요청, 2026-07).

메인 검증(backtest_costs.py)은 두 가지 단순화를 뒀다:
  1) z-score를 S&P500 전체 종목 한 번에 계산 — 업종별 사업구조 차이(고마진 소프트웨어 vs
     저마진 소매 등)를 구분하지 않음 → 특정 업종에 암묵적으로 쏠릴 위험.
  2) 팩터 후보를 IC 상위 6개(+mom12_1 강제포함)로 한정 — mom6(모멘텀, IC+0.043)처럼 퀄리티
     군집과 안 겹치면서 기댓값에 보탤 수 있는 팩터가 후보에도 못 들어감.

이 스크립트는 그 두 가지를 바꿔서 다시 검증한다:
  · z-score를 섹터(GICS) 내에서 계산(섹터 표본이 너무 작으면 전체 z로 폴백) —
    ⚠ 섹터 분류는 현재(위키피디아) 기준 1회 스냅샷을 10년 전체에 그대로 적용한다(비PIT
    한계 — SCORE_MODEL_DESIGN.md 부록 B와 동일한 종류의 한계, 결과에 그대로 명기).
  · 후보를 IC>0 상위 9개로 넓힘(mom12_1 강제포함 규칙 없이 순수 IC 순위) — mom6·cop·
    roa·div_yield 등이 자연히 포함됨.
메인 게이트 파일(backtest_costs.py·overfit_stats.py의 output/*.json)은 건드리지 않고
별도 파일에 저장 — 이 실험이 실제로 더 나은 결과를 보이면(PBO/DSR 통과) 그때 메인 전략
교체를 검토할 근거가 된다.

실행: python backtest_sector_neutral.py --years 10 --oos 0.4
      python backtest_sector_neutral.py --self-test
결과: output/backtest_sector_neutral_compare.json
      output/trial_returns_sector_neutral.json
      output/pbo_report_sector_neutral.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS

COMPARE_PATH = "output/backtest_sector_neutral_compare.json"
TRIAL_PATH = "output/trial_returns_sector_neutral.json"
REPORT_PATH = "output/pbo_report_sector_neutral.json"

KEEP = 9
LEVELS = (0, 1, 2)
MIN_SECTOR_GROUP = 10   # 섹터 표본이 이보다 적으면 전체(글로벌) z로 폴백


def _log(m): print(f"[섹터중립] {m}", file=sys.stderr)


def _pick_broad(ic_sorted, keep=KEEP):
    """mom12_1 강제포함 규칙 없이 순수 IC>0 상위 keep개 — mom6 등 비중복 팩터가 자연 포함되게."""
    sel = [f for f, ic in ic_sorted if ic > 0][:keep]
    return sel if len(sel) >= 2 else [f for f, _ in ic_sorted[:2]]


def _z_sector_neutral(raw: pd.DataFrame, sector_map: dict, min_group=MIN_SECTOR_GROUP) -> pd.DataFrame:
    """섹터 내 상대 z-score. 섹터 표본이 min_group 미만이면 전체(글로벌) z로 폴백."""
    sectors = pd.Series({s: sector_map.get(s, "") for s in raw.index})
    out = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    for col in raw.columns:
        col_data = raw[col]
        global_z = BW._z(col_data)
        z = pd.Series(index=raw.index, dtype=float)
        for sec, idx in sectors.groupby(sectors).groups.items():
            sub = col_data.loc[idx]
            if sec and len(sub.dropna()) >= min_group:
                z.loc[idx] = BW._z(sub)
            else:
                z.loc[idx] = global_z.loc[idx]
        out[col] = z
    return out.fillna(0.0)


def _filter_snaps_sector_neutral(snaps, pit, sector_map, min_group=MIN_SECTOR_GROUP):
    """backtest_costs._filter_snaps(mode='pit')와 동일 구조, z만 섹터중립으로 교체."""
    out, covs = [], []
    for s in snaps:
        members = BC.membership_asof(pit, s["date"])
        idx = s["raw"].index.intersection(members)
        if len(members):
            covs.append(len(idx) / len(members))
        if len(idx) < 10:
            continue
        raw = s["raw"].loc[idx]
        out.append({**s, "raw": raw, "z": _z_sector_neutral(raw, sector_map, min_group)})
    cov = {"mean": round(100 * float(np.mean(covs)), 1),
          "min": round(100 * float(np.min(covs)), 1)} if covs else None
    return out, cov


def run(panel, spy, funds, opens, pit, sector_map, cost, years=10, topn=30,
       rebal_days=63, keep=KEEP, levels=LEVELS, oos_frac=0.4):
    snaps = BC.build_snaps(panel, spy, funds, opens, rebal_days)
    pit_snaps, cov = _filter_snaps_sector_neutral(snaps, pit, sector_map)
    n_sector_hit = sum(1 for s in sector_map if s in panel.columns)
    _log(f"섹터 매핑 확보 {n_sector_hit}종목 · PIT 이벤트 {len(pit_snaps)}개 · "
         f"커버리지 평균 {cov['mean'] if cov else '-'}%")

    allidx = list(range(len(pit_snaps)))
    if oos_frac and 0 < oos_frac < 0.9:
        cut = int(len(pit_snaps) * (1 - oos_frac))
        train, test = allidx[:cut], allidx[cut:]
    else:
        train, test = allidx, None

    ic_sorted = BC._agg_ic(pit_snaps, train)
    selected = _pick_broad(ic_sorted, keep)
    _log(f"후보 팩터(IC>0 상위 {keep}개, 강제포함 없음): {selected}")

    grid = [BC.eval_config(w, pit_snaps, train, cost, topn) for w in BW._weight_grid(selected, levels)]
    best = max(grid, key=BW.score_config)
    oos = None
    if test:
        o = BC.eval_config(best["weights"], pit_snaps, test, cost, topn)
        oos = {"train": {k: best.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
              "test": {k: o.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
              "n_train": len(train), "n_test": len(test)}
        best = BC.eval_config(best["weights"], pit_snaps, allidx, cost, topn)

    # PBO 입력: 전체 이벤트에 대한 조합별 6m 순초과수익 행렬
    trials, matrix = [], []
    for w in BW._weight_grid(selected, levels):
        row, ev6 = BC.eval_config(w, pit_snaps, allidx, cost, topn, collect_6m=True)
        trials.append(BW._wstr(w)); matrix.append(ev6)
    n_ev = min(len(m) for m in matrix) if matrix else 0
    trial_data = {"horizon": "6m", "universe": "pit_sector_neutral", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": BW.TD["6m"],
                 "dates": [pit_snaps[i]["date"] for i in range(n_ev)],
                 "trials": trials, "excess_returns": [m[:n_ev] for m in matrix]}
    os.makedirs("output", exist_ok=True)
    with open(TRIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    rpt = OS.analyze(trial_data, save=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2)

    payload = {"as_of": panel.index[-1].date().isoformat(),
              "method": "섹터중립 z-score + IC상위9(강제포함 없음) 그리드",
              "n_sector_mapped": n_sector_hit, "pit_coverage_pct": cov,
              "candidate_factors": selected, "n_trials": len(trials), "n_events": n_ev,
              "best": best, "oos": oos,
              "pbo": rpt["pbo"]["pbo"], "pbo_verdict": rpt["pbo_verdict"],
              "dsr": rpt["dsr"].get("dsr"), "dsr_verdict": rpt["dsr_verdict"],
              "passed": rpt["passed"],
              "limitations": ["섹터(GICS) 분류는 현재 위키피디아 스냅샷 1회를 10년 전체에 "
                             "동일 적용(비PIT) — 과거 시점 실제 섹터와 다를 수 있음",
                             f"섹터 표본이 {MIN_SECTOR_GROUP}종목 미만이면 전체(글로벌) z로 폴백"]}
    with open(COMPARE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {COMPARE_PATH} · {TRIAL_PATH} · {REPORT_PATH} "
         f"(최선: {BW._wstr(best['weights'])} · PBO {rpt['pbo']['pbo']:.1%} · DSR {rpt['dsr'].get('dsr')})")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 섹터중립 z-score + 넓은 후보군 로직 점검")
    panel, spy, funds, opens = BW._synthetic()
    pit = BC._synthetic_pit(panel)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    rng = np.random.default_rng(5)
    sectors = ["Tech", "Health", "Financials"]
    sector_map = {s: rng.choice(sectors) for s in panel.columns}

    raw = pd.DataFrame({"f1": rng.normal(0, 1, len(panel.columns))}, index=panel.columns)
    z = _z_sector_neutral(raw, sector_map, min_group=5)
    assert z.shape == raw.shape
    for sec in sectors:
        members = [s for s in panel.columns if sector_map[s] == sec]
        if len(members) >= 5:
            assert abs(float(z.loc[members, "f1"].mean())) < 1.0, "섹터 내 평균이 0 근처여야 함"

    payload = run(panel, spy, funds, opens, pit, sector_map, cost,
                 years=8, topn=15, rebal_days=63, keep=6, oos_frac=0.4)
    assert payload["n_trials"] > 0 and payload["n_events"] > 4
    assert "pbo" in payload and "dsr" in payload
    _log(f"[self-test] 통과: 후보 {payload['candidate_factors']} · 이벤트 {payload['n_events']}개")


def main():
    ap = argparse.ArgumentParser(description="섹터중립 z-score + 넓은 팩터 후보군 재검증")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--keep", type=int, default=KEEP)
    ap.add_argument("--oos", type=float, default=0.4)
    ap.add_argument("--market", default="us", choices=["us", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    import sp500_daily_report as R
    pit = BC.load_pit(args.pit_file)
    panel, spy, opens = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    sector_map = R.fetch_wikipedia_sectors()
    if not sector_map:
        _log("섹터 맵 확보 실패(네트워크/파싱) — 중단"); sys.exit(1)
    cost = BC.CostModel(args.market, args.commission_bps, args.slippage_bps)
    run(panel, spy, funds, opens, pit, sector_map, cost,
       years=args.years, topn=args.topn, rebal_days=args.rebal_days,
       keep=args.keep, oos_frac=args.oos)


if __name__ == "__main__":
    main()
