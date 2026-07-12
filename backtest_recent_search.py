#!/usr/bin/env python3
"""
backtest_recent_search.py — 최근 구간에서 "여러 방법 중 최선"을 찾는 탐색(지호 님 요청, 2026-07).

탐색 축 3가지(가중치 조합 / 상위 N종목 / 리밸주기 — 보유기간은 build_snaps가 이미 계산해두는
1개월·3개월·6개월 컬럼을 재사용하므로 별도 재계산 없이 같이 비교됨):
  · 가중치: 최근 구간 내 스냅샷 IC 기준 상위 4개 팩터(backtest_weights._pick)를 골라 0/1/2 그리드
  · 상위 N: 4 / 10 / 20 / 30
  · 리밸주기: 주간(5거래일) / 격주(10거래일)

⚠ 경고(먼저 읽을 것) — SCORE_MODEL_DESIGN.md D4 그대로 재현되는 상황이다: "최적 조합을
고르는 행위 자체가 다중검정을 키운다." 게다가 표본 기간이 짧아(최근 1~2년) 원래도 독립
정보량(T_eff)이 작다(backtest_recent.py 단일 설정 테스트에서 n_eff 3~7 수준이었음). 그런
얇은 데이터에 수십~수백 개 조합을 태우면 "1등"은 순수 잡음일 위험이 크다 — 그래서 조합별
숫자표만이 아니라 리밸주기×보유기간 그룹별로 overfit_stats.analyze()를 돌려 PBO/DSR까지
같이 보고한다. group_pbo_dsr의 passed=false면 "1등"을 신뢰하지 말 것.

실행: python backtest_recent_search.py
      python backtest_recent_search.py --self-test
결과: output/backtest_recent_search.json (조합별 표 + 그룹별 PBO/DSR 요약)
      output/pbo_report_recent_<그룹>.json (그룹별 상세)
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS

OUT_PATH = "output/backtest_recent_search.json"
REPORT_PATH_TMPL = "output/pbo_report_recent_{tag}.json"

TOPN_GRID = [4, 10, 20, 30]
REBAL_GRID = [5, 10]     # 주간·격주(거래일)
HORIZONS = ["1m", "3m", "6m"]
KEEP = 4
LEVELS = (0, 1, 2)


def _log(m): print(f"[최근탐색] {m}", file=sys.stderr)


def _window_snaps(panel, spy, funds, opens, pit, rebal_days, start, end):
    snaps = BC.build_snaps(panel, spy, funds, opens, rebal_days)
    pit_snaps, cov = BC._filter_snaps(snaps, pit, "pit")
    win = [s for s in pit_snaps if start <= pd.Timestamp(s["date"]) <= end]
    return win, cov


def _eval_multi_horizon(w, snaps, idxs, cost, topn):
    """가중치 w·상위 topn 조합의 이벤트별 순초과수익(net−bench)을 1m/3m/6m 전부 계산."""
    wv = pd.Series(w); cols = list(w)
    out = {h: [] for h in HORIZONS}
    for i in idxs:
        s = snaps[i]; z = s["z"]
        top = (z[cols] * wv).sum(axis=1).sort_values(ascending=False).index[:topn]
        for h in HORIZONS:
            r = s["fwd"][h].reindex(top).dropna()
            if len(r):
                net = float(np.mean([cost.net(x) for x in r]))
                out[h].append(net - s["bench"][h])
            else:
                out[h].append(None)
    return out


def run_search(panel, spy, funds, opens, pit, cost, start_months_ago=24, end_months_ago=3,
              topn_grid=TOPN_GRID, rebal_grid=REBAL_GRID, keep=KEEP, levels=LEVELS,
              min_events=8):
    today = panel.index[-1]
    start = today - pd.DateOffset(months=start_months_ago)
    end = today - pd.DateOffset(months=end_months_ago)

    summary_rows = []
    group_reports = {}
    for rebal_days in rebal_grid:
        win_snaps, cov = _window_snaps(panel, spy, funds, opens, pit, rebal_days, start, end)
        if len(win_snaps) < min_events:
            _log(f"리밸 {rebal_days}d — 구간 내 이벤트 {len(win_snaps)}개(부족, 건너뜀)")
            continue
        idxs = list(range(len(win_snaps)))
        ic_sorted = BC._agg_ic(win_snaps, idxs)
        selected = BC._pick(ic_sorted, keep)
        min_n = min(len(s["raw"]) for s in win_snaps)
        labels, per_h_matrix = [], {h: [] for h in HORIZONS}
        for topn in topn_grid:
            if topn > min_n:
                continue
            for w in BW._weight_grid(selected, levels):
                res = _eval_multi_horizon(w, win_snaps, idxs, cost, topn)
                if any(any(v is None for v in res[h]) for h in HORIZONS):
                    continue
                label = f"rebal{rebal_days}d_top{topn}_" + BW._wstr(w)
                labels.append(label)
                row = {"rebal_days": rebal_days, "topn": topn, "weights": w, "label": label}
                for h in HORIZONS:
                    row[f"mean_excess_{h}_pct"] = round(100 * float(np.mean(res[h])), 2)
                    per_h_matrix[h].append([round(v, 6) for v in res[h]])
                summary_rows.append(row)

        dates = [s["date"] for s in win_snaps]
        for h in HORIZONS:
            M = per_h_matrix[h]
            if len(M) < 4:
                continue
            trial_data = {"horizon": h, "universe": "pit_recent_window", "cost": cost.describe(),
                         "rebal_days": rebal_days, "hold_days": BW.TD[h],
                         "dates": dates, "trials": labels[:len(M)], "excess_returns": M}
            try:
                rpt = OS.analyze(trial_data, save=False)
            except RuntimeError as e:
                _log(f"리밸{rebal_days}d·{h} — 중첩(embargo)이 너무 커서 PBO 계산 불가({e}) → 그룹 건너뜀")
                continue
            tag = f"rebal{rebal_days}d_{h}"
            group_reports[tag] = {"n_trials": len(M), "n_events": len(dates),
                                  "pbo": rpt["pbo"]["pbo"], "pbo_verdict": rpt["pbo_verdict"],
                                  "dsr": rpt["dsr"].get("dsr"), "dsr_verdict": rpt["dsr_verdict"],
                                  "passed": rpt["passed"], "best_trial": rpt["dsr"]["best_trial"]}
            os.makedirs("output", exist_ok=True)
            with open(REPORT_PATH_TMPL.format(tag=tag), "w", encoding="utf-8") as f:
                json.dump(rpt, f, ensure_ascii=False, indent=2)

    if not summary_rows:
        raise RuntimeError("유효한 조합이 하나도 없음 — 구간·그리드를 조정하세요.")
    summary_rows.sort(key=lambda r: r.get("mean_excess_3m_pct", -999), reverse=True)
    payload = {
        "as_of": today.date().isoformat(),
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "topn_grid": topn_grid, "rebal_grid_days": rebal_grid, "horizons": HORIZONS,
        "n_combos_total": len(summary_rows), "top10_by_mean_excess_3m": summary_rows[:10],
        "group_pbo_dsr": group_reports,
        "warning": ("탐색 조합 수가 많고 표본 기간이 짧아 다중검정 과최적화 위험이 큼 — "
                   "group_pbo_dsr의 passed=false면 표에서 1등처럼 보이는 조합을 신뢰하지 "
                   "말 것(SCORE_MODEL_DESIGN.md D4). 그룹(리밸×보유기간)마다 별도 검정이며, "
                   "그룹 간 비교 자체도 추가 다중검정이라 완전히 보정되진 않음.")}
    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {OUT_PATH} (조합 {len(summary_rows)}개 · 그룹 {len(group_reports)}개)")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 탐색 엔진(가중치×topN×리밸) 점검")
    panel, spy, funds, opens = BW._synthetic()
    pit = BC._synthetic_pit(panel)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    payload = run_search(panel, spy, funds, opens, pit, cost,
                         start_months_ago=30, end_months_ago=3,
                         topn_grid=[6, 15], rebal_grid=[5, 10], min_events=6)
    assert payload["n_combos_total"] > 0
    assert payload["group_pbo_dsr"], "그룹별 PBO/DSR 리포트가 비어있음"
    for tag, g in payload["group_pbo_dsr"].items():
        assert 0.0 <= g["pbo"] <= 1.0, f"{tag} PBO 범위 이상: {g['pbo']}"
    _log(f"[self-test] 통과: 조합 {payload['n_combos_total']}개 · 그룹 {len(payload['group_pbo_dsr'])}개")


def main():
    ap = argparse.ArgumentParser(description="최근 구간 그리드 탐색(가중치×상위N×리밸) + 그룹별 PBO/DSR")
    ap.add_argument("--years", type=float, default=3.5)
    ap.add_argument("--start-months-ago", type=int, default=24)
    ap.add_argument("--end-months-ago", type=int, default=3)
    ap.add_argument("--market", default="us", choices=["us", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    pit = BC.load_pit(args.pit_file)
    panel, spy, opens = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel(args.market, args.commission_bps, args.slippage_bps)
    run_search(panel, spy, funds, opens, pit, cost,
              start_months_ago=args.start_months_ago, end_months_ago=args.end_months_ago)


if __name__ == "__main__":
    main()
