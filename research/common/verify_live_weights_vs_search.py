#!/usr/bin/env python3
"""
verify_live_weights_vs_search.py — 라이브 가중치(1:2:2)를 backtest_costs.py의 자동탐색
결과와 같은 데이터셋 위에서 직접(apples-to-apples) 비교 (2026-08-22~23, 지호 님 요청:
"다시 백테스트 돌려서 최적의 전략을 찾아볼래").

배경: backtest_costs.py의 legacy/pit_legacy_weights/pit_best 세 필드 전부 매 실행마다
IC 상위 팩터를 다시 골라 자동탐색하는 것이라, 어느 것도 라이브 1:2:2(int_gp_assets:1+
rd_mktcap:2+shareholder_yield:2)를 그대로 고정해 평가하지 않는다(§9-K-1 참고). 이 스크립트가
`backtest_costs.build_panel_pit`→`build_snaps`→`_filter_snaps`로 똑같은 PIT 패널을 만들고,
라이브 가중치를 `run_scenario(..., fixed_weights=...)`로 직접 고정평가해 자동탐색 결과와
나란히 비교한다.

실행: python verify_live_weights_vs_search.py --years 10 --keep 5 --levels 0,1,2,3
"""
from __future__ import annotations
import argparse, json, sys, math
import numpy as np
import backtest_costs as BC
import backtest_weights as BW

LIVE_WEIGHTS = {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}
# 2026-08-23 지호 님 지시: "승률도 중요하지만 순초과·샤프, 특히 샤프를 우선" — overfit_stats.py의
# PBO/DSR은 내부적으로 6개월 초과수익의 샤프로 이미 후보를 뽑는다(pbo_cscv._sharpe_all) —
# score_config(순초과 중심, 샤프 미반영)와는 다른 기준. 이번 961조합 탐색에서 그 샤프 기준
# 최고점으로 나온 조합을 직접 대조 평가한다.
SHARPE_BEST_CANDIDATE = {"int_gp_assets": 1, "shareholder_yield": 2, "roa": 1}


def _log(m): print(f"[라이브가중치검증] {m}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--levels", default="0,1,2,3")
    args = ap.parse_args()

    pit = BC.load_pit()
    panel, spy, opens = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", 0.0, 5.0)
    snaps = BC.build_snaps(panel, spy, funds, opens, args.rebal_days)
    pit_snaps, cov = BC._filter_snaps(snaps, pit, "pit")
    _log(f"PIT 이벤트 {len(pit_snaps)}회 · 커버리지 평균 {cov['mean']}%(최저 {cov['min']}%)")

    levels = tuple(int(x) for x in args.levels.split(","))
    allidx = list(range(len(pit_snaps)))

    live_row, live_ev6 = BC.eval_config(LIVE_WEIGHTS, pit_snaps, allidx, cost, args.topn, collect_6m=True)
    best_row, ic_sorted, _, _ = BC.run_scenario(pit_snaps, cost, args.topn, args.keep, levels)
    sharpe_row, sharpe_ev6 = BC.eval_config(SHARPE_BEST_CANDIDATE, pit_snaps, allidx, cost,
                                            args.topn, collect_6m=True)

    def show(name, row):
        _log(f"[{name}] 가중치={BW._wstr(row['weights'])} 회전율={row.get('turnover')}%")
        for h in ("3m", "6m", "12m"):
            _log(f"    {h}: 승률={row.get('win_'+h)}% 순초과={row.get('excess_'+h)}%p "
                f"샤프={row.get('sharpe_'+h)}")

    show("라이브 1:2:2(고정)", live_row)
    show("score_config 최우수(keep={})".format(args.keep), best_row)
    show("DSR/샤프 최우수(int_gp_assets1·shareholder_yield2·roa1)", sharpe_row)
    _log(f"IC 상위 10: {ic_sorted[:10]}")

    # ------------------------- 짝지은(paired) 블록부트스트랩: 라이브 vs 샤프최우수 -------------------------
    # §2 "0%와 65%" 비교와 동일 방법론(같은 리샘플을 두 후보 모두에 동시 적용 — 공통 시장위험
    # 공유를 반영). 이벤트 32회뿐이라 블록=2이벤트, 5000회.
    a, b = np.array(live_ev6), np.array(sharpe_ev6)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    rng = np.random.default_rng(7)
    block, n_boot = 2, 5000
    n_blocks = n // block
    diffs = np.empty(n_boot)          # 순초과6M(라이브) - 순초과6M(샤프후보)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([np.arange(s * block, s * block + block) for s in starts])[:n]
        diffs[i] = a[idx].mean() - b[idx].mean()
    ci = (round(float(np.percentile(diffs, 5)), 4), round(float(np.percentile(diffs, 95)), 4))
    pct_live_higher_excess = round(float((diffs > 0).mean()) * 100, 1)
    # 샤프 차이도 같은 리샘플로(리샘플된 6M초과수익 시리즈 자체의 평균/표준편차비)
    sharpe_diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([np.arange(s * block, s * block + block) for s in starts])[:n]
        sd_a, sd_b = a[idx].std(ddof=1), b[idx].std(ddof=1)
        sh_a = a[idx].mean() / sd_a if sd_a > 0 else 0.0
        sh_b = b[idx].mean() / sd_b if sd_b > 0 else 0.0
        sharpe_diffs[i] = sh_b - sh_a     # 양수 = 샤프후보가 라이브보다 샤프 높음
    sharpe_ci = (round(float(np.percentile(sharpe_diffs, 5)), 4), round(float(np.percentile(sharpe_diffs, 95)), 4))
    pct_candidate_higher_sharpe = round(float((sharpe_diffs > 0).mean()) * 100, 1)
    _log(f"[짝지은 부트스트랩, n={n}이벤트·block={block}·5000회] "
        f"6M순초과 차이(라이브-후보) 90%CI {ci} · 라이브가 더 높을 확률 {pct_live_higher_excess}%")
    _log(f"[짝지은 부트스트랩] 6M샤프 차이(후보-라이브) 90%CI {sharpe_ci} · "
        f"후보가 샤프 더 높을 확률 {pct_candidate_higher_sharpe}%")

    # ------------------------- 이상치(소수 대박 이벤트) 민감도 진단 -------------------------
    # 지호 님 질문: "과거 데이터라 편향일 수 있다 - 소수 대폭등 구간이 평균을 덮는 거 아니냐"
    def outlier_diag(name, arr):
        s = np.sort(arr)[::-1]           # 내림차순
        mean, med = float(arr.mean()), float(np.median(arr))
        top3_share = float(s[:3].sum() / arr.sum()) * 100 if arr.sum() != 0 else float("nan")
        # 최고 1~3개 이벤트 제외 시 평균이 얼마나 바뀌는지
        trimmed1 = float(np.delete(arr, arr.argmax()).mean())
        trimmed3 = float(s[3:].mean())
        skew = float(((arr - mean) ** 3).mean() / (arr.std(ddof=0) ** 3)) if arr.std(ddof=0) > 0 else 0.0
        _log(f"[이상치진단:{name}] 평균={mean*100:.2f}%p 중앙값={med*100:.2f}%p "
            f"최고1개={s[0]*100:.2f}%p 최저1개={s[-1]*100:.2f}%p 왜도={skew:.2f}")
        _log(f"    상위3개이벤트가 전체합의 {top3_share:.1f}% 차지 · "
            f"최고1개 제외시 평균 {trimmed1*100:.2f}%p(원래 {mean*100:.2f}%p) · "
            f"상위3개 제외시 평균 {trimmed3*100:.2f}%p")
        return {"mean_pct": round(mean*100,2), "median_pct": round(med*100,2),
                "max_pct": round(s[0]*100,2), "min_pct": round(s[-1]*100,2), "skew": round(skew,2),
                "top3_share_of_sum_pct": round(top3_share,1),
                "mean_excl_top1_pct": round(trimmed1*100,2), "mean_excl_top3_pct": round(trimmed3*100,2)}

    live_diag = outlier_diag("라이브1:2:2", a)
    cand_diag = outlier_diag("샤프후보(roa)", b)

    payload = {"live_weights_fixed": live_row, "auto_search_best_by_score_config": best_row,
              "outlier_diagnostics": {"live": live_diag, "candidate": cand_diag},
              "raw_event_excess_6m": {"live": a.round(6).tolist(), "candidate": b.round(6).tolist()},
              "paired_bootstrap": {"excess_diff_ci90_live_minus_candidate": ci,
                                   "pct_live_higher_excess": pct_live_higher_excess,
                                   "sharpe_diff_ci90_candidate_minus_live": sharpe_ci,
                                   "pct_candidate_higher_sharpe": pct_candidate_higher_sharpe},
              "sharpe_dsr_best_candidate": sharpe_row,
              "ic_top10": ic_sorted[:10], "n_events": len(pit_snaps),
              "pit_coverage": cov, "keep": args.keep, "levels": args.levels}
    with open("output/live_weights_vs_search.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log("저장: output/live_weights_vs_search.json")


if __name__ == "__main__":
    main()
