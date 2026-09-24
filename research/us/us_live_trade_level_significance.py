#!/usr/bin/env python3
"""
us_live_trade_level_significance.py — 지호 님 요청(2026-09-24): "바스켓 평균 32개 이벤트 대신
종목별 개별 거래로 유의성을 다시 보자".

us_full_stack_exec_validation.py의 21~32조합 스윕과는 목적이 다르다: 거긴 "어떤 진입·청산
규칙이 제일 나은가"를 탐색하므로 PBO/DSR로 다중검정을 보정해야 한다(그래서 T_eff가 6~11까지
줄어듦). 여기선 이미 확정된 라이브 규칙(entry_live__exit_live) 단 하나만 검정한다 — 여러
후보 중 고르는 게 아니므로 다중검정 문제가 없고, 대신 "같은 리밸런싱 날짜에 뽑힌 종목들은
서로 독립이 아니다"(같은 시장 국면·같은 팩터 점수 분포를 공유)라는 점만 클러스터로 보정하면
된다.

방법: us_full_stack_exec_validation._live_ranked와 동일한 라이브 선정을 재사용해 매 리밸런싱
시점(p)의 topn 바스켓을 뽑고, backtest_exec._simulate_trade(entry_live/exit_live)로 종목별
개별 net·excess(=net-SPY)를 계산한다(바스켓 평균으로 뭉개지 않고 종목 단위로 보존, 현금
슬롯은 제외 — 슬롯 희석까지 포함한 계좌 기준 수치는 backtest_exec_compare 쪽 참고).
리밸런싱 날짜(p)를 클러스터로 묶어 클러스터-로버스트 t-검정(=클러스터 평균에 대한 표준
t-검정과 동치, 균형패널 기준)과 리밸런싱 날짜를 통째로 리샘플하는 블록부트스트랩(정규분포
가정 없이 p-value·신뢰구간)을 계산한다. naive(개별 종목을 독립으로 취급) t-검정도 같이
내지만 — 이건 과신을 유발하는 틀린 방법이라 대조용으로만 표시한다.

한계: 그래도 원천 표본(리밸런싱 날짜)은 32개뿐이라 검정력 자체의 한계는 그대로다 — 이
스크립트는 "숨어있던 표본을 찾아낸다"가 아니라 "이미 가진 표본을 CSCV 임베고 없이 정직하게
재는" 목적.

실행: python -m research.us.us_live_trade_level_significance [--years 10] [--topn 10] [--boot 20000]
결과: output/us_live_trade_level_significance_top{N}.json
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

import backtest_costs as BC
import backtest_weights as BW
import backtest_exec as BE
from research.us.us_full_stack_exec_validation import _live_ranked, POOL_N


def _log(m): print(f"[US종목단위유의성]  {m}", file=sys.stderr)


def run(years: float = 10, topn: int = 10, n_boot: int = 20000, seed: int = 20260924) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = BE._load_exec_weights()
    _log(f"라이브 가중치 사용: {weights} · topn={topn}")

    import tech_factors as T
    cross = T.build_panels(panel)
    ranked = {}

    def _rank(p):
        if p not in ranked:
            ranked[p] = _live_ranked(panel, p, funds, cross, pit, weights)
        return ranked[p]

    ma20, ma50, ma200, atr = BE._ma(panel, 20), BE._ma(panel, 50), BE._ma(panel, 200), BE._atr_close(panel)
    spy_r = spy.reindex(panel.index).ffill()
    n = len(panel)

    def _bench(d0, d1):
        if not (np.isfinite(spy_r.iloc[d1]) and np.isfinite(spy_r.iloc[d0])):
            return 0.0
        return float(spy_r.iloc[d1] / spy_r.iloc[d0] - 1)

    ps = list(range(BW.LOOKBACK, n - BE.MAX_HOLD - 1, 63))
    pool_fn = lambda d: None if _rank(d) is None else set(_rank(d)[:POOL_N])

    cluster_ids, excess_all, net_all, beat_spy = [], [], [], []
    n_events_used = 0
    for p in ps:
        r = _rank(p)
        if r is None:
            continue
        basket = r[:topn]
        if not basket:
            continue
        n_events_used += 1
        entry_day = p + 1
        for sym in basket:
            t = BE._simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day,
                                    "entry_live", "exit_live", pool_fn=pool_fn)
            if t is None:
                continue
            net = t["filled_frac"] * cost.net(t["exit_price"] / t["entry_price"] - 1)
            bench = _bench(entry_day, t["exit_day"])
            ex = net - bench
            cluster_ids.append(p)
            net_all.append(net)
            excess_all.append(ex)
            beat_spy.append(1.0 if ex > 0 else 0.0)

    cluster_ids = np.array(cluster_ids)
    excess_all = np.array(excess_all)
    net_all = np.array(net_all)
    n_trades = len(excess_all)
    clusters = sorted(set(cluster_ids.tolist()))
    n_clusters = len(clusters)
    _log(f"개별 종목-이벤트 {n_trades}건 (리밸런싱 {n_events_used}회 · 클러스터 {n_clusters}개)")

    # ---- (A) naive — 개별 종목을 독립으로 취급(과신 유발, 대조용) ----
    naive_mean = float(excess_all.mean())
    naive_se = float(excess_all.std(ddof=1) / np.sqrt(n_trades))
    naive_t = naive_mean / naive_se if naive_se else 0.0

    # ---- (B) 클러스터-로버스트 — 리밸런싱 날짜별 평균을 관측치로(균형패널이면 표준 t검정과 동치) ----
    cluster_means = np.array([excess_all[cluster_ids == c].mean() for c in clusters])
    cr_mean = float(cluster_means.mean())
    cr_se = float(cluster_means.std(ddof=1) / np.sqrt(n_clusters))
    cr_t = cr_mean / cr_se if cr_se else 0.0
    cr_df = n_clusters - 1

    # ---- (C) 블록부트스트랩 — 리밸런싱 날짜를 통째로 복원추출(정규분포 가정 없음) ----
    rng = np.random.RandomState(seed)
    idx_by_cluster = {c: np.where(cluster_ids == c)[0] for c in clusters}
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(clusters, size=n_clusters, replace=True)
        vals = [excess_all[idx_by_cluster[c]].mean() for c in picked]
        boot_means[i] = float(np.mean(vals))
    boot_ci90 = (float(np.percentile(boot_means, 5)), float(np.percentile(boot_means, 95)))
    boot_ci95 = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    boot_p_le0 = float((boot_means <= 0).mean())   # 부트스트랩 분포에서 평균초과수익<=0 인 비율

    hit_rate = float(np.mean(beat_spy))

    try:
        from scipy import stats as _st
        naive_p = float(2 * _st.t.sf(abs(naive_t), df=n_trades - 1))
        cr_p = float(2 * _st.t.sf(abs(cr_t), df=cr_df))
    except Exception:
        # scipy 없으면 정규근사(표본이 작을수록 부정확 — 참고용, boot_p_le0을 더 신뢰)
        from math import erfc, sqrt
        naive_p = float(erfc(abs(naive_t) / sqrt(2)))
        cr_p = float(erfc(abs(cr_t) / sqrt(2)))

    out = {
        "as_of": panel.index[-1].date().isoformat(),
        "topn": topn, "n_events": n_events_used, "n_clusters": n_clusters, "n_trades": n_trades,
        "scope_note": "현금 슬롯(필터 부족분) 제외 — 종목이 실제로 뽑힌 개별 거래만. "
                      "계좌 슬롯 희석까지 포함한 수치는 backtest_exec_compare_us_livestack_top"
                      f"{topn}.json의 entry_live__exit_live 행 참고.",
        "hit_rate_pct": round(100 * hit_rate, 1),
        "naive_pooled_invalid": {
            "mean_excess_pct": round(100 * naive_mean, 3), "t": round(naive_t, 3),
            "p": round(naive_p, 4), "n": n_trades,
            "warning": "종목 간 상관을 무시한 틀린 검정 — 과신 유발, 대조용으로만 표시"},
        "cluster_robust": {
            "mean_excess_pct": round(100 * cr_mean, 3), "se_pct": round(100 * cr_se, 3),
            "t": round(cr_t, 3), "p": round(cr_p, 4), "df": cr_df, "n_clusters": n_clusters},
        "block_bootstrap": {
            "n_boot": n_boot, "mean_excess_pct": round(100 * float(boot_means.mean()), 3),
            "ci90_pct": [round(100 * boot_ci90[0], 3), round(100 * boot_ci90[1], 3)],
            "ci95_pct": [round(100 * boot_ci95[0], 3), round(100 * boot_ci95[1], 3)],
            "p_mean_le_zero": round(boot_p_le0, 4)},
        "note": "리밸런싱 날짜(클러스터) 자체는 여전히 32개뿐 — 종목 단위로 쪼개도 진짜 독립정보가 "
                "그만큼 늘어나는 건 아니다(균형패널에선 클러스터-로버스트 결과가 기존 바스켓평균 "
                "DSR과 방향이 같아야 정상). 이 스크립트의 목적은 표본을 늘리는 게 아니라 CSCV의 "
                "embargo/purge로 T_eff가 6~11까지 깎이는 것 없이, 32개 클러스터를 그대로 쓰는 더 "
                "표준적인 검정(및 정규분포 가정이 없는 부트스트랩)으로 같은 결론이 재현되는지 "
                "교차확인하는 것.",
    }
    _log(f"hit_rate={out['hit_rate_pct']}%")
    _log(f"naive(틀림) mean={out['naive_pooled_invalid']['mean_excess_pct']}%p t={naive_t:.3f} p={naive_p:.4f}")
    _log(f"클러스터-로버스트 mean={out['cluster_robust']['mean_excess_pct']}%p "
        f"t={cr_t:.3f}(df={cr_df}) p={cr_p:.4f}")
    _log(f"블록부트스트랩({n_boot}회) CI90={out['block_bootstrap']['ci90_pct']}%p "
        f"P(평균<=0)={out['block_bootstrap']['p_mean_le_zero']}")

    path = f"output/us_live_trade_level_significance_top{topn}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log(f"저장: {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    run(years=args.years, topn=args.topn, n_boot=args.boot)


if __name__ == "__main__":
    main()
