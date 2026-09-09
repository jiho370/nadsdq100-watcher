#!/usr/bin/env python3
"""
us_factor_formula_pit_sweep.py — us_factor_formula_sweep.py의 정정판(2026-07-17).

지호 님이 지적한 문제(전작 스크립트가 backtest_weights.build_panel()의 현재 S&P500
구성종목만 써서 생존편향이 있었고, PBO 41.2%가 원 검증 PBO 15.0%보다 나쁘게 나온 원인이
표본 크기가 아니라 이 패널 질 차이였음 — STRATEGY.md §2 정정 기록 참고)를 Fable 5 자문
반영해 다음과 같이 고친다:

  1. 패널 = backtest_costs.build_panel_pit()(PIT 유니버스, 상장폐지 종목 포함)
  2. 펀더멘탈 = 과거 편입 후 제외된 198종목을 fundamentals_edgar.py로 사전 백필한 캐시
     (없으면 그 종목은 z=0으로 왜곡되던 문제 — Fable 5가 실측으로 발견)
  3. z/rank/qgate는 그 시점 실제 멤버십(backtest_costs.membership_asof)으로 필터링한
     '뒤'에 계산 — 원 검증(backtest_costs._filter_snaps 방식)과 동일 순서
  4. 63일 리밸런싱 + 6개월 forward 유지(원 인증과 같은 잣대 — 21일/daily로 바꾸면 비교 불가)
  5. 시행은 5개(현행 포함)로 사전등록 — 격자 스윕 금지:
       기준  gp1·rd2·sy2 [raw]  (현행 라이브)
       C1    gp1·rd2·sy2 [rank] (가중치 그대로, rd_mktcap만 순위변환 — 극단치 지배 완화)
       C2    gp1·rd2·sy2 [qgate](가중치 그대로, rd_mktcap만 품질게이트)
       C3    gp1·rd1·sy2 [raw]  (rd 가중치 절반 — 쏠림 원인 자체를 완화)
       C4    gp1·rd0·sy2 [raw]  (반증용 — rd를 빼도 되는지)
     (지호 님 반응 "shareholder_yield 단독이 잘 나왔다"는 원 PIT 검증(output/pbo_report.json,
     665개 시행)에서 이미 진 조합이라 후보에서 제외 — Fable 5 확인)
  6. 판정: 이벤트 단위 페어드 비교(도전자-기준의 6M 순초과수익 차이, 도전자 4개 본페로니
     보정) + 평균 최대섹터쏠림 비열등 — "수익 개선"이 아니라 "쏠림 완화 + 성과 비열등"
     프레임(Fable 5, 리스크 관리가 동기이므로)
  7. 주 평가 = topn8+섹터캡2(라이브 조건), 보조 강건성 = topn30 무캡(원 인증 잣대)

실행: python us_factor_formula_pit_sweep.py
결과: output/us_factor_formula_pit_sweep.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS

TD_DAYS = 126
LOOKBACK = 252
REBAL_DAYS = 63
FACTORS = ["int_gp_assets", "rd_mktcap", "shareholder_yield"]

TRIALS = [
    {"label": "기준(gp1·rd2·sy2[raw])",  "weights": {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}, "rd_mode": "raw"},
    {"label": "C1(gp1·rd2·sy2[rank])",   "weights": {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}, "rd_mode": "rank"},
    {"label": "C2(gp1·rd2·sy2[qgate])",  "weights": {"int_gp_assets": 1, "rd_mktcap": 2, "shareholder_yield": 2}, "rd_mode": "qgate"},
    {"label": "C3(gp1·rd1·sy2[raw])",    "weights": {"int_gp_assets": 1, "rd_mktcap": 1, "shareholder_yield": 2}, "rd_mode": "raw"},
    {"label": "C4(gp1·rd0·sy2[raw])",    "weights": {"int_gp_assets": 1, "rd_mktcap": 0, "shareholder_yield": 2}, "rd_mode": "raw"},
]


def _log(m): print(f"[PIT팩터] {m}", file=sys.stderr)


def _rd_variant(raw_rd: pd.Series, z_gp: pd.Series, z_sy: pd.Series, mode: str) -> pd.Series:
    if mode == "raw" or raw_rd.std(ddof=0) == 0 or raw_rd.isna().all():
        sd = raw_rd.std()
        return ((raw_rd - raw_rd.mean()) / sd).clip(-3, 3).fillna(0.0) if sd and not np.isnan(sd) else raw_rd * 0.0
    if mode == "rank":
        n = len(raw_rd)
        rank = raw_rd.rank(method="average")
        return (((rank - 0.5) / n - 0.5) * 6).fillna(0.0) if n > 1 else raw_rd * 0.0
    if mode == "qgate":
        sd = raw_rd.std()
        z_raw = ((raw_rd - raw_rd.mean()) / sd).clip(-3, 3).fillna(0.0) if sd and not np.isnan(sd) else raw_rd * 0.0
        return z_raw.where((z_gp > 0) | (z_sy > 0), 0.0)
    raise ValueError(mode)


def build_snaps(panel: pd.DataFrame, spy: pd.Series, funds: dict, pit):
    """PIT 멤버십으로 그 시점 실제 구성종목만 남긴 뒤 3팩터 원값 계산."""
    import fundamentals_edgar as F
    n = len(panel); max_h = max(TD_DAYS, LOOKBACK)
    ps = list(range(LOOKBACK, n - TD_DAYS - 1, REBAL_DAYS))
    snaps = []
    n_cov, n_tot = [], []
    for p in ps:
        date_iso = panel.index[p].date().isoformat()
        members = BC.membership_asof(pit, date_iso)
        price = panel.iloc[p]
        valid = [s for s in price.dropna().index
                 if s in members and not np.isnan(panel.iloc[p - LOOKBACK][s])]
        if not valid:
            continue
        v = pd.Index(valid)
        rows = {}
        has_fund = 0
        for s in v:
            fd = funds.get(s) or {}
            fv = F.factor_values(fd, date_iso, float(price[s]))
            rows[s] = {f: fv.get(f) for f in FACTORS}
            if fd:
                has_fund += 1
        raw = pd.DataFrame(rows).T.astype(float)
        mom = (panel.iloc[p] / panel.iloc[p - 126] - 1).reindex(v)
        raw = raw[mom.notna()]
        if raw.empty:
            continue
        e = p + 1
        if e + TD_DAYS >= n:
            continue
        fwd = panel.iloc[e + TD_DAYS][raw.index] / panel.iloc[e][raw.index] - 1
        bench = float(spy.iloc[e + TD_DAYS] / spy.iloc[e] - 1)
        snaps.append({"date": date_iso, "raw": raw, "fwd": fwd, "bench": bench})
        n_cov.append(has_fund); n_tot.append(len(v))
    cov_pct = round(100 * sum(n_cov) / max(sum(n_tot), 1), 1)
    _log(f"스냅 {len(snaps)}개 · 평균 펀더멘탈 커버리지 {cov_pct}%(그 시점 실제 멤버 대비)")
    return snaps


def sector_of_map():
    import sp500_daily_report as R
    return R.fetch_wikipedia_sectors()


def eval_trial(snaps, weights: dict, rd_mode: str, sector_map: dict, topn: int, sector_cap):
    import sp500_daily_report as R
    excess, max_sec = [], []
    for snap in snaps:
        raw = snap["raw"]
        z_gp = ((raw["int_gp_assets"] - raw["int_gp_assets"].mean()) / raw["int_gp_assets"].std()).clip(-3, 3).fillna(0.0)
        z_sy = ((raw["shareholder_yield"] - raw["shareholder_yield"].mean()) / raw["shareholder_yield"].std()).clip(-3, 3).fillna(0.0)
        z_rd = _rd_variant(raw["rd_mktcap"], z_gp, z_sy, rd_mode)
        comp = weights.get("int_gp_assets", 0) * z_gp + weights.get("rd_mktcap", 0) * z_rd \
             + weights.get("shareholder_yield", 0) * z_sy
        ranked = [(s, comp[s]) for s in comp.sort_values(ascending=False).index]
        if sector_cap is not None:
            scored_tuples = [(s, sc, "") for s, sc in ranked]
            picked = R.pick_with_sector_cap(scored_tuples, sector_map, topn, sector_cap)
            top = [s for s, _, _ in picked]
        else:
            top = [s for s, _ in ranked[:topn]]
        r = snap["fwd"].reindex(top).dropna()
        if len(r):
            net = float(r.mean()) - BW_COST
            excess.append(net - snap["bench"])
        else:
            excess.append(np.nan)
        from collections import Counter
        c = Counter(sector_map.get(s, "(미상)") for s in top)
        max_sec.append(c.most_common(1)[0][1] if c else 0)
    return excess, max_sec


BW_COST = 0.001   # backtest_weights.py와 동일 관례(왕복 10bp) — 실제 CostModel과 별개로 excess 계산용 근사


def paired_report(snaps, sector_map, topn, sector_cap, tag):
    base = TRIALS[0]
    base_ex, base_ms = eval_trial(snaps, base["weights"], base["rd_mode"], sector_map, topn, sector_cap)
    rows = [{"label": base["label"], "mean_excess_6m_pct": round(100*np.nanmean(base_ex), 2),
            "avg_max_sector": round(float(np.nanmean(base_ms)), 2)}]
    diffs_for_pbo = [base_ex]
    labels_for_pbo = [base["label"]]
    n_challengers = len(TRIALS) - 1
    for t in TRIALS[1:]:
        ex, ms = eval_trial(snaps, t["weights"], t["rd_mode"], sector_map, topn, sector_cap)
        diffs_for_pbo.append(ex)
        labels_for_pbo.append(t["label"])
        pair = np.array([a - b if (a == a and b == b) else np.nan for a, b in zip(ex, base_ex)])
        pair = pair[~np.isnan(pair)]
        mean_diff = float(pair.mean()) if len(pair) else 0.0
        se = float(pair.std(ddof=1) / math.sqrt(len(pair))) if len(pair) > 1 else 0.0
        tstat = mean_diff / se if se > 0 else 0.0
        # 본페로니 보정(도전자 4개): 유의수준 0.05/4=0.0125 양측 근사 임계 t≈2.5(자유도 낮아 보수적으로 2.8 사용)
        sig = abs(tstat) > 2.8
        rows.append({"label": t["label"], "mean_excess_6m_pct": round(100*np.nanmean(ex), 2),
                    "avg_max_sector": round(float(np.nanmean(ms)), 2),
                    "paired_diff_vs_baseline_pct": round(100*mean_diff, 2),
                    "paired_tstat": round(tstat, 2), "bonferroni_significant": sig})
        _log(f"[{tag}] {t['label']:28s} 페어드차이 {100*mean_diff:+.2f}%p (t={tstat:+.2f}, "
             f"{'유의' if sig else '유의아님'}) · 평균최대섹터쏠림 {np.nanmean(ms):.2f} "
             f"(기준 {np.nanmean(base_ms):.2f})")
    n_ev = min(len(d) for d in diffs_for_pbo)
    clean = [[v if v == v else 0.0 for v in d[:n_ev]] for d in diffs_for_pbo]
    trial_data = {"horizon": f"us_factor_pit_{tag}", "universe": "sp500_pit", "cost": {"approx_bps": BW_COST*10000},
                 "rebal_days": REBAL_DAYS, "hold_days": TD_DAYS,
                 "dates": [s["date"] for s in snaps[:n_ev]], "trials": labels_for_pbo,
                 "excess_returns": clean}
    rpt = OS.analyze(trial_data, save=False)
    return rows, rpt


def run(years=10, save=True):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = build_snaps(panel, spy, funds, pit)
    sector_map = sector_of_map()

    _log("=== 주 평가: topn8 + 섹터캡2(라이브 조건) ===")
    rows_primary, rpt_primary = paired_report(snaps, sector_map, 8, 2, "primary_top8cap2")
    _log("=== 보조 강건성: topn30 무캡(원 인증 잣대) ===")
    rows_robust, rpt_robust = paired_report(snaps, sector_map, 30, None, "robust_top30nocap")

    payload = {"as_of": panel.index[-1].date().isoformat(), "n_snaps": len(snaps),
              "trials_registered": [t["label"] for t in TRIALS],
              "primary_top8_cap2": {"rows": rows_primary,
                                    "pbo": rpt_primary.get("pbo", {}).get("pbo"),
                                    "dsr": rpt_primary.get("dsr", {}).get("dsr"),
                                    "passed": rpt_primary.get("passed", False)},
              "robust_top30_nocap": {"rows": rows_robust,
                                     "pbo": rpt_robust.get("pbo", {}).get("pbo"),
                                     "dsr": rpt_robust.get("dsr", {}).get("dsr"),
                                     "passed": rpt_robust.get("passed", False)},
              "note": "펀더멘탈 사전 백필(과거 편입 후 제외 종목 EDGAR 조회) 완료 후 실행. "
                      "판정: 페어드 비교(본페로니 보정, 도전자 4개) + 섹터쏠림 비열등."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_factor_formula_pit_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_factor_formula_pit_sweep.json")
    return payload


if __name__ == "__main__":
    run()
