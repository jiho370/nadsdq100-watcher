#!/usr/bin/env python3
"""
score_calibration.py (v2) — 0~10 추천강도 점수 캘리브레이션.
설계 근거: SCORE_MODEL_DESIGN.md (v1 실패 원인 D1~D5의 대응이 그대로 코드가 됨).

v1 → v2 변경 요약:
  · 호라이즌 1m → 6m (D1: 팩터는 느린 신호 — 1m은 단기반전이 지배해 ρ≈0이었음)
  · 임의 고정가중(0.5/0.3/0.2) 폐지 → 워크포워드 IR-가중 팩터 군집 점수 (D2·D4)
    - 후보: 퀄리티·주주환원·현금흐름 + 2023~24 문헌 추가 팩터(무형조정·이익성장·부채발행)
    - 각 시점 가중치 w_f = mean(IC_f)/std(IC_f), 그 시점 이전에 '완결된' 스냅샷 IC만 사용
      (forward 구간이 안 끝난 스냅샷의 IC는 모름 → look-ahead 원천 차단), 음수 IC는 0
  · 검증: 풀링 Spearman(허위 정밀) → 스냅샷 단위 Spearman ρ_t 의 평균·t-stat (D3)
  · 분위표: 전체 기간 + 최근 5년 분리 (D5: 레짐 의존 감시)
  · 게이트: ρ̄>0 & t≥2 & (D10−D1) > 왕복비용 — 전부 통과해야 display_allowed=true

진입 게이트: output/pbo_report.json 의 passed=true (T_eff 보정 DSR 기준) 필요. --force 로 우회(연구용).

실행(PC): python fundamentals_edgar.py            # 신규 팩터(rnd·sga) 증분 수집
          python backtest_costs.py --years 10 --oos 0.4
          python overfit_stats.py
          python score_calibration.py --years 10   # 기본 horizon 6m
          python score_calibration.py --self-test
결과: output/score_calibration.json
"""
from __future__ import annotations
import os, sys, json, math, argparse, warnings
from statistics import NormalDist
import numpy as np
import pandas as pd

# 상수 컬럼 상관계산에서 나오는 무해한 divide 경고 억제(결과는 NaN 처리로 걸러짐)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

import backtest_weights as BW
import backtest_costs as BC

CALIB_PATH = "output/score_calibration.json"
GATE_PATH = "output/pbo_report.json"
ND = NormalDist()

# 팩터 군집 후보(재무 계열만 — mom12_1은 이 표본 IC<0이라 제외, D2).
# 실제 채택 여부·가중치는 훈련구간 IC가 결정한다(IC≤0이면 자동 0).
CLUSTER_CANDIDATES = [
    "gp_assets", "gross_margin", "op_margin", "roa", "roe", "cop",
    "accruals", "shareholder_yield", "fcf_ev", "fcf_yield", "asset_growth",
    # 2023~24 문헌 추가(fundamentals_edgar.py 참고)
    "droe", "debt_issuance", "rd_mktcap", "int_gp_assets", "int_value",
]
MIN_HISTORY = 8          # IR-가중 산출에 필요한 최소 '완결' 스냅샷 수(그 전엔 점수 없음)
ROUNDTRIP_COST_PCT = 0.11  # D10−D1 스프레드 최소 요구치(%p) — 미국 왕복 ~10.3bp


def _log(m): print(m, file=sys.stderr)


# ------------------------- 라이브 리포트용 API -------------------------
def load_calibration(path=CALIB_PATH):
    """모든 검증을 통과한 캘리브레이션만 반환. 아니면 None → 리포트는 점수 표시 생략."""
    try:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        return cal if cal.get("display_allowed") else None
    except Exception:
        return None


def score_from_percentile(pct: float) -> int:
    """군집 점수 백분위(0~1) → 0~10 점수."""
    return int(min(max(round(pct * 10), 0), 10))


# ------------------------- 워크포워드 IR-가중 군집 점수 -------------------------
def build_pool(panel, spy, funds, pit, rebal_days=21, horizon="6m",
               candidates=CLUSTER_CANDIDATES, min_history=MIN_HISTORY, extra_cross=None):
    """스냅샷마다: (1) 과거 완결 스냅샷들의 IC로 IR-가중 산출 → (2) 군집 점수 백분위
    → (3) forward 수익률과 함께 수집. 반환: (pct, ret, date, rho_t 리스트, 사용 가중치 로그)
    extra_cross: 부록 A2-(a) SR_CANDIDATES 등 추가 팩터 패널(dict[name->DataFrame])을
    tech_factors 크로스오버 패널에 병합 — 기본 실행(candidates=CLUSTER_CANDIDATES)에는
    영향 없음(예산 분리, NEXT_STEPS_SONNET.md 트랙 C)."""
    import tech_factors as T
    hd = BW.TD[horizon]
    use_fund = bool(funds)
    spy = spy.reindex(panel.index).ffill()
    n = len(panel)
    cross = T.build_panels(panel)
    if extra_cross:
        cross = {**cross, **extra_cross}
    ps = list(range(BW.LOOKBACK, n - hd - 1, rebal_days))

    # 1패스: 스냅샷별 원팩터·forward수익 준비 + 스냅샷 IC 기록
    snaps = []
    for p in ps:
        raw = BW._raw_frame(panel, p, funds, use_fund, cross)
        if raw is None or len(raw) < 20:
            continue
        date = panel.index[p].date().isoformat()
        idx = raw.index.intersection(BC.membership_asof(pit, date))
        if len(idx) < 20:
            continue
        raw = raw.loc[idx]
        e = p + 1
        fwd = (panel.iloc[e + hd][idx] / panel.iloc[e][idx] - 1).dropna()
        cols = [c for c in candidates if c in raw.columns]
        fr = fwd.rank()
        ics = {}
        for c in cols:
            v = raw[c].rank().corr(fr.reindex(raw.index))
            if pd.notna(v):
                ics[c] = float(v)
        snaps.append({"p": p, "date": date, "raw": raw, "fwd": fwd, "ics": ics})

    # 2패스: 워크포워드 — t 시점 가중치는 't 이전에 forward 구간이 끝난' 스냅샷 IC만 사용
    pool_pct, pool_ret, pool_date, rhos, w_log = [], [], [], [], []
    for t, s in enumerate(snaps):
        done = [x for x in snaps if x["p"] + hd <= s["p"]]       # 완결된 스냅샷만
        if len(done) < min_history:
            continue
        acc = {}
        for x in done:
            for f, v in x["ics"].items():
                acc.setdefault(f, []).append(v)
        w = {}
        for f, arr in acc.items():
            if len(arr) >= 3:
                sd = float(np.std(arr, ddof=1))
                ir = float(np.mean(arr)) / sd if sd > 0 else 0.0
                if ir > 0:
                    w[f] = ir
        w = {f: v for f, v in w.items() if f in s["raw"].columns}   # 현 스냅샷에 있는 팩터만
        if not w:
            continue
        tot = sum(w.values())
        w = {f: v / tot for f, v in w.items()}
        cols = list(w)
        # 펀더멘털 결측 종목 제외: 전부 NaN→z=0이면 점수 0 동점 블록이 분포 중앙에 뭉쳐
        # 순위상관을 왜곡한다(실측: 6분위 표본 급감·수익 급락 이상 현상의 원인)
        present = s["raw"][cols].notna().sum(axis=1)
        keep = present[present >= max(2, len(cols) // 3)].index
        if len(keep) < 20:
            continue
        z = s["raw"].loc[keep, cols].apply(BW._z).fillna(0.0)
        score = (z * pd.Series(w)).sum(axis=1)
        pct = score.rank(pct=True)
        ok = s["fwd"].index.intersection(pct.index)
        if len(ok) < 20:
            continue
        # Spearman = 순위의 Pearson (pandas method="spearman"은 scipy 의존 → 직접 계산)
        ra = pct.reindex(ok).rank().to_numpy()
        rb = s["fwd"].reindex(ok).rank().to_numpy()
        rho = float(np.corrcoef(ra, rb)[0, 1])
        if not np.isfinite(rho):
            continue
        rhos.append({"date": s["date"], "rho": round(rho, 4), "n": len(ok)})
        pool_pct += list(pct.reindex(ok)); pool_ret += list(s["fwd"].reindex(ok))
        pool_date += [s["date"]] * len(ok)
        w_log.append({"date": s["date"], "weights": {f: round(v, 3) for f, v in w.items()}})
    return (np.array(pool_pct, float), np.array(pool_ret, float),
            pool_date, rhos, w_log)


# ------------------------- 캘리브레이션 + 검증 -------------------------
def _decile_table(pool_pct, pool_ret):
    dec = np.minimum((pool_pct * 10).astype(int), 9) + 1
    table = []
    for d in range(1, 11):
        r = pool_ret[dec == d]
        if len(r) == 0:
            table.append({"decile": d, "n": 0}); continue
        wins, losses = r[r > 0], r[r <= 0]
        payoff = (float(wins.mean() / abs(losses.mean()))
                  if len(wins) and len(losses) and losses.mean() != 0 else None)
        table.append({"decile": d, "score_range": f"{d-1}~{d}", "n": int(len(r)),
                      "mean_ret_pct": round(100 * float(r.mean()), 2),
                      "win_rate_pct": round(100 * float((r > 0).mean()), 1),
                      "worst_pct": round(100 * float(r.min()), 1),
                      "payoff": round(payoff, 2) if payoff else None})
    return table


def calibrate(pool_pct, pool_ret, pool_date, rhos, horizon, alpha=0.05,
              min_spread_pct=ROUNDTRIP_COST_PCT, rebal_days=21):
    # 검증 ①: 스냅샷 단위 Spearman — 독립 단위는 '스냅샷'(풀링 p값의 허위 정밀 제거, D3)
    # 단, 월간 스냅샷의 forward 구간이 서로 겹치므로(보유>리밸) 유효 표본으로 보정한다.
    rv = np.array([r["rho"] for r in rhos], float)
    n_snap = len(rv)
    hold_days = BW.TD[horizon]
    n_eff = max(int(round(n_snap * min(rebal_days / hold_days, 1.0))), 3)
    rho_mean = float(rv.mean()) if n_snap else 0.0
    t_stat = (rho_mean / (float(rv.std(ddof=1)) / math.sqrt(n_eff))
              if n_snap > 2 and rv.std(ddof=1) > 0 else 0.0)
    # 분위표: 전체 + 최근 5년(D5)
    full = _decile_table(pool_pct, pool_ret)
    dates = np.array(pool_date)
    cutoff = (pd.Timestamp(max(pool_date)) - pd.DateOffset(years=5)).date().isoformat()
    recent_mask = dates >= cutoff
    recent = _decile_table(pool_pct[recent_mask], pool_ret[recent_mask])
    # 검증 ②: D10−D1 스프레드가 왕복비용 초과
    spread = ((full[-1].get("mean_ret_pct") or 0) - (full[0].get("mean_ret_pct") or 0))
    checks = {"rho_positive": rho_mean > 0,
              "t_stat_ge_2": t_stat >= 2.0,
              "spread_gt_cost": spread > min_spread_pct}
    allowed = all(checks.values())
    return {"horizon": horizon, "n_samples": int(len(pool_ret)),
            "method": "workforward IR-weighted quality/shareholder cluster (v2)",
            "deciles": full, "recent_5y_deciles": recent, "recent_5y_cutoff": cutoff,
            "spearman_by_snapshot": {"rho_mean": round(rho_mean, 4),
                                     "t_stat": round(t_stat, 2),
                                     "n_snapshots": n_snap, "n_eff": n_eff,
                                     "alpha": alpha,
                                     "note": "t는 중첩 보정 유효표본(n_eff) 기준"},
            "d10_d1_spread_pct": round(spread, 2),
            "min_spread_required_pct": min_spread_pct,
            "checks": checks, "display_allowed": bool(allowed),
            "note": ("전 검증 통과 — 리포트 노출 가능" if allowed else
                     "검증 실패 — 점수를 리포트에 노출하지 말 것(SCORE_MODEL_DESIGN.md §2.5)")}


def report(cal, save=True, path=CALIB_PATH):
    _log(f"\n=== v2 점수 캘리브레이션 (horizon {cal['horizon']} · 표본 {cal['n_samples']}) ===")
    for tag, tbl in (("전체", cal["deciles"]), (f"최근5년(≥{cal['recent_5y_cutoff']})",
                                               cal["recent_5y_deciles"])):
        _log(f"  [{tag}]  분위 | 표본 | 평균% | 승률% | 손익비")
        for r in tbl:
            _log(f"    {r['decile']:>2d} | {r['n']:>6d} | {r.get('mean_ret_pct','-'):>6} | "
                 f"{r.get('win_rate_pct','-'):>5} | {r.get('payoff') or '-'}")
    sp = cal["spearman_by_snapshot"]
    _log(f"  스냅샷 Spearman: ρ̄={sp['rho_mean']} t={sp['t_stat']} (스냅샷 {sp['n_snapshots']}개)")
    _log(f"  D10−D1 스프레드 {cal['d10_d1_spread_pct']}%p (요구 >{cal['min_spread_required_pct']}%p)")
    _log(f"  검증: {cal['checks']} → display_allowed={cal['display_allowed']}")
    _log(f"  {cal['note']}")
    if save:
        os.makedirs("output", exist_ok=True)
        prev_attempts = []
        try:
            with open(path, encoding="utf-8") as f:
                prev_attempts = json.load(f).get("attempted_horizons", [])
        except Exception:
            pass
        cal["attempted_horizons"] = sorted(set(prev_attempts + [cal["horizon"]]))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        _log(f"\n>>> 저장: {path} (시도한 호라이즌: {cal['attempted_horizons']} — 다중검정 예산 기록)")


def _gate(force=False) -> bool:
    try:
        with open(GATE_PATH, encoding="utf-8") as f:
            rep = json.load(f)
    except Exception:
        _log(f"[게이트] {GATE_PATH} 없음 — 먼저 backtest_costs.py → overfit_stats.py 실행.")
        return force
    if rep.get("passed"):
        _log(f"[게이트] PBO {rep['pbo']['pbo']:.1%} · DSR(T_eff) {rep['dsr'].get('dsr')} → 통과.")
        return True
    _log(f"[게이트] 미통과({rep.get('pbo_verdict')} / {rep.get('dsr_verdict')}) — 중단. "
         f"(연구용 우회: --force)")
    return force


# ------------------------- self-test -------------------------
def _planted(n_days=2000, n_syms=80, seed=9):
    """gross_margin(→gp_assets 계열)이 미래수익을 진짜로 예측하는 합성 세계."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    quality = rng.normal(0, 1, n_syms)
    drift = 0.0002 + 0.0005 * quality
    panel = pd.DataFrame({f"S{i:02d}": 100 * np.exp(np.cumsum(
        rng.normal(drift[i], 0.018, n_days))) for i in range(n_syms)}, index=dates)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n_days))), index=dates)
    funds = {}
    for i in range(n_syms):
        s = f"S{i:02d}"; rev, assets = 1e10, 2e10
        gross = rev * (0.35 + 0.08 * quality[i]); ni = rev * 0.05
        funds[s] = {}
        for k, v in (("revenue", rev), ("gross", gross), ("ni", ni), ("assets", assets),
                     ("equity", assets * 0.5), ("ocf", ni * 1.1), ("debt", assets * 0.4),
                     ("eps", 5.0), ("capex", ni * 0.3)):
            funds[s][k] = [{"end": f"{y}-12-31", "filed": f"{y+1}-03-01", "val": float(v)}
                           for y in range(2015, 2025)]
    pit = [(dates[0].date().isoformat(), frozenset(panel.columns))]
    return panel, spy, funds, pit


def self_test():
    _log("[self-test] ① 심은 신호 세계 → 통과 / ② 노이즈 세계 → 차단 / ③ API")
    panel, spy, funds, pit = _planted()
    pct, ret, dts, rhos, wlog = build_pool(panel, spy, funds, pit, rebal_days=21,
                                           horizon="6m", min_history=5)
    assert len(pct) > 1000 and len(rhos) >= 10, f"풀 부족: {len(pct)}, {len(rhos)}"
    cal = calibrate(pct, ret, dts, rhos, "6m")
    report(cal, save=False)
    assert cal["display_allowed"], f"심은 신호인데 차단됨: {cal['checks']}"
    assert cal["deciles"][-1]["mean_ret_pct"] > cal["deciles"][0]["mean_ret_pct"]
    # 워크포워드 확인: 가중치 로그가 존재하고 첫 스냅샷들은 (완결 IC 부족으로) 제외됐어야 함
    assert wlog and len(rhos) < 90, "burn-in이 적용되지 않음"

    panel2, spy2, funds2, pit2 = _planted(seed=21)
    rng = np.random.default_rng(3)                       # 노이즈 세계: 수익과 무관한 펀더멘탈
    for s in funds2:
        q = rng.normal()
        for k in ("gross",):
            for pnt in funds2[s][k]:
                pnt["val"] = 1e10 * (0.35 + 0.08 * q)    # quality와 무관하게 재배정
    # 수익 드리프트도 무작위 재생성
    dates = panel2.index
    panel2 = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, len(dates))))
                           for c in panel2.columns}, index=dates)
    pct2, ret2, dts2, rhos2, _ = build_pool(panel2, spy2, funds2, pit2, rebal_days=21,
                                            horizon="6m", min_history=5)
    if len(pct2):
        cal2 = calibrate(pct2, ret2, dts2, rhos2, "6m")
        assert not cal2["display_allowed"], f"노이즈인데 통과: {cal2['checks']}"
    assert load_calibration("/nonexistent") is None
    assert score_from_percentile(0.97) == 10 and score_from_percentile(0.0) == 0
    _log("[self-test] 통과: 신호 통과 · 노이즈 차단 · 워크포워드 burn-in OK")


def main():
    ap = argparse.ArgumentParser(description="v2 점수 캘리브레이션(IR-가중 군집·스냅샷 Spearman)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--horizon", default="6m", choices=list(BW.TD))
    ap.add_argument("--rebal-days", type=int, default=21)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--force", action="store_true", help="PBO/DSR 게이트 무시(연구용)")
    ap.add_argument("--candidates", default="cluster", choices=["cluster", "sr"],
                    help="cluster=기본 퀄리티·주주환원 군집 / "
                         "sr=지지·저항 A1 신호(부록 A2-a, 연구용·예산 분리, 별도 파일에 저장)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not _gate(args.force):
        sys.exit(1)
    pit = BC.load_pit(args.pit_file)
    panel, spy, _ = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    if not funds:
        _log("펀더멘탈 캐시 없음 — 먼저 python fundamentals_edgar.py"); sys.exit(1)
    if args.candidates == "sr":
        import backtest_exec as BE
        candidates, extra_cross, save_path = BE.SR_CANDIDATES, BE.sr_signal_panels(panel), BE.SR_CALIB_PATH
        _log(f"[예산분리] --candidates sr — SR_CANDIDATES {len(candidates)}종, "
             f"결과는 {save_path}에 저장(기본 게이트에 영향 없음)")
    else:
        candidates, extra_cross, save_path = CLUSTER_CANDIDATES, None, CALIB_PATH
    pct, ret, dts, rhos, wlog = build_pool(panel, spy, funds, pit, args.rebal_days, args.horizon,
                                           candidates=candidates, extra_cross=extra_cross)
    _log(f"[풀] 표본 {len(ret)}건 · 점수산출 스냅샷 {len(rhos)}개 · "
         f"최근 가중치 {wlog[-1]['weights'] if wlog else '-'}")
    cal = calibrate(pct, ret, dts, rhos, args.horizon, rebal_days=args.rebal_days)
    cal["as_of"] = pd.Timestamp.today().date().isoformat()
    cal["latest_weights"] = wlog[-1]["weights"] if wlog else None
    report(cal, path=save_path)


if __name__ == "__main__":
    main()
