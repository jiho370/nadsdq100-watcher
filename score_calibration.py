#!/usr/bin/env python3
"""
score_calibration.py — STRATEGY_UPGRADE_PROPOSAL.md 5장: 0~10 추천강도 점수 캘리브레이션.

원칙: 점수는 "과거 데이터가 보여준 기대값의 순위"여야 한다(임의 가중합 금지).

진입 게이트(로드맵 순서 강제): output/pbo_report.json 의 passed=true 여야 실행됨.
  → PBO/DSR 검증을 통과하지 못한 백테스트 위에 점수를 캘리브레이션하는 것을 코드로 차단.
  (연구 목적 우회: --force)

절차(5장 설계 그대로):
  1단계 구성요소(모두 스냅샷 내 백분위):
     · factor_pct  — 채택 가중치(w·z 합성 = select_by_weights 와 동일 구조) 백분위
     · trend_pct   — z(12-1 모멘텀)+z(52주 고점 근접도) 백분위
     · lowvol_pct  — 60일 실현변동성 낮을수록 높음(리스크 감점)
     복합점수 = 0.5·factor + 0.3·trend + 0.2·lowvol
     (AI 가감점 ±1은 라이브 전용 — 과거 데이터에 AI 판정이 없어 캘리브레이션에서 제외)
  2단계 캘리브레이션: 과거 후보 풀 전체에서 복합점수 10분위별
     실측 {평균 수익률, 승률, 평균손익비} 테이블 산출 (horizon 기본 1m)
  3단계 검증: Spearman(점수, 실현수익) 단조성 — ρ>0 이고 p<0.05 일 때만
     display_allowed=true. 아니면 점수를 리포트에 노출하지 않음.
  4단계 표시: 리포트 코드는 load_calibration() 이 None 이면 점수 표시 생략(통합은 추후).

실행(PC): python backtest_costs.py ... && python overfit_stats.py    # 게이트 선행
          python score_calibration.py --years 10 --horizon 1m
          python score_calibration.py --self-test
결과: output/score_calibration.json
"""
from __future__ import annotations
import os, sys, json, math, argparse
from statistics import NormalDist
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC

CALIB_PATH = "output/score_calibration.json"
GATE_PATH = "output/pbo_report.json"
COMP_W = {"factor": 0.5, "trend": 0.3, "lowvol": 0.2}
ND = NormalDist()


def _log(m): print(m, file=sys.stderr)


# ------------------------- 라이브 리포트용 API -------------------------
def load_calibration(path=CALIB_PATH):
    """단조성 검증을 통과한 캘리브레이션만 반환. 아니면 None → 리포트는 점수 표시 생략."""
    try:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        return cal if cal.get("display_allowed") else None
    except Exception:
        return None


def score_from_percentile(pct: float) -> int:
    """복합점수 백분위(0~1) → 0~10 점수."""
    return int(min(max(round(pct * 10), 0), 10))


# ------------------------- 풀 구축 -------------------------
def build_pool(panel, spy, funds, pit, weights, rebal_days=21, horizon="1m"):
    """스냅샷마다 (복합점수 백분위, forward return) 수집. PIT 멤버십·공시시차는
    backtest_costs/fundamentals_edgar 로직을 그대로 재사용(look-ahead 없음)."""
    import tech_factors as T
    hd = BW.TD[horizon]
    use_fund = bool(funds)
    spy = spy.reindex(panel.index).ffill()
    n = len(panel)
    cross = T.build_panels(panel)
    pool_pct, pool_ret, dates = [], [], []
    for p in range(BW.LOOKBACK, n - hd - 1, rebal_days):
        raw = BW._raw_frame(panel, p, funds, use_fund, cross)
        if raw is None or len(raw) < 20:
            continue
        date = panel.index[p].date().isoformat()
        members = BC.membership_asof(pit, date)
        idx = raw.index.intersection(members)
        if len(idx) < 20:
            continue
        raw = raw.loc[idx]
        z = raw.apply(BW._z).fillna(0.0)
        # ① 팩터 합성(w·z) ② 추세(모멘텀+52주 근접) ③ 저변동성
        wv = pd.Series({k: v for k, v in weights.items() if v and k in z.columns})
        if wv.empty:
            continue
        factor = (z[list(wv.index)] * wv).sum(axis=1)
        sub = panel.iloc[p - 251:p + 1][idx]
        prox = sub.iloc[-1] / sub.max() - 1.0
        trend = z["mom12_1"] + BW._z(prox.reindex(idx)).fillna(0.0)
        vol60 = panel.iloc[p - 60:p + 1][idx].pct_change().std()
        comp = (COMP_W["factor"] * factor.rank(pct=True)
                + COMP_W["trend"] * trend.rank(pct=True)
                + COMP_W["lowvol"] * (1.0 - vol60.rank(pct=True).reindex(idx).fillna(0.5)))
        pct = comp.rank(pct=True)                     # 스냅샷 내 백분위(레짐 혼합 방지)
        e = p + 1
        fwd = panel.iloc[e + hd][idx] / panel.iloc[e][idx] - 1
        ok = fwd.dropna().index
        pool_pct += list(pct.reindex(ok)); pool_ret += list(fwd.reindex(ok))
        dates.append(date)
    return np.array(pool_pct, float), np.array(pool_ret, float), dates


# ------------------------- 캘리브레이션 + 단조성 검증 -------------------------
def spearman(x: np.ndarray, y: np.ndarray):
    """Spearman ρ + 양측 p(대표본 t→정규 근사)."""
    n = len(x)
    if n < 10:
        return 0.0, 1.0
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    p = 2 * (1 - ND.cdf(abs(t)))
    return rho, p


def calibrate(pool_pct: np.ndarray, pool_ret: np.ndarray, horizon: str, alpha=0.05):
    dec = np.minimum((pool_pct * 10).astype(int), 9) + 1          # 1~10분위
    table = []
    for d in range(1, 11):
        r = pool_ret[dec == d]
        if len(r) == 0:
            table.append({"decile": d, "n": 0}); continue
        wins, losses = r[r > 0], r[r <= 0]
        payoff = (float(wins.mean() / abs(losses.mean()))
                  if len(wins) and len(losses) and losses.mean() != 0 else None)
        table.append({"decile": d, "score_range": f"{d-1}~{d}",
                      "n": int(len(r)),
                      "mean_ret_pct": round(100 * float(r.mean()), 2),
                      "win_rate_pct": round(100 * float((r > 0).mean()), 1),
                      "payoff": round(payoff, 2) if payoff else None})
    rho, p = spearman(pool_pct, pool_ret)
    allowed = bool(rho > 0 and p < alpha)
    return {"horizon": horizon, "n_samples": int(len(pool_ret)),
            "component_weights": COMP_W, "deciles": table,
            "spearman": {"rho": round(rho, 4), "p_value": round(p, 6), "alpha": alpha},
            "display_allowed": allowed,
            "note": ("단조성 검증 통과 — 리포트 노출 가능" if allowed else
                     "단조성 검증 실패 — 점수를 리포트에 노출하지 말 것(§5 3단계)")}


def report(cal, save=True):
    _log(f"\n=== 0~10 점수 캘리브레이션 (horizon {cal['horizon']} · 표본 {cal['n_samples']}) ===")
    _log(f"{'분위':>4s}{'점수':>7s}{'표본':>8s}{'평균수익%':>10s}{'승률%':>8s}{'손익비':>8s}")
    for row in cal["deciles"]:
        _log(f"{row['decile']:>4d}{row.get('score_range','-'):>7s}{row['n']:>8d}"
             f"{row.get('mean_ret_pct','-'):>10}{row.get('win_rate_pct','-'):>8}"
             f"{row.get('payoff') if row.get('payoff') is not None else '-':>8}")
    sp = cal["spearman"]
    _log(f"  Spearman ρ={sp['rho']} (p={sp['p_value']}) → "
         f"display_allowed={cal['display_allowed']}")
    _log(f"  {cal['note']}")
    if save:
        os.makedirs("output", exist_ok=True)
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        _log(f"\n>>> 저장: {CALIB_PATH}")


def _gate(force=False) -> bool:
    """PBO/DSR 게이트: 통과된 pbo_report.json 없으면 진행 금지(로드맵 3→4 체크포인트)."""
    try:
        with open(GATE_PATH, encoding="utf-8") as f:
            rep = json.load(f)
    except Exception:
        _log(f"[게이트] {GATE_PATH} 없음 — 먼저 backtest_costs.py → overfit_stats.py 실행.")
        return force
    if rep.get("passed"):
        _log(f"[게이트] PBO {rep['pbo']['pbo']:.1%} · DSR {rep['dsr'].get('dsr')} → 통과. 진행.")
        return True
    _log(f"[게이트] 미통과({rep.get('pbo_verdict')} / {rep.get('dsr_verdict')}) — "
         f"검증되지 않은 백테스트 위 점수는 허상(§5). 중단.")
    return force


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] ① 단조 신호 풀 → 노출 허용 / ② 노이즈 풀 → 노출 차단 / ③ 파이프라인")
    rng = np.random.default_rng(42)
    n = 4000
    pct = rng.uniform(0, 1, n)
    good = 0.05 * pct + rng.normal(0, 0.05, n)         # 점수가 진짜로 예측
    cal_g = calibrate(pct, good, "1m")
    assert cal_g["display_allowed"], "단조 신호인데 노출 차단됨"
    top = cal_g["deciles"][-1]["mean_ret_pct"]; bot = cal_g["deciles"][0]["mean_ret_pct"]
    assert top > bot, f"10분위({top}) ≤ 1분위({bot}) — 단조성 위반"
    noise = rng.normal(0, 0.05, n)                     # 점수와 무관
    cal_n = calibrate(pct, noise, "1m")
    assert not cal_n["display_allowed"], \
        f"노이즈인데 노출 허용됨(ρ={cal_n['spearman']['rho']}, p={cal_n['spearman']['p_value']})"
    assert load_calibration("/nonexistent") is None
    assert score_from_percentile(0.97) == 10 and score_from_percentile(0.0) == 0

    panel, spy, funds, _ = BW._synthetic()             # ③ 실제 파이프라인 경로
    pit = BC._synthetic_pit(panel)
    pool_pct, pool_ret, dates = build_pool(panel, spy, funds, pit,
                                           weights={"mom12_1": 2, "gross_margin": 1},
                                           rebal_days=42, horizon="1m")
    assert len(pool_pct) == len(pool_ret) > 500 and len(dates) > 10
    cal = calibrate(pool_pct, pool_ret, "1m")
    report(cal, save=False)
    _log("[self-test] 통과: 단조 허용 · 노이즈 차단 · 풀 구축 OK")


def main():
    ap = argparse.ArgumentParser(description="0~10 점수 캘리브레이션(분위별 실측 기대값)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--horizon", default="1m", choices=list(BW.TD))
    ap.add_argument("--rebal-days", type=int, default=21)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--force", action="store_true", help="PBO/DSR 게이트 무시(연구용)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not _gate(args.force):
        sys.exit(1)
    # 가중치: PIT+비용 재탐색 결과 우선, 없으면 기존 best_weights.json
    weights = None
    for path, key in (("output/backtest_costs_compare.json", "pit_best"),
                      ("output/best_weights.json", None)):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            weights = (d[key]["weights"] if key else d["weights"]); break
        except Exception:
            continue
    if not weights:
        _log("가중치 없음 — 먼저 backtest_costs.py 실행."); sys.exit(1)
    _log(f"[가중치] {BW._wstr(weights)} ({path})")
    pit = BC.load_pit(args.pit_file)
    panel, spy, _ = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    pool_pct, pool_ret, dates = build_pool(panel, spy, funds, pit, weights,
                                           args.rebal_days, args.horizon)
    _log(f"[풀] 스냅샷 {len(dates)}회 · 표본 {len(pool_ret)}건")
    cal = calibrate(pool_pct, pool_ret, args.horizon)
    cal["weights_used"] = weights
    cal["as_of"] = pd.Timestamp.today().date().isoformat()
    report(cal)


if __name__ == "__main__":
    main()
