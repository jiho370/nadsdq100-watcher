#!/usr/bin/env python3
"""
overfit_stats.py — STRATEGY_UPGRADE_PROPOSAL.md 3장·로드맵 3단계: 과최적화 통계 검증.

입력: output/trial_returns.json (backtest_costs.py가 저장 — 가중치 조합별 이벤트 순초과수익 행렬)
계산:
  1) PBO (Probability of Backtest Overfitting) — Bailey, Borwein, López de Prado & Zhu(2017)
     CSCV: 이벤트를 S개 블록으로 나눠 C(S, S/2)개 조합마다
       IS에서 샤프 최고 조합 선택 → 그 조합의 OOS 순위 ω → λ = ln(ω/(1−ω))
     PBO = P(λ ≤ 0) = "IS 1등이 OOS에서 중앙값 이하로 떨어질 확률". 0.5면 순수 운.
  2) DSR (Deflated Sharpe Ratio) — Bailey & López de Prado(2014)
     시도한 조합 수 N·조합 간 샤프 분산·표본 수·왜도·첨도를 반영해
     "보고된 샤프가 다중검정을 감안해도 0보다 유의하게 큰가"의 확률.
     DSR ≥ 0.95 = 95% 신뢰수준 통과.
주의: 조합들이 서로 상관되어 있어(같은 팩터 공유) N은 '독립 시행 수'보다 과대 → DSR은 보수적.

실행(PC): python backtest_costs.py --years 10 ...   # 먼저 (trial_returns.json 생성)
          python overfit_stats.py
          python overfit_stats.py --self-test
결과: output/pbo_report.json
"""
from __future__ import annotations
import os, sys, json, math, argparse, itertools
from statistics import NormalDist
import numpy as np

TRIALS_PATH = "output/trial_returns.json"
REPORT_PATH = "output/pbo_report.json"
ND = NormalDist()
EULER = 0.5772156649015329


def _log(m): print(m, file=sys.stderr)


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


# ------------------------- PBO (CSCV) -------------------------
def pbo_cscv(M: np.ndarray, n_blocks: int = 12):
    """M: (N조합 × T이벤트). 반환: dict(pbo, n_combos, λ 요약, 성과저하, OOS손실확률)."""
    N, T = M.shape
    S = min(n_blocks, T if T % 2 == 0 else T - 1)
    S = max(S - (S % 2), 4)                       # 짝수, 최소 4
    if T < S:
        raise RuntimeError(f"이벤트 {T}회 < 블록 {S}개 — 데이터가 너무 짧음")
    edges = np.array_split(np.arange(T), S)       # 블록 경계(순서 유지 — 시계열 구조 보존)

    def _sharpe_all(idx):                          # 전 조합 벡터화 샤프
        sub = M[:, idx]
        sd = sub.std(axis=1, ddof=1)
        return np.where(sd > 0, sub.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0)

    lambdas, is_best_oos, pairs = [], [], []
    for combo in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([edges[i] for i in combo])
        oos_idx = np.concatenate([edges[i] for i in range(S) if i not in combo])
        sr_is = _sharpe_all(is_idx)
        sr_oos = _sharpe_all(oos_idx)
        n_star = int(np.argmax(sr_is))
        # OOS 상대순위 ω ∈ (0,1): 1에 가까울수록 OOS에서도 1등
        omega = np.sum(sr_oos <= sr_oos[n_star]) / (N + 1)
        omega = min(max(omega, 1.0 / (N + 1)), N / (N + 1))
        lambdas.append(math.log(omega / (1 - omega)))
        is_best_oos.append(float(M[n_star, oos_idx].mean()))
        pairs.append((float(sr_is[n_star]), float(sr_oos[n_star])))
    lam = np.array(lambdas)
    a = np.array(pairs)
    slope = None                                   # IS 최고 조합의 IS→OOS 샤프 저하 기울기
    if a[:, 0].std() > 0:
        slope = float(np.polyfit(a[:, 0], a[:, 1], 1)[0])
    return {"pbo": round(float((lam <= 0).mean()), 4),
            "n_combos": len(lambdas), "n_blocks": S,
            "lambda_mean": round(float(lam.mean()), 4),
            "is_sharpe_mean": round(float(a[:, 0].mean()), 4),
            "oos_sharpe_mean": round(float(a[:, 1].mean()), 4),
            "degradation_slope": round(slope, 4) if slope is not None else None,
            "prob_oos_loss": round(float((np.array(is_best_oos) < 0).mean()), 4)}


# ------------------------- DSR -------------------------
def deflated_sharpe(best: np.ndarray, all_sharpes: np.ndarray):
    """best: 채택 조합의 이벤트 수익률. all_sharpes: 전체 조합의 (비연율화) 샤프."""
    T = len(best)
    sr = _sharpe(best)
    x = best - best.mean()
    sd = best.std(ddof=1)
    g3 = float((x ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    g4 = float((x ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    N = len(all_sharpes)
    var_sr = float(np.var(all_sharpes, ddof=1)) if N > 1 else 0.0
    # 기대 최대 샤프(순수 운으로 N회 시도): E[max] ≈ √V·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e)))
    if N > 1 and var_sr > 0:
        sr0 = math.sqrt(var_sr) * ((1 - EULER) * ND.inv_cdf(1 - 1.0 / N)
                                   + EULER * ND.inv_cdf(1 - 1.0 / (N * math.e)))
    else:
        sr0 = 0.0
    denom = 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2
    if denom <= 0 or T < 3:
        return {"sr": round(sr, 4), "sr0": round(sr0, 4), "dsr": None,
                "skew": round(g3, 3), "kurtosis": round(g4, 3), "T": T, "N": N,
                "note": "분모≤0 또는 표본 부족 — DSR 계산 불가"}
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)
    return {"sr": round(sr, 4), "sr0": round(sr0, 4), "dsr": round(ND.cdf(z), 4),
            "skew": round(g3, 3), "kurtosis": round(g4, 3), "T": T, "N": N}


# ------------------------- 실행 -------------------------
def analyze(data: dict, n_blocks=12, save=True):
    trials = data["trials"]
    M = np.array(data["excess_returns"], dtype=float)
    _log(f"[입력] 조합 {M.shape[0]}개 × 이벤트 {M.shape[1]}회 "
         f"(horizon {data.get('horizon')} · {data.get('universe')} · {data.get('cost')})")
    pbo = pbo_cscv(M, n_blocks)
    all_sr = np.array([_sharpe(M[i]) for i in range(M.shape[0])])
    best_i = int(np.argmax(all_sr))
    dsr = deflated_sharpe(M[best_i], all_sr)

    pbo_verdict = ("낮음(과최적화 가능성 작음)" if pbo["pbo"] < 0.2 else
                   "중간(주의)" if pbo["pbo"] < 0.5 else
                   "높음(IS 1등이 운일 확률 높음 — 채택 보류 권장)")
    dsr_verdict = ("계산 불가" if dsr.get("dsr") is None else
                   "95% 신뢰수준 통과" if dsr["dsr"] >= 0.95 else
                   "통과 실패(다중검정 감안 시 샤프가 유의하지 않음)")
    _log(f"\n=== PBO (CSCV, 블록 {pbo['n_blocks']}·조합 {pbo['n_combos']}) ===")
    _log(f"  PBO = {pbo['pbo']:.1%} → {pbo_verdict}")
    _log(f"  IS샤프 평균 {pbo['is_sharpe_mean']} → OOS샤프 평균 {pbo['oos_sharpe_mean']} "
         f"(저하 기울기 {pbo['degradation_slope']}) · OOS 손실확률 {pbo['prob_oos_loss']:.1%}")
    _log(f"\n=== DSR (최고샤프 조합: {trials[best_i]}) ===")
    _log(f"  SR {dsr['sr']} vs 운으로 기대되는 최대 SR₀ {dsr['sr0']} "
         f"(시행 {dsr['N']}회 · 표본 {dsr['T']}) → DSR = {dsr.get('dsr')} → {dsr_verdict}")

    report = {"input": {k: data.get(k) for k in ("horizon", "universe", "cost")},
              "n_trials": M.shape[0], "n_events": M.shape[1],
              "pbo": pbo, "pbo_verdict": pbo_verdict,
              "dsr": {**dsr, "best_trial": trials[best_i]}, "dsr_verdict": dsr_verdict,
              "passed": bool(dsr.get("dsr") is not None and dsr["dsr"] >= 0.95
                             and pbo["pbo"] < 0.5),
              "references": ["Bailey et al. 2017 (PBO/CSCV)",
                             "Bailey & López de Prado 2014 (DSR)"],
              "note": "조합 간 상관으로 N이 과대계상될 수 있어 DSR은 보수적(실제보다 낮게) 추정됨."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _log(f"\n>>> 저장: {REPORT_PATH} (passed={report['passed']} — "
             f"score_calibration.py 진행 게이트)")
    return report


# ------------------------- self-test -------------------------
def self_test():
    """순수 노이즈 vs 진짜 알파 시나리오에서 PBO/DSR 방향성 점검(시드 5개 평균 —
    CSCV 조합들이 서로 겹쳐 단일 시드 PBO는 분산이 크기 때문)."""
    _log("[self-test] 합성 시나리오 2종 × 시드 5개로 PBO/DSR 방향성 점검")
    N, T = 60, 48
    pn, ps, dn, ds = [], [], [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.05, (N, T))                   # 전부 순수 노이즈
        signal = noise.copy()
        signal[:5] += 0.04                                      # 5개 조합엔 진짜 알파
        for M, ps_, ds_ in ((noise, pn, dn), (signal, ps, ds)):
            ps_.append(pbo_cscv(M)["pbo"])
            sr = np.array([_sharpe(M[i]) for i in range(N)])
            ds_.append(deflated_sharpe(M[int(np.argmax(sr))], sr).get("dsr") or 0.0)
    p_n, p_s, d_n, d_s = map(lambda v: float(np.mean(v)), (pn, ps, dn, ds))
    assert p_n > 0.3, f"노이즈 평균 PBO가 너무 낮음({p_n:.2f}) — 0.5 근처여야 정상"
    assert p_s < p_n, f"진짜 알파 PBO({p_s:.2f})가 노이즈({p_n:.2f})보다 낮아야 함"
    assert d_s > d_n, f"진짜 알파 DSR({d_s:.2f})이 노이즈 DSR({d_n:.2f})보다 높아야 함"
    _log(f"[self-test] 통과: PBO 노이즈 {p_n:.2f} vs 알파 {p_s:.2f} · "
         f"DSR 노이즈 {d_n:.2f} vs 알파 {d_s:.2f}")


def main():
    ap = argparse.ArgumentParser(description="PBO(CSCV) + Deflated Sharpe Ratio")
    ap.add_argument("--input", default=TRIALS_PATH)
    ap.add_argument("--blocks", type=int, default=12, help="CSCV 블록 수(짝수)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not os.path.exists(args.input):
        _log(f"입력 없음: {args.input} — 먼저 python backtest_costs.py 를 실행하세요."); sys.exit(1)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    analyze(data, n_blocks=args.blocks)


if __name__ == "__main__":
    main()
