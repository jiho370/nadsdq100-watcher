#!/usr/bin/env python3
"""
us_spmo_vs_spy_prereg.py — "SPMO가 SPY보다 장기적으로 낫다"는 주장 자체의 사전등록 검증
(2026-07-23, 지호 님 요청)

배경: STRATEGY.md §6-C는 "SPMO 절대샤프 1.193 > SPY 1.091, 전체기간 누적 SPMO +523.7%
vs SPY +303.2%"라고 적었지만 이건 점추정치 나열일 뿐 유의성 검정을 거친 적이 없다.
us_spmo_blend_prereg.py(알고리즘 vs 알고리즘+SPMO 블렌드)와는 별개 질문 — 이 스크립트는
"SPMO 자체가 SPY 자체를 유의하게 이기는가"만 독립적으로 검증한다.

사전등록(실행 전 확정):
  가설: SPMO(모멘텀 ETF) 원시가격 buy&hold가 SPY 원시가격 buy&hold보다 위험조정
        성과(CAGR)가 통계적으로 유의하게 낫다.
  비교 대상: SPMO vs SPY 단 1개 쌍 — 다른 모멘텀 ETF·다른 벤치마크 추가 스캔 없음.
  데이터: 둘 다 yfinance 수정종가(배당·분할 반영) buy&hold, 레짐타이밍 없음, SPMO 상장
          (2015-10) 이후 공통 최대 가용기간.
  방법: 월간(21거래일) 비중첩 수익률 페어드 t검정 + 짝지은 블록부트스트랩(6개월 블록,
        5000회) — us_spmo_blend_prereg.py와 동일 방법론.
  판정규칙(셋 다 충족해야 "SPMO 우위 확인" — 전부 "SPMO가 SPY를 이긴다" 방향 기준):
    ① CAGR 차이(SPMO-SPY) 95%CI 하한 > 0
    ② 페어드 t ≥ +1.96
    ③ 서브기간 3구간(~2019 / 2020-2023 / 2024+) 전부 SPMO 우위
  단일 사전등록 시행 — PBO/DSR 대상 아님(후보가 1개뿐).

실행: python us_spmo_vs_spy_prereg.py
결과: output/us_spmo_vs_spy_prereg.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import core_satellite_kr as CS
import sp500_daily_report as R

BLOCK = 6
N_BOOT = 5000
SEED = 42
MONTH = 21
SUBS = [("~2019", None, "2019-12-31"), ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024+", "2024-01-01", None)]


def _log(m): print(f"[SPMOvsSPY사전등록] {m}", file=sys.stderr)


def _load():
    hist = R.download_histories(["SPY", "SPMO"], period="max")
    spy, spmo = hist.get("SPY"), hist.get("SPMO")
    if spy is None or spmo is None or spy.empty or spmo.empty:
        raise RuntimeError("SPY/SPMO 시세 조회 실패")
    idx = spy.index.intersection(spmo.reindex(spy.index).ffill().dropna().index)
    if len(idx) < 60:
        raise RuntimeError(f"공통구간 부족(n={len(idx)})")
    spy_nav = (spy.reindex(idx)); spy_nav = spy_nav / spy_nav.iloc[0]
    spmo_nav = (spmo.reindex(idx)); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")
    return spy_nav, spmo_nav


def _monthly_returns(nav: pd.Series) -> np.ndarray:
    return np.array([nav.iloc[t + MONTH] / nav.iloc[t] - 1
                     for t in range(0, len(nav) - MONTH, MONTH)])


def _cagr_from_monthly(sample: np.ndarray) -> float:
    yrs = len(sample) / 12
    return float(np.prod(1 + sample) ** (1 / yrs) - 1) * 100


def _sharpe_from_monthly(sample: np.ndarray) -> float:
    return float(sample.mean() / sample.std() * np.sqrt(12)) if sample.std() else 0.0


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    n = len(d)
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = float(d.mean()) / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def run(save=True):
    spy_nav, spmo_nav = _load()

    full_spy, full_spmo = CS.stats(spy_nav), CS.stats(spmo_nav)
    _log(f"SPY: CAGR {full_spy['cagr_pct']}% 샤프 {full_spy['sharpe']} MDD {full_spy['mdd_pct']}%")
    _log(f"SPMO: CAGR {full_spmo['cagr_pct']}% 샤프 {full_spmo['sharpe']} MDD {full_spmo['mdd_pct']}%")

    r_spy, r_spmo = _monthly_returns(spy_nav), _monthly_returns(spmo_nav)
    n = min(len(r_spy), len(r_spmo))
    r_spy, r_spmo = r_spy[:n], r_spmo[:n]

    tstat, pval = _paired_ttest(r_spmo, r_spy)
    _log(f"페어드 t검정(SPMO-SPY, 월간수익률, n={n}): t={tstat:+.2f} p={pval:.3f}")

    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    cagr_diffs = np.empty(N_BOOT)
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = _cagr_from_monthly(r_spmo[bidx]) - _cagr_from_monthly(r_spy[bidx])
        sharpe_diffs[i] = _sharpe_from_monthly(r_spmo[bidx]) - _sharpe_from_monthly(r_spy[bidx])

    cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    cagr_mean = float(cagr_diffs.mean())
    cagr_pos = float((cagr_diffs > 0).mean()) * 100
    sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    sharpe_mean = float(sharpe_diffs.mean())

    _log(f"CAGR 차이(SPMO-SPY) 95%CI: [{cagr_lo:+.2f}%p, {cagr_hi:+.2f}%p] (평균 {cagr_mean:+.2f}%p, "
         f"{N_BOOT}회 중 {cagr_pos:.1f}%가 양수)")
    _log(f"샤프 차이(SPMO-SPY) 95%CI: [{sharpe_lo:+.3f}, {sharpe_hi:+.3f}] (평균 {sharpe_mean:+.3f})")

    sub_rows, signs = [], []
    for label, a, b in SUBS:
        s_spy, s_spmo = CS.stats(spy_nav, a, b), CS.stats(spmo_nav, a, b)
        if s_spy is None or s_spmo is None:
            sub_rows.append({"period": label, "note": "표본 부족"})
            continue
        d_cagr = s_spmo["cagr_pct"] - s_spy["cagr_pct"]
        d_sharpe = s_spmo["sharpe"] - s_spy["sharpe"]
        signs.append(d_cagr > 0)
        sub_rows.append({"period": label, "spy_cagr": s_spy["cagr_pct"], "spmo_cagr": s_spmo["cagr_pct"],
                         "cagr_diff": round(d_cagr, 2), "spy_sharpe": s_spy["sharpe"],
                         "spmo_sharpe": s_spmo["sharpe"], "sharpe_diff": round(d_sharpe, 3)})
        _log(f"{label}: CAGR차이 {d_cagr:+.2f}%p · 샤프차이 {d_sharpe:+.3f}")
    subperiod_consistent = len(signs) >= 2 and all(signs)

    gate1_ci_positive = cagr_lo > 0
    gate2_ttest_sig = tstat >= 1.96
    gate3_subperiod = subperiod_consistent
    passed = gate1_ci_positive and gate2_ttest_sig and gate3_subperiod
    direction = "SPMO 우위" if cagr_mean > 0 else "SPY 우위"
    rejected_opposite = (not passed) and cagr_hi < 0
    verdict = "SPMO 우위 확인" if passed else ("가설 기각(SPY가 유의하게 우위)" if rejected_opposite else "판정 보류(유의한 차이 없음)")

    _log(f"판정 — ①CI하한>0:{gate1_ci_positive} ②t≥+1.96:{gate2_ttest_sig}({tstat:+.2f}) "
         f"③서브기간 SPMO우위 일관:{gate3_subperiod} → 방향:{direction} → 최종:{verdict}")

    payload = {
        "as_of": spy_nav.index[-1].date().isoformat(), "n_months": n,
        "prereg": {
            "hypothesis": "SPMO(모멘텀ETF) buy&hold가 SPY buy&hold보다 CAGR이 유의하게 낫다",
            "decision_rule": "①CAGR차이 95%CI 하한>0 AND ②paired t>=+1.96 AND ③서브기간 3개 전부 SPMO우위",
        },
        "full_period": {"spy": full_spy, "spmo": full_spmo},
        "paired_ttest": {"t": round(float(tstat), 3), "p": round(float(pval), 4), "n": n},
        "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1),
                                "n_boot": N_BOOT, "block_months": BLOCK},
        "sharpe_diff_bootstrap": {"mean": round(sharpe_mean, 3), "ci95_lo": round(sharpe_lo, 3),
                                  "ci95_hi": round(sharpe_hi, 3), "n_boot": N_BOOT, "block_months": BLOCK},
        "subperiods": sub_rows,
        "gates": {"g1_ci_lo_positive": gate1_ci_positive, "g2_ttest_significant_favorable": gate2_ttest_sig,
                 "g3_subperiod_consistent_favorable": gate3_subperiod},
        "direction": direction, "passed": passed, "verdict": verdict,
        "note": "단일 사전등록 시행 — PBO/DSR 다중검정 게이트 적용 대상 아님. 게이트는 전부 "
                "'SPMO가 SPY를 이긴다'는 가설 방향으로 정의(반대로 유의해도 채택 아님).",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_vs_spy_prereg.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    run()
