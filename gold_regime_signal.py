#!/usr/bin/env python3
"""
gold_regime_signal.py — 금 배분타이밍 레짐신호 2·3단계: 분류 로직 + 백테스트.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §레짐 분류 로직 참고.

실행: python gold_regime_signal.py --self-test
"""
from __future__ import annotations
import sys, argparse
import numpy as np
import pandas as pd

CORR_THRESHOLDS = (0.25, 0.35, 0.45)
REBAL_FREQS = ("weekly", "monthly")
WEIGHT_SCALES = {"base": {"strong": 0.20, "mild": 0.10},
                 "wide": {"strong": 0.30, "mild": 0.15}}
DEFAULT_CORR_THRESHOLD = 0.35
DEFAULT_REBAL_FREQ = "weekly"
DEFAULT_WEIGHT_SCALE = "base"
COST_BPS = 5.0


def _log(m): print(f"[금레짐신호] {m}", file=sys.stderr)


def _direction(mom_3m: float, mom_6m: float, mom_12m: float) -> str:
    s = int(np.sign(mom_3m)) + int(np.sign(mom_6m)) + int(np.sign(mom_12m))
    if s >= 2:
        return "UP"
    if s <= -2:
        return "DOWN"
    return "MIXED"


def _score_to_verdict(total: int) -> tuple[str, str | None]:
    if total >= 3:
        return "ADD", "strong"
    if total >= 1:
        return "ADD", "mild"
    if total == 0:
        return "HOLD", None
    if total >= -2:
        return "REDUCE", "mild"
    return "REDUCE", "strong"


def classify(row: pd.Series, corr_threshold: float) -> dict:
    gold_dir = _direction(row["gold_mom_3m"], row["gold_mom_6m"], row["gold_mom_12m"])
    base_score = {"UP": 2, "DOWN": -2, "MIXED": 0}[gold_dir]

    dxy_dir = _direction(row["dxy_mom_3m"], row["dxy_mom_6m"], row["dxy_mom_12m"])
    dxy_alive = abs(row["gold_dxy_corr60"]) >= corr_threshold
    dxy_confirm = (1 if dxy_dir == "DOWN" else -1 if dxy_dir == "UP" else 0) if dxy_alive else 0

    rr_dir = _direction(row["real_rate_mom_3m"], row["real_rate_mom_6m"], row["real_rate_mom_12m"])
    rr_alive = abs(row["gold_realrate_corr60"]) >= corr_threshold
    rr_confirm = (1 if rr_dir == "DOWN" else -1 if rr_dir == "UP" else 0) if rr_alive else 0

    ief_dir = _direction(row["ief_mom_3m"], row["ief_mom_6m"], row["ief_mom_12m"])
    ief_sync = 1 if (gold_dir == "UP" and ief_dir == "UP") else \
               -1 if (gold_dir == "DOWN" and ief_dir == "DOWN") else 0

    unexplained = (not dxy_alive) and (not rr_alive)
    total = base_score if unexplained else base_score + dxy_confirm + rr_confirm + ief_sync

    verdict, strength = _score_to_verdict(total)
    if unexplained and strength == "strong":
        strength = "mild"

    return {"score": total, "verdict": verdict, "strength": strength,
            "gold_direction": gold_dir,
            "dxy_alive": bool(dxy_alive), "dxy_direction": dxy_dir,
            "realrate_alive": bool(rr_alive), "realrate_direction": rr_dir,
            "ief_sync": ief_sync, "unexplained": bool(unexplained),
            "confidence": "low" if unexplained else "normal"}


def target_exposure(verdict: str, strength: str | None, weight_scale: str) -> float:
    if verdict == "HOLD":
        return 1.0
    delta = WEIGHT_SCALES[weight_scale][strength]
    sign = 1.0 if verdict == "ADD" else -1.0
    return round(1.0 + sign * delta, 4)


def build_regime_series(features: pd.DataFrame, corr_threshold: float, rebal_freq: str,
                         weight_scale: str) -> pd.DataFrame:
    if rebal_freq == "weekly":
        decision_idx = features.index
    elif rebal_freq == "monthly":
        decision_idx = features.groupby(features.index.to_period("M")).tail(1).index
    else:
        raise ValueError(f"알 수 없는 rebal_freq: {rebal_freq}")

    rows = []
    for ts in decision_idx:
        c = classify(features.loc[ts], corr_threshold)
        exp = target_exposure(c["verdict"], c["strength"], weight_scale)
        rows.append({"date": ts, "exposure": exp, **c})
    decisions = pd.DataFrame(rows).set_index("date")
    return decisions.reindex(features.index).ffill()


def _mk_row(**kw) -> pd.Series:
    base = {"gold_mom_3m": 0.0, "gold_mom_6m": 0.0, "gold_mom_12m": 0.0,
            "dxy_mom_3m": 0.0, "dxy_mom_6m": 0.0, "dxy_mom_12m": 0.0,
            "real_rate_mom_3m": 0.0, "real_rate_mom_6m": 0.0, "real_rate_mom_12m": 0.0,
            "ief_mom_3m": 0.0, "ief_mom_6m": 0.0, "ief_mom_12m": 0.0,
            "gold_dxy_corr60": 0.0, "gold_realrate_corr60": 0.0}
    base.update(kw)
    return pd.Series(base)


def self_test_regime_series():
    idx = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    rng = np.random.default_rng(3)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 30), "gold_mom_6m": rng.normal(0, 0.05, 30),
        "gold_mom_12m": rng.normal(0, 0.05, 30),
        "dxy_mom_3m": rng.normal(0, 0.02, 30), "dxy_mom_6m": rng.normal(0, 0.02, 30),
        "dxy_mom_12m": rng.normal(0, 0.02, 30),
        "real_rate_mom_3m": rng.normal(0, 0.3, 30), "real_rate_mom_6m": rng.normal(0, 0.3, 30),
        "real_rate_mom_12m": rng.normal(0, 0.3, 30),
        "ief_mom_3m": rng.normal(0, 0.02, 30), "ief_mom_6m": rng.normal(0, 0.02, 30),
        "ief_mom_12m": rng.normal(0, 0.02, 30),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.3, 30),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.3, 30),
    }, index=idx)

    weekly = build_regime_series(feat, 0.35, "weekly", "base")
    assert list(weekly.index) == list(feat.index)
    assert weekly["exposure"].nunique() > 1, "weekly는 주마다 다른 판정이 나올 수 있어야 함"

    monthly = build_regime_series(feat, 0.35, "monthly", "base")
    assert list(monthly.index) == list(feat.index)
    # 첫 월말 결정일 이전 주는 NaN이어야 함(아직 판정 근거가 없음 — 룩어헤드 아님)
    first_decision = feat.groupby(feat.index.to_period("M")).tail(1).index[0]
    before_first = monthly.index < first_decision
    assert monthly.loc[before_first, "exposure"].isna().all(), "첫 결정일 이전은 NaN이어야 함(룩어헤드 방지)"
    after_first = ~before_first
    assert not monthly.loc[after_first, "exposure"].isna().any(), "첫 결정일 이후는 NaN이 없어야 함"

    _log("통과: build_regime_series weekly/monthly 배선 정상")


def self_test():
    # 1) 강한 ADD: 금 UP 다수결 + DXY 살아있고 DOWN + 실질금리 살아있고 DOWN + IEF 동조
    row = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                  dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                  real_rate_mom_3m=-0.3, real_rate_mom_6m=-0.5, real_rate_mom_12m=-0.4,
                  ief_mom_3m=0.02, ief_mom_6m=0.01, ief_mom_12m=0.03,
                  gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c = classify(row, 0.35)
    assert c["verdict"] == "ADD" and c["strength"] == "strong", c
    assert c["score"] == 2 + 1 + 1 + 1, c  # 금+DXY확인+실질금리확인+IEF동조
    assert target_exposure(c["verdict"], c["strength"], "base") == 1.20

    # 2) HOLD: 모든 신호 혼조
    row2 = _mk_row(gold_mom_3m=0.01, gold_mom_6m=-0.01, gold_mom_12m=0.005,
                   gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c2 = classify(row2, 0.35)
    assert c2["verdict"] == "HOLD" and c2["score"] == 0, c2
    assert target_exposure("HOLD", None, "base") == 1.0

    # 3) 설명 안 되는 구간: 상관 둘 다 죽음 → 금 방향만 사용, 강한 등급도 mild로 다운캡
    row3 = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                   dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                   ief_mom_3m=0.02, ief_mom_6m=0.01, ief_mom_12m=0.03,
                   gold_dxy_corr60=-0.1, gold_realrate_corr60=0.05)  # 둘 다 임계값 미만
    c3 = classify(row3, 0.35)
    assert c3["unexplained"] is True and c3["confidence"] == "low", c3
    assert c3["verdict"] == "ADD" and c3["strength"] == "mild", c3  # 금만으로는 score=2→mild
    assert c3["score"] == 2, c3  # DXY/실질금리/IEF 기여 전부 0

    # 4) REDUCE 강: 금 DOWN + DXY 살아있고 UP(약세컨펌) + 실질금리 살아있고 UP + IEF 동조(둘다하락)
    row4 = _mk_row(gold_mom_3m=-0.05, gold_mom_6m=-0.08, gold_mom_12m=-0.10,
                   dxy_mom_3m=0.02, dxy_mom_6m=0.03, dxy_mom_12m=0.01,
                   real_rate_mom_3m=0.3, real_rate_mom_6m=0.5, real_rate_mom_12m=0.4,
                   ief_mom_3m=-0.02, ief_mom_6m=-0.01, ief_mom_12m=-0.03,
                   gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c4 = classify(row4, 0.35)
    assert c4["verdict"] == "REDUCE" and c4["strength"] == "strong", c4
    assert target_exposure(c4["verdict"], c4["strength"], "base") == 0.80

    _log("통과: classify/target_exposure 4개 시나리오(ADD강/HOLD/설명안됨/REDUCE강) 정상")
    self_test_regime_series()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
