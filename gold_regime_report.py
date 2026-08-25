#!/usr/bin/env python3
"""
gold_regime_report.py — 금 배분타이밍 레짐신호 5단계: 최종 리포트 + 현재 시점 판정.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §5단계 참고.

실행: python gold_regime_report.py
      python gold_regime_report.py --self-test
결과: output/gold_regime_summary.json, output/gold_regime_report.md
"""
from __future__ import annotations
import os, sys, json, argparse
import pandas as pd

from gold_regime_data import load_or_build, fetch_all
from gold_regime_signal import (
    classify, target_exposure, build_regime_series, era_performance,
    DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE, COST_BPS,
)
import gold_regime_overfit_gate as OG

SUMMARY_PATH = "output/gold_regime_summary.json"
REPORT_PATH = "output/gold_regime_report.md"


def _log(m): print(f"[금레짐리포트] {m}", file=sys.stderr)


def current_judgment(features: pd.DataFrame, corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> dict:
    last_ts = features.index[-1]
    c = classify(features.loc[last_ts], corr_threshold)
    return {"as_of": str(last_ts.date()), **c,
            "target_exposure": target_exposure(c["verdict"], c["strength"], DEFAULT_WEIGHT_SCALE)}


def _render_markdown(summary: dict) -> str:
    lines = ["# 금 배분타이밍 레짐신호 — 백테스트 리포트\n"]
    cj = summary["current_judgment"]
    lines.append(f"## 현재 판정 ({cj['as_of']} 기준)\n")
    lines.append(f"- **{cj['verdict']}"
                 f"{'(' + cj['strength'] + ')' if cj['strength'] else ''}** "
                 f"— 권고 노출 {cj['target_exposure']*100:.0f}%")
    lines.append(f"- 금 방향: {cj['gold_direction']} · "
                 f"DXY {'살아있음' if cj['dxy_alive'] else '죽음'}({cj['dxy_direction']}) · "
                 f"실질금리 {'살아있음' if cj['realrate_alive'] else '죽음'}({cj['realrate_direction']}) · "
                 f"IEF동조 {cj['ief_sync']}")
    lines.append(f"- 신뢰도: {cj['confidence']}"
                 f"{' (상관관계 붕괴 — 금 추세만으로 보수적 판단)' if cj['unexplained'] else ''}\n")

    lines.append("## 국면별 성과\n")
    lines.append("| 국면 | 일수 | 신호CAGR | Buy&Hold CAGR | 고정비중CAGR | 신호Ulcer | "
                 "BH Ulcer | 설명안됨% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in summary["eras"]:
        if "note" in e:
            lines.append(f"| {e['era']} | {e['n_days']} | {e['note']} | | | | | |")
            continue
        lines.append(f"| {e['era']} | {e['n_days']} | {e['signal_cagr']}% | {e['buyhold_cagr']}% | "
                     f"{e['fixed_weight_cagr']}% | {e['signal_ulcer']} | {e['buyhold_ulcer']} | "
                     f"{e['unexplained_pct']}% |")

    gate = summary["overfit_gate"]
    dsr = gate["dsr_pbo"]["dsr"]
    lines.append(f"\n## 과최적화 검증\n")
    lines.append(f"- 시도한 파라미터 조합: {gate['n_param_combos']}개")
    lines.append(f"- DSR: {dsr.get('dsr')} (95% 통과 기준 0.95) — {gate['dsr_pbo']['dsr_verdict']}")
    lines.append(f"- PBO(참고): {gate['dsr_pbo']['pbo']['pbo']} — {gate['dsr_pbo']['pbo_verdict']}")
    lines.append(f"- Walk-forward: {gate['walk_forward']['n_folds']}개 폴드, "
                 f"OOS 평균 CAGR {gate['walk_forward']['oos_cagr_mean']}%")
    lines.append(f"- 비용 민감도: {gate['cost_sensitivity']}")
    lines.append("\n## 리스크\n")
    lines.append("- DXY/실질금리-금 상관관계가 앞으로도 붕괴 상태를 유지하거나 다시 강화될지 불확실.")
    lines.append("- 2022년 이후 구조변화(중앙은행 매입 급증)가 지속될지는 이 백테스트로 확정할 수 없음.")
    return "\n".join(lines) + "\n"


def build_report(save: bool = True) -> dict:
    features = load_or_build()
    gold_daily = fetch_all()["gold"]
    regime = build_regime_series(features, DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE)
    eras = era_performance(gold_daily, regime, COST_BPS)
    gate = OG.run(save=False)
    current = current_judgment(features)
    summary = {"eras": eras, "overfit_gate": gate, "current_judgment": current}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(_render_markdown(summary))
        _log(f"저장: {SUMMARY_PATH}, {REPORT_PATH}")
    return summary


def self_test():
    from gold_regime_signal import _mk_row  # 재사용 — 합성 row
    row = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                  dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                  gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    feat = pd.DataFrame([row], index=[pd.Timestamp("2026-08-21")])
    j = current_judgment(feat)
    assert j["as_of"] == "2026-08-21"
    assert j["verdict"] in ("ADD", "HOLD", "REDUCE")
    assert "target_exposure" in j
    _log("통과: current_judgment 배선 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        build_report()
