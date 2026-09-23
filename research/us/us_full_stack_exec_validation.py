#!/usr/bin/env python3
"""
us_full_stack_exec_validation.py — 지호 님 요청(2026-09-24, ChatGPT 리뷰 계기):
"최종 조합(가중치+floor+100일선+topn10+분할매수+6개월재평가) 전체를 한 번에 검증".

배경: backtest_exec.py의 원래 21조합(entry×exit) 검증(Stage 2/7, HISTORY.md)은 floor·
100일선 필터가 생기기 전(7월) 선정 로직(_select_basket — 가중합성 점수 topN만, 필터 없음)
으로 돌렸다. floor=3.25·100일선 필터는 그 뒤(9월) 별도의 가벼운 "이벤트평균" 프레임으로만
검증됐지, 실제 체결(분할매수 채움 여부·손절·MDD)까지 시뮬레이션하는 backtest_exec.py의
무거운 프레임으로 "지금 라이브에 켜진 선정 로직 + 지금 라이브에 켜진 진입/청산 방식"을
한 번에 합쳐서 검증한 적은 없었다 — 이 공백을 메운다.

방법: backtest_exec.run_exec()의 select_fn 주입을 이용해 라이브와 동일한 선정
(가중합성 점수 + floor 3.25 + 100일선 위 + 섹터캡 없음 + topn10)을 넣고, 기존 21조합
(entry1_full/entry2_pullback2/entry3_pullback3 × 7 exit)을 그대로 돌려 PBO/DSR까지
동일 파이프라인(overfit_stats.OS.analyze)으로 낸다.

한계(정직히 명시): entry_plan.py의 실제 라이브 규칙은 "과열(RSI≥72 등)이면 3분할, 아니면
2분할"인 종목별 조건분기인데, backtest_exec.py의 entry2/3은 전체 바스켓에 하나의 규칙을
고정 적용하는 전역 스윕이라 이 조건분기 자체는 여기서도 재현 못 한다(HISTORY.md §7.1에서
이미 별도 검증 시도됐고 결론 보류 상태 — 동일 한계 승계).

실행: python -m research.us.us_full_stack_exec_validation [--years 10]
결과: output/backtest_exec_compare_us_livestack.json,
      output/backtest_exec_trials_us_livestack.json,
      output/backtest_exec_pbo_us_livestack.json
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import backtest_exec as BE
from research.us.us_momentum_overlay import _mom_features

FLOOR = 3.25


def _log(m): print(f"[US풀스택검증]  {m}", file=sys.stderr)


def _select_basket_livestack(panel, p, funds, cross, pit, weights, topn):
    """라이브와 동일: 가중합성 점수 + floor(3.25) + 100일선 위 + 섹터캡 없음 + topn.
    backtest_exec._select_basket에 floor·100일선 필터만 추가한 버전."""
    raw = BW._raw_frame(panel, p, funds, bool(funds), cross)
    if raw is None or raw.empty:
        return []
    date = panel.index[p].date().isoformat()
    idx = raw.index.intersection(BC.membership_asof(pit, date))
    if len(idx) < topn:
        return []
    raw = raw.loc[idx]
    w = {k: v for k, v in weights.items() if k in raw.columns}
    if not w:
        return []
    z = raw[list(w)].apply(BW._z).fillna(0.0)
    score = (z * pd.Series(w)).sum(axis=1)
    score = score[score >= FLOOR]
    if len(score) < topn:
        return []
    mom = _mom_features(panel, date, score.index)
    above = mom["ma100_gap"] > 0 if "ma100_gap" in mom.columns else pd.Series(True, index=score.index)
    score = score[above.reindex(score.index).fillna(False)]
    if len(score) < topn:
        return []
    return list(score.sort_values(ascending=False).index[:topn])


def run(years: float = 10, topn: int = 10) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = BE._load_exec_weights()
    _log(f"라이브 가중치 사용: {weights} · topn={topn}")

    import tech_factors as T
    cross = T.build_panels(panel)
    select_fn = lambda p: _select_basket_livestack(panel, p, funds, cross, pit, weights, topn)

    payload, report = BE.run_exec(panel, spy, funds, pit, rebal_days=63, topn=topn, cost=cost,
                                  select_fn=select_fn, out_suffix=f"_us_livestack_top{topn}")
    _log(f"[결과] baseline={payload['baseline']}")
    for row in payload["rows"]:
        _log(f"{row['entry']}__{row['exit']}: net {row['net_pct']:+.2f}% MDD {row['mdd_pct']:.1f}% "
            f"미체결 {row['unfilled_pct']:.1f}% n={row['n_events']}")
    if report:
        _log(f"PBO={report.get('pbo_pct')}% DSR={report.get('dsr')} 통과={report.get('passed')} "
            f"최고시행={report.get('best_trial')}")
    return {"exec": payload, "pbo_dsr": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()
    run(years=args.years, topn=args.topn)


if __name__ == "__main__":
    main()
