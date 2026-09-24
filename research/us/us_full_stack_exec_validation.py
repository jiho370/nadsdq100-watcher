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

2026-09-24 개정(전략 검토 E — 이 검증이 라이브와 달랐던 5가지를 라이브 함수로 통일):
  · 주주환원 z 클립 ±3 → 라이브와 같은 export_data.live_z(±5)
  · 후보가 topn보다 적으면 이벤트를 통째로 버리던 것 → 있는 만큼만 사고 빈 슬롯은 현금
    (후보 0이면 전액 현금 이벤트 — 라이브도 필터 통과 0이면 하이브리드 폴백 없이 현금 대기)
  · 100일선 값이 없는 종목 통과 → 제외(라이브도 동일하게 변경)
  · 전 종목 고정 2/3분할 → entry_live(과열이면 3분할·아니면 2분할, entry_plan.tranche_targets
    가격·200일선 하한) 추가
  · 126거래일 고정 청산 → exit_live(180달력일 경과 AND 후보풀(상위 60) 이탈 시 매도) 추가
  기존 21조합도 그대로 같이 돌려 4×8=32조합으로 PBO/DSR을 낸다(시행 수가 늘어난 만큼 DSR
  보정이 더 엄격해진다 — 라이브 조합만 따로 떼어 보는 것보다 정직).
남은 한계: 이벤트별 독립 평가라 보유상한 10·"팔아야 산다"·이벤트 간 재투자는 반영 안 됨
  (계좌 NAV 검증은 research/us/backtest_portfolio.py 몫), 후보풀 재확인은 5거래일 간격 근사,
  재무 스냅샷의 분할 기준은 fundamentals_cache의 splits 수집 여부에 좌우됨.

실행: python -m research.us.us_full_stack_exec_validation [--years 10]
결과: output/backtest_exec_compare_us_livestack_top{N}.json,
      output/trial_returns_exec_us_livestack_top{N}.json,
      output/pbo_report_exec_us_livestack_top{N}.json
"""
from __future__ import annotations
import argparse
import sys

import backtest_costs as BC
import backtest_weights as BW
import backtest_exec as BE

POOL_N = 60         # 라이브 후보풀 크기(daily_ai_report.MAX_CANDIDATES 기본값) — 재평가 매도 기준


def _log(m): print(f"[US풀스택검증]  {m}", file=sys.stderr)


def _live_ranked(panel, p, funds, cross, pit, weights):
    """라이브 export_data.select_by_weights와 같은 규칙으로 필터 통과 종목을 점수순으로.
    데이터 자체가 없으면 None(이벤트 제외), 필터 통과 0이면 [](현금)."""
    import export_data as E
    raw = BW._raw_frame(panel, p, funds, bool(funds), cross)
    if raw is None or raw.empty:
        return None
    date = panel.index[p].date().isoformat()
    idx = raw.index.intersection(BC.membership_asof(pit, date))
    if not len(idx):
        return None
    raw = raw.loc[idx]
    w = {k: v for k, v in weights.items() if k in raw.columns}
    if not w:
        return None
    score = sum(float(v) * E.live_z(raw[k], k) for k, v in w.items())
    score = score[score >= E.SCORE_FLOOR]
    gap = raw["ma100_gap"].reindex(score.index) if "ma100_gap" in raw.columns else None
    if gap is not None:
        score = score[gap > 0]          # 값 없음(NaN)은 통과 못 함 — 라이브와 동일
    return list(score.sort_values(ascending=False).index)


def run(years: float = 10, topn: int = 10) -> dict:
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = BE._load_exec_weights()
    _log(f"라이브 가중치 사용: {weights} · topn={topn} · 후보풀 {POOL_N}")

    import tech_factors as T
    cross = T.build_panels(panel)
    ranked = {}

    def _rank(p):
        if p not in ranked:
            ranked[p] = _live_ranked(panel, p, funds, cross, pit, weights)
        return ranked[p]

    select_fn = lambda p: None if _rank(p) is None else _rank(p)[:topn]
    pool_fn = lambda d: None if _rank(d) is None else set(_rank(d)[:POOL_N])

    payload, report = BE.run_exec(panel, spy, funds, pit, rebal_days=63, topn=topn, cost=cost,
                                  select_fn=select_fn, out_suffix=f"_us_livestack_top{topn}",
                                  entries=BE.ENTRY_RULES + [BE.LIVE_ENTRY],
                                  exits=BE.EXIT_RULES + [BE.LIVE_EXIT], pool_fn=pool_fn)
    _log(f"[결과] 라이브 조합 = {BE.LIVE_ENTRY}__{BE.LIVE_EXIT}")
    for row in payload["rows"]:
        _log(f"{row['entry']}__{row['exit']}: net {row['net_pct']:+.2f}% 바스켓MDD {row['basket_mdd_pct']:.1f}% "
            f"미체결 {row['unfilled_pct']:.1f}% n={row['n_events']}")
    if report:
        pbo, dsr = report.get("pbo") or {}, report.get("dsr") or {}
        _log(f"PBO={pbo.get('pbo')} DSR={dsr.get('dsr')} 통과={report.get('passed')} "
            f"최고시행={dsr.get('best_trial')}")
    return {"exec": payload, "pbo_dsr": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()
    run(years=args.years, topn=args.topn)


if __name__ == "__main__":
    main()
