#!/usr/bin/env python3
"""
kr_entry_exit_sweep.py — 국장 새틀라이트 매수/매도 '집행 방식' 검증 (2026-07-16).

배경(지호 님 질문): "매수 분할비율·매도 분할여부도 한국 따로 검증해야지. 웬만하면 매수/매도
맞추는 게 좋을거 같은데." — 미국(S&P500)은 backtest_exec.py로 이미 두 질문을 검증했다:
  · 진입 비율(--entry-ratio-sweep): 분할(50/50·30/30/40)이 전량매수보다 낫다는 걸 확인했고,
    정확한 비율도 스윕했으나 효과가 작아 라이브 비율(50/50) 유지.
  · 처분 방식(--disposal-sweep): 분할+반등대기 vs 즉시전량 — 차이가 신뢰도 게이트를 통과 못해
    "구분 안 되면 더 단순한 쪽" 원칙으로 즉시전량 채택.
그런데 이 두 스윕은 `_select_basket`(미국 팩터 가중치)로 종목을 고정 선정해 **미국 데이터로만**
실행됐다. entry_plan.py는 US/KR 공용 모듈이라 이 결론(분할매수 50/50, 즉시전량 매도)을 한국
쪽으로 검증 없이 그대로 썼다 — 이 프로젝트에서 이미 여러 번 반복된 패턴(topn·매도규칙도
처음엔 미국 결론을 복사했다가 한국 전용 재검증에서 결론이 달라짐, STRATEGY.md §3 Stage 6·6.1).

설계: backtest_exec.py의 트레이드 엔진(_simulate_trade, ENTRY_RATIO_2/3, DISPOSAL_SWEEP)은
종목 선정과 무관한 범용 엔진이다 — run_entry_ratio_sweep/run_disposal_sweep에 select_fn을
주입하는 기존 패턴(run_exec의 KR 모드와 동일)을 그대로 재사용해 valuediv 랭킹(현재 라이브
팩터, backtest_kr_strategies.build_decisions)을 꽂는다. 처음부터 13년 데이터
(output/kr_panel_cache_13y.pkl, kr_sell_algo_sweep.py의 _load_long과 동일) — 이 세션에서
8년 표본 결론이 13년 재검증에서 여러 번 뒤집힌 전례(Stage 3.1·6.1.1) 때문에 8년은 건너뛴다.

topn=15(미국 entry-ratio-sweep 기본값과 동일 — 이 질문은 "몇 종목을 담나"가 아니라
"각 포지션을 어떻게 채우고 비우나"이므로 라이브 topn=5보다 넓혀 이벤트 수를 확보한다,
POOL_KR=30의 절반).

실행: python kr_entry_exit_sweep.py --entry-ratio-sweep
      python kr_entry_exit_sweep.py --disposal-sweep
      python kr_entry_exit_sweep.py --self-test
결과: output/backtest_entry_ratio_compare_kr.json · output/pbo_report_entry_ratio_kr.json
      output/backtest_disposal_compare_kr.json · output/pbo_report_disposal_kr.json
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_exec as BE
import backtest_kr_strategies as KS

TOPN = 15
REBAL_DAYS = 63


def _log(m): print(f"[매수매도집행KR] {m}", file=sys.stderr)


def _load_long(rebal_days=REBAL_DAYS, cache_path="output/kr_panel_cache_13y.pkl"):
    """13년 캐시 로더 — kr_sell_algo_sweep.py._load_long과 동일(8년 공용 캐시와 별도)."""
    import pickle
    import backtest_kr as BK
    from benchmarks_kr import build_benchmarks
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    panel, bench = d["panel"], d["bench"]
    snaps, _, _ = BK.build_kr_snaps(panel, bench, d["membership"], d["fundamentals"],
                                    rebal_days=rebal_days, flows=d["flows"], mktcaps=d["mktcaps"])
    navs_bm = build_benchmarks(panel, d["membership"], d["mktcaps"], bench)
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    return panel, snaps, navs_bm, cost


def _make_select_fn(panel: pd.DataFrame, snaps: list, topn=TOPN):
    """valuediv 랭킹(POOL_KR=30) → 결정 시점별 상위 topn. backtest_exec._simulate_trade가
    쓰는 panel 위치 인덱스(p)로 조회하는 dict 클로저(main()의 기존 KR select_fn과 동일 패턴,
    단 모멘텀이 아니라 현재 라이브 팩터인 valuediv 사용 — main()의 KR 경로는 옛 모멘텀을
    아직도 쓰고 있어 이 스크립트가 그 공백도 우회한다)."""
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    by_p = {p: ranked[:topn] for p, ranked in decisions}
    _log(f"valuediv 결정 시점 {len(by_p)}개 (풀 30 → topn {topn})")
    return lambda p: by_p.get(p, [])


def run_entry_ratio_sweep_kr(save=True):
    """entry1_full(전량)을 포함시켜 '분할 vs 전량' 자체 + 비율까지 한 번에 비교(2026-07-16,
    지호 님 질문 — 한국도 분할이 맞는지부터 확인). 미국은 이 질문을 run_exec(21조합)으로
    따로 봤지만 여기선 run_entry_ratio_sweep 하나로 합친다(entries 파라미터 신규 지원)."""
    import backtest_kr as BK
    panel, snaps, navs_bm, cost = _load_long()
    bench = navs_bm["B2_equal"].reindex(panel.index).ffill()
    select_fn = _make_select_fn(panel, snaps)
    entries = ["entry1_full"] + list(BE.ENTRY_RATIO_SWEEP)
    return BE.run_entry_ratio_sweep(panel, bench, None, None, rebal_days=REBAL_DAYS, topn=TOPN,
                                    cost=cost, select_fn=select_fn, out_suffix="_kr",
                                    lookback=BK.LOOKBACK, entries=entries)


def run_disposal_sweep_kr(save=True):
    import backtest_kr as BK
    panel, snaps, navs_bm, cost = _load_long()
    bench = navs_bm["B2_equal"].reindex(panel.index).ffill()
    select_fn = _make_select_fn(panel, snaps)
    return BE.run_disposal_sweep(panel, bench, None, None, rebal_days=REBAL_DAYS, topn=TOPN,
                                 cost=cost, select_fn=select_fn, out_suffix="_kr",
                                 lookback=BK.LOOKBACK)


# ------------------------- self-test -------------------------
def self_test():
    """13년 캐시·pykrx 없이도 배선(select_fn 주입 → run_entry_ratio_sweep/run_disposal_sweep)이
    정상 동작하는지 합성 데이터로 확인. 트레이드 엔진 내부 로직(_simulate_trade)은
    backtest_exec.py 자체 self-test가 이미 검증하므로 여기선 KR 주입 지점만 본다."""
    _log("[self-test] 합성 데이터로 select_fn 주입 배선 확인")
    rng = np.random.default_rng(3)
    n, m = 3000, 25
    idx = pd.bdate_range("2013-01-01", periods=n)
    rets = rng.normal(0.0003, 0.02, (n, m))
    cols = [f"K{i:03d}" for i in range(m)]
    panel = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    bench = panel.mean(axis=1)
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    ps = list(range(260, n - 300, REBAL_DAYS))
    assert len(ps) >= 4, "합성 데이터 결정 시점 부족 — self-test 설계 확인 필요"
    by_p = {p: list(rng.permutation(cols))[:10] for p in ps}
    select_fn = lambda p: by_p.get(p, [])

    payload, report = BE.run_entry_ratio_sweep(panel, bench, None, None, rebal_days=REBAL_DAYS,
                                               topn=10, cost=cost, select_fn=select_fn,
                                               out_suffix="_kr_selftest", lookback=260)
    assert payload["n_combos"] == len(BE.ENTRY_RATIO_SWEEP)
    assert all("net_pct" in r for r in payload["rows"])
    assert "pbo" in report or "gate" in report or isinstance(report, dict)
    _log(f"[self-test] 통과: entry-ratio 배선 정상({payload['n_combos']}종 조합, "
         f"{payload['rows'][0]['n_events']}건 이벤트)")

    payload2, _ = BE.run_disposal_sweep(panel, bench, None, None, rebal_days=REBAL_DAYS,
                                        topn=10, cost=cost, select_fn=select_fn,
                                        out_suffix="_kr_selftest", lookback=260)
    assert payload2["n_combos"] == len(BE.DISPOSAL_SWEEP)
    assert all("net_pct" in r for r in payload2["rows"])
    _log(f"[self-test] 통과: disposal 배선 정상({payload2['n_combos']}종 조합, "
         f"{payload2['rows'][0]['n_events']}건 이벤트)")

    # 정리(합성 self-test 산출물이 실제 결과 파일을 덮어쓰지 않도록 out_suffix로 분리했지만,
    # 굳이 output/에 남길 필요 없는 임시 산출물이므로 삭제)
    for f in ("backtest_entry_ratio_compare_kr_selftest.json", "trial_returns_entry_ratio_kr_selftest.json",
             "pbo_report_entry_ratio_kr_selftest.json", "backtest_disposal_compare_kr_selftest.json",
             "trial_returns_disposal_kr_selftest.json", "pbo_report_disposal_kr_selftest.json"):
        try:
            os.remove(os.path.join("output", f))
        except FileNotFoundError:
            pass
    _log("[self-test] 전부 통과")


def main():
    ap = argparse.ArgumentParser(description="국장 새틀라이트 매수/매도 집행방식(분할비율·처분방식) 검증")
    ap.add_argument("--entry-ratio-sweep", action="store_true", help="매수 분할비율 스윕")
    ap.add_argument("--disposal-sweep", action="store_true", help="매도 처분방식(즉시 vs 분할) 스윕")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.entry_ratio_sweep:
        run_entry_ratio_sweep_kr(); return
    if args.disposal_sweep:
        run_disposal_sweep_kr(); return
    ap.error("--entry-ratio-sweep, --disposal-sweep, --self-test 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()
