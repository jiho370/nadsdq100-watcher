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
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_exec as BE
import backtest_kr_strategies as KS
import overfit_stats as OS

TOPN = 15
REBAL_DAYS = 63
HOT_RSI = 72          # kr_stocks._hot()과 동일 기준
HOT_GAP50 = 15        # 50일선 대비 +15% 이상
CORE_ENTRIES = ["entry1_full", "entry2_5050", "entry3_303040"]  # 전량 vs 현행 2분할 vs 현행 3분할


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


def _classify_hot(panel: pd.DataFrame, p: int, syms: list) -> tuple[list, list]:
    """kr_stocks._hot()과 동일 기준(RSI≥72 또는 50일선 대비 +15% 이상)으로 hot/normal 분류.
    p 시점까지의 종가만 사용(미래 참조 없음)."""
    from market_signals import _rsi
    hot, normal = [], []
    for sym in syms:
        closes = panel[sym].iloc[:p + 1].dropna()
        if len(closes) < 51:
            normal.append(sym); continue
        price = float(closes.iloc[-1])
        ma50 = float(closes.iloc[-50:].mean())
        rsi = _rsi(closes.iloc[-30:].tolist())
        gap50 = (price / ma50 - 1) * 100 if ma50 else 0.0
        is_hot = (rsi is not None and rsi >= HOT_RSI) or gap50 >= HOT_GAP50
        (hot if is_hot else normal).append(sym)
    return hot, normal


def run_hot_split_sweep(save=True):
    """지호 님 질문(2026-07-16): "과열이면 3분할, 평시면 2분할"이라는 조건분기 자체가
    검증된 적이 있나 — 없었다. 기존 entry-ratio-sweep은 바스켓 전체에 규칙 하나를 통째로
    적용해 "분할이 전량매수를 이긴다"만 확인했지, "과열 종목이 진짜로 3분할에서 더 득을
    보는가"는 안 봤다. 여기서는 매 결정 시점마다 후보(valuediv 풀 30)를 hot/normal로 나눠
    각 그룹 안에서 entry1_full·entry2_5050·entry3_303040 3종을 따로 비교한다."""
    import backtest_kr as BK
    panel, snaps, navs_bm, cost = _load_long()
    bench = navs_bm["B2_equal"].reindex(panel.index).ffill()
    decisions = KS.build_decisions(panel, snaps, "valuediv")   # 풀 30(전체) — 두 그룹으로 나눌 여유 확보
    ma20, ma50, ma200, atr = BE._ma(panel, 20), BE._ma(panel, 50), BE._ma(panel, 200), BE._atr_close(panel)
    bench_r = bench.reindex(panel.index).ffill()

    per_combo = {("hot", e): {"excess": [], "dates": []} for e in CORE_ENTRIES}
    per_combo.update({("normal", e): {"excess": [], "dates": []} for e in CORE_ENTRIES})
    stats = {k: {"net": [], "mdd": [], "n_syms": []} for k in per_combo}
    exit_rule = "exit_time6m"
    n_hot_events, n_normal_events = 0, 0

    for p, ranked in decisions:
        date = panel.index[p].date().isoformat()
        entry_day = p + 1
        hot_syms, normal_syms = _classify_hot(panel, p, ranked)
        n_hot_events += len(hot_syms); n_normal_events += len(normal_syms)
        for group, syms in (("hot", hot_syms), ("normal", normal_syms)):
            if not syms:
                continue
            for e in CORE_ENTRIES:
                evs = []
                for sym in syms:
                    r = BE._simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, e, exit_rule)
                    if r is None:
                        continue
                    net = cost.net(r["exit_price"] / r["entry_price"] - 1)
                    b_ret = 0.0
                    if np.isfinite(bench_r.iloc[r["exit_day"]]) and np.isfinite(bench_r.iloc[entry_day]):
                        b_ret = float(bench_r.iloc[r["exit_day"]] / bench_r.iloc[entry_day] - 1)
                    evs.append({"net": net, "excess": net - b_ret, "mdd": r["mdd"]})
                if not evs:
                    continue
                key = (group, e)
                per_combo[key]["excess"].append(round(float(np.mean([x["excess"] for x in evs])), 6))
                per_combo[key]["dates"].append(date)
                stats[key]["net"].append(float(np.mean([x["net"] for x in evs])))
                stats[key]["mdd"].append(float(np.mean([x["mdd"] for x in evs])))
                stats[key]["n_syms"].append(len(evs))

    _log(f"결정 시점 {len(decisions)}개 · 종목-이벤트 누적 hot {n_hot_events}건 · normal {n_normal_events}건")

    results = {}
    for group in ("hot", "normal"):
        n_ev = min((len(per_combo[(group, e)]["excess"]) for e in CORE_ENTRIES), default=0)
        if n_ev < 4:
            _log(f"[{group}] 이벤트 부족(n_ev={n_ev}) — 판정 생략"); results[group] = None; continue
        matrix = [per_combo[(group, e)]["excess"][:n_ev] for e in CORE_ENTRIES]
        dates0 = per_combo[(group, CORE_ENTRIES[0])]["dates"][:n_ev]
        rows = [{"entry": e, "net_pct": round(100 * float(np.mean(stats[(group, e)]["net"])), 2),
                "mdd_pct": round(100 * float(np.mean(stats[(group, e)]["mdd"])), 1),
                "avg_n_syms": round(float(np.mean(stats[(group, e)]["n_syms"])), 1),
                "n_events": n_ev} for e in CORE_ENTRIES]
        trial_data = {"horizon": f"hotsplit_{group}", "universe": "pit", "cost": cost.describe(),
                     "rebal_days": REBAL_DAYS, "hold_days": BE.MAX_HOLD,
                     "dates": dates0, "trials": CORE_ENTRIES, "excess_returns": matrix}
        report = OS.analyze(trial_data, save=False)
        results[group] = {"rows": rows, "n_events": n_ev, "report": report}
        _log(f"[{group}] " + " · ".join(f"{r['entry']}={r['net_pct']}%p(MDD{r['mdd_pct']}%,평균{r['avg_n_syms']}종목)"
                                        for r in rows))

    payload = {"as_of": panel.index[-1].date().isoformat(), "market": "kr",
              "note": "결정 시점마다 valuediv 풀(30)을 hot(RSI≥72 또는 50일선+15%)/normal로 나눠 "
                      "entry1_full·entry2_5050·entry3_303040을 그룹별로 독립 비교 — '과열이면 3분할'"
                      "이라는 조건분기 자체가 근거 있는지 검증(2026-07-16, 지호 님 질문)",
              "hot": results["hot"], "normal": results["normal"]}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_hot_split_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/kr_hot_split_sweep.json")
    return payload


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

    # _classify_hot 배선 확인 — RSI 급등 종목(HOT)과 평탄한 종목(NORMAL)을 합성으로 만들어
    # 정확히 갈리는지 확인(kr_stocks._hot()과 동일 기준: RSI≥72 또는 50일선 대비 +15%).
    idx3 = pd.bdate_range("2020-01-01", periods=120)
    rng3 = np.random.default_rng(11)
    flat_px = 100.0 * np.exp(np.cumsum(rng3.normal(0.0, 0.005, 120)))   # 잔잔한 등락(진짜 non-hot)
    surge_px = np.concatenate([np.full(60, 100.0), 100.0 * np.exp(np.cumsum(np.full(60, 0.006)))])
    panel3 = pd.DataFrame({"FLAT": flat_px, "SURGE": surge_px}, index=idx3)
    hot3, normal3 = _classify_hot(panel3, 119, ["FLAT", "SURGE"])
    assert hot3 == ["SURGE"] and normal3 == ["FLAT"], f"hot/normal 분류 오류: hot={hot3} normal={normal3}"
    _log("[self-test] 통과: _classify_hot 배선 정상(급등 종목만 hot 판정)")

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
    ap.add_argument("--hot-split-sweep", action="store_true",
                    help="'과열이면 3분할' 조건분기 자체가 근거 있는지(hot/normal 그룹별 비교)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.entry_ratio_sweep:
        run_entry_ratio_sweep_kr(); return
    if args.disposal_sweep:
        run_disposal_sweep_kr(); return
    if args.hot_split_sweep:
        run_hot_split_sweep(); return
    ap.error("--entry-ratio-sweep, --disposal-sweep, --hot-split-sweep, --self-test 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()
