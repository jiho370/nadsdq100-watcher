#!/usr/bin/env python3
"""
kr_sell_algo_sweep.py — 국장 새틀라이트 매도 알고리즘 강건성 검증 (2026-07-15).

배경: 미국(S&P500)은 21조합 백테스트(backtest_exec.py)로 매도 규칙을 정밀 검증했지만,
한국은 그 결론(6개월 재평가 + 200일선 -3% 백업)을 근거 없이 그대로 복사했다. 이후
밸류 전략의 매수 진입필터(종가>200일선)를 폐기(2026-07-14)하면서 "200일선 아래 종목을
의도적으로 매수 → 다음날 200일선 백업에 즉시 매도"되는 버그가 실제로 발생, 2026-07-15에
200일선 백업을 임시로 꺼버렸다(SELL_MA200_BACKUP 기본 OFF) — 이것도 "검증해서 채택"이
아니라 "버그 회피"에 가깝다. 지호 님 지시(2026-07-15)로 Claude(Fable 5) 설계자문을 받아
매도 규칙을 제대로 검증한다.

자문 반영 설계:
  · 후보 9개(다중검정 예산 등록) + 대조군 1개(옛 버그 버전, 채점 제외) — 10개 이내로 제한.
  · reeval_days 90/120/180(현행)/270 — MA 개입 없이 재평가 주기 자체가 최적인지.
  · state_gated MA백업(버퍼 -3/-5/-10%) — 매수 시 이미 버퍼 아래면 면제, 이후 한 번이라도
    회복("armed")해야 재이탈 시 매도(진입시점 스냅샷 방식보다 갈등 재발 여지가 적음, 2026-
    07-15 backtest_portfolio.simulate() ma_stop_mode="state_gated" 신규 배선).
  · entry_stop_pct 트레일링(-25%, 미국은 이게 나빴지만 종목수가 적은 한국은 다를 수 있음)·
    재난스톱(-40%, 6종목 집중 구조의 개별종목 꼬리위험 방어 — DART 이벤트 연동 대신 가격
    임계값으로 근사, 관리종목 지정 등 이벤트 기반은 후속 과제로 남김).
  · 판정: 베이스라인(현행 라이브 reeval180·MA개입없음) 대비 전체기간 CAGR +0.5%p 이상+
    MDD 5%p 이내 + 최근3년 하위사분위 아님 — 셋 다 만족해야 "채택후보"(Fable 자문 — 샤프
    1등 뽑기 대신 기준선 격파 조건부 채택). PBO/DSR은 전체기간에 등록 후보 9개로 계산.
  · 전방수익률 진단(스톱 발동 후 63거래일 그 종목이 어떻게 됐는지) — "안 팔았으면 어땠나"로
    방어가 승자를 잘라내는 건지(과최적화 신호) 손실을 막은 건지 구분.
  · 1/2/3/5년 + 전체기간 다기간 비교 — 최근 구간 1등이 아니라 전체기간으로 순위, 최근
    2~3년은 거부권(veto)으로만 사용(Fable: 최근 1등 뽑기는 표본이 작아 과최적화 지름길).

실행: python kr_sell_algo_sweep.py
      python kr_sell_algo_sweep.py --self-test
결과: output/kr_sell_algo_sweep.json · output/pbo_report_kr_sell_algo.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP
import backtest_kr_strategies as KS
import core_satellite_kr as CS

TOPN = 6           # 현행 라이브값 고정 — 매도알고리즘 질문과 topN 질문을 안 섞음
CORE_WEIGHT = 0.65  # 현행 라이브값 고정

CANDIDATES = {
    "reeval90":           dict(reeval_days=90,  ma200_backup=False),
    "reeval120":          dict(reeval_days=120, ma200_backup=False),
    "reeval180_baseline": dict(reeval_days=180, ma200_backup=False),   # 현행 라이브 기본값
    "reeval270":          dict(reeval_days=270, ma200_backup=False),
    "gated_buf3":         dict(reeval_days=180, ma200_backup=True,
                               ma_stop_mode="state_gated", ma_buffer=0.03),
    "gated_buf5":         dict(reeval_days=180, ma200_backup=True,
                               ma_stop_mode="state_gated", ma_buffer=0.05),
    "gated_buf10":        dict(reeval_days=180, ma200_backup=True,
                               ma_stop_mode="state_gated", ma_buffer=0.10),
    "trail25":            dict(reeval_days=180, ma200_backup=False, entry_stop_pct=0.25),
    "disaster40":         dict(reeval_days=180, ma200_backup=False, entry_stop_pct=0.40),
}
CONTROL = {"old_buggy_unconditional": dict(reeval_days=180, ma200_backup=True,
                                           ma_stop_mode="unconditional", ma_buffer=0.03)}
BASELINE_KEY = "reeval180_baseline"
WINDOWS_YEARS = [1, 2, 3, 5]
FWD_HORIZON = 63    # 스톱 발동 후 전방수익률 진단 구간(약 3개월)


def _log(m): print(f"[매도알고리즘] {m}", file=sys.stderr)


def _load():
    from benchmarks_kr import load_research_data, load_benchmarks
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    navs_bm = load_benchmarks()
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    return panel, snaps, navs_bm, ma200, cost


def _forward_return_diag(trade_log: list, panel: pd.DataFrame, horizon=FWD_HORIZON) -> dict:
    """스톱 발동(ma200_stop·entry_stop) 이후 그 종목의 향후 horizon거래일 수익률 —
    "안 팔았으면 어땠나"(Fable 자문 진단). 평균이 크게 양수면 승자를 잘라낸 것(과최적화
    신호), 0 이하/약한 양수면 방어가 제 몫을 한 것."""
    px = panel.ffill()
    fwd_rets = []
    for e in trade_log:
        if e.get("action") != "sell" or e.get("reason") not in ("ma200_stop", "entry_stop"):
            continue
        sym, date_s = e.get("sym"), e.get("date")
        if sym not in px.columns:
            continue
        d = pd.Timestamp(date_s)
        if d not in px.index:
            continue
        i = px.index.get_loc(d)
        if i + horizon >= len(px):
            continue
        p0, p1 = px[sym].iloc[i], px[sym].iloc[i + horizon]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            fwd_rets.append(float(p1 / p0 - 1))
    if not fwd_rets:
        return {"n_events": 0, "avg_forward_return_pct": None, "median_forward_return_pct": None}
    return {"n_events": len(fwd_rets),
           "avg_forward_return_pct": round(float(np.mean(fwd_rets)) * 100, 2),
           "median_forward_return_pct": round(float(np.median(fwd_rets)) * 100, 2)}


def run(save=True):
    panel, snaps, navs_bm, ma200, cost = _load()
    b1 = navs_bm["B1_kospi200"].dropna()
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    reg = CS.regime_series(b1.reindex(panel.index).ffill())
    core = CS.timed_nav(b1.reindex(panel.index).ffill(), reg)
    last_date = panel.index[-1]

    all_candidates = {**CANDIDATES, **CONTROL}
    results, trial_parts = {}, []
    for name, kwargs in all_candidates.items():
        trade_log = []
        sat_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, trade_log=trade_log, **kwargs)
        if sat_nav is None:
            _log(f"{name}: NAV 산출 실패"); continue
        sat = sat_nav / sat_nav.iloc[0]
        mixed = CS.mix_nav(core, sat, CORE_WEIGHT)
        full = CS.stats(mixed)
        if full is None:
            _log(f"{name}: 통계 산출 실패"); continue
        windows = {"full": full}
        for y in WINDOWS_YEARS:
            cutoff = (last_date - pd.Timedelta(days=int(y * 365.25))).date().isoformat()
            windows[f"{y}y"] = CS.stats(mixed, cutoff, None)
        diag = _forward_return_diag(trade_log, panel)
        n_sells = sum(1 for e in trade_log if e.get("action") == "sell")
        n_ma = sum(1 for e in trade_log if e.get("reason") == "ma200_stop")
        n_entry = sum(1 for e in trade_log if e.get("reason") == "entry_stop")
        n_reeval = sum(1 for e in trade_log if e.get("reason") == "reeval")
        results[name] = {"params": kwargs, "windows": windows,
                         "n_sells": n_sells, "n_ma_sells": n_ma,
                         "n_entry_sells": n_entry, "n_reeval_sells": n_reeval,
                         "forward_return_diag": diag}
        _log(f"{name}: 전체 CAGR {full['cagr_pct']:6.2f}% 샤프 {full['sharpe']:5.2f} "
             f"MDD {full['mdd_pct']:6.1f}% · 매도 {n_sells}건(MA {n_ma}/진입스톱 {n_entry}/재평가 "
             f"{n_reeval}) · 스톱후 {FWD_HORIZON}일 평균수익률 {diag['avg_forward_return_pct']}%")
        if name in CANDIDATES:   # 대조군(옛 버그 버전)은 다중검정 예산·PBO/DSR 채점 대상 아님
            d, r = BP.monthly_excess(mixed, b1.reindex(mixed.index).ffill())
            trial_parts.append({"name": name, "dates": d, "matrix": r})

    if len(trial_parts) < 2:
        raise RuntimeError("후보 결과 부족 — PBO/DSR 계산 불가")
    n_ev = min(len(p["matrix"]) for p in trial_parts)
    trial_data = {"horizon": "kr_sell_algo", "universe": "kospi200_pit", "cost": cost.describe(),
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": trial_parts[0]["dates"][:n_ev],
                 "trials": [p["name"] for p in trial_parts],
                 "excess_returns": [p["matrix"][:n_ev] for p in trial_parts]}
    rpt = OS.analyze(trial_data, save=False)

    # 결정규칙(Fable 자문): 베이스라인 대비 전체기간 CAGR +0.5%p 이상 + MDD 5%p 이내 +
    # 최근3년 샤프가 후보군 하위사분위 아님 — 셋 다 만족해야 "채택후보".
    base_full = results[BASELINE_KEY]["windows"]["full"]
    recent_vals = sorted(results[n]["windows"]["3y"]["sharpe"] for n in CANDIDATES
                         if results[n]["windows"].get("3y"))
    q1_cut = recent_vals[max(0, len(recent_vals) // 4 - 1)] if len(recent_vals) >= 4 else recent_vals[0]
    verdicts = {}
    for name in CANDIDATES:
        if name not in results:
            continue
        f = results[name]["windows"]["full"]
        r3 = results[name]["windows"].get("3y")
        beats_baseline = f["cagr_pct"] >= base_full["cagr_pct"] + 0.5
        mdd_ok = f["mdd_pct"] >= base_full["mdd_pct"] - 5.0
        not_bottom_q = (r3 is not None and r3["sharpe"] >= q1_cut)
        verdicts[name] = {"beats_baseline_cagr_pct": round(f["cagr_pct"] - base_full["cagr_pct"], 2),
                          "beats_baseline": beats_baseline, "mdd_within_5pp": mdd_ok,
                          "recent_3y_not_bottom_quartile": not_bottom_q,
                          "adopt_worthy": beats_baseline and mdd_ok and not_bottom_q}

    payload = {"as_of": last_date.date().isoformat(), "topn": TOPN, "core_weight": CORE_WEIGHT,
              "baseline": BASELINE_KEY,
              "judgment": "베이스라인(현행 라이브: 재평가만, MA백업 없음) 대비 전체기간 +0.5%p CAGR "
                          "우위 + MDD 5%p 이내 + 최근3년 하위사분위 아님 — 셋 다 만족해야 채택후보"
                          "(Fable 5 자문, 2026-07-15)",
              "results": results, "verdicts": verdicts,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "n_trials_registered": len(CANDIDATES)}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_sell_algo_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open("output/pbo_report_kr_sell_algo.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/kr_sell_algo_sweep.json (등록 시행수 {len(CANDIDATES)} · "
             f"PBO {payload['pbo']} · DSR {payload['dsr']})")
    return payload


ENTRY_STOP_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def run_entry_stop_grid(save=True):
    """2026-07-15 지호 님 질문 대응: -25%는 그리드 스윕 없이 고른 값이었다는 지적이 맞음.
    Fable 5 자문: 10~40%로 스윕해 평탄 고지대(고르면 신뢰)인지 뾰족한 최적점(과적합 신호로
    오히려 신뢰 하락)인지 확인, 발동 빈도도 같이 기록(연 1회 미만이면 통계 아니라 일화)."""
    panel, snaps, navs_bm, ma200, cost = _load()
    b1 = navs_bm["B1_kospi200"].dropna()
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    reg = CS.regime_series(b1.reindex(panel.index).ffill())
    core = CS.timed_nav(b1.reindex(panel.index).ffill(), reg)
    n_years = (panel.index[-1] - panel.index[0]).days / 365.25

    rows = []
    for pct in ENTRY_STOP_GRID:
        trade_log = []
        sat_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, reeval_days=180,
                              ma200_backup=False, entry_stop_pct=pct, trade_log=trade_log)
        if sat_nav is None:
            continue
        sat = sat_nav / sat_nav.iloc[0]
        mixed = CS.mix_nav(core, sat, CORE_WEIGHT)
        f = CS.stats(mixed)
        if f is None:
            continue
        diag = _forward_return_diag(trade_log, panel)
        n_stops = sum(1 for e in trade_log if e.get("reason") == "entry_stop")
        rows.append({"entry_stop_pct": pct, **f, "n_stops": n_stops,
                    "stops_per_year": round(n_stops / n_years, 2),
                    "forward_return_diag": diag})
        _log(f"entry_stop=-{pct*100:.0f}%: CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} "
             f"MDD {f['mdd_pct']:6.1f}% · 발동 {n_stops}회(연 {n_stops/n_years:.2f}회) · "
             f"발동후{FWD_HORIZON}일 평균수익률 {diag['avg_forward_return_pct']}%")
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "core_weight": CORE_WEIGHT,
              "judgment": "10~40% 그리드 — 평탄고지대 vs 뾰족한최적점 확인(Fable 5 자문, "
                          "2026-07-16). 이론적 참고: 코스피 개별주 연변동성 ~35%면 6개월 보유 "
                          "시 반기 시그마 ≈25% — -25%는 대략 1시그마(정상 등락과 구분 어려움), "
                          "-40%가 대략 2시그마(진짜 이상 신호에 가까움)",
              "rows": rows}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_entry_stop_grid.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/kr_entry_stop_grid.json")
    return payload


YEAR_END_REEVAL_TEST = [90, 180]


def run_year_end_stage(save=True):
    """2026-07-15 지호 님 질문: 연말에 절세 목적으로 수익 포지션을 강제 정리해야 하는
    실제 관행이 있는데, 이걸 반영하면 3개월(90일) vs 6개월(180일) 재평가 결론이 바뀔까?
    (직전 백테스트에서 reeval 90~270일이 전부 동일했던 건 이 제약이 없어서였을 가능성 —
    Fable 5 예측: 연말 제약을 넣으면 드디어 갈릴 것, 180일이 1월매수+7월재평가+12월강제
    정리로 자연스럽게 정렬돼 90일보다 나을 것으로 예상). year_end_rebuy="wait"(재매수 안
    하고 다음 리밸런싱까지 현금 보유) vs "immediate"(같은 날 재매수, 세금 이벤트만 발생)
    두 변형 비교 — 그 차이가 이 제약의 실질 비용."""
    panel, snaps, navs_bm, ma200, cost = _load()
    b1 = navs_bm["B1_kospi200"].dropna()
    decisions = KS.build_decisions(panel, snaps, "valuediv")
    reg = CS.regime_series(b1.reindex(panel.index).ffill())
    core = CS.timed_nav(b1.reindex(panel.index).ffill(), reg)

    rows = []
    for reeval in YEAR_END_REEVAL_TEST:
        for rebuy in ("wait", "immediate"):
            trade_log = []
            sat_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, reeval_days=reeval,
                                  ma200_backup=False, year_end_liquidate=True,
                                  year_end_rebuy=rebuy, trade_log=trade_log)
            if sat_nav is None:
                continue
            sat = sat_nav / sat_nav.iloc[0]
            mixed = CS.mix_nav(core, sat, CORE_WEIGHT)
            f = CS.stats(mixed)
            if f is None:
                continue
            n_ye = sum(1 for e in trade_log if e.get("reason") == "year_end_taxharvest")
            rows.append({"reeval_days": reeval, "year_end_rebuy": rebuy, **f,
                        "n_year_end_sells": n_ye})
            _log(f"reeval={reeval} rebuy={rebuy}: CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} "
                 f"MDD {f['mdd_pct']:6.1f}% · 연말강제매도 {n_ye}회")
    sat_nav0 = BP.simulate(panel, ma200, decisions, TOPN, cost, reeval_days=180, ma200_backup=False)
    sat0 = sat_nav0 / sat_nav0.iloc[0]
    mixed0 = CS.mix_nav(core, sat0, CORE_WEIGHT)
    f0 = CS.stats(mixed0)
    _log(f"(참고) 연말 제약 없는 현행(reeval180): CAGR {f0['cagr_pct']:.2f}% 샤프 {f0['sharpe']:.2f} "
         f"MDD {f0['mdd_pct']:.1f}%")
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "core_weight": CORE_WEIGHT,
              "judgment": "연말 강제정리(수익종목만 매도, 손실종목은 유지) 제약 하에서 재평가 "
                          "90일 vs 180일, 즉시재매수 vs 대기재매수 비교(Fable 5 자문, 2026-07-16)",
              "no_constraint_baseline": f0, "rows": rows}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/kr_year_end_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/kr_year_end_sweep.json")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 후보별 배선·다기간 슬라이싱·전방수익률 진단 검증(네트워크 없음)")
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=2600)
    n_stocks = 20
    rets = rng.normal(0.0003, 0.02, (len(idx), n_stocks))
    panel = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i}" for i in range(n_stocks)])
    ma200 = panel.rolling(200, min_periods=200).mean()
    cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
    decisions = [(p, list(rng.permutation(panel.columns))) for p in range(260, len(idx) - 1, 63)]

    trade_log = []
    nav = BP.simulate(panel, ma200, decisions, 6, cost, reeval_days=180, ma200_backup=False,
                      trade_log=trade_log)
    assert nav is not None
    last = nav.index[-1]
    cutoff = (last - pd.Timedelta(days=365)).date().isoformat()
    w1, wfull = CS.stats(nav, cutoff, None), CS.stats(nav)
    assert w1 is not None and wfull is not None
    diag = _forward_return_diag(trade_log, panel)
    assert "n_events" in diag
    _log(f"[self-test] 통과: 윈도우 슬라이싱(1y CAGR {w1['cagr_pct']} vs 전체 {wfull['cagr_pct']}), "
         f"전방수익률진단 n={diag['n_events']}")

    # state_gated 후보가 unconditional 대조군보다 MA스톱 발동 횟수가 적거나 같아야 함
    # (면제 구간이 있으므로) — 합성 데이터로 배선만 확인.
    log_gated, log_uncond = [], []
    BP.simulate(panel, ma200, decisions, 6, cost, reeval_days=99999, ma200_backup=True,
               ma_stop_mode="state_gated", trade_log=log_gated)
    BP.simulate(panel, ma200, decisions, 6, cost, reeval_days=99999, ma200_backup=True,
               ma_stop_mode="unconditional", trade_log=log_uncond)
    n_gated = sum(1 for e in log_gated if e.get("reason") == "ma200_stop")
    n_uncond = sum(1 for e in log_uncond if e.get("reason") == "ma200_stop")
    assert n_gated <= n_uncond, f"state_gated가 unconditional보다 스톱이 많으면 안 됨: {n_gated} vs {n_uncond}"
    _log(f"[self-test] 통과: state_gated MA스톱 {n_gated}건 ≤ unconditional {n_uncond}건")


def main():
    ap = argparse.ArgumentParser(description="국장 새틀라이트 매도알고리즘 강건성 검증")
    ap.add_argument("--stage", choices=["main", "entry-stop-grid", "year-end"], default="main")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.stage == "main":
        run()
    elif args.stage == "entry-stop-grid":
        run_entry_stop_grid()
    else:
        run_year_end_stage()


if __name__ == "__main__":
    main()
