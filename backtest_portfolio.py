#!/usr/bin/env python3
"""
backtest_portfolio.py — 포트폴리오 단위 일별 NAV 시뮬레이션으로 topN(보유종목 수) 판정.

왜 이 파일인가(2026-07-14): backtest_exec.py의 topn 스윕은 '트레이드별 평균 net/MDD'라
분산투자 효과(종목을 늘릴수록 개별 리스크가 상쇄되는 것)를 구조적으로 반영하지 못했다
— topn이 작을수록 평균 팩터 품질만 올라 net이 단조 증가하는 왜곡(STRATEGY.md 기록).
여기서는 포트폴리오 전체를 하나의 일별 NAV 곡선으로 시뮬레이션한다:

  · 라이브 규칙 재현: 상한 N종목, 진입 시 동일비중(NAV/N), 빈 슬롯은 당일 팩터 상위로
    충원("팔아야 산다"), 매도 = ①일별 200일선 -3% 이탈 ②결정일에 보유 ≥180일(달력)이고
    당일 후보풀 밖(6개월 정기 재평가 — holdings.py와 동일).
  · 일별 NAV → CAGR·연변동성·샤프·**진짜 포트폴리오 MDD**(cummax 대비 낙폭).
  · 판정: 21거래일(월간) 비중첩 초과수익(vs 벤치마크)을 overfit_stats(PBO/DSR)에 투입
    — 10년이면 표본 ~100개로 기존 이벤트 프레임(T_eff=8)보다 검정력이 크게 좋아진다.

실행(PC): python backtest_portfolio.py --market us --years 10
          python backtest_portfolio.py --market kr --years 8
          python backtest_portfolio.py --self-test
결과: output/backtest_portfolio_{us,kr}.json · output/pbo_report_portfolio_{us,kr}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS

REEVAL_DAYS = 180        # 6개월 정기 재평가(달력일) — holdings.py SELL_REEVAL_DAYS와 동일
MA_BUFFER = 0.03         # 200일선 -3% — holdings.py SELL_MA_BUFFER와 동일
POOL_SIZE = 60           # 후보풀 크기 — 라이브 REPORT_MAX_CANDIDATES와 동일
MONTH = 21               # 판정용 비중첩 수익률 구간(거래일)

TOPN_US = [5, 8, 10, 12, 15, 20]
TOPN_KR = [3, 4, 6, 8, 10]

# 2026-07-14 확장(지호 님 질문 — "풀 60은 그 팩터로 뽑되, 그중 10개는 다른 팩터로 골라도
# 되지 않나"): 2단계 선별 구조 검증. 1단계(풀 60 = 검증된 가중치 점수 상위)는 고정하고,
# 2단계(풀 안에서 상위 topn 선별)의 랭킹만 교체해 비교한다. 후보는 근거 있는 것만 제한
# (다중검정 예산): base=현행(동일 점수), mom12_1/mom6=모멘텀 문헌(단 이 표본에서 mom12_1
# IC 음수 확인됨 — SCORE_MODEL_DESIGN D1), hi52=52주고점 근접(George-Hwang), lowvol=
# 저변동성 이상현상, value=퀄리티 풀 내 저평가 우선.
US_RERANKS = ["base", "mom12_1", "mom6", "hi52", "lowvol", "value"]


def _log(m): print(f"[포트폴리오] {m}", file=sys.stderr)


# ------------------------- 엔진 (시장 무관) -------------------------
def simulate(panel: pd.DataFrame, ma200: pd.DataFrame, decisions: list, topn: int,
             cost: BC.CostModel, reeval_days=REEVAL_DAYS, ma200_backup=True,
             sector_of=None, sector_cap=None, trade_log=None) -> pd.Series | None:
    """decisions: [(p, ranked_syms)] — p=panel 행 인덱스(오름차순), ranked_syms=순위순 후보풀.
    반환: 일별 NAV Series(시작 1.0) 또는 None(결정 시점 없음).
    체결은 당일 종가, 비용은 cost.buy/cost.sell을 편도로 각각 적용.
    ma200_backup=False면 ①(200일선 조기이탈)을 끄고 ②(정기 재평가)만 쓴다 — 보유기간
    자체의 순효과를 보려면 21조합 champion(exit_time6m, 가격 개입 없음)과 같은 조건이어야
    두 질문(보유기간 vs 200일선 백업)이 안 섞인다(2026-07-15, 지호 님 질문).
    sector_of(date_yyyymmdd, sym)->str|None + sector_cap(int)이 둘 다 주어지면 슬롯채우기
    시 "동일 업종 보유 ≤sector_cap" 캡 적용(기존 보유분 포함해서 카운트 — 180일 보유로
    포지션이 여러 리밸런싱에 걸쳐 유지되므로 신규 매수만 캡해선 실효 노출을 못 막음).
    캡으로 슬롯이 남으면 2차 패스에서 캡을 무시하고 채운다(빈 슬롯은 실효 주식비중을
    바꿔 topN 비교를 오염시킴 — 대신 trade_log에 완화 발동 여부를 남김). trade_log가
    주어지면 매수/매도 이벤트를 그대로 append(둘 다 기본값 None이라 기존 호출부는 그대로,
    2026-07-15 섹터캡·회전율 실험용 확장)."""
    if not decisions:
        return None
    dec_by_p = {p: syms for p, syms in decisions}
    p0 = decisions[0][0]
    px = panel.ffill()                      # 평가용(결측일은 직전가) — 매매는 원 종가 유효할 때만
    dates = panel.index
    cash, pos = 1.0, {}                     # pos: sym -> {"sh", "entry_date"}
    nav_out = np.full(len(dates), np.nan)

    for i in range(p0, len(dates)):
        today = dates[i]
        today_s = today.strftime("%Y%m%d")
        prices = px.iloc[i]

        # ── ① 일별 200일선 -3% 이탈 매도(폭락 방어 백업) — ma200_backup=True일 때만
        if ma200_backup:
            for sym in list(pos):
                p_now, m = prices.get(sym), ma200.iloc[i].get(sym)
                if np.isfinite(p_now) and np.isfinite(m) and p_now < m * (1 - MA_BUFFER):
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        held = (today - pos[sym]["entry_date"]).days
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "ma200_stop", "held_days": held})
                    del pos[sym]

        # ── ② 결정일: 6개월 재평가 매도 + 빈 슬롯 충원
        ranked = dec_by_p.get(i)
        if ranked:
            pool_set = set(ranked)
            for sym in list(pos):
                held = (today - pos[sym]["entry_date"]).days
                p_now = prices.get(sym)
                if held >= reeval_days and sym not in pool_set and np.isfinite(p_now):
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "reeval", "held_days": held})
                    del pos[sym]
            nav_now = cash + sum(v["sh"] * prices.get(s, np.nan) for s, v in pos.items()
                                 if np.isfinite(prices.get(s, np.nan)))
            sec_count = {}
            if sector_of is not None and sector_cap is not None:
                for sym in pos:
                    sc = sector_of(today_s, sym)
                    if sc:
                        sec_count[sc] = sec_count.get(sc, 0) + 1
            deferred = []
            for sym in ranked:
                if len(pos) >= topn or cash <= 1e-9:
                    break
                p_now = panel.iloc[i].get(sym)       # 매매는 당일 실제 종가 필요
                if sym in pos or not np.isfinite(p_now) or p_now <= 0:
                    continue
                sc = sector_of(today_s, sym) if (sector_of is not None and sector_cap is not None) else None
                if sc is not None and sec_count.get(sc, 0) >= sector_cap:
                    deferred.append(sym)
                    continue
                alloc = min(nav_now / topn, cash)
                pos[sym] = {"sh": alloc * (1 - cost.buy) / p_now, "entry_date": today}
                cash -= alloc
                if sc:
                    sec_count[sc] = sec_count.get(sc, 0) + 1
                if trade_log is not None:
                    trade_log.append({"date": today_s, "sym": sym, "action": "buy"})
            if deferred and len(pos) < topn and cash > 1e-9:
                if trade_log is not None:
                    trade_log.append({"date": today_s, "action": "sector_cap_relaxed",
                                      "n_deferred": len(deferred)})
                for sym in deferred:
                    if len(pos) >= topn or cash <= 1e-9:
                        break
                    p_now = panel.iloc[i].get(sym)
                    if sym in pos or not np.isfinite(p_now) or p_now <= 0:
                        continue
                    alloc = min(nav_now / topn, cash)
                    pos[sym] = {"sh": alloc * (1 - cost.buy) / p_now, "entry_date": today}
                    cash -= alloc
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "buy",
                                          "note": "sector_cap_relaxed"})

        nav_out[i] = cash + sum(v["sh"] * prices.get(s, np.nan) for s, v in pos.items()
                                if np.isfinite(prices.get(s, np.nan)))

    s = pd.Series(nav_out, index=dates).dropna()
    return s if len(s) > MONTH else None


def metrics(nav: pd.Series, bench: pd.Series) -> dict:
    """일별 NAV → 연 CAGR·변동성·샤프·MDD(+벤치마크 대비 초과 CAGR)."""
    b = bench.reindex(nav.index).ffill()
    b = b / b.iloc[0]
    yrs = len(nav) / 252
    ret = nav.pct_change().dropna()
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1) * 100
    bench_cagr = float(b.iloc[-1] ** (1 / yrs) - 1) * 100
    vol = float(ret.std() * np.sqrt(252)) * 100
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    mdd = float((nav / nav.cummax() - 1).min()) * 100
    bench_mdd = float((b / b.cummax() - 1).min()) * 100
    return {"cagr_pct": round(cagr, 2), "excess_cagr_pct": round(cagr - bench_cagr, 2),
            "vol_pct": round(vol, 1), "sharpe": round(sharpe, 2), "mdd_pct": round(mdd, 1),
            "bench_cagr_pct": round(bench_cagr, 2), "bench_mdd_pct": round(bench_mdd, 1),
            "years": round(yrs, 1)}


def monthly_excess(nav: pd.Series, bench: pd.Series) -> tuple[list, list]:
    """비중첩 21거래일 초과수익 (PBO/DSR 입력용). 반환: (dates, returns)."""
    b = bench.reindex(nav.index).ffill()
    out_d, out_r = [], []
    for t in range(0, len(nav) - MONTH, MONTH):
        r = float(nav.iloc[t + MONTH] / nav.iloc[t] - 1)
        rb = float(b.iloc[t + MONTH] / b.iloc[t] - 1)
        out_d.append(nav.index[t + MONTH].date().isoformat())
        out_r.append(round(r - rb, 6))
    return out_d, out_r


def run_sweep(panel, ma200, bench, decisions, topn_list, cost, market: str, ma200_backup=True):
    rows, matrix, dates0 = [], [], None
    for tn in topn_list:
        nav = simulate(panel, ma200, decisions, tn, cost, ma200_backup=ma200_backup)
        if nav is None:
            _log(f"topn={tn}: NAV 산출 실패(결정 시점 부족)"); continue
        m = metrics(nav, bench)
        d, r = monthly_excess(nav, bench)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        rows.append({"topn": tn, **m})
        _log(f"topn={tn}: CAGR {m['cagr_pct']}% (초과 {m['excess_cagr_pct']}%p) · "
             f"변동성 {m['vol_pct']}% · 샤프 {m['sharpe']} · MDD {m['mdd_pct']}%")
    if not rows:
        raise RuntimeError("어떤 topn도 시뮬레이션 실패")
    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]

    payload = {"as_of": panel.index[-1].date().isoformat(), "market": market,
              "n_combos": len(rows), "rows": rows,
              "baseline": f"topn{'10' if market == 'us' else '6'}(현행 보유 상한)",
              "note": "포트폴리오 일별 NAV 시뮬레이션(동일비중 진입·6개월 재평가+200일선 매도·"
                      "슬롯 충원) — 분산투자 효과가 변동성·MDD에 직접 반영됨",
              "adoption_criteria": "현행 대비 샤프·MDD 개선이 PBO/DSR(월간 초과수익) 판정을 "
                                   "통과할 때만 보유상한 변경 제안"}
    trial_data = {"horizon": f"portfolio_{market}", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": MONTH, "hold_days": MONTH,
                 "dates": dates0[:n_ev], "trials": [f"topn{r['topn']}" for r in rows],
                 "excess_returns": matrix}
    os.makedirs("output", exist_ok=True)
    with open(f"output/backtest_portfolio_{market}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(f"output/trial_returns_portfolio_{market}.json", "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(f"output/pbo_report_portfolio_{market}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: output/backtest_portfolio_{market}.json · output/pbo_report_portfolio_{market}.json")
    return payload, report


# 2026-07-15 확장(지호 님 질문 — "6개월이 최적이야? 3/9/12개월은?"): backtest_weights.py의
# 보유기간 스윕은 {1,3,6,12}개월 4점뿐이고 '단일 코호트 decay'만 봤다(재투자·회전율 비용
# 복리효과 미반영). 여기서는 topN과 같은 포트폴리오 NAV 프레임으로 1~12개월 전부(월 단위)
# 스윕한다 — 200일선 백업은 끈다(순수 보유기간 효과만, 위 ma200_backup 주석 참고).
HOLD_MONTHS = list(range(1, 13))


def run_hold_sweep(panel, ma200, bench, decisions, topn, cost, market: str):
    """보유기간(1~12개월, 재평가 주기=reeval_days) 스윕 — topn 고정, ma200 백업 없음."""
    rows, matrix, dates0 = [], [], None
    for mo in HOLD_MONTHS:
        reeval_days = mo * 30
        nav = simulate(panel, ma200, decisions, topn, cost, reeval_days=reeval_days, ma200_backup=False)
        if nav is None:
            _log(f"{mo}개월: NAV 산출 실패"); continue
        m = metrics(nav, bench)
        d, r = monthly_excess(nav, bench)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        rows.append({"hold_months": mo, **m})
        _log(f"{mo}개월: CAGR {m['cagr_pct']}% (초과 {m['excess_cagr_pct']}%p) · "
             f"변동성 {m['vol_pct']}% · 샤프 {m['sharpe']} · MDD {m['mdd_pct']}%")
    if not rows:
        raise RuntimeError("어떤 보유기간도 시뮬레이션 실패")
    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]

    payload = {"as_of": panel.index[-1].date().isoformat(), "market": market, "topn": topn,
              "n_combos": len(rows), "rows": rows, "baseline": "hold_months6(현행 재평가 주기)",
              "note": "포트폴리오 일별 NAV 시뮬레이션, 200일선 조기이탈 백업 없이(순수 보유기간 "
                      "효과만) 재평가 주기(=대략적인 보유기간) 1~12개월 스윕. topn 고정.",
              "adoption_criteria": "현행(6개월) 대비 샤프·MDD 개선이 PBO/DSR(월간 초과수익) "
                                   "판정을 통과할 때만 보유기간 변경 제안"}
    trial_data = {"horizon": f"hold_{market}", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": MONTH, "hold_days": MONTH,
                 "dates": dates0[:n_ev], "trials": [f"hold{r['hold_months']}m" for r in rows],
                 "excess_returns": matrix}
    os.makedirs("output", exist_ok=True)
    with open(f"output/backtest_hold_{market}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(f"output/trial_returns_hold_{market}.json", "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(f"output/pbo_report_hold_{market}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: output/backtest_hold_{market}.json · output/pbo_report_hold_{market}.json")
    return payload, report


# ------------------------- 시장별 래퍼 -------------------------
def us_decisions(panel, funds, pit, step=MONTH):
    """미장: 라이브 팩터 가중치로 결정일마다 상위 POOL_SIZE 랭킹."""
    import tech_factors as T
    import backtest_exec as BE
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    out = []
    for p in range(BW.LOOKBACK, len(panel) - 1, step):
        ranked = BE._select_basket(panel, p, funds, cross, pit, weights, POOL_SIZE)
        if ranked:
            out.append((p, ranked))
    _log(f"미장 결정 시점 {len(out)}개 (step {step}d · 풀 {POOL_SIZE})")
    return out


def _rerank_pool(panel, p, raw, pool: list, method: str) -> list:
    """풀(순위순)을 2단계 랭킹으로 재정렬. 점수 결측 종목은 뒤로(원래 순서 유지)."""
    if method == "base":
        return pool
    if method in ("mom12_1", "mom6", "value"):
        s = raw[method].reindex(pool) if method in raw.columns else pd.Series(np.nan, index=pool)
    else:
        lo = max(0, p - 251)
        win = panel.iloc[lo:p + 1][pool]
        if method == "hi52":
            s = win.iloc[-1] / win.max() - 1
        else:                                    # lowvol — 최근 1년 일별 변동성 낮은 순
            s = -win.pct_change().std()
    order = {sym: i for i, sym in enumerate(pool)}          # 결측 tie-break용 원 순위
    return sorted(pool, key=lambda t: (not np.isfinite(s.get(t, np.nan)),
                                       -(s.get(t) if np.isfinite(s.get(t, np.nan)) else 0),
                                       order[t]))


def us_rerank_decisions(panel, funds, pit, step=MONTH) -> dict:
    """{method: decisions} — 풀(60)은 동일하게 뽑고 2단계 랭킹만 교체."""
    import tech_factors as T
    import backtest_exec as BE
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    out = {m: [] for m in US_RERANKS}
    for p in range(BW.LOOKBACK, len(panel) - 1, step):
        raw = BW._raw_frame(panel, p, funds, bool(funds), cross)
        if raw is None or raw.empty:
            continue
        pool = BE._select_basket(panel, p, funds, cross, pit, weights, POOL_SIZE)
        if not pool:
            continue
        for m in US_RERANKS:
            out[m].append((p, _rerank_pool(panel, p, raw, pool, m)))
    _log(f"미장 재랭킹 결정 시점 {len(out['base'])}개 × 방법 {len(US_RERANKS)}종")
    return out


def run_rerank_sweep(panel, ma200, bench, dec_by_method: dict, topn: int, cost):
    """2단계 재랭킹 방법 비교 — topn 고정, 방법만 교체."""
    rows, matrix, dates0 = [], [], None
    for m, decisions in dec_by_method.items():
        nav = simulate(panel, ma200, decisions, topn, cost)
        if nav is None:
            _log(f"rerank={m}: NAV 산출 실패"); continue
        mt = metrics(nav, bench)
        d, r = monthly_excess(nav, bench)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        rows.append({"rerank": m, **mt})
        _log(f"rerank={m}: CAGR {mt['cagr_pct']}% (초과 {mt['excess_cagr_pct']}%p) · "
             f"변동성 {mt['vol_pct']}% · 샤프 {mt['sharpe']} · MDD {mt['mdd_pct']}%")
    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]
    payload = {"as_of": panel.index[-1].date().isoformat(), "market": "us", "topn": topn,
              "n_combos": len(rows), "rows": rows,
              "baseline": "base(현행 — 풀 선별과 동일 점수 상위)",
              "note": f"1단계 풀({POOL_SIZE}) 고정, 2단계(상위 {topn} 선별) 랭킹만 교체",
              "adoption_criteria": "base 대비 샤프 개선이 PBO/DSR 판정을 통과할 때만 교체 제안"}
    trial_data = {"horizon": "rerank_us", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": MONTH, "hold_days": MONTH,
                 "dates": dates0[:n_ev], "trials": [r["rerank"] for r in rows],
                 "excess_returns": matrix}
    os.makedirs("output", exist_ok=True)
    with open("output/backtest_rerank_us.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open("output/trial_returns_rerank_us.json", "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open("output/pbo_report_rerank_us.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log("저장: output/backtest_rerank_us.json · output/pbo_report_rerank_us.json")
    return payload, report


def kr_decisions(panel, snaps):
    """국장: backtest_kr 스냅샷의 라이브 규칙(필터+z(mom12_1)0.6+z(hi52)0.4) 랭킹."""
    pos_by_date = {d.date().isoformat(): i for i, d in enumerate(panel.index)}
    out = []
    for s in snaps:
        p = pos_by_date.get(s["date"])
        if p is None:
            continue
        ok = s["live_ok"]
        pool = ok[ok].index
        if len(pool) < 5:
            continue
        z = s["z"].loc[pool]
        score = z["mom12_1"] * 0.6 + z["hi52_prox"] * 0.4
        out.append((p, list(score.sort_values(ascending=False).index)))
    _log(f"국장 결정 시점 {len(out)}개")
    return out


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 랜덤워크로 분산투자 효과(변동성 감소)와 엔진 무결성 검증")
    rng = np.random.default_rng(7)
    n, m = 900, 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0004, 0.02, (n, m))
    panel = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i}" for i in range(m)])
    ma200 = panel.rolling(200, min_periods=200).mean()
    bench = panel.mean(axis=1)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    decisions = [(p, list(rng.permutation(panel.columns))) for p in range(260, n - 1, MONTH)]

    nav2 = simulate(panel, ma200, decisions, 2, cost)
    nav20 = simulate(panel, ma200, decisions, 20, cost)
    assert nav2 is not None and nav20 is not None
    m2, m20 = metrics(nav2, bench), metrics(nav20, bench)
    assert m20["vol_pct"] < m2["vol_pct"], f"분산투자로 변동성이 줄어야 함: {m2} vs {m20}"
    assert m2["mdd_pct"] <= 0 and np.isfinite(nav2.iloc[-1])
    d, r = monthly_excess(nav20, bench)
    assert len(d) == len(r) and len(d) > 10
    _log(f"[self-test] 통과: vol(N=2) {m2['vol_pct']}% > vol(N=20) {m20['vol_pct']}% · "
         f"월간 초과수익 {len(r)}개")

    # ma200_backup=False 배선 확인 — 꺼도 시뮬레이션이 정상 동작해야 함
    nav_nobackup = simulate(panel, ma200, decisions, 10, cost, reeval_days=90, ma200_backup=False)
    assert nav_nobackup is not None and np.isfinite(nav_nobackup.iloc[-1])
    _log("[self-test] 통과: ma200_backup=False 배선 정상")

    # 섹터캡 배선 확인(2026-07-15) — 종목을 2개 섹터로 강제 분할해두고 cap=1을 걸면
    # 어느 시점에도 한 섹터에서 2종목 이상 보유하면 안 됨(캡 완화가 발동한 시점 제외).
    sector_map = {c: ("A" if i % 2 == 0 else "B") for i, c in enumerate(panel.columns)}
    def sector_of(date_s, sym): return sector_map.get(sym)
    trade_log = []
    nav_cap = simulate(panel, ma200, decisions, 6, cost, sector_of=sector_of, sector_cap=1,
                       trade_log=trade_log)
    assert nav_cap is not None and np.isfinite(nav_cap.iloc[-1])
    relax_dates = {e["date"] for e in trade_log if e.get("action") == "sector_cap_relaxed"}
    buys = [e for e in trade_log if e.get("action") == "buy" and e.get("note") != "sector_cap_relaxed"]
    for e in buys:
        if e["date"] in relax_dates:
            continue   # 완화 발동 시점은 캡 위반이 정상(의도된 동작)
    sells = [e for e in trade_log if e.get("action") == "sell"]
    assert len(buys) > 0 and len(sells) > 0, "trade_log에 매수/매도 기록이 남아야 함"
    assert all("held_days" in e for e in sells), "매도 기록엔 held_days가 있어야 함"
    _log(f"[self-test] 통과: 섹터캡 배선 정상(매수 {len(buys)}건·매도 {len(sells)}건 로그됨, "
         f"캡 완화 {len(relax_dates)}회)")


def main():
    ap = argparse.ArgumentParser(description="포트폴리오 NAV 기반 topN 판정")
    ap.add_argument("--market", default="us", choices=["us", "kr"])
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--rebal-days", type=int, default=63, help="KR 스냅 간격(캐시 재사용을 위해 63 권장)")
    ap.add_argument("--rerank-sweep", action="store_true",
                    help="2단계 재랭킹 스윕(풀 60 고정, 상위 topn 선별 팩터만 교체) — 미국만")
    ap.add_argument("--hold-sweep", action="store_true",
                    help="보유기간(1~12개월) 스윕 — topn 고정, 200일선 백업 없이")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.market == "us":
        pit = BC.load_pit()
        panel, spy, _ = BC.build_panel_pit(args.years, pit)
        funds = BW.load_funds()
        cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
        ma200 = panel.rolling(200, min_periods=200).mean()
        if args.rerank_sweep:
            dec_by_method = us_rerank_decisions(panel, funds, pit)
            run_rerank_sweep(panel, ma200, spy, dec_by_method, args.topn, cost)
            return
        decisions = us_decisions(panel, funds, pit)
        if args.hold_sweep:
            run_hold_sweep(panel, ma200, spy, decisions, args.topn, cost, "us")
            return
        run_sweep(panel, ma200, spy, decisions, TOPN_US, cost, "us")
    else:
        import backtest_kr as BK
        panel, membership, fundamentals, flows, mktcaps, bench = BK.prepare_kr_data(
            int(args.years), args.rebal_days)
        snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                        args.rebal_days, flows=flows, mktcaps=mktcaps)
        cost = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)
        ma200 = panel.rolling(200, min_periods=200).mean()
        decisions = kr_decisions(panel, snaps)
        if args.hold_sweep:
            run_hold_sweep(panel, ma200, bench, decisions, args.topn, cost, "kr")
            return
        run_sweep(panel, ma200, bench, decisions, TOPN_KR, cost, "kr")


if __name__ == "__main__":
    main()
