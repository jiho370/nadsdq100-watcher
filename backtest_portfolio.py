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


def _year_end_trigger_dates(dates: pd.DatetimeIndex, cutoff_month=12, cutoff_day=28,
                            lookback_days=7) -> set:
    """연도별 (월,일) 컷오프 직전(±lookback_days 내) 마지막 거래일 집합 — 절세 목적 연말
    강제 정리 트리거(2026-07-15, 지호 님 질문). lookback_days 창 안에 실제 거래일이
    있어야만 트리거로 인정 — 데이터가 12월까지 안 닿는 마지막 부분연도를 "연말"로
    오인하는 버그 방지(self-test로 재현·수정 확인)."""
    trig = set()
    for y in sorted(set(dates.year)):
        cutoff = pd.Timestamp(year=y, month=cutoff_month, day=cutoff_day)
        window = dates[(dates <= cutoff) & (dates >= cutoff - pd.Timedelta(days=int(lookback_days)))]
        if len(window):
            trig.add(window.max())
    return trig


# ------------------------- 엔진 (시장 무관) -------------------------
def simulate(panel: pd.DataFrame, ma200: pd.DataFrame, decisions: list, topn: int,
             cost: BC.CostModel, reeval_days=REEVAL_DAYS, ma200_backup=True,
             sector_of=None, sector_cap=None, trade_log=None,
             ma_buffer=MA_BUFFER, ma_stop_mode="unconditional",
             entry_stop_pct=None, year_end_liquidate=False,
             year_end_rebuy="wait", full_rebalance=False) -> pd.Series | None:
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
    2026-07-15 섹터캡·회전율 실험용 확장).
    ma_stop_mode="state_gated"(2026-07-15, Fable 5 자문 — 매도알고리즘 실험)면 매수 시점에
    이미 200일선 버퍼 아래였던 종목(밸류 전략이 의도적으로 매수하는 케이스)은 MA 백업에서
    처음엔 면제되고, 이후 한 번이라도 버퍼 위로 회복("armed")한 다음에야 재이탈 시 매도된다
    — "매수 직후 즉시 매도" 버그의 근본 해법(진입 시점 조건부 면제보다 갈등 재발 여지가
    적음). entry_stop_pct가 주어지면 MA와 무관하게 "진입가 대비 -X% 하락 시 매도"를
    독립적으로 적용(트레일링/재난 스톱 — 둘 다 이 파라미터로 표현, 값만 다름: -25%=완만한
    트레일링, -40%=재난 스톱 근사, DART 이벤트 데이터 미연동이라 가격 임계값으로 근사).
    year_end_liquidate=True(2026-07-15, 지호 님 질문 — 연말 절세 관행)면 매년 12/28
    이전 마지막 거래일에 **수익 포지션만** 강제 매도(손실 포지션은 유지 — 손익 비대칭이
    핵심, Fable 5 자문). year_end_rebuy="immediate"면 같은 날 같은 종목·비중으로 즉시
    재매수(세금 이벤트만 발생, 시장노출 유지) — "wait"(기본)면 재매수 안 하고 다음
    정기 리밸런싱의 빈 슬롯 충원 로직이 자연스럽게 채우도록 둔다(현금 보유 구간 발생).
    full_rebalance=True(2026-07-16, 지호 님 정정 — "재평가 주기"가 뜻한 건 held_days
    끈적한 보유가 아니라 "매 결정일마다 전량 정리 후 그날 팩터 랭킹대로 재구성"이었음)면
    reeval_days·풀 소속 여부와 무관하게 매 결정일에 전원 매도 후 topn을 다시 채운다 —
    이 모드에서는 "재평가 주기"의 의미가 reeval_days가 아니라 decisions 자체의 간격
    (build_kr_snaps의 rebal_days)이 된다."""
    if not decisions:
        return None
    dec_by_p = {p: syms for p, syms in decisions}
    p0 = decisions[0][0]
    px = panel.ffill()                      # 평가용(결측일은 직전가) — 매매는 원 종가 유효할 때만
    dates = panel.index
    year_end_dates = _year_end_trigger_dates(dates) if year_end_liquidate else set()
    cash, pos = 1.0, {}                     # pos: sym -> {"sh", "entry_date", "entry_price", "armed"}
    nav_out = np.full(len(dates), np.nan)

    for i in range(p0, len(dates)):
        today = dates[i]
        today_s = today.strftime("%Y%m%d")
        prices = px.iloc[i]

        # ── ① 일별 200일선 -버퍼 이탈 매도(폭락 방어 백업) — ma200_backup=True일 때만
        if ma200_backup:
            for sym in list(pos):
                p_now, m = prices.get(sym), ma200.iloc[i].get(sym)
                if not (np.isfinite(p_now) and np.isfinite(m)):
                    continue
                above_line = p_now >= m * (1 - ma_buffer)
                if ma_stop_mode == "state_gated" and not pos[sym].get("armed", False):
                    if above_line:
                        pos[sym]["armed"] = True   # 버퍼 위로 회복 — 이제부터 스톱 적용 대상
                    continue                       # 아직 미회복(면제 구간)이면 이번 스텝은 건너뜀
                if not above_line:
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        held = (today - pos[sym]["entry_date"]).days
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "ma200_stop", "held_days": held})
                    del pos[sym]

        # ── ①b 진입가 대비 -entry_stop_pct% 하락 매도(트레일링/재난 스톱, MA와 무관)
        if entry_stop_pct is not None:
            for sym in list(pos):
                p_now = prices.get(sym)
                ep = pos[sym].get("entry_price")
                if np.isfinite(p_now) and ep and p_now < ep * (1 - entry_stop_pct):
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        held = (today - pos[sym]["entry_date"]).days
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "entry_stop", "held_days": held})
                    del pos[sym]

        # ── ①c 연말 강제 정리(절세 관행) — 수익 포지션만 매도, 손실은 유지
        if year_end_liquidate and today in year_end_dates:
            rebuy = []
            for sym in list(pos):
                p_now = prices.get(sym)
                ep = pos[sym].get("entry_price")
                if np.isfinite(p_now) and ep and p_now > ep:
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        held = (today - pos[sym]["entry_date"]).days
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "year_end_taxharvest", "held_days": held})
                    armed_prev = pos[sym].get("armed", True)
                    del pos[sym]
                    if year_end_rebuy == "immediate":
                        rebuy.append((sym, p_now, armed_prev))
            if rebuy:
                nav_now = cash + sum(v["sh"] * prices.get(s, np.nan) for s, v in pos.items()
                                     if np.isfinite(prices.get(s, np.nan)))
                for sym, p_now, armed_prev in rebuy:
                    if cash <= 1e-9 or not np.isfinite(p_now) or p_now <= 0:
                        continue
                    alloc = min(nav_now / topn, cash)
                    pos[sym] = {"sh": alloc * (1 - cost.buy) / p_now, "entry_date": today,
                               "entry_price": p_now, "armed": armed_prev}
                    cash -= alloc
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "buy",
                                          "note": "year_end_immediate_rebuy"})

        # ── ② 결정일: 6개월 재평가 매도 + 빈 슬롯 충원 (또는 full_rebalance면 전량 정리 후 재구성)
        ranked = dec_by_p.get(i)
        if ranked:
            pool_set = set(ranked)
            for sym in list(pos):
                held = (today - pos[sym]["entry_date"]).days
                p_now = prices.get(sym)
                if not np.isfinite(p_now):
                    continue
                # full_rebalance=True(2026-07-16, 지호 님 정정 — "재평가 시점마다 전량
                # 정리하고 그날 팩터 랭킹대로 다시 산다"는 의미였음): held_days·풀 소속
                # 무관하게 결정일마다 전원 매도 후 재구성. False(기본, 라이브 방식)면
                # 기존처럼 "보유 ≥reeval_days AND 풀 밖"인 종목만 선별 매도(끈적한 보유).
                if full_rebalance or (held >= reeval_days and sym not in pool_set):
                    cash += pos[sym]["sh"] * p_now * (1 - cost.sell)
                    if trade_log is not None:
                        trade_log.append({"date": today_s, "sym": sym, "action": "sell",
                                          "reason": "full_rebalance" if full_rebalance else "reeval",
                                          "held_days": held})
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
                m_now = ma200.iloc[i].get(sym)
                armed0 = not (np.isfinite(m_now) and p_now < m_now * (1 - ma_buffer))  # 매수 시 이미 버퍼 아래면 미무장
                pos[sym] = {"sh": alloc * (1 - cost.buy) / p_now, "entry_date": today,
                           "entry_price": p_now, "armed": armed0}
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
                    m_now = ma200.iloc[i].get(sym)
                    armed0 = not (np.isfinite(m_now) and p_now < m_now * (1 - ma_buffer))
                    pos[sym] = {"sh": alloc * (1 - cost.buy) / p_now, "entry_date": today,
                               "entry_price": p_now, "armed": armed0}
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

    # 2026-07-15 "매수 직후 즉시매도" 버그 재현·수정 확인 — 지호 님 리포트 대응.
    # 종목 A를 200일선 버퍼 아래에서 매수(밸류 전략이 실제로 하는 행동)하도록 가격을 설계:
    # 처음 260일은 평탄(MA 형성용), 이후 급락시켜 매수 시점엔 이미 버퍼 아래.
    idx2 = pd.bdate_range("2020-01-01", periods=400)
    flat = np.full(180, 100.0)
    predecline = 100.0 * np.exp(np.cumsum(np.full(80, -0.0025)))    # 매수 전 완만한 하락
    postdecline = predecline[-1] * np.exp(np.cumsum(np.full(140, -0.001)))  # 매수 후도 계속 하락(회복 없음)
    priceA = pd.Series(np.concatenate([flat, predecline, postdecline]), index=idx2)
    # 매수 시점(day 260) 가격이 이미 MA 버퍼(-3%) 아래인지 확인(테스트 전제 조건)
    assert priceA.iloc[260] < priceA.rolling(200, min_periods=200).mean().iloc[260] * 0.97
    panelA = pd.DataFrame({"A": priceA})
    ma200A = panelA.rolling(200, min_periods=200).mean()
    decA = [(260, ["A"])]
    costA = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    log_unconditional = []
    simulate(panelA, ma200A, decA, 1, costA, reeval_days=99999, ma200_backup=True,
            ma_stop_mode="unconditional", trade_log=log_unconditional)
    ma200_sells_uncond = [e for e in log_unconditional if e.get("reason") == "ma200_stop"]
    assert ma200_sells_uncond and ma200_sells_uncond[0]["held_days"] <= 10, (
        f"unconditional 모드는 버퍼 아래 매수 시 즉시(수일 내) 매도돼야 버그 재현: {log_unconditional}")

    log_gated = []
    simulate(panelA, ma200A, decA, 1, costA, reeval_days=99999, ma200_backup=True,
            ma_stop_mode="state_gated", trade_log=log_gated)
    ma200_sells_gated = [e for e in log_gated if e.get("reason") == "ma200_stop"]
    assert not ma200_sells_gated, (
        f"state_gated 모드는 회복 없이 계속 버퍼 아래면 MA스톱이 아예 발동하면 안 됨(면제 유지): {log_gated}")
    _log(f"[self-test] 통과: 매수직후즉시매도 버그 — unconditional 재현(held_days="
         f"{ma200_sells_uncond[0]['held_days']}) vs state_gated 면제(스톱 0건) 확인")

    # entry_stop_pct(트레일링/재난 스톱) 배선 확인 — MA와 무관하게 진입가 대비 하락으로 매도
    log_entry_stop = []
    simulate(panelA, ma200A, decA, 1, costA, reeval_days=99999, ma200_backup=False,
            entry_stop_pct=0.10, trade_log=log_entry_stop)
    entry_stop_sells = [e for e in log_entry_stop if e.get("reason") == "entry_stop"]
    assert entry_stop_sells, f"진입가 -10% 하락 시 entry_stop 매도가 발동해야 함: {log_entry_stop}"
    _log(f"[self-test] 통과: entry_stop_pct 배선 정상(held_days={entry_stop_sells[0]['held_days']})")

    # 2026-07-15 연말 강제 정리(절세) 배선 확인 — 수익 종목(W)만 매도, 손실 종목(L)은 유지.
    idxY = pd.bdate_range("2020-01-01", periods=300)
    up = 100.0 * np.exp(np.cumsum(np.full(250, 0.003)))     # 수익 종목
    down = 100.0 * np.exp(np.cumsum(np.full(250, -0.003)))  # 손실 종목
    panelY = pd.DataFrame({"W": np.concatenate([np.full(50, 100.0), up]),
                           "L": np.concatenate([np.full(50, 100.0), down])}, index=idxY)
    ma200Y = panelY.rolling(50, min_periods=1).mean()   # ma200_backup 안 씀 — armed 초기화용 더미
    decY = [(50, ["W", "L"])]
    costY = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    log_wait = []
    simulate(panelY, ma200Y, decY, 2, costY, reeval_days=99999, ma200_backup=False,
            year_end_liquidate=True, year_end_rebuy="wait", trade_log=log_wait)
    ye_sells = [e for e in log_wait if e.get("reason") == "year_end_taxharvest"]
    assert len(ye_sells) >= 1 and all(e["sym"] == "W" for e in ye_sells), (
        f"연말 강제정리는 수익종목(W)만 팔아야 함: {ye_sells}")
    assert not any(e.get("action") == "buy" and e.get("note") == "year_end_immediate_rebuy"
                  for e in log_wait), "year_end_rebuy='wait'는 즉시 재매수하면 안 됨"

    log_immediate = []
    simulate(panelY, ma200Y, decY, 2, costY, reeval_days=99999, ma200_backup=False,
            year_end_liquidate=True, year_end_rebuy="immediate", trade_log=log_immediate)
    rebuys = [e for e in log_immediate if e.get("note") == "year_end_immediate_rebuy"]
    assert len(rebuys) >= 1 and all(e["sym"] == "W" for e in rebuys), (
        f"year_end_rebuy='immediate'는 매도된 수익종목을 같은 날 재매수해야 함: {rebuys}")
    # 부분연도(12월까지 안 닿는 마지막 구간) 오탐 방지 확인 — 패널이 2021-02-23에서 끝나므로
    # 2021년엔 실제 12월 데이터가 없어 트리거가 없어야 함(있으면 버그).
    trig = _year_end_trigger_dates(panelY.index)
    assert all(d.month == 12 for d in trig), f"12월 데이터 없는 부분연도가 트리거로 오탐: {trig}"
    _log(f"[self-test] 통과: 연말 강제정리 배선 정상(수익종목만 매도 {len(ye_sells)}회, "
         f"즉시재매수 모드 재매수 {len(rebuys)}회, 트리거일 {sorted(trig)})")

    # 2026-07-16 full_rebalance 배선 확인 — 여전히 풀 안에 있어도(끈적한 보유였다면 안 팔림)
    # 결정일마다 무조건 전량 매도 후 재구성돼야 함.
    idxF = pd.bdate_range("2020-01-01", periods=400)
    flatF = np.full(400, 100.0)
    panelF = pd.DataFrame({"A": flatF, "B": flatF}, index=idxF)
    ma200F = panelF.rolling(50, min_periods=1).mean()
    decF = [(50, ["A", "B"]), (150, ["A", "B"]), (250, ["A", "B"])]   # 매 결정일 동일 풀(끈적하면 안 팔릴 케이스)
    costF = BC.CostModel("kospi", commission_bps=1.5, slippage_bps=5.0)

    log_sticky = []
    simulate(panelF, ma200F, decF, 2, costF, reeval_days=99999, ma200_backup=False,
            full_rebalance=False, trade_log=log_sticky)
    sticky_sells = [e for e in log_sticky if e.get("action") == "sell"]
    assert not sticky_sells, f"끈적한 모드는 풀에 계속 있으면 안 팔려야 함: {sticky_sells}"

    log_full = []
    simulate(panelF, ma200F, decF, 2, costF, reeval_days=99999, ma200_backup=False,
            full_rebalance=True, trade_log=log_full)
    full_sells = [e for e in log_full if e.get("reason") == "full_rebalance"]
    assert len(full_sells) == 4, (   # 결정일 150·250에서 A·B 각각 매도 = 2×2=4건(첫 결정일은 신규매수뿐)
        f"full_rebalance는 2번째·3번째 결정일마다 보유 2종목을 전부 팔아야 함(기대 4건): {full_sells}")
    _log(f"[self-test] 통과: full_rebalance 배선 정상(끈적한 모드 매도 0건 vs "
         f"full_rebalance 매도 {len(full_sells)}건)")


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
