#!/usr/bin/env python3
"""
backtest_exec.py — 트랙 C: 지지·저항(S/R) 실행규칙 백테스트 (SCORE_MODEL_DESIGN.md 부록 A).

원칙(부록 A): "많이들 쓴다"는 채택도 기각도 아니다 — 코드로 재현 가능한 정의를 만들고
판정은 백테스트(PBO/DSR)가 한다.

A1. S/R 신호(전부 종가·거래량만으로 계산, 7종) — score_calibration.py --candidates sr 가
    SR_CANDIDATES 로 가져다 쓴다(A2-a, 횡단면 팩터 검정, 예산 분리 — 기본 실행엔 안 섞임).
      sr_support_dist / sr_resist_dist : 로컬 극값(±2% 밴드 군집, 터치 2회 이상) 최근접 레벨까지 거리%
      hi52_prox / lo52_prox / ath_prox : 52주 고점·저점·역사적 신고가 근접도
      round_prox   : 라운드넘버(1·2·5×10^k)까지 거리(음수=근접)
      vol_poc_dist : 거래량 프로파일 최빈가 근사(252일 거래량가중평균가 VWAP) 대비 거리%
      한계: round_prox/S-R는 종가만, vol_poc_dist는 진짜 POC(히스토그램 최빈 구간) 대신
      계산비용이 낮은 VWAP로 근사 — 정확한 POC가 필요하면 별도 개선 필요.

A2-(b). 실행 규칙 비교("진입·청산 타이밍이 순수익을 개선하는가") — 동일 종목·동일 선정
    시점(종목 선정은 output/best_weights.json 고정, 없으면 모멘텀 폴백)에서 규칙만 바꾼다:
      진입 ① entry1_full     : 신호일 익일 전량
           ② entry2_pullback2: 1차 50% 익일 / 2차 50% 20일선(-3%) 눌림 대기(10거래일, 못 채우면 미체결)
      청산 ① exit1_trail20     : 현재 시스템과 동일(holdings.py) — 고점대비 -20% 또는 200일선 -3%
           ② exit2_atr2stage   : 피크대비 -1.5×ATR60 절반청산 → -2.5×ATR60(또는 200일선-3%) 전량
           ③ exit3_support2stage: 최근접 지지 -2% 절반 → -4% 전량(진입 시점 지지선 고정 사용)
    평가: net 수익(backtest_costs.CostModel 재사용), 회전율, 손절빈도, MDD(트레이드별 최대낙폭),
    미체결률. 판정은 overfit_stats.analyze()로 동일 프레임(PBO/DSR) 재사용.

실행(PC): python backtest_costs.py --years 10 ...        # PIT 패널(사전 확인용, 재사용은 자체 로드)
          python backtest_exec.py --years 10
          python backtest_exec.py --self-test
결과: output/backtest_exec_compare.json (규칙조합 비교표 · 회전율·손절빈도·MDD·미체결률)
      output/trial_returns_exec.json    (overfit_stats 입력, 조합=진입×청산 6종)
      output/pbo_report_exec.json       (PBO/DSR 판정 — 채택 기준은 SCORE_MODEL_DESIGN.md 부록 A3)
"""
from __future__ import annotations
import os, sys, re, json, math, argparse
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC
import overfit_stats as OS

TRIAL_PATH = "output/trial_returns_exec.json"
REPORT_PATH = "output/pbo_report_exec.json"
COMPARE_PATH = "output/backtest_exec_compare.json"
SR_CALIB_PATH = "output/score_calibration_sr.json"   # A2-(a) 연구용 — 기본 게이트(score_calibration.json)와 분리

SR_CANDIDATES = ["sr_support_dist", "sr_resist_dist", "hi52_prox", "lo52_prox",
                 "ath_prox", "round_prox", "vol_poc_dist"]

# 2026-07 확장(지호 님 요청): 진입 3종 × 청산 7종 = 21조합.
#   entry3_pullback3 = 라이브 '과열' 규칙(30/30/40 — 현재가/20일선/50일선) 그대로
#   exit_time6m      = 검증된 백테스트의 원형(고정 6개월) — 현행 트레일링과의 핵심 대조군
#   exit_trail15/25  = 현행 -20%의 파라미터 민감도 스윕
ENTRY_RULES = ["entry1_full", "entry2_pullback2", "entry3_pullback3"]
EXIT_RULES = ["exit_trail15", "exit_trail20", "exit_trail25", "exit_ma200only",
              "exit_time6m", "exit_atr2stage", "exit_support2stage"]
BASELINE = "entry1_full__exit_trail20"

# 2026-07-13 확장(지호 님 질문 — "분할매수 비율도 백테스트 근거 있나"): 위 21조합은
# entry1_full(전량)과의 '분할 vs 전량' 비교만 했을 뿐, 정확한 비율(50/50·30/30/40)
# 자체는 스윕한 적이 없었다. entry2_<w1w2>·entry3_<w1w2w3> 형식(각 2자리%, 합 100)의
# 이름은 _parse_entry_ratio가 파싱해 비율만 다르게 시뮬레이션한다 — 트리거 기준선
# (20일선-3%/50일선-8%)은 라이브와 동일하게 고정, 비율만 변수.
ENTRY_RATIO_2 = {"entry2_5050": (0.50, 0.50), "entry2_3070": (0.30, 0.70), "entry2_7030": (0.70, 0.30)}
ENTRY_RATIO_3 = {"entry3_303040": (0.30, 0.30, 0.40), "entry3_502525": (0.50, 0.25, 0.25),
                 "entry3_204040": (0.20, 0.40, 0.40)}
ENTRY_RATIO_SWEEP = list(ENTRY_RATIO_2) + list(ENTRY_RATIO_3)

# 2026-07-13 확장(지호 님 질문 — "매도도 매수 분할이랑 맞춰봐야 하지 않나"): 위 EXIT_RULES는
# '언제 매도를 결정하는가'(트레일링/200일선/6개월/ATR/지지선)만 검정했다. entry_plan.sell_plan()의
# 실제 라이브 처분 방식("50% 즉시 + 50% 반등(20일선) 대기, 2주 내 미반등 시 전량, 단 -15%
# 초과손실이면 즉시 전량")은 '결정된 매도를 어떻게 집행하는가'라는 별개 질문인데 한 번도
# 검증된 적이 없었다 — exit_time6m(전량 즉시)을 대조군으로, 채택된 6개월 트리거 시점 이후의
# 처분 비율/대기기간만 스윕한다. 이름 규칙: exit_time6m_<w1w2>w<대기거래일수>.
DISPOSAL_SWEEP = {
    "exit_time6m": None,                          # 대조군 — 트리거 즉시 전량(현행 exit_time6m)
    "exit_time6m_5050w10": ((0.50, 0.50), 10),     # 라이브 현행: 50%즉시+50%반등대기(2주=10거래일)
    "exit_time6m_3070w10": ((0.30, 0.70), 10),
    "exit_time6m_7030w10": ((0.70, 0.30), 10),
    "exit_time6m_5050w5":  ((0.50, 0.50), 5),      # 대기기간 민감도(1주)
    "exit_time6m_5050w20": ((0.50, 0.50), 20),     # 대기기간 민감도(4주)
}
DISPOSAL_LOSS_OVERRIDE = -0.15   # entry_plan.sell_plan()과 동일 — 초과 손실이면 대기 없이 즉시 전량

# 2026-07-14 확장(지호 님 질문 — "몇 종목을 보유할지도 백테스트 근거가 있나"): 지금까지의
# 스윕은 전부 '고른 종목을 어떻게 사고 파는가'만 봤다. '몇 종목을 고르는가'(topn, 현재 미국10·
# 한국6은 순수 제품 판단)는 다른 문제 — 매 리밸런싱 시점의 팩터 상위 N을 그대로 바꿔가며
# 비교한다. 진입 entry1_full·청산 exit_time6m 고정(topn 효과만 순수 비교).
TOPN_SWEEP = [5, 8, 10, 12, 15, 20]

PULLBACK_WINDOW = 10     # 2차 트랜치 눌림 대기 거래일
ATR_WINDOW = 60
MAX_HOLD = 252           # 강제청산 상한(12m) — STRATEGY.md 장기보유 취지상 이 이상은 안 봄
TRAIL = 0.20             # holdings.py와 동일(env SELL_TRAIL로 조정 가능하나 여기선 고정 비교)
MA_BUFFER = 0.03


def _log(m): print(f"[실행규칙] {m}", file=sys.stderr)


# ------------------------- A1: S/R 신호 (종가·거래량만) -------------------------
def _local_extrema_idx(vals: np.ndarray, order=5):
    """1일 확정 지연이 아니라 order일 확정 지연(양옆 order일과 비교) — 표준적 피벗 확정
    관행이며 몇 거래일 지연일 뿐 미래(수개월) 참조가 아니다. 결측 구간은 극값 판정에서 제외."""
    n = len(vals)
    mins = np.zeros(n, dtype=bool)
    maxs = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        w = vals[i - order:i + order + 1]
        if not np.isfinite(w).all():
            continue
        if vals[i] <= w.min():
            mins[i] = True
        if vals[i] >= w.max():
            maxs[i] = True
    return mins, maxs


def _nearest_cluster(prices: np.ndarray, ref_price: float, band_pct=0.02, min_touches=2, below=True):
    """가격들을 ±band_pct 이내로 군집화 → min_touches 이상 터치한 레벨 중 ref_price에 최근접."""
    if len(prices) == 0:
        return None
    prices = np.sort(prices)
    clusters, cur = [], [prices[0]]
    for p in prices[1:]:
        if abs(p - cur[-1]) / cur[-1] <= band_pct:
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    levels = [float(np.mean(c)) for c in clusters if len(c) >= min_touches]
    if not levels:
        return None
    cand = [lv for lv in levels if (lv < ref_price if below else lv > ref_price)]
    if not cand:
        return None
    return max(cand) if below else min(cand)


def support_level_asof(vals: np.ndarray, t: int, lookback=252, order=5, band_pct=0.02, min_touches=2):
    """단일 시점(t) 지지선 1회 조회 — 실행엔진이 트레이드당 1회만 호출(전체 패널 계산 아님)."""
    lo = max(0, t - lookback)
    if t - lo < order * 2 + 1:
        return None
    mins, _ = _local_extrema_idx(vals[lo:t + 1], order)
    idx = np.where(mins)[0]
    idx = idx[idx <= (t - lo - order)]
    if len(idx) == 0:
        return None
    px = vals[lo:t + 1][idx]
    price = vals[t]
    if not np.isfinite(price):
        return None
    return _nearest_cluster(px, price, band_pct, min_touches, below=True)


def _round_level_arr(vals: np.ndarray) -> np.ndarray:
    out = np.full(vals.shape, np.nan)
    flat = vals.ravel()
    outf = out.ravel()
    for i, p in enumerate(flat):
        if np.isfinite(p) and p > 0:
            k = math.floor(math.log10(p))
            cands = [b * 10 ** kk for kk in (k - 1, k, k + 1) for b in (1, 2, 5)]
            outf[i] = min(cands, key=lambda c: abs(c - p))
    return out


def sr_signal_panels(panel: pd.DataFrame, vol_panel: pd.DataFrame | None = None,
                     lookback=252, order=5, band_pct=0.02, min_touches=2) -> dict:
    """A1 — 7개 S/R 신호를 패널 전체(dates×syms)로 계산. 연구용 스크립트(수 분 소요 가능) —
    score_calibration.py --candidates sr 가 사용(기본 실행에는 안 섞임, 예산 분리)."""
    hi = panel.rolling(252, min_periods=60).max()
    lo = panel.rolling(252, min_periods=60).min()
    ath = panel.expanding(min_periods=60).max()
    hi52_prox = panel / hi - 1
    lo52_prox = panel / lo - 1
    ath_prox = panel / ath - 1

    lvl_arr = _round_level_arr(panel.to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        round_prox = pd.DataFrame(-np.abs(panel.to_numpy(dtype=float) / lvl_arr - 1),
                                  index=panel.index, columns=panel.columns)

    if vol_panel is not None:
        vp = vol_panel.reindex(index=panel.index, columns=panel.columns)
        pv = (panel * vp).rolling(252, min_periods=60).sum()
        vsum = vp.rolling(252, min_periods=60).sum()
        vwap = pv / vsum.replace(0, np.nan)
        vol_poc_dist = panel / vwap - 1
    else:
        vol_poc_dist = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)

    sup_out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    res_out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for ci, col in enumerate(panel.columns):
        vals = panel[col].to_numpy(dtype=float)
        n = len(vals)
        if n < lookback + 2 * order:
            continue
        mins_mask, maxs_mask = _local_extrema_idx(vals, order)
        min_idx = np.where(mins_mask)[0]; max_idx = np.where(maxs_mask)[0]
        min_px = vals[min_idx]; max_px = vals[max_idx]
        for t in range(lookback, n):
            price = vals[t]
            if not np.isfinite(price):
                continue
            cut, lo_b = t - order, t - lookback
            mm = (min_idx <= cut) & (min_idx >= lo_b)
            Mm = (max_idx <= cut) & (max_idx >= lo_b)
            s_lvl = _nearest_cluster(min_px[mm], price, band_pct, min_touches, below=True)
            r_lvl = _nearest_cluster(max_px[Mm], price, band_pct, min_touches, below=False)
            if s_lvl is not None:
                sup_out.iat[t, ci] = price / s_lvl - 1
            if r_lvl is not None:
                res_out.iat[t, ci] = price / r_lvl - 1

    return {"sr_support_dist": sup_out, "sr_resist_dist": res_out,
            "hi52_prox": hi52_prox, "lo52_prox": lo52_prox, "ath_prox": ath_prox,
            "round_prox": round_prox, "vol_poc_dist": vol_poc_dist}


# ------------------------- A2-(b): 진입·청산 규칙 엔진 -------------------------
def _ma(panel, w):
    return panel.rolling(w, min_periods=w).mean()


def _atr_close(panel, w=ATR_WINDOW):
    """종가만으로 근사한 ATR — 고가/저가 데이터가 없어 일별 절대수익률×가격의 이동평균으로
    대체(진짜 True Range보다 변동성을 다소 과소평가할 수 있음 — 근사임을 명시)."""
    ret = panel.pct_change().abs()
    return ret.rolling(w, min_periods=max(w // 2, 5)).mean() * panel


def _load_exec_weights():
    """실행규칙 비교는 '어떤 종목을 뽑는가'가 아니라 '어떻게 사고 파는가'만 검정하는 것이 목적
    — 종목 선정은 라이브와 동일하게 output/best_weights.json(발행된 검증 가중치) 고정, 없으면
    모멘텀 폴백(export_data.load_best_weights()와 동일 우선순위)."""
    try:
        with open("output/best_weights.json", encoding="utf-8") as f:
            w = (json.load(f).get("weights") or {})
        if any(w.values()):
            return {k: v for k, v in w.items() if v}
    except Exception:
        pass
    return {"mom6": 1, "mom12_1": 1}


def _select_basket(panel, p, funds, cross, pit, weights, topn):
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
    return list(score.sort_values(ascending=False).index[:topn])


def _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, entry_rule, exit_rule):
    """단일 종목·단일 이벤트의 진입~청산 시뮬레이션(일별 경로). 반환: dict 또는 None(가격 결측)."""
    vals = panel[sym].to_numpy(dtype=float)
    n = len(vals)
    cap = min(entry_day + MAX_HOLD, n - 1)
    if entry_day >= n or not np.isfinite(vals[entry_day]):
        return None
    p1 = vals[entry_day]

    def _wait_fill(target, window):
        for d in range(entry_day + 1, min(entry_day + 1 + window, n)):
            if np.isfinite(vals[d]) and vals[d] <= target:
                return vals[d], d
        return None, None

    # ---- 진입 ----
    if entry_rule == "entry1_full":
        entry_price, filled_frac, entry_ref_day = p1, 1.0, entry_day
    elif entry_rule == "entry2_pullback2":
        ma20v = ma20[sym].to_numpy(dtype=float)
        base = ma20v[entry_day] * 0.97 if np.isfinite(ma20v[entry_day]) else p1 * 0.97
        fill2, day2 = _wait_fill(min(base, p1), PULLBACK_WINDOW)
        if fill2 is not None:
            entry_price, filled_frac, entry_ref_day = 0.5 * p1 + 0.5 * fill2, 1.0, day2
        else:
            entry_price, filled_frac, entry_ref_day = p1, 0.5, entry_day
    elif entry_rule == "entry3_pullback3":  # 라이브 과열 규칙: 30% 즉시 / 30% 20일선-3% / 40% 50일선-8%
        ma20v = ma20[sym].to_numpy(dtype=float)
        ma50v = ma50[sym].to_numpy(dtype=float)
        t2 = min(ma20v[entry_day] * 0.97 if np.isfinite(ma20v[entry_day]) else p1 * 0.97, p1)
        t3 = min(ma50v[entry_day] * 0.92 if np.isfinite(ma50v[entry_day]) else p1 * 0.92, p1)
        f2, d2 = _wait_fill(t2, PULLBACK_WINDOW)
        f3, d3 = _wait_fill(t3, PULLBACK_WINDOW * 2)
        fills = [(0.3, p1, entry_day)]
        if f2 is not None:
            fills.append((0.3, f2, d2))
        if f3 is not None:
            fills.append((0.4, f3, d3))
        filled_frac = sum(w for w, _, _ in fills)
        entry_price = sum(w * px for w, px, _ in fills) / filled_frac
        entry_ref_day = max(d for _, _, d in fills)
    elif entry_rule in ENTRY_RATIO_2:   # 2분할 비율 스윕 — 트리거는 entry2_pullback2와 동일(20일선-3%)
        w1, w2 = ENTRY_RATIO_2[entry_rule]
        ma20v = ma20[sym].to_numpy(dtype=float)
        base = ma20v[entry_day] * 0.97 if np.isfinite(ma20v[entry_day]) else p1 * 0.97
        fill2, day2 = _wait_fill(min(base, p1), PULLBACK_WINDOW)
        if fill2 is not None:
            entry_price, filled_frac, entry_ref_day = w1 * p1 + w2 * fill2, 1.0, day2
        else:
            entry_price, filled_frac, entry_ref_day = p1, w1, entry_day
    elif entry_rule in ENTRY_RATIO_3:   # 3분할 비율 스윕 — 트리거는 entry3_pullback3와 동일
        w1, w2, w3 = ENTRY_RATIO_3[entry_rule]
        ma20v = ma20[sym].to_numpy(dtype=float)
        ma50v = ma50[sym].to_numpy(dtype=float)
        t2 = min(ma20v[entry_day] * 0.97 if np.isfinite(ma20v[entry_day]) else p1 * 0.97, p1)
        t3 = min(ma50v[entry_day] * 0.92 if np.isfinite(ma50v[entry_day]) else p1 * 0.92, p1)
        f2, d2 = _wait_fill(t2, PULLBACK_WINDOW)
        f3, d3 = _wait_fill(t3, PULLBACK_WINDOW * 2)
        fills = [(w1, p1, entry_day)]
        if f2 is not None:
            fills.append((w2, f2, d2))
        if f3 is not None:
            fills.append((w3, f3, d3))
        filled_frac = sum(w for w, _, _ in fills)
        entry_price = sum(w * px for w, px, _ in fills) / filled_frac
        entry_ref_day = max(d for _, _, d in fills)
    else:
        raise ValueError(f"알 수 없는 entry_rule: {entry_rule}")

    # ---- 청산 ----
    peak = entry_price
    exit_price, exit_day, stop_triggered = None, cap, False
    ma200v = ma200[sym].to_numpy(dtype=float)

    if exit_rule.startswith("exit_trail"):        # 트레일링 스윕(15/20/25%) + 200일선 백업(현행 구조)
        trail = int(exit_rule[-2:]) / 100.0
        for d in range(entry_day + 1, cap + 1):
            if not np.isfinite(vals[d]):
                continue
            peak = max(peak, vals[d])
            if (np.isfinite(ma200v[d]) and vals[d] < ma200v[d] * (1 - MA_BUFFER)) or \
               vals[d] < peak * (1 - trail):
                exit_price, exit_day, stop_triggered = vals[d], d, True
                break
        if exit_price is None:
            exit_price, exit_day = vals[cap], cap

    elif exit_rule == "exit_ma200only":           # 200일선 이탈만(트레일링 없음)
        for d in range(entry_day + 1, cap + 1):
            if np.isfinite(vals[d]) and np.isfinite(ma200v[d]) and vals[d] < ma200v[d] * (1 - MA_BUFFER):
                exit_price, exit_day, stop_triggered = vals[d], d, True
                break
        if exit_price is None:
            exit_price, exit_day = vals[cap], cap

    elif exit_rule == "exit_time6m":              # 고정 6개월 — 검증된 백테스트의 원형(대조군)
        exit_day = min(entry_day + 126, cap)
        while exit_day > entry_day and not np.isfinite(vals[exit_day]):
            exit_day -= 1
        exit_price = vals[exit_day]

    elif exit_rule in DISPOSAL_SWEEP and DISPOSAL_SWEEP[exit_rule] is not None:
        # 6개월 트리거는 exit_time6m과 동일하게 확정하되, 그 이후 '처분'만 라이브 sell_plan()
        # 방식(분할+반등대기, 손실 -15% 초과 시 즉시 전량)으로 시뮬레이션.
        (w1, w2), window = DISPOSAL_SWEEP[exit_rule]
        trig_day = min(entry_day + 126, cap)
        while trig_day > entry_day and not np.isfinite(vals[trig_day]):
            trig_day -= 1
        trig_ret = vals[trig_day] / entry_price - 1
        if trig_ret <= DISPOSAL_LOSS_OVERRIDE:
            exit_price, exit_day, stop_triggered = vals[trig_day], trig_day, True
        else:
            d1 = min(trig_day + 1, cap)
            while d1 > trig_day and not np.isfinite(vals[d1]):
                d1 -= 1
            p1_exit = vals[d1] if np.isfinite(vals[d1]) else vals[trig_day]
            ma20v = ma20[sym].to_numpy(dtype=float)
            target = ma20v[d1] if np.isfinite(ma20v[d1]) else p1_exit
            p2_exit, p2_day = None, None
            for dd in range(d1 + 1, min(d1 + 1 + window, cap + 1)):
                if np.isfinite(vals[dd]) and vals[dd] >= target:
                    p2_exit, p2_day = vals[dd], dd
                    break
            if p2_exit is None:
                p2_day = min(d1 + window, cap)
                while p2_day > d1 and not np.isfinite(vals[p2_day]):
                    p2_day -= 1
                p2_exit = vals[p2_day]
            exit_price, exit_day = w1 * p1_exit + w2 * p2_exit, p2_day

    elif exit_rule == "exit_atr2stage":
        atrv = atr[sym].to_numpy(dtype=float)
        half_done, half_price = False, None
        for d in range(entry_day + 1, cap + 1):
            if not np.isfinite(vals[d]):
                continue
            peak = max(peak, vals[d])
            a = atrv[d] if np.isfinite(atrv[d]) else peak * 0.02
            stage1 = vals[d] <= peak - 1.5 * a
            stage2 = vals[d] <= peak - 2.5 * a or \
                (np.isfinite(ma200v[d]) and vals[d] < ma200v[d] * (1 - MA_BUFFER))
            if not half_done and stage1:
                half_done, half_price = True, vals[d]
            if half_done and stage2:
                exit_price, exit_day, stop_triggered = 0.5 * half_price + 0.5 * vals[d], d, True
                break
        if exit_price is None:
            if half_done:
                exit_price, exit_day, stop_triggered = 0.5 * half_price + 0.5 * vals[cap], cap, True
            else:
                exit_price, exit_day = vals[cap], cap

    else:  # exit_support2stage
        sup = support_level_asof(vals, entry_ref_day)
        half_done, half_price = False, None
        if sup is None:                       # 지지선 미발견 — 200일선 이탈만 백업으로 사용
            for d in range(entry_day + 1, cap + 1):
                if np.isfinite(vals[d]) and np.isfinite(ma200v[d]) and vals[d] < ma200v[d] * (1 - MA_BUFFER):
                    exit_price, exit_day, stop_triggered = vals[d], d, True
                    break
        else:
            stage1_lvl, stage2_lvl = sup * 0.98, sup * 0.96
            for d in range(entry_day + 1, cap + 1):
                if not np.isfinite(vals[d]):
                    continue
                if not half_done and vals[d] <= stage1_lvl:
                    half_done, half_price = True, vals[d]
                if half_done and vals[d] <= stage2_lvl:
                    exit_price, exit_day, stop_triggered = 0.5 * half_price + 0.5 * vals[d], d, True
                    break
            if exit_price is None and half_done:
                exit_price, exit_day, stop_triggered = 0.5 * half_price + 0.5 * vals[cap], cap, True
        if exit_price is None:
            exit_price, exit_day = vals[cap], cap

    span = vals[entry_day:exit_day + 1]
    span = span[np.isfinite(span)]
    mdd = float(span.min() / entry_price - 1) if len(span) else 0.0
    return {"entry_price": entry_price, "exit_price": exit_price, "entry_day": entry_day,
            "exit_day": exit_day, "filled_frac": filled_frac, "stop": stop_triggered, "mdd": mdd}


def run_exec(panel, spy, funds, pit, rebal_days=63, topn=15, cost=None,
             select_fn=None, out_suffix="", lookback=None):
    """select_fn(p)->basket 지정 시 자체 선정(미장 가중치) 대신 사용 — KR 모드가 라이브
    규칙 선정을 주입한다. out_suffix로 결과 파일 분리(예: "_kr")."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = _load_exec_weights()
    if select_fn is None:
        import tech_factors as T
        cross = T.build_panels(panel)
        select_fn = lambda p: _select_basket(panel, p, funds, cross, pit, weights, topn)
    ma20, ma50, ma200, atr = _ma(panel, 20), _ma(panel, 50), _ma(panel, 200), _atr_close(panel)
    spy = spy.reindex(panel.index).ffill() if spy is not None else None
    n = len(panel)
    ps = list(range(lookback or BW.LOOKBACK, n - MAX_HOLD - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")

    combos = [(e, x) for e in ENTRY_RULES for x in EXIT_RULES]
    per_combo = {c: {"excess": [], "dates": []} for c in combos}
    stats = {c: {"stop": [], "mdd": [], "unfilled": [], "net": []} for c in combos}
    turns, prev_basket = [], None

    for p in ps:
        basket = select_fn(p)
        if not basket:
            continue
        date = panel.index[p].date().isoformat()
        if prev_basket is not None:
            turns.append(1 - len(set(basket) & prev_basket) / max(len(basket), 1))
        prev_basket = set(basket)
        entry_day = p + 1
        for combo in combos:
            entry_rule, exit_rule = combo
            evs = []
            for sym in basket:
                r = _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, entry_rule, exit_rule)
                if r is None:
                    continue
                net = cost.net(r["exit_price"] / r["entry_price"] - 1)
                spy_ret = 0.0
                if spy is not None and np.isfinite(spy.iloc[r["exit_day"]]) and np.isfinite(spy.iloc[entry_day]):
                    spy_ret = float(spy.iloc[r["exit_day"]] / spy.iloc[entry_day] - 1)
                evs.append({"net": net, "excess": net - spy_ret, "stop": r["stop"],
                           "mdd": r["mdd"], "unfilled": 1 - r["filled_frac"]})
            if not evs:
                continue
            per_combo[combo]["excess"].append(round(float(np.mean([e["excess"] for e in evs])), 6))
            per_combo[combo]["dates"].append(date)
            stats[combo]["stop"].append(float(np.mean([e["stop"] for e in evs])))
            stats[combo]["mdd"].append(float(np.mean([e["mdd"] for e in evs])))
            stats[combo]["unfilled"].append(float(np.mean([e["unfilled"] for e in evs])))
            stats[combo]["net"].append(float(np.mean([e["net"] for e in evs])))

    n_ev = min((len(v["excess"]) for v in per_combo.values()), default=0)
    if n_ev < 4:
        raise RuntimeError(f"이벤트 수 부족(n_ev={n_ev}) — 기간을 늘리세요.")
    trials = [f"{e}__{x}" for e, x in combos]
    matrix = [per_combo[c]["excess"][:n_ev] for c in combos]
    dates0 = per_combo[combos[0]]["dates"][:n_ev]
    turnover = round(100 * float(np.mean(turns)), 1) if turns else None

    rows = [{"entry": e, "exit": x,
            "net_pct": round(100 * float(np.mean(stats[(e, x)]["net"])), 2),
            "stop_rate_pct": round(100 * float(np.mean(stats[(e, x)]["stop"])), 1),
            "mdd_pct": round(100 * float(np.mean(stats[(e, x)]["mdd"])), 1),
            "unfilled_pct": round(100 * float(np.mean(stats[(e, x)]["unfilled"])), 1),
            "n_events": len(stats[(e, x)]["net"])}
            for e, x in combos]

    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "turnover_pct": turnover,
              "n_combos": len(combos), "rows": rows,
              "baseline": BASELINE + " (현행 시스템과 동일 규칙)",
              "adoption_criteria": "청산 규칙은 baseline 대비 net 개선 & 손절빈도 감소가 "
                                   "T_eff 보정 후에도 유지될 때만 채택 제안(SCORE_MODEL_DESIGN.md 부록 A3)",
              "limitations": ["ATR은 종가 기반 근사(고가/저가 데이터 미사용)",
                              "지지선은 진입 시점 1회 계산 후 보유기간 동안 고정(매일 재계산 아님)",
                              "vol_poc_dist(A1)는 실제 거래량 히스토그램 최빈가 대신 "
                              "252일 거래량가중평균가(VWAP)로 근사"]}

    os.makedirs("output", exist_ok=True)
    compare_path = COMPARE_PATH.replace(".json", f"{out_suffix}.json")
    trial_path = TRIAL_PATH.replace(".json", f"{out_suffix}.json")
    report_path = REPORT_PATH.replace(".json", f"{out_suffix}.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    trial_data = {"horizon": "exec", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: {compare_path} · {trial_path} · {report_path} "
         f"(조합 {len(combos)}개 × 이벤트 {n_ev}회 · 회전율 {turnover}%)")
    return payload, report


def run_entry_ratio_sweep(panel, spy, funds, pit, rebal_days=63, topn=15, cost=None, lookback=None,
                          select_fn=None, out_suffix="", entries=None):
    """분할매수 '비율' 자체를 스윕(트리거 기준선 20일선-3%/50일선-8%는 라이브와 동일하게 고정,
    비율만 변수) — 청산은 채택된 exit_time6m 1종 고정(entry×exit 교차 아님, entry만 순수 비교).

    배경: run_exec의 21조합은 entry1_full(전량) vs entry2/3_pullback(분할)만 비교했다 —
    분할이 이긴다는 결론은 있지만, '왜 50/50과 30/30/40이라는 정확한 비율인가'는 검증된
    적이 없었다(다른 비율은 시도조차 안 함). 이 함수가 그 공백을 메운다.

    select_fn(p)->basket 지정 시 자체 선정(미장 가중치) 대신 사용 — run_exec()과 동일한 주입
    패턴(2026-07-16, KR 전용 검증용 kr_entry_exit_sweep.py가 valuediv 랭킹을 주입).
    entries 지정 시 ENTRY_RATIO_SWEEP 대신 사용 — 예: entry1_full을 포함시켜 '분할 vs 전량'
    자체를 이 함수 하나로 같이 비교(2026-07-16, KR은 21조합을 따로 안 돌려도 되게)."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = None
    if select_fn is None:
        weights = _load_exec_weights()
        import tech_factors as T
        cross = T.build_panels(panel)
        select_fn = lambda p: _select_basket(panel, p, funds, cross, pit, weights, topn)
    ma20, ma50, ma200, atr = _ma(panel, 20), _ma(panel, 50), _ma(panel, 200), _atr_close(panel)
    spy = spy.reindex(panel.index).ffill() if spy is not None else None
    n = len(panel)
    ps = list(range(lookback or BW.LOOKBACK, n - MAX_HOLD - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")

    exit_rule = "exit_time6m"
    entries = entries or ENTRY_RATIO_SWEEP
    per_combo = {e: {"excess": [], "dates": []} for e in entries}
    stats = {e: {"stop": [], "mdd": [], "unfilled": [], "net": []} for e in entries}

    for p in ps:
        basket = select_fn(p)
        if not basket:
            continue
        date = panel.index[p].date().isoformat()
        entry_day = p + 1
        for e in entries:
            evs = []
            for sym in basket:
                r = _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, e, exit_rule)
                if r is None:
                    continue
                net = cost.net(r["exit_price"] / r["entry_price"] - 1)
                spy_ret = 0.0
                if spy is not None and np.isfinite(spy.iloc[r["exit_day"]]) and np.isfinite(spy.iloc[entry_day]):
                    spy_ret = float(spy.iloc[r["exit_day"]] / spy.iloc[entry_day] - 1)
                evs.append({"net": net, "excess": net - spy_ret, "stop": r["stop"],
                           "mdd": r["mdd"], "unfilled": 1 - r["filled_frac"]})
            if not evs:
                continue
            per_combo[e]["excess"].append(round(float(np.mean([x["excess"] for x in evs])), 6))
            per_combo[e]["dates"].append(date)
            stats[e]["stop"].append(float(np.mean([x["stop"] for x in evs])))
            stats[e]["mdd"].append(float(np.mean([x["mdd"] for x in evs])))
            stats[e]["unfilled"].append(float(np.mean([x["unfilled"] for x in evs])))
            stats[e]["net"].append(float(np.mean([x["net"] for x in evs])))

    n_ev = min((len(v["excess"]) for v in per_combo.values()), default=0)
    if n_ev < 4:
        raise RuntimeError(f"이벤트 수 부족(n_ev={n_ev}) — 기간을 늘리세요.")
    trials = list(entries)
    matrix = [per_combo[e]["excess"][:n_ev] for e in entries]
    dates0 = per_combo[entries[0]]["dates"][:n_ev]

    rows = [{"entry": e, "exit": exit_rule,
            "net_pct": round(100 * float(np.mean(stats[e]["net"])), 2),
            "stop_rate_pct": round(100 * float(np.mean(stats[e]["stop"])), 1),
            "mdd_pct": round(100 * float(np.mean(stats[e]["mdd"])), 1),
            "unfilled_pct": round(100 * float(np.mean(stats[e]["unfilled"])), 1),
            "n_events": len(stats[e]["net"])}
            for e in entries]

    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "n_combos": len(entries), "rows": rows,
              "baseline": "entry2_5050(현행 평시 50/50) · entry3_303040(현행 과열 30/30/40)",
              "note": "청산 exit_time6m 고정 — 진입 '비율'만 순수 비교(진입×청산 교차 스윕 아님)",
              "adoption_criteria": "현행 비율(entry2_5050/entry3_303040) 대비 net 개선이 "
                                   "T_eff 보정 후에도 유지될 때만 비율 변경 제안"}

    os.makedirs("output", exist_ok=True)
    compare_path = f"output/backtest_entry_ratio_compare{out_suffix}.json"
    trial_path = f"output/trial_returns_entry_ratio{out_suffix}.json"
    report_path = f"output/pbo_report_entry_ratio{out_suffix}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    trial_data = {"horizon": "entry_ratio", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: {compare_path} · {trial_path} · {report_path} (진입비율 {len(entries)}종 × 이벤트 {n_ev}회)")
    return payload, report


def run_disposal_sweep(panel, spy, funds, pit, rebal_days=63, topn=15, cost=None, lookback=None,
                       select_fn=None, out_suffix=""):
    """매도 '처분' 방식 스윕 — 트리거(6개월 시점)는 고정, 그 이후 전량즉시 vs 분할+반등대기를
    비교한다. 진입은 entry1_full로 고정(처분 효과만 순수 비교, 진입 방식과 섞지 않음).

    select_fn(p)->basket 지정 시 자체 선정(미장 가중치) 대신 사용(2026-07-16, KR 전용
    검증용 — run_entry_ratio_sweep과 동일 주입 패턴)."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = None
    if select_fn is None:
        weights = _load_exec_weights()
        import tech_factors as T
        cross = T.build_panels(panel)
        select_fn = lambda p: _select_basket(panel, p, funds, cross, pit, weights, topn)
    ma20, ma50, ma200, atr = _ma(panel, 20), _ma(panel, 50), _ma(panel, 200), _atr_close(panel)
    spy = spy.reindex(panel.index).ffill() if spy is not None else None
    n = len(panel)
    ps = list(range(lookback or BW.LOOKBACK, n - MAX_HOLD - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")

    entry_rule = "entry1_full"
    exits = list(DISPOSAL_SWEEP)
    per_combo = {e: {"excess": [], "dates": []} for e in exits}
    stats = {e: {"stop": [], "mdd": [], "unfilled": [], "net": []} for e in exits}

    for p in ps:
        basket = select_fn(p)
        if not basket:
            continue
        date = panel.index[p].date().isoformat()
        entry_day = p + 1
        for e in exits:
            evs = []
            for sym in basket:
                r = _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, entry_rule, e)
                if r is None:
                    continue
                net = cost.net(r["exit_price"] / r["entry_price"] - 1)
                spy_ret = 0.0
                if spy is not None and np.isfinite(spy.iloc[r["exit_day"]]) and np.isfinite(spy.iloc[entry_day]):
                    spy_ret = float(spy.iloc[r["exit_day"]] / spy.iloc[entry_day] - 1)
                evs.append({"net": net, "excess": net - spy_ret, "stop": r["stop"],
                           "mdd": r["mdd"], "unfilled": 1 - r["filled_frac"]})
            if not evs:
                continue
            per_combo[e]["excess"].append(round(float(np.mean([x["excess"] for x in evs])), 6))
            per_combo[e]["dates"].append(date)
            stats[e]["stop"].append(float(np.mean([x["stop"] for x in evs])))
            stats[e]["mdd"].append(float(np.mean([x["mdd"] for x in evs])))
            stats[e]["unfilled"].append(float(np.mean([x["unfilled"] for x in evs])))
            stats[e]["net"].append(float(np.mean([x["net"] for x in evs])))

    n_ev = min((len(v["excess"]) for v in per_combo.values()), default=0)
    if n_ev < 4:
        raise RuntimeError(f"이벤트 수 부족(n_ev={n_ev}) — 기간을 늘리세요.")
    trials = list(exits)
    matrix = [per_combo[e]["excess"][:n_ev] for e in exits]
    dates0 = per_combo[exits[0]]["dates"][:n_ev]

    rows = [{"entry": entry_rule, "exit": e,
            "net_pct": round(100 * float(np.mean(stats[e]["net"])), 2),
            "stop_rate_pct": round(100 * float(np.mean(stats[e]["stop"])), 1),
            "mdd_pct": round(100 * float(np.mean(stats[e]["mdd"])), 1),
            "unfilled_pct": round(100 * float(np.mean(stats[e]["unfilled"])), 1),
            "n_events": len(stats[e]["net"])}
            for e in exits]

    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "n_combos": len(exits), "rows": rows,
              "baseline": "exit_time6m(트리거 즉시 전량) · exit_time6m_5050w10(현행 라이브 처분)",
              "note": "진입 entry1_full 고정 — 트리거(6개월) 이후 '처분 방식'만 순수 비교",
              "adoption_criteria": "현행 처분(exit_time6m_5050w10) 대비 net 개선이 "
                                   "T_eff 보정 후에도 유지될 때만 처분 방식 변경 제안"}

    os.makedirs("output", exist_ok=True)
    compare_path = f"output/backtest_disposal_compare{out_suffix}.json"
    trial_path = f"output/trial_returns_disposal{out_suffix}.json"
    report_path = f"output/pbo_report_disposal{out_suffix}.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    trial_data = {"horizon": "disposal", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: {compare_path} · {trial_path} · {report_path} (처분방식 {len(exits)}종 × 이벤트 {n_ev}회)")
    return payload, report


def run_topn_sweep(panel, spy, funds, pit, rebal_days=63, topn_list=None, cost=None, lookback=None):
    """보유종목 수(topn) 자체를 스윕 — 매 리밸런싱 시점의 팩터 상위 N을 바꿔가며 비교.
    진입 entry1_full·청산 exit_time6m 고정(topn 효과만 순수 비교, 어떻게 사고 파는지는 안 건드림)."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = _load_exec_weights()
    topn_list = topn_list or TOPN_SWEEP
    import tech_factors as T
    cross = T.build_panels(panel)
    ma20, ma50, ma200, atr = _ma(panel, 20), _ma(panel, 50), _ma(panel, 200), _atr_close(panel)
    spy = spy.reindex(panel.index).ffill() if spy is not None else None
    n = len(panel)
    ps = list(range(lookback or BW.LOOKBACK, n - MAX_HOLD - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")

    entry_rule, exit_rule = "entry1_full", "exit_time6m"
    per_combo = {tn: {"excess": [], "dates": []} for tn in topn_list}
    stats = {tn: {"stop": [], "mdd": [], "unfilled": [], "net": []} for tn in topn_list}
    turns = {tn: [] for tn in topn_list}
    prev_basket = {tn: None for tn in topn_list}

    for p in ps:
        date = panel.index[p].date().isoformat()
        entry_day = p + 1
        for tn in topn_list:
            basket = _select_basket(panel, p, funds, cross, pit, weights, tn)
            if not basket:
                continue
            if prev_basket[tn] is not None:
                turns[tn].append(1 - len(set(basket) & prev_basket[tn]) / max(len(basket), 1))
            prev_basket[tn] = set(basket)
            evs = []
            for sym in basket:
                r = _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, entry_rule, exit_rule)
                if r is None:
                    continue
                net = cost.net(r["exit_price"] / r["entry_price"] - 1)
                spy_ret = 0.0
                if spy is not None and np.isfinite(spy.iloc[r["exit_day"]]) and np.isfinite(spy.iloc[entry_day]):
                    spy_ret = float(spy.iloc[r["exit_day"]] / spy.iloc[entry_day] - 1)
                evs.append({"net": net, "excess": net - spy_ret, "stop": r["stop"],
                           "mdd": r["mdd"], "unfilled": 1 - r["filled_frac"]})
            if not evs:
                continue
            per_combo[tn]["excess"].append(round(float(np.mean([x["excess"] for x in evs])), 6))
            per_combo[tn]["dates"].append(date)
            stats[tn]["stop"].append(float(np.mean([x["stop"] for x in evs])))
            stats[tn]["mdd"].append(float(np.mean([x["mdd"] for x in evs])))
            stats[tn]["unfilled"].append(float(np.mean([x["unfilled"] for x in evs])))
            stats[tn]["net"].append(float(np.mean([x["net"] for x in evs])))

    n_ev = min((len(v["excess"]) for v in per_combo.values()), default=0)
    if n_ev < 4:
        raise RuntimeError(f"이벤트 수 부족(n_ev={n_ev}) — 기간을 늘리세요.")
    trials = [f"topn{tn}" for tn in topn_list]
    matrix = [per_combo[tn]["excess"][:n_ev] for tn in topn_list]
    dates0 = per_combo[topn_list[0]]["dates"][:n_ev]

    rows = [{"topn": tn,
            "net_pct": round(100 * float(np.mean(stats[tn]["net"])), 2),
            "stop_rate_pct": round(100 * float(np.mean(stats[tn]["stop"])), 1),
            "mdd_pct": round(100 * float(np.mean(stats[tn]["mdd"])), 1),
            "turnover_pct": round(100 * float(np.mean(turns[tn])), 1) if turns[tn] else None,
            "n_events": len(stats[tn]["net"])}
            for tn in topn_list]

    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "entry": entry_rule, "exit": exit_rule, "rebal_days": rebal_days,
              "n_combos": len(topn_list), "rows": rows,
              "baseline": "topn10(현행 미국 보유 상한)",
              "note": "진입 entry1_full·청산 exit_time6m 고정 — 보유종목 수(topn)만 순수 비교",
              "adoption_criteria": "현행(topn=10) 대비 net 개선 & MDD 축소가 T_eff 보정 후에도 "
                                   "유지될 때만 보유상한 변경 제안",
              "limitations": ["MDD·net은 트레이드별 평균(포트폴리오 전체를 하나의 자산으로 본 "
                              "일별 equity curve 낙폭이 아님) — 분산투자 효과(topn이 커질수록 "
                              "개별 종목 리스크가 상쇄되는 정도)는 이 지표로 완전히 포착 안 됨"]}

    os.makedirs("output", exist_ok=True)
    compare_path = "output/backtest_topn_compare.json"
    trial_path = "output/trial_returns_topn.json"
    report_path = "output/pbo_report_topn.json"
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    trial_data = {"horizon": "topn", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"저장: {compare_path} · {trial_path} · {report_path} (topn {len(topn_list)}종 × 이벤트 {n_ev}회)")
    return payload, report


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] ① 합성 데이터 규칙비교 엔진 ② 트레일링 스톱 발동 ③ 미체결 추적 ④ A1 신호 shape")
    panel, spy, funds, opens = BW._synthetic()
    pit = BC._synthetic_pit(panel)
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    payload, report = run_exec(panel, spy, funds, pit, rebal_days=63, topn=8, cost=cost)
    assert payload["rows"] and all(r["n_events"] > 0 for r in payload["rows"]), payload["rows"]
    assert "pbo" in report and "dsr" in report

    tn_payload, tn_report = run_topn_sweep(panel, spy, funds, pit, rebal_days=63,
                                           topn_list=[5, 8, 12], cost=cost)
    assert tn_payload["rows"] and all(r["n_events"] > 0 for r in tn_payload["rows"]), tn_payload["rows"]
    assert "pbo" in tn_report and "dsr" in tn_report

    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    crash = np.concatenate([100 * np.ones(200), 100 * (1 - 0.004) ** np.arange(200)])
    norebound = np.full(n, 100.0); norebound[337:] = 90.0            # 트리거 다음날 급락 후 미반등
    rebound = np.full(n, 100.0); rebound[337:341] = 90.0; rebound[341:] = 100.0  # 급락 후 곧 반등
    p2 = pd.DataFrame({"CRASH": crash, "FLAT": np.full(n, 100.0),
                       "NOREBOUND": norebound, "REBOUND": rebound}, index=idx)
    ma20, ma50, ma200, atr = _ma(p2, 20), _ma(p2, 50), _ma(p2, 200), _atr_close(p2)

    r = _simulate_trade(p2, ma20, ma50, ma200, atr, "CRASH", 210, "entry1_full", "exit_trail20")
    assert r["stop"] and r["exit_price"] < r["entry_price"], f"급락 경로인데 스톱 미발동: {r}"

    flat_r = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry2_pullback2", "exit_trail20")
    assert flat_r["filled_frac"] == 0.5, f"되돌림 없는데 100% 체결됨: {flat_r}"

    t6 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry1_full", "exit_time6m")
    assert t6["exit_day"] == 210 + 126 and not t6["stop"], f"고정 6개월 청산 오류: {t6}"
    p3 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry3_pullback3", "exit_trail20")
    assert abs(p3["filled_frac"] - 0.3) < 1e-9, f"횡보인데 2·3차 체결됨: {p3}"

    # 비율 스윕 엔트리(entry2_*/entry3_*) — 되돌림 없을 때 filled_frac이 정확히 1차 비율과 같아야
    r2_7030 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry2_7030", "exit_trail20")
    assert abs(r2_7030["filled_frac"] - 0.7) < 1e-9, f"entry2_7030 1차 비율 불일치: {r2_7030}"
    r3_502525 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry3_502525", "exit_trail20")
    assert abs(r3_502525["filled_frac"] - 0.5) < 1e-9, f"entry3_502525 1차 비율 불일치: {r3_502525}"
    # entry2_5050/entry3_303040은 기존 entry2_pullback2/entry3_pullback3과 동일 비율이어야 함
    r2_5050 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry2_5050", "exit_trail20")
    assert abs(r2_5050["filled_frac"] - flat_r["filled_frac"]) < 1e-9, (r2_5050, flat_r)
    r3_303040 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry3_303040", "exit_trail20")
    assert abs(r3_303040["filled_frac"] - p3["filled_frac"]) < 1e-9, (r3_303040, p3)
    try:
        _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry9_bogus", "exit_trail20")
        assert False, "알 수 없는 entry_rule인데 예외 미발생"
    except ValueError:
        pass

    # 처분(disposal) 스윕 — 트리거(entry_day+126=336) 이후 처분 방식 검증
    dcrash = _simulate_trade(p2, ma20, ma50, ma200, atr, "CRASH", 210, "entry1_full", "exit_time6m_5050w10")
    assert dcrash["stop"] and dcrash["exit_day"] == 336, f"손실 -15% 초과인데 즉시전량 미발동: {dcrash}"
    dnoreb = _simulate_trade(p2, ma20, ma50, ma200, atr, "NOREBOUND", 210, "entry1_full", "exit_time6m_5050w10")
    assert not dnoreb["stop"] and dnoreb["exit_day"] == 347, f"미반등인데 강제청산(10거래일) 시점 오류: {dnoreb}"
    assert abs(dnoreb["exit_price"] - 90.0) < 1e-6, f"미반등 처분가 계산 오류: {dnoreb}"
    dreb = _simulate_trade(p2, ma20, ma50, ma200, atr, "REBOUND", 210, "entry1_full", "exit_time6m_5050w10")
    assert dreb["exit_day"] < 347, f"반등했는데 강제청산 시점까지 대기: {dreb}"
    assert abs(dreb["exit_price"] - 95.0) < 1e-6, f"반등 처분가 블렌딩 오류: {dreb}"

    small = panel.iloc[:400, :6]
    sig = sr_signal_panels(small)
    assert set(sig) == set(SR_CANDIDATES), set(sig)
    for name, df in sig.items():
        assert df.shape == small.shape, f"{name} 패널 shape 불일치: {df.shape} vs {small.shape}"
    assert sig["hi52_prox"].to_numpy()[np.isfinite(sig["hi52_prox"].to_numpy())].max() <= 1e-9, \
        "hi52_prox는 정의상 0 이하여야 함(현재가 ≤ 52주 고점)"

    _log("[self-test] 통과: 규칙비교 엔진 · 트레일링 스톱 발동 · 미체결 추적 · A1 신호 shape/부호 OK")


def main():
    ap = argparse.ArgumentParser(description="트랙 C — 지지·저항 실행규칙 백테스트(A2-b)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--market", default="us", choices=["us", "kr", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--entry-ratio-sweep", action="store_true",
                    help="분할매수 '비율'만 스윕(청산 exit_time6m 고정) — 미국만 지원")
    ap.add_argument("--disposal-sweep", action="store_true",
                    help="매도 '처분 방식'만 스윕(6개월 트리거 고정, 진입 entry1_full) — 미국만 지원")
    ap.add_argument("--topn-sweep", action="store_true",
                    help="보유종목 수(topn)만 스윕(진입 entry1_full·청산 exit_time6m 고정) — 미국만 지원")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.entry_ratio_sweep:
        pit = BC.load_pit(args.pit_file)
        panel, spy, _ = BC.build_panel_pit(args.years, pit)
        funds = BW.load_funds()
        cost = BC.CostModel("us", args.commission_bps, args.slippage_bps)
        run_entry_ratio_sweep(panel, spy, funds, pit, rebal_days=args.rebal_days, topn=args.topn, cost=cost)
        return
    if args.disposal_sweep:
        pit = BC.load_pit(args.pit_file)
        panel, spy, _ = BC.build_panel_pit(args.years, pit)
        funds = BW.load_funds()
        cost = BC.CostModel("us", args.commission_bps, args.slippage_bps)
        run_disposal_sweep(panel, spy, funds, pit, rebal_days=args.rebal_days, topn=args.topn, cost=cost)
        return
    if args.topn_sweep:
        pit = BC.load_pit(args.pit_file)
        panel, spy, _ = BC.build_panel_pit(args.years, pit)
        funds = BW.load_funds()
        cost = BC.CostModel("us", args.commission_bps, args.slippage_bps)
        run_topn_sweep(panel, spy, funds, pit, rebal_days=args.rebal_days, cost=cost)
        return
    if args.market == "us":
        pit = BC.load_pit(args.pit_file)
        panel, spy, _ = BC.build_panel_pit(args.years, pit)
        funds = BW.load_funds()
        cost = BC.CostModel("us", args.commission_bps, args.slippage_bps)
        run_exec(panel, spy, funds, pit, rebal_days=args.rebal_days, topn=args.topn, cost=cost)
        return
    # KR 모드: 선정은 라이브 규칙(펀더멘탈 필터 + z(mom12_1)0.6+z(hi52)0.4) 주입 — 규칙만 비교
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = BK.prepare_kr_data(
        int(args.years), args.rebal_days)
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals, args.rebal_days)
    by_date = {}
    for s in snaps:
        pool = s["live_ok"][s["live_ok"]].index
        if len(pool) < 5:
            continue
        z = s["z"].loc[pool]
        score = z["mom12_1"] * 0.6 + z["hi52_prox"] * 0.4
        by_date[s["date"]] = list(score.sort_values(ascending=False).index[:args.topn])
    select_fn = lambda p: by_date.get(panel.index[p].date().isoformat(), [])
    cost = BC.CostModel("kospi", max(args.commission_bps, 1.5), args.slippage_bps)
    run_exec(panel, bench, None, None, rebal_days=args.rebal_days, topn=args.topn,
             cost=cost, select_fn=select_fn, out_suffix="_kr", lookback=BK.LOOKBACK)


if __name__ == "__main__":
    main()
