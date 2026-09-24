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
# 2026-09-24(전략 검토 E): 라이브 규칙 그대로의 진입·청산 — entry_live(entry_plan.tranche_targets —
# §17 이후 현재가 전량 1회) / exit_live(180달력일 경과 AND 후보풀 이탈 시 매도,
# pool_fn 필요). 기본 21조합에는 안 넣고 research/us/us_full_stack_exec_validation.py가 추가한다.
LIVE_ENTRY, LIVE_EXIT = "entry_live", "exit_live"
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
POOL_CHECK_DAYS = 5      # exit_live: 재평가 기간 경과 후 후보풀 재확인 간격(거래일, 라이브는 매 영업일)
ATR_WINDOW = 60
MAX_HOLD = 252           # 강제청산 상한(12m) — HISTORY.md 장기보유 취지상 이 이상은 안 봄
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
        return None                      # 데이터 없음 → 이벤트 제외(현금 이벤트와 구분)
    date = panel.index[p].date().isoformat()
    idx = raw.index.intersection(BC.membership_asof(pit, date))
    if len(idx) < topn:
        return None
    raw = raw.loc[idx]
    w = {k: v for k, v in weights.items() if k in raw.columns}
    if not w:
        return None
    z = raw[list(w)].apply(BW._z).fillna(0.0)
    score = (z * pd.Series(w)).sum(axis=1)
    return list(score.sort_values(ascending=False).index[:topn])


def _last_valid(vals, day, lo):
    """day 이하에서 가장 최근 유효가격의 인덱스 — 거래가 끝난 종목(인수·상장폐지)은 마지막
    거래가로 청산된 것으로 간주한다(인수 현금지급 근사. 파산은 과대평가될 수 있음)."""
    while day > lo and not np.isfinite(vals[day]):
        day -= 1
    return day


def _planned_fills(vals, ma20, ma50, ma200, sym, entry_day, entry_rule, n):
    """진입 규칙별 '체결 후보' [(비중, 체결가, 체결일)] — 날짜순 정리(청산 이후 체결 제거)는
    _simulate_trade가 청산일을 정한 뒤에 한다. 1차 트랜치는 항상 entry_day 종가."""
    p1 = vals[entry_day]

    def _wait_fill(target, window):
        for d in range(entry_day + 1, min(entry_day + 1 + window, n)):
            if np.isfinite(vals[d]) and vals[d] <= target:
                return vals[d], d
        return None, None

    def _col(df, t):
        v = df[sym].to_numpy(dtype=float)[t]
        return v if np.isfinite(v) else None

    if entry_rule == "entry1_full":
        return [(1.0, p1, entry_day)]
    if entry_rule == "entry_live":
        # 라이브 규칙 그대로(entry_plan.tranche_targets). 2026-09-24(HISTORY.md §17)부터 라이브가
        # 현재가 전량 1회라 entry1_full과 같다 — 그 전엔 과열 3분할·평시 2분할(대기 10/20거래일).
        import entry_plan as EP
        return [(pct / 100.0, p1, entry_day) for pct, _, _ in EP.tranche_targets(p1)]
    if entry_rule == "entry2_pullback2" or entry_rule in ENTRY_RATIO_2:
        w1, w2 = ENTRY_RATIO_2.get(entry_rule, (0.5, 0.5))
        m20 = _col(ma20, entry_day)
        base = m20 * 0.97 if m20 is not None else p1 * 0.97
        f2, d2 = _wait_fill(min(base, p1), PULLBACK_WINDOW)
        return [(w1, p1, entry_day)] + ([(w2, f2, d2)] if f2 is not None else [])
    if entry_rule == "entry3_pullback3" or entry_rule in ENTRY_RATIO_3:
        w1, w2, w3 = ENTRY_RATIO_3.get(entry_rule, (0.3, 0.3, 0.4))
        m20, m50 = _col(ma20, entry_day), _col(ma50, entry_day)
        t2 = min(m20 * 0.97 if m20 is not None else p1 * 0.97, p1)
        t3 = min(m50 * 0.92 if m50 is not None else p1 * 0.92, p1)
        f2, d2 = _wait_fill(t2, PULLBACK_WINDOW)
        f3, d3 = _wait_fill(t3, PULLBACK_WINDOW * 2)
        fills = [(w1, p1, entry_day)]
        if f2 is not None:
            fills.append((w2, f2, d2))
        if f3 is not None:
            fills.append((w3, f3, d3))
        return fills
    raise ValueError(f"알 수 없는 entry_rule: {entry_rule}")


def _avg_entry(fills):
    frac = sum(w for w, _, _ in fills)
    return sum(w * px for w, px, _ in fills) / frac, frac


def _simulate_trade(panel, ma20, ma50, ma200, atr, sym, entry_day, entry_rule, exit_rule,
                    pool_fn=None):
    """단일 종목·단일 이벤트의 진입~청산 시뮬레이션(일별 경로). 반환: dict 또는 None(가격 결측).

    2026-09-24 전략 검토 A·B 반영:
      · 날짜순 처리 — 추가 매수 트랜치는 청산일 '이전'에 체결된 것만 포지션에 넣는다(예전엔
        미래 구간의 추가 매수를 전부 먼저 계산해 청산 이후의 저가 매수가 평균단가에 섞였다).
        트레일링 고점은 1차 체결가에서 시작하고(평균단가에 미래 체결이 섞이지 않게), 지지선은
        진입 결정 시점(entry_day)에 계산한다.
      · 배정자금 기준 수익 — alloc_ret = filled_frac × 순수익(미체결분은 현금, 수익 0 가정).
        예전엔 체결분 수익률만 써서 부분체결 전략과 전량매수 전략의 자본 기준이 달랐다.
      · path = 배정자금 1의 일별 가치(현금+보유분, 비용 제외) — 이벤트 바스켓 MDD 계산용.
        예전 'mdd'(최저가/평균단가-1)는 고점대비 낙폭이 아니었다."""
    vals = panel[sym].to_numpy(dtype=float)
    n = len(vals)
    cap = min(entry_day + MAX_HOLD, n - 1)
    if entry_day >= n or not np.isfinite(vals[entry_day]):
        return None
    p1 = vals[entry_day]
    fills_all = _planned_fills(vals, ma20, ma50, ma200, sym, entry_day, entry_rule, n)

    # ---- 청산(청산 판정은 평균단가가 아니라 가격경로·1차 체결가만 쓴다 → 미래 체결과 무관) ----
    peak = p1
    exit_price, exit_day, stop_triggered = None, cap, False
    ma200v = ma200[sym].to_numpy(dtype=float)

    def _at_cap():
        d = _last_valid(vals, cap, entry_day)
        return vals[d], d

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
            exit_price, exit_day = _at_cap()

    elif exit_rule == "exit_ma200only":           # 200일선 이탈만(트레일링 없음)
        for d in range(entry_day + 1, cap + 1):
            if np.isfinite(vals[d]) and np.isfinite(ma200v[d]) and vals[d] < ma200v[d] * (1 - MA_BUFFER):
                exit_price, exit_day, stop_triggered = vals[d], d, True
                break
        if exit_price is None:
            exit_price, exit_day = _at_cap()

    elif exit_rule == "exit_time6m":              # 고정 6개월 — 검증된 백테스트의 원형(대조군)
        exit_day = _last_valid(vals, min(entry_day + 126, cap), entry_day)
        exit_price = vals[exit_day]

    elif exit_rule == "exit_live":
        # 라이브 매도 규칙 그대로(holdings.update): 보유 REEVAL_DAYS(달력일) 경과 후 그날의
        # 팩터 후보풀(pool_fn) 밖이면 매도. 라이브는 매 영업일 확인하지만 후보풀 재계산 비용
        # 때문에 POOL_CHECK_DAYS 거래일 간격으로 확인(근사). 후보풀에 계속 남으면 MAX_HOLD
        # 에서 평가 종료(시가평가 — 라이브엔 상한이 없으므로 매도 신호가 아님).
        import holdings as H
        if pool_fn is None:
            raise ValueError("exit_live에는 pool_fn(day)->후보풀 set 이 필요")
        t0 = panel.index[entry_day]
        d = entry_day + 1
        while d <= cap and (panel.index[d] - t0).days < H.REEVAL_DAYS:
            d += 1
        while d <= cap:
            pool = pool_fn(d)
            if pool is not None and np.isfinite(vals[d]) and sym not in pool:
                exit_price, exit_day, stop_triggered = vals[d], d, False
                break
            d += POOL_CHECK_DAYS
        if exit_price is None:
            exit_price, exit_day = _at_cap()

    elif exit_rule in DISPOSAL_SWEEP and DISPOSAL_SWEEP[exit_rule] is not None:
        # 6개월 트리거는 exit_time6m과 동일하게 확정하되, 그 이후 '처분'만 라이브 sell_plan()
        # 방식(분할+반등대기, 손실 -15% 초과 시 즉시 전량)으로 시뮬레이션.
        (w1, w2), window = DISPOSAL_SWEEP[exit_rule]
        trig_day = _last_valid(vals, min(entry_day + 126, cap), entry_day)
        entry_at_trig, _ = _avg_entry([f for f in fills_all if f[2] <= trig_day])
        trig_ret = vals[trig_day] / entry_at_trig - 1
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
            last_px, exit_day = _at_cap()
            if half_done:
                exit_price, stop_triggered = 0.5 * half_price + 0.5 * last_px, True
            else:
                exit_price = last_px

    elif exit_rule == "exit_support2stage":
        sup = support_level_asof(vals, entry_day)
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
                last_px, exit_day = _at_cap()
                exit_price, stop_triggered = 0.5 * half_price + 0.5 * last_px, True
        if exit_price is None:
            exit_price, exit_day = _at_cap()
    else:
        raise ValueError(f"알 수 없는 exit_rule: {exit_rule}")

    # ---- 날짜순 정리: 청산일 이후(당일 포함)의 추가 매수는 체결되지 않은 것으로 본다 ----
    fills = [f for f in fills_all if f[2] == entry_day or f[2] < exit_day]
    entry_price, filled_frac = _avg_entry(fills)

    # ---- 배정자금 1의 일별 가치(현금 + 보유분) — 청산 후엔 현금으로 고정 ----
    seg = pd.Series(vals[entry_day:cap + 1]).ffill().to_numpy()
    path = np.empty(len(seg))
    cash, shares = 1.0, 0.0
    by_day = {}
    for w, px, d in fills:
        by_day.setdefault(d, []).append((w, px))
    for i, d in enumerate(range(entry_day, cap + 1)):
        for w, px in by_day.get(d, []):
            cash -= w
            shares += w / px
        if d >= exit_day:
            path[i] = cash + shares * exit_price
        else:
            path[i] = cash + shares * seg[i]
    return {"entry_price": entry_price, "exit_price": exit_price, "entry_day": entry_day,
            "exit_day": exit_day, "filled_frac": filled_frac, "stop": stop_triggered,
            "path": path}


def _eval_event(panel, ind, basket, entry_day, entry_rule, exit_rule, cost, spy, slots, pool_fn=None):
    """리밸런싱 이벤트 1회 = 배정자금 1을 slots(목표 종목수)개 슬롯에 균등 배분해 평가.

    2026-09-24 전략 검토 A·B·E 반영: 후보가 목표보다 적거나(필터 탈락) 0개여도 이벤트를
    버리지 않고 빈 슬롯은 현금(수익 0)으로 둔다 — 예전엔 후보 부족 시점을 통째로 삭제해 그
    기간의 현금·손익이 결과에서 사라졌다. 초과수익은 같은 자금을 SPY(벤치마크)에 전액 투자한
    경우와 비교(종목 슬롯은 그 종목 보유기간, 빈 슬롯은 표준 6개월=126거래일 동안).
    basket_mdd = 슬롯 평균 가치경로의 고점대비 최대낙폭(이 이벤트 바스켓 기준 — 겹치는 여러
    이벤트를 합친 계좌 MDD는 아님, 계좌 NAV MDD는 research/us/backtest_portfolio.py)."""
    ma20, ma50, ma200, atr = ind
    n = len(panel)

    def _bench(d0, d1):
        if spy is None or not (np.isfinite(spy.iloc[d1]) and np.isfinite(spy.iloc[d0])):
            return 0.0
        return float(spy.iloc[d1] / spy.iloc[d0] - 1)

    slots = max(slots, len(basket), 1)
    trades = [_simulate_trade(panel, ma20, ma50, ma200, atr, s, entry_day, entry_rule, exit_rule,
                              pool_fn=pool_fn) for s in basket]
    trades = [t for t in trades if t is not None]
    n_cash = slots - len(trades)
    cash_bench = _bench(entry_day, min(entry_day + 126, n - 1)) if n_cash else 0.0
    nets = [t["filled_frac"] * cost.net(t["exit_price"] / t["entry_price"] - 1) for t in trades]
    excess = [nt - _bench(entry_day, t["exit_day"]) for nt, t in zip(nets, trades)]
    horizon = MAX_HOLD + 1
    paths = [np.pad(t["path"], (0, horizon - len(t["path"])), mode="edge")[:horizon] for t in trades]
    paths += [np.ones(horizon)] * n_cash
    basket_path = np.mean(paths, axis=0)
    mdd = float((basket_path / np.maximum.accumulate(basket_path) - 1).min())
    return {"net": (sum(nets)) / slots,
            "excess": (sum(excess) - n_cash * cash_bench) / slots,
            "stop": (sum(t["stop"] for t in trades)) / slots,
            "unfilled": (sum(1 - t["filled_frac"] for t in trades) + n_cash) / slots,
            "basket_mdd": mdd}


def _run_grid(panel, spy, ps, select_fn, trials, cost, slots, pool_fn=None):
    """trials=[(entry, exit, key)] 전부를 같은 이벤트들에서 평가. select_fn(p)가 None이면
    (데이터 없음) 그 이벤트는 건너뛰고, 빈 리스트면(후보 0) 전액 현금 이벤트로 평가한다.
    slots는 정수 또는 key->정수 함수(topn 스윕)."""
    ind = (_ma(panel, 20), _ma(panel, 50), _ma(panel, 200), _atr_close(panel))
    spy = spy.reindex(panel.index).ffill() if spy is not None else None
    per = {k: {"excess": [], "dates": []} for _, _, k in trials}
    stats = {k: {"stop": [], "mdd": [], "unfilled": [], "net": []} for _, _, k in trials}
    for p in ps:
        date = panel.index[p].date().isoformat()
        for e, x, k in trials:
            basket = select_fn(p, k)
            if basket is None:
                continue
            ns = slots(k) if callable(slots) else slots
            ev = _eval_event(panel, ind, basket, p + 1, e, x, cost, spy, ns, pool_fn=pool_fn)
            per[k]["excess"].append(round(float(ev["excess"]), 6))
            per[k]["dates"].append(date)
            for f, src in (("stop", "stop"), ("mdd", "basket_mdd"), ("unfilled", "unfilled"), ("net", "net")):
                stats[k][f].append(float(ev[src]))
    return per, stats


def _row(st):
    return {"net_pct": round(100 * float(np.mean(st["net"])), 2),
            "stop_rate_pct": round(100 * float(np.mean(st["stop"])), 1),
            "basket_mdd_pct": round(100 * float(np.mean(st["mdd"])), 1),
            "unfilled_pct": round(100 * float(np.mean(st["unfilled"])), 1),
            "n_events": len(st["net"])}


ACCOUNTING_NOTE = ("2026-09-24 이후 산식: net·excess는 배정자금 기준(미체결·빈 슬롯은 현금 수익 0), "
                   "추가매수는 청산 이전 체결분만 반영, basket_mdd_pct는 이벤트 바스켓 가치경로의 "
                   "고점대비 최대낙폭 평균(계좌 NAV MDD 아님 — research/us/backtest_portfolio.py 참고)")


def _save(payload, trial_data, compare_path, trial_path, report_path):
    os.makedirs("output", exist_ok=True)
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(trial_path, "w", encoding="utf-8") as f:
        json.dump(trial_data, f, ensure_ascii=False)
    report = OS.analyze(trial_data, save=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _matrix(per, keys):
    n_ev = min((len(per[k]["excess"]) for k in keys), default=0)
    if n_ev < 4:
        raise RuntimeError(f"이벤트 수 부족(n_ev={n_ev}) — 기간을 늘리세요.")
    return n_ev, [per[k]["excess"][:n_ev] for k in keys], per[keys[0]]["dates"][:n_ev]


def _default_select(panel, funds, pit, topn):
    weights = _load_exec_weights()
    import tech_factors as T
    cross = T.build_panels(panel)
    return weights, (lambda p: _select_basket(panel, p, funds, cross, pit, weights, topn))


def _rebal_points(panel, rebal_days, lookback):
    ps = list(range(lookback or BW.LOOKBACK, len(panel) - MAX_HOLD - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")
    return ps


def run_exec(panel, spy, funds, pit, rebal_days=63, topn=15, cost=None,
             select_fn=None, out_suffix="", lookback=None, entries=None, exits=None, pool_fn=None):
    """select_fn(p)->basket 지정 시 자체 선정(미장 가중치) 대신 사용 — KR 모드가 라이브
    규칙 선정을 주입한다. out_suffix로 결과 파일 분리(예: "_kr"). entries/exits로 규칙
    목록을 바꿀 수 있고(예: entry_live·exit_live 추가), exit_live는 pool_fn(day)->set 필요."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = _load_exec_weights()
    if select_fn is None:
        weights, select_fn = _default_select(panel, funds, pit, topn)
    ps = _rebal_points(panel, rebal_days, lookback)
    combos = [(e, x) for e in (entries or ENTRY_RULES) for x in (exits or EXIT_RULES)]
    trials = [f"{e}__{x}" for e, x in combos]
    cache = {}

    def sel(p, _k):
        if p not in cache:
            cache[p] = select_fn(p)
        return cache[p]

    per, stats = _run_grid(panel, spy, ps, sel, [(e, x, f"{e}__{x}") for e, x in combos],
                           cost, topn, pool_fn=pool_fn)
    n_ev, matrix, dates0 = _matrix(per, trials)
    baskets = [cache[p] for p in ps if cache.get(p) is not None]
    turns = [1 - len(set(b) & set(a)) / max(len(b), 1) for a, b in zip(baskets, baskets[1:]) if b]
    turnover = round(100 * float(np.mean(turns)), 1) if turns else None

    rows = [{"entry": e, "exit": x, **_row(stats[f"{e}__{x}"])} for e, x in combos]
    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "turnover_pct": turnover,
              "n_combos": len(combos), "rows": rows,
              "baseline": BASELINE + " (옛 트레일링 규칙 — 라이브 규칙은 entry_live__exit_live)",
              "accounting": ACCOUNTING_NOTE,
              "adoption_criteria": "청산 규칙은 baseline 대비 net 개선 & 손절빈도 감소가 "
                                   "T_eff 보정 후에도 유지될 때만 채택 제안(SCORE_MODEL_DESIGN.md 부록 A3)",
              "limitations": ["ATR은 종가 기반 근사(고가/저가 데이터 미사용)",
                              "지지선은 진입 시점 1회 계산 후 보유기간 동안 고정(매일 재계산 아님)",
                              "vol_poc_dist(A1)는 실제 거래량 히스토그램 최빈가 대신 "
                              "252일 거래량가중평균가(VWAP)로 근사",
                              "이벤트별 독립 평가 — 보유상한·'팔아야 산다'·이벤트 간 자금 재투자는 "
                              "반영 안 됨(계좌 단위 검증은 backtest_portfolio.py)"]}
    trial_data = {"horizon": "exec", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": trials, "excess_returns": matrix}
    compare_path = COMPARE_PATH.replace(".json", f"{out_suffix}.json")
    trial_path = TRIAL_PATH.replace(".json", f"{out_suffix}.json")
    report_path = REPORT_PATH.replace(".json", f"{out_suffix}.json")
    report = _save(payload, trial_data, compare_path, trial_path, report_path)
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
        weights, select_fn = _default_select(panel, funds, pit, topn)
    ps = _rebal_points(panel, rebal_days, lookback)
    exit_rule = "exit_time6m"
    entries = list(entries or ENTRY_RATIO_SWEEP)
    cache = {}
    sel = lambda p, _k: cache[p] if p in cache else cache.setdefault(p, select_fn(p))
    per, stats = _run_grid(panel, spy, ps, sel, [(e, exit_rule, e) for e in entries], cost, topn)
    n_ev, matrix, dates0 = _matrix(per, entries)
    rows = [{"entry": e, "exit": exit_rule, **_row(stats[e])} for e in entries]
    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "n_combos": len(entries), "rows": rows,
              "baseline": "entry2_5050(현행 평시 50/50) · entry3_303040(현행 과열 30/30/40)",
              "accounting": ACCOUNTING_NOTE,
              "note": "청산 exit_time6m 고정 — 진입 '비율'만 순수 비교(진입×청산 교차 스윕 아님)",
              "adoption_criteria": "현행 비율(entry2_5050/entry3_303040) 대비 net 개선이 "
                                   "T_eff 보정 후에도 유지될 때만 비율 변경 제안"}
    trial_data = {"horizon": "entry_ratio", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": list(entries), "excess_returns": matrix}
    paths = (f"output/backtest_entry_ratio_compare{out_suffix}.json",
             f"output/trial_returns_entry_ratio{out_suffix}.json",
             f"output/pbo_report_entry_ratio{out_suffix}.json")
    report = _save(payload, trial_data, *paths)
    _log(f"저장: {' · '.join(paths)} (진입비율 {len(entries)}종 × 이벤트 {n_ev}회)")
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
        weights, select_fn = _default_select(panel, funds, pit, topn)
    ps = _rebal_points(panel, rebal_days, lookback)
    entry_rule = "entry1_full"
    exits = list(DISPOSAL_SWEEP)
    cache = {}
    sel = lambda p, _k: cache[p] if p in cache else cache.setdefault(p, select_fn(p))
    per, stats = _run_grid(panel, spy, ps, sel, [(entry_rule, x, x) for x in exits], cost, topn)
    n_ev, matrix, dates0 = _matrix(per, exits)
    rows = [{"entry": entry_rule, "exit": x, **_row(stats[x])} for x in exits]
    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "topn": topn, "rebal_days": rebal_days, "n_combos": len(exits), "rows": rows,
              "baseline": "exit_time6m(트리거 즉시 전량) · exit_time6m_5050w10(옛 라이브 처분)",
              "accounting": ACCOUNTING_NOTE,
              "note": "진입 entry1_full 고정 — 트리거(6개월) 이후 '처분 방식'만 순수 비교",
              "adoption_criteria": "현행 처분(exit_time6m_5050w10) 대비 net 개선이 "
                                   "T_eff 보정 후에도 유지될 때만 처분 방식 변경 제안"}
    trial_data = {"horizon": "disposal", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": exits, "excess_returns": matrix}
    paths = (f"output/backtest_disposal_compare{out_suffix}.json",
             f"output/trial_returns_disposal{out_suffix}.json",
             f"output/pbo_report_disposal{out_suffix}.json")
    report = _save(payload, trial_data, *paths)
    _log(f"저장: {' · '.join(paths)} (처분방식 {len(exits)}종 × 이벤트 {n_ev}회)")
    return payload, report


def run_topn_sweep(panel, spy, funds, pit, rebal_days=63, topn_list=None, cost=None, lookback=None):
    """보유종목 수(topn) 자체를 스윕 — 매 리밸런싱 시점의 팩터 상위 N을 바꿔가며 비교.
    진입 entry1_full·청산 exit_time6m 고정(topn 효과만 순수 비교, 어떻게 사고 파는지는 안 건드림)."""
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    weights = _load_exec_weights()
    topn_list = topn_list or TOPN_SWEEP
    import tech_factors as T
    cross = T.build_panels(panel)
    ps = _rebal_points(panel, rebal_days, lookback)
    entry_rule, exit_rule = "entry1_full", "exit_time6m"
    keys = [f"topn{tn}" for tn in topn_list]
    cache = {}

    def sel(p, k):
        if (p, k) not in cache:
            cache[(p, k)] = _select_basket(panel, p, funds, cross, pit, weights, int(k[4:]))
        return cache[(p, k)]

    per, stats = _run_grid(panel, spy, ps, sel, [(entry_rule, exit_rule, k) for k in keys], cost,
                           slots=lambda k: int(k[4:]))
    n_ev, matrix, dates0 = _matrix(per, keys)
    rows = []
    for tn, k in zip(topn_list, keys):
        bs = [cache[(p, k)] for p in ps if cache.get((p, k)) is not None]
        turns = [1 - len(set(b) & set(a)) / max(len(b), 1) for a, b in zip(bs, bs[1:]) if b]
        rows.append({"topn": tn, **_row(stats[k]),
                     "turnover_pct": round(100 * float(np.mean(turns)), 1) if turns else None})
    payload = {"as_of": panel.index[-1].date().isoformat(), "weights_used": weights,
              "entry": entry_rule, "exit": exit_rule, "rebal_days": rebal_days,
              "n_combos": len(topn_list), "rows": rows,
              "baseline": "topn10(현행 미국 보유 상한)",
              "accounting": ACCOUNTING_NOTE,
              "note": "진입 entry1_full·청산 exit_time6m 고정 — 보유종목 수(topn)만 순수 비교",
              "adoption_criteria": "현행(topn=10) 대비 net 개선 & MDD 축소가 T_eff 보정 후에도 "
                                   "유지될 때만 보유상한 변경 제안"}
    trial_data = {"horizon": "topn", "universe": "pit", "cost": cost.describe(),
                 "rebal_days": rebal_days, "hold_days": MAX_HOLD,
                 "dates": dates0, "trials": keys, "excess_returns": matrix}
    paths = ("output/backtest_topn_compare.json", "output/trial_returns_topn.json",
             "output/pbo_report_topn.json")
    report = _save(payload, trial_data, *paths)
    _log(f"저장: {' · '.join(paths)} (topn {len(topn_list)}종 × 이벤트 {n_ev}회)")
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

    # 2026-09-24 전략 검토 재현 사례 — ①부분체결 자본기준 ②날짜순 체결 ③바스켓 MDD
    m = 300
    ix = pd.bdate_range("2020-01-01", periods=m)
    up = np.full(m, 100.0); up[211:] = 110.0                         # 100→110, 20일선-3% 눌림 없음
    dip = np.full(m, 100.0); dip[:150] = 60.0; dip[211] = 96.0; dip[212:] = 50.0   # 100→96→50(200일선은 아래)
    hump = np.full(m, 100.0); hump[211:250] = 200.0; hump[250:] = 150.0  # 100→200→150
    p3 = pd.DataFrame({"UP": up, "DIP": dip, "HUMP": hump}, index=ix)
    ind3 = (_ma(p3, 20), _ma(p3, 50), _ma(p3, 200), _atr_close(p3))
    ev = _eval_event(p3, ind3, ["UP"], 210, "entry2_pullback2", "exit_time6m", cost, None, 1)
    assert abs(ev["net"] - 0.5 * cost.net(0.10)) < 1e-9 and abs(ev["net"] - 0.04943) < 5e-5, \
        f"부분체결인데 배정자금 기준 수익이 아님: {ev}"
    tr = _simulate_trade(p3, *ind3[:3], ind3[3], "DIP", 210, "entry3_pullback3", "exit_trail20")
    assert tr["exit_day"] == 212 and abs(tr["filled_frac"] - 0.6) < 1e-9 and \
        abs(tr["entry_price"] - 98.0) < 1e-9, f"청산 이후 체결이 평균단가에 섞임: {tr}"
    ev = _eval_event(p3, ind3, ["HUMP"], 210, "entry1_full", "exit_time6m", cost, None, 1)
    assert abs(ev["basket_mdd"] + 0.25) < 1e-9, f"고점대비 낙폭이 아님: {ev}"
    ev = _eval_event(p3, ind3, [], 210, "entry1_full", "exit_time6m", cost, None, 10)
    assert ev["net"] == 0.0 and ev["unfilled"] == 1.0, f"후보 0개 이벤트는 전액 현금이어야: {ev}"
    # 라이브 규칙(§17): 현재가 전량 1회 / exit_live는 180일 경과 후 후보풀 이탈 시
    lv = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry_live", "exit_live",
                         pool_fn=lambda d: set())
    assert abs(lv["filled_frac"] - 1.0) < 1e-9, lv
    held = (p2.index[lv["exit_day"]] - p2.index[210]).days
    assert 180 <= held < 190 and not lv["stop"], f"exit_live 재평가 시점 오류: {held}일 {lv}"
    lv2 = _simulate_trade(p2, ma20, ma50, ma200, atr, "FLAT", 210, "entry_live", "exit_live",
                          pool_fn=lambda d: {"FLAT"})
    assert lv2["exit_day"] == min(210 + MAX_HOLD, n - 1), f"후보풀 잔류인데 청산: {lv2}"

    small = panel.iloc[:400, :6]
    sig = sr_signal_panels(small)
    assert set(sig) == set(SR_CANDIDATES), set(sig)
    for name, df in sig.items():
        assert df.shape == small.shape, f"{name} 패널 shape 불일치: {df.shape} vs {small.shape}"
    assert sig["hi52_prox"].to_numpy()[np.isfinite(sig["hi52_prox"].to_numpy())].max() <= 1e-9, \
        "hi52_prox는 정의상 0 이하여야 함(현재가 ≤ 52주 고점)"

    _log("[self-test] 통과: 규칙비교 엔진 · 트레일링 스톱 발동 · 미체결 추적 · 배정자금/날짜순/MDD · "
         "라이브 규칙 · A1 신호 shape/부호 OK")


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
    # KR 모드(레거시): 2026-07-14 이전 한국 규칙(펀더멘탈 필터 + z(mom12_1)0.6+z(hi52)0.4) 주입 —
    # 현행 valuediv 선정의 집행 검증은 research/kr/kr_entry_exit_sweep.py
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = BK.prepare_kr_data(
        int(args.years), args.rebal_days)
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals, args.rebal_days)
    by_date = {}   # 스냅샷 없는 날짜는 None(데이터 없음 → 이벤트 제외)
    for s in snaps:
        pool = s["live_ok"][s["live_ok"]].index
        if len(pool) < 5:
            continue
        z = s["z"].loc[pool]
        score = z["mom12_1"] * 0.6 + z["hi52_prox"] * 0.4
        by_date[s["date"]] = list(score.sort_values(ascending=False).index[:args.topn])
    select_fn = lambda p: by_date.get(panel.index[p].date().isoformat())
    cost = BC.CostModel("kospi", max(args.commission_bps, 1.5), args.slippage_bps)
    run_exec(panel, bench, None, None, rebal_days=args.rebal_days, topn=args.topn,
             cost=cost, select_fn=select_fn, out_suffix="_kr", lookback=BK.LOOKBACK)


if __name__ == "__main__":
    main()
