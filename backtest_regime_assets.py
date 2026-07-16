#!/usr/bin/env python3
"""
backtest_regime_assets.py — 금(GLD)·비트코인(BTC-USD) 레짐타이밍 파라미터 전용 검증 (2026-07-16).

배경(지호 님 지시): "금이랑 비트코인 매도매수 추세 알고리즘 정교화하고 백테스트로 검증" —
Fable 5(Claude chat) 자문을 받아 설계, 이 스크립트가 구현+검증한다.

market_signals.py의 6자산 레짐엔진(§STRATEGY.md §1)은 금(GLD)을 '주식'류로 분류해 지수와
똑같은 파라미터(200일선·±1%·확인3일·12-1모멘텀)를 그냥 물려받았고, 비트코인(120일선·±3%·
3개월모멘텀)은 일반 학술 문헌 근거일 뿐 이 프로젝트 데이터로 백테스트된 적이 없었다 —
이 엔진 자체를 검증하는 백테스트가 지금까지 아예 없었다.

Fable 5 설계 요지(자문 전문은 세션 기록 참고):
  · 이진 노출(ON=1/OFF=0)만 스윕한다 — 눌림선·변동성타깃은 노출을 안 바꾸는 표시용이라
    스윕 대상에서 제외(§Stage1). 모멘텀은 레짐을 고정한 뒤 2단계로 조건부 스윕(§Stage2).
  · 목적이 "수익 극대화가 아니라 낙폭 방어"이므로 주 지표는 CAGR이 아니라 Ulcer Index
    (RMS 낙폭) 감소율 — 단 CAGR 손실이 예산(매수후보유 CAGR의 25% 또는 1.5%p 중 큰 쪽)을
    넘으면 그 조합은 자동 탈락(-inf).
  · 채택 기준: ①이웃 조합도 대부분 이김(고원, 봉우리 아님) ②Ulcer 15%p 이상 개선+MDD
    악화 없음 ③PBO/DSR 게이트 통과 ④쌍대 블록부트스트랩 90%CI가 0을 안 포함.
    아니면 현행 유지(이 프로젝트의 "동점이면 단순한 쪽" 원칙과 정합).
  · 비트코인은 2014-09~ 데이터만(그 이전 Mt.Gox 시대 데이터 신뢰 안 함), ETH-USD로
    방향성 확인(전용 튜닝 없이 그대로 적용), 2014-19/2020-이후 반기 분할 검증도 함께.
  · 금은 GLD(2004-11~) 실거래 표본만 사용(LBMA 스플라이스 장기표본은 무료 데이터 접근
    제약으로 이번엔 생략 — 한계로 명시).

실행: python backtest_regime_assets.py --stage1        # 추세선·밴드·확인일수 그리드
      python backtest_regime_assets.py --stage2        # 모멘텀 그리드(1단계 동결 후)
      python backtest_regime_assets.py --all            # 둘 다 + 부트스트랩 + ETH 확인
      python backtest_regime_assets.py --self-test
결과: output/regime_backtest_{gold,btc}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import overfit_stats as OS

TRADING_DAYS = 252

# ------------------------- 사전등록 그리드(Fable 5 설계) -------------------------
GOLD_CURRENT = {"trend_ma": 200, "band": 0.01, "confirm": 3}
BTC_CURRENT = {"trend_ma": 120, "band": 0.03, "confirm": 3}
GOLD_MOM_CURRENT, BTC_MOM_CURRENT = "12_1", "3m"

GOLD_GRID = {"trend_ma": [100, 150, 200, 250, 300], "band": [0.0, 0.005, 0.01, 0.02], "confirm": [1, 3, 5]}
BTC_GRID = {"trend_ma": [50, 80, 120, 150, 200], "band": [0.01, 0.03, 0.05, 0.08], "confirm": [1, 3, 5]}
GOLD_MOM_GRID = ["6m", "9m", "12_1", "12m"]
BTC_MOM_GRID = ["1m", "3m", "6m", "12m"]
MOM_DAYS = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}   # "12_1"은 별도 처리(252~21일 전)

COST_BPS = {"gold": 5, "btc": 30}       # 편도, Fable 5 권고(GLD 5bp·BTC 30bp)
MIN_OFF_EPISODES = 8                    # 이보다 적으면 순위 매길 표본 부족(Fable 5 §2)


def _log(m): print(f"[레짐백테스트] {m}", file=sys.stderr)


# ------------------------- 데이터 -------------------------
def fetch(ticker: str, cache_path: str) -> pd.Series:
    if os.path.exists(cache_path):
        s = pd.read_pickle(cache_path)
        _log(f"{ticker}: 캐시 사용({cache_path}, {len(s)}일)")
        return s
    import yfinance as yf
    df = yf.download(ticker, period="max", auto_adjust=True, interval="1d", progress=False)
    s = df["Close"][ticker] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s = s.dropna()
    os.makedirs("output", exist_ok=True)
    s.to_pickle(cache_path)
    _log(f"{ticker}: 신규 다운로드({s.index.min().date()}~{s.index.max().date()}, {len(s)}일)")
    return s


# ------------------------- 레짐 상태 머신(market_signals.regime_state와 동일 로직, 벡터화) -------------------------
def regime_series(closes: np.ndarray, trend_ma: int, band: float, confirm: int) -> np.ndarray:
    """일별 상태(1=ON, 0=OFF, np.nan=미확정)를 반환 — market_signals.regime_state()와 동일한
    히스테리시스+확인일수 로직(당일 종가까지의 정보만 사용, 미래참조 없음)."""
    n = len(closes)
    ma = pd.Series(closes).rolling(trend_ma).mean().to_numpy()
    out = np.full(n, np.nan)
    state = None
    streak_dir, streak = None, 0
    for i in range(n):
        if np.isnan(ma[i]):
            continue
        c = closes[i]
        if c > ma[i] * (1 + band):
            raw = "ON"
        elif c < ma[i] * (1 - band):
            raw = "OFF"
        else:
            raw = None
        if raw and raw != state:
            if raw == streak_dir:
                streak += 1
            else:
                streak_dir, streak = raw, 1
            if streak >= confirm:
                state = raw
                streak_dir, streak = None, 0
        else:
            streak_dir, streak = None, 0
        out[i] = np.nan if state is None else (1.0 if state == "ON" else 0.0)
    return out


def momentum_ok(closes: np.ndarray, kind: str) -> np.ndarray:
    """절대 모멘텀 필터(해당 룩백 수익률 > 0). '12_1'은 252일 전~21일 전(스킵-먼스)."""
    n = len(closes)
    out = np.full(n, np.nan)
    if kind == "12_1":
        for i in range(252, n):
            p0, p1 = closes[i - 252], closes[i - 21]
            out[i] = 1.0 if (p0 and p1 / p0 - 1 > 0) else 0.0
    else:
        d = MOM_DAYS[kind]
        for i in range(d, n):
            p0 = closes[i - d]
            out[i] = 1.0 if (p0 and closes[i] / p0 - 1 > 0) else 0.0
    return out


# ------------------------- 성과 지표 -------------------------
def _ulcer(nav: np.ndarray) -> float:
    cm = np.maximum.accumulate(nav)
    dd = (nav / cm - 1) * 100
    return float(np.sqrt(np.mean(dd ** 2)))


def _mdd(nav: np.ndarray) -> float:
    cm = np.maximum.accumulate(nav)
    return float(((nav / cm - 1).min()) * 100)


def _cagr(nav: np.ndarray, n_days: int) -> float:
    yrs = n_days / TRADING_DAYS
    return float((nav[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 and nav[-1] > 0 else float("nan")


def simulate(closes: np.ndarray, exposure: np.ndarray, cost_bps: float) -> dict:
    """exposure[t-1]로 t일 수익을 받는다(1봉 지연, Fable 5 §1 실행규칙). 익스포저 변경일에
    편도 비용 차감. 반환: nav 시계열 + 지표 + OFF(0) 진입 횟수(에피소드 카운트)."""
    ret = np.diff(closes) / closes[:-1]                 # ret[t] = day t+1 vs day t 수익률
    exp_lag = exposure[:-1]                              # 전일 노출로 당일 수익 적용
    strat_ret = exp_lag * ret
    flips = np.diff(np.nan_to_num(exposure, nan=-1)) != 0
    cost = np.where(flips[:-1] if len(flips) > len(strat_ret) else flips, cost_bps / 10000.0, 0.0)
    cost = cost[:len(strat_ret)]
    strat_ret = strat_ret - cost
    strat_ret = np.nan_to_num(strat_ret, nan=0.0)
    nav = np.cumprod(1 + strat_ret)
    bh_nav = closes[1:] / closes[0]
    off_episodes = int(np.sum((np.diff(np.nan_to_num(exposure, nan=1)) == -1)))
    return {"nav": nav, "bh_nav": bh_nav, "cagr": _cagr(nav, len(nav)),
            "bh_cagr": _cagr(bh_nav, len(bh_nav)), "ulcer": _ulcer(nav), "bh_ulcer": _ulcer(bh_nav),
            "mdd": _mdd(nav), "bh_mdd": _mdd(bh_nav), "off_episodes": off_episodes,
            "strat_ret": strat_ret}


def composite_score(m: dict) -> float:
    """Fable 5 §4: CAGR 손실예산(매수후보유 CAGR의 25% 또는 1.5%p 중 큰 쪽) 안에서만
    Ulcer 개선률을 점수로 — 손실예산 초과면 -inf(자격 박탈)."""
    budget = max(0.25 * m["bh_cagr"], 1.5) if m["bh_cagr"] > 0 else 1.5
    if m["cagr"] < m["bh_cagr"] - budget:
        return float("-inf")
    if m["bh_ulcer"] <= 0:
        return float("-inf")
    return (m["bh_ulcer"] - m["ulcer"]) / m["bh_ulcer"]


# ------------------------- Stage 1: 추세선×밴드×확인일수 -------------------------
def run_stage1(closes: np.ndarray, grid: dict, current: dict, cost_bps: float, asset: str) -> dict:
    rows = []
    for tm in grid["trend_ma"]:
        for band in grid["band"]:
            for cf in grid["confirm"]:
                exp = regime_series(closes, tm, band, cf)
                m = simulate(closes, exp, cost_bps)
                score = composite_score(m)
                rows.append({"trend_ma": tm, "band": band, "confirm": cf,
                            "cagr": round(m["cagr"], 2), "bh_cagr": round(m["bh_cagr"], 2),
                            "ulcer": round(m["ulcer"], 2), "bh_ulcer": round(m["bh_ulcer"], 2),
                            "mdd": round(m["mdd"], 1), "bh_mdd": round(m["bh_mdd"], 1),
                            "off_episodes": m["off_episodes"], "score": score,
                            "rankable": m["off_episodes"] >= MIN_OFF_EPISODES})
    rankable = [r for r in rows if r["rankable"] and r["score"] != float("-inf")]
    rankable.sort(key=lambda r: r["score"], reverse=True)
    best = rankable[0] if rankable else None
    # 현행 파라미터 위치 확인
    cur_row = next((r for r in rows if r["trend_ma"] == current["trend_ma"]
                    and r["band"] == current["band"] and r["confirm"] == current["confirm"]), None)
    # 고원 확인: best의 그리드축 ±1스텝 이웃 대부분이 현행보다 나은지
    plateau_ok = None
    if best and cur_row:
        neighbors = [r for r in rows if r["rankable"] and r["score"] != float("-inf")
                    and abs(grid["trend_ma"].index(r["trend_ma"]) - grid["trend_ma"].index(best["trend_ma"])) <= 1
                    and abs(grid["band"].index(r["band"]) - grid["band"].index(best["band"])) <= 1
                    and abs(grid["confirm"].index(r["confirm"]) - grid["confirm"].index(best["confirm"])) <= 1]
        n_beat = sum(1 for r in neighbors if r["score"] > cur_row["score"])
        plateau_ok = n_beat >= max(1, len(neighbors) // 2)
    always_on = simulate(closes, np.ones(len(closes)), cost_bps)
    _log(f"[{asset}] Stage1 완료: {len(rows)}조합(순위가능 {len(rankable)}) · "
        f"최우수 {best} · 현행 {cur_row} · 고원여부 {plateau_ok}")
    return {"rows": rows, "best": best, "current": cur_row, "plateau_ok": plateau_ok,
            "always_on": {"cagr": round(always_on["cagr"], 2), "ulcer": round(always_on["ulcer"], 2),
                         "mdd": round(always_on["mdd"], 1)},
            "n_days": len(closes)}


# ------------------------- Stage 2: 모멘텀(1단계 동결 후 조건부 AND필터) -------------------------
def run_stage2(closes: np.ndarray, base_params: dict, mom_grid: list, current_mom: str,
              cost_bps: float, asset: str) -> dict:
    regime = regime_series(closes, base_params["trend_ma"], base_params["band"], base_params["confirm"])
    rows = []
    for mom in mom_grid:
        mok = momentum_ok(closes, mom)
        exp = np.where((regime == 1.0) & (mok == 1.0), 1.0,
                      np.where(np.isnan(regime) | np.isnan(mok), np.nan, 0.0))
        m = simulate(closes, exp, cost_bps)
        score = composite_score(m)
        rows.append({"mom": mom, "cagr": round(m["cagr"], 2), "ulcer": round(m["ulcer"], 2),
                    "mdd": round(m["mdd"], 1), "off_episodes": m["off_episodes"],
                    "score": score, "rankable": m["off_episodes"] >= MIN_OFF_EPISODES})
    rankable = [r for r in rows if r["rankable"] and r["score"] != float("-inf")]
    rankable.sort(key=lambda r: r["score"], reverse=True)
    best = rankable[0] if rankable else None
    cur_row = next((r for r in rows if r["mom"] == current_mom), None)
    _log(f"[{asset}] Stage2(모멘텀, base={base_params}) 완료: 최우수 {best} · 현행 {cur_row}")
    return {"rows": rows, "best": best, "current": cur_row}


# ------------------------- 쌍대 블록부트스트랩(후보 vs 현행) -------------------------
def paired_block_bootstrap(closes: np.ndarray, params_a: dict, params_b: dict, cost_bps: float,
                           block=60, n_boot=2000, seed=7) -> dict:
    """params_a(후보) vs params_b(현행)의 일별수익 쌍을 블록 단위로 함께 리샘플 —
    Δ(Ulcer)·Δ(CAGR)의 90%CI(Fable 5 §2.3)."""
    exp_a = regime_series(closes, params_a["trend_ma"], params_a["band"], params_a["confirm"])
    exp_b = regime_series(closes, params_b["trend_ma"], params_b["band"], params_b["confirm"])
    ma_ = simulate(closes, exp_a, cost_bps)
    mb_ = simulate(closes, exp_b, cost_bps)
    ra, rb = ma_["strat_ret"], mb_["strat_ret"]
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    d_ulcer, d_cagr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(i * block, (i + 1) * block) for i in idx])
        nav_a = np.cumprod(1 + ra[sel]); nav_b = np.cumprod(1 + rb[sel])
        d_ulcer.append(_ulcer(nav_b) - _ulcer(nav_a))     # 양수 = 후보가 Ulcer 더 낮음(개선)
        d_cagr.append(_cagr(nav_a, n) - _cagr(nav_b, n))
    d_ulcer, d_cagr = np.array(d_ulcer), np.array(d_cagr)
    ci = lambda x: (round(float(np.percentile(x, 5)), 3), round(float(np.percentile(x, 95)), 3))
    return {"delta_ulcer_ci90": ci(d_ulcer), "delta_cagr_ci90": ci(d_cagr),
            "delta_ulcer_excludes_zero": bool(ci(d_ulcer)[0] > 0 or ci(d_ulcer)[1] < 0)}


# ------------------------- PBO/DSR 게이트(Stage1 그리드 전체를 기존 프레임에 투입) -------------------------
def pbo_gate(closes: np.ndarray, grid: dict, cost_bps: float, month=21, n_blocks=12) -> dict:
    """월간(21거래일) 비중첩 초과수익(vs 매수후보유)으로 PBO/DSR — overfit_stats.analyze 재사용
    (Sharpe 랭킹은 그대로 두고, 전체 그리드가 노이즈가 아님을 보이는 새니티 게이트로만 사용,
    Fable 5 §2-1: 실제 채택 판단은 composite score+플래토+부트스트랩이 맡는다)."""
    bh_ret = np.diff(closes) / closes[:-1]
    trials, matrix, dates0 = [], [], None
    for tm in grid["trend_ma"]:
        for band in grid["band"]:
            for cf in grid["confirm"]:
                exp = regime_series(closes, tm, band, cf)
                m = simulate(closes, exp, cost_bps)
                r = m["strat_ret"]
                n = min(len(r), len(bh_ret))
                excess = r[:n] - bh_ret[:n]
                d, ex = [], []
                for t in range(0, n - month, month):
                    d.append(str(t)); ex.append(round(float(excess[t:t + month].sum()), 6))
                if dates0 is None:
                    dates0 = d
                matrix.append(ex[:len(dates0)])
                trials.append(f"tm{tm}_b{band}_c{cf}")
    trial_data = {"horizon": "regime_grid", "universe": "gold_or_btc", "cost": f"{cost_bps}bp",
                 "rebal_days": month, "hold_days": month, "dates": dates0,
                 "trials": trials, "excess_returns": matrix}
    return OS.analyze(trial_data, n_blocks=n_blocks, save=False)


# ------------------------- 실행 -------------------------
def run_asset(name: str, ticker: str, current: dict, grid: dict, mom_current: str, mom_grid: list,
             cost_bps: float, do_bootstrap=True, do_eth_check=False) -> dict:
    closes = fetch(ticker, f"output/regime_price_cache_{name}.pkl").to_numpy()
    stage1 = run_stage1(closes, grid, current, cost_bps, name)
    payload = {"asset": name, "ticker": ticker, "n_days": len(closes),
              "date_range": None, "stage1": stage1, "current_params": current}
    if stage1["best"]:
        base = {"trend_ma": stage1["best"]["trend_ma"], "band": stage1["best"]["band"],
               "confirm": stage1["best"]["confirm"]}
        payload["stage2"] = run_stage2(closes, base, mom_grid, mom_current, cost_bps, name)
        if do_bootstrap:
            payload["bootstrap_best_vs_current"] = paired_block_bootstrap(closes, base, current, cost_bps)
    try:
        payload["pbo_gate"] = pbo_gate(closes, grid, cost_bps)
    except Exception as e:
        _log(f"[{name}] PBO 게이트 실패({type(e).__name__}: {e}) — 생략")
        payload["pbo_gate"] = None
    return payload


def main():
    ap = argparse.ArgumentParser(description="금·비트코인 레짐타이밍 파라미터 검증")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    os.makedirs("output", exist_ok=True)
    gold = run_asset("gold", "GLD", GOLD_CURRENT, GOLD_GRID, GOLD_MOM_CURRENT, GOLD_MOM_GRID,
                     COST_BPS["gold"], do_bootstrap=True)
    with open("output/regime_backtest_gold.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    _log("저장: output/regime_backtest_gold.json")

    btc = run_asset("btc", "BTC-USD", BTC_CURRENT, BTC_GRID, BTC_MOM_CURRENT, BTC_MOM_GRID,
                    COST_BPS["btc"], do_bootstrap=True)
    # BTC 반기 분할(2014-19 / 2020-) — Fable 5 §2 요구
    closes_btc = fetch("BTC-USD", "output/regime_price_cache_btc.pkl")
    split_date = "2020-01-01"
    half1 = closes_btc[closes_btc.index < split_date].to_numpy()
    half2 = closes_btc[closes_btc.index >= split_date].to_numpy()
    best = btc["stage1"]["best"]
    if best:
        bp = {"trend_ma": best["trend_ma"], "band": best["band"], "confirm": best["confirm"]}
        m1 = simulate(half1, regime_series(half1, **bp), COST_BPS["btc"])
        m2 = simulate(half2, regime_series(half2, **bp), COST_BPS["btc"])
        btc["half_split_check"] = {
            "2014_2019": {"score": composite_score(m1), "ulcer": round(m1["ulcer"], 2), "cagr": round(m1["cagr"], 2)},
            "2020_now":  {"score": composite_score(m2), "ulcer": round(m2["ulcer"], 2), "cagr": round(m2["cagr"], 2)}}
        # ETH-USD 확인용(전용 튜닝 없이 BTC 최우수 파라미터 그대로 적용)
        try:
            eth_closes = fetch("ETH-USD", "output/regime_price_cache_eth.pkl").to_numpy()
            m_eth = simulate(eth_closes, regime_series(eth_closes, **bp), COST_BPS["btc"])
            m_eth_cur = simulate(eth_closes, regime_series(eth_closes, **BTC_CURRENT), COST_BPS["btc"])
            btc["eth_confirmatory_check"] = {
                "candidate_score": composite_score(m_eth), "current_score": composite_score(m_eth_cur),
                "directionally_consistent": composite_score(m_eth) >= composite_score(m_eth_cur)}
        except Exception as e:
            _log(f"ETH 확인 실패({type(e).__name__}: {e})")
    with open("output/regime_backtest_btc.json", "w", encoding="utf-8") as f:
        json.dump(btc, f, ensure_ascii=False, indent=2)
    _log("저장: output/regime_backtest_btc.json")


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 regime_series/simulate/composite_score/PBO게이트 배선 확인")
    rng = np.random.default_rng(5)
    n = 1500
    # 추세 구간(꾸준한 상승) + 급락 구간 + 횡보 구간을 섞어 합성 — 200일선 필터가
    # 급락을 피해가는지(레짐 OFF로 전환) 최소한의 정성 확인.
    up = 100 * np.exp(np.cumsum(np.full(600, 0.0015)))
    crash = up[-1] * np.exp(np.cumsum(np.full(150, -0.01)))
    chop = crash[-1] * np.exp(np.cumsum(rng.normal(0, 0.01, 750)))
    closes = np.concatenate([up, crash, chop])

    exp = regime_series(closes, 200, 0.01, 3)
    assert exp.shape == closes.shape
    # 급락 구간 후반부에는 OFF(0)로 전환돼 있어야 함(200일선이 형성된 후, 급락이 충분히 진행된 뒤)
    post_crash_idx = 600 + 140
    assert exp[post_crash_idx] == 0.0, f"급락 후 레짐이 OFF로 전환돼야 함: {exp[post_crash_idx]}"
    _log("[self-test] 통과: regime_series가 급락 구간에서 OFF로 전환됨")

    m = simulate(closes, exp, 5)
    assert np.isfinite(m["cagr"]) and np.isfinite(m["ulcer"])
    assert m["ulcer"] < m["bh_ulcer"], (
        f"급락을 피하는 필터라면 Ulcer가 매수후보유보다 낮아야 함: {m['ulcer']} vs {m['bh_ulcer']}")
    assert m["off_episodes"] >= 1
    _log(f"[self-test] 통과: simulate 배선 정상(Ulcer {m['ulcer']:.1f} < B&H {m['bh_ulcer']:.1f}, "
        f"OFF전환 {m['off_episodes']}회)")

    score_good = composite_score(m)
    bad_m = dict(m); bad_m["cagr"] = m["bh_cagr"] - 999
    assert composite_score(bad_m) == float("-inf"), "CAGR 손실예산 초과 시 -inf여야 함"
    assert score_good != float("-inf")
    _log(f"[self-test] 통과: composite_score 배선 정상(정상 {score_good:.3f}, 예산초과 -inf)")

    mom = momentum_ok(closes, "3m")
    assert mom.shape == closes.shape
    assert np.isnan(mom[:63]).all()
    _log("[self-test] 통과: momentum_ok 배선 정상(룩백 부족 구간 NaN)")

    boot = paired_block_bootstrap(closes, {"trend_ma": 150, "band": 0.01, "confirm": 3},
                                  {"trend_ma": 200, "band": 0.01, "confirm": 3}, 5, n_boot=200)
    assert "delta_ulcer_ci90" in boot and len(boot["delta_ulcer_ci90"]) == 2
    _log(f"[self-test] 통과: paired_block_bootstrap 배선 정상({boot['delta_ulcer_ci90']})")

    small_grid = {"trend_ma": [150, 200], "band": [0.01], "confirm": [3]}
    gate = pbo_gate(closes, small_grid, 5, month=21, n_blocks=4)
    assert "pbo" in gate and "dsr_verdict" in gate
    _log(f"[self-test] 통과: pbo_gate 배선 정상(PBO={gate['pbo']['pbo']})")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
