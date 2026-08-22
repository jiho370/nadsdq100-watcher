#!/usr/bin/env python3
"""
fx_hedge_validation.py — 원달러 환노출 필터, 사전등록 프로토콜 검증 (2026-07-30).

지호 님이 작성한 사전등록 스펙("원달러 환노출 필터 — 백테스트 및 검증 프로토콜")을
그대로 구현한다. §6-R(backtest_regime_fx.py)의 단독자산 백테스트는 이 스펙의 §0-1
(표본 오염)·§0-2(포트폴리오 레벨 재정의 누락)·Gate1(무작위 대조군 누락) 문제가 있었음이
지적됐다 — 이 스크립트가 그 결함을 고친 버전이다.

구현 범위(이번 실행): §9 실행순서의 1~4단계.
  1. §2-1 헤지 캐리 수익률(한미 금리차 근사) 데이터 확보
  2. §0-2/§3 포트폴리오 레벨 목적함수 재정의(Primary=ΔUlcer, 미국주식 슬리브=SPY로 근사)
  3. §1 A안(전체그리드 동일가중 앙상블) 노출비율 h_t 시계열 생성 — argmax 선택 없음
  4. §4 Gate1(시장체류시간 매칭 무작위 대조군) — 가장 먼저 돌릴 것으로 지정된 게이트
Gate2~4(블록부트스트랩·집중도/레짐·고원)와 §5(워크포워드)·§6(외부 통화쌍)은 Gate1
결과에 따라 후속 실행(§8 사전확약: Gate1 미달 시 전체 중단, 재시도는 다중검정이므로 안 함).

가정(명시, 전부 §2/§0-2가 허용한 근사):
  - 미국주식 슬리브 = SPY(실제 라이브 topn8 알고리즘이 아니라 지수로 근사 — 알고리즘
    고유의 과최적화 이슈와 환헤지 질문을 분리하기 위함). "나머지 슬리브"(한국주식·채권·
    금·코인)는 이번 검증 범위 밖 — 목적함수는 "미국주식 슬리브 자체의 Ulcer"에 한정.
  - 헤지 캐리 수익률 ≈ (한국 3개월 은행간금리 − 미국 3개월 국채금리)/100/252 (FRED
    DTB3·IR3TIB01KRM156N, 실제 스왑포인트 미확보 — §2-1이 명시적으로 허용한 근사).
  - 실행: t일 신호로 t+1일 수익 적용(1봉 지연, look-ahead 방지 — RA.simulate과 동일 관행).
  - 캘린더: SPY 거래일을 기준 달력으로, KRW=X·금리 시계열은 그 위에 ffill로 정렬.

실행: python fx_hedge_validation.py --gate1
      python fx_hedge_validation.py --self-test
결과: output/fx_hedge_gate1.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_regime_assets as RA
from backtest_regime_fx import FX_GRID_WIDE

TRADING_DAYS = 252
N_REP_DEFAULT = 2000
COST_BPS = 10.0


def _log(m): print(f"[FX헤지검증] {m}", file=sys.stderr)


# ------------------------- 데이터 -------------------------
def _is_fresh(path: str, max_stale_days: int = 5) -> bool:
    """2026-08-22 재검증: fetch 캐시들이 무기한 재사용돼 수 주 stale한 채로 쓰이고 있었음
    (backtest_regime_assets.fetch()와 동일 버그) — 여기도 동일 기준으로 신선도 확인."""
    if not os.path.exists(path):
        return False
    s = pd.read_pickle(path)
    return (pd.Timestamp.now().normalize() - s.index.max().normalize()).days <= max_stale_days


def fetch_spy() -> pd.Series:
    path = "output/regime_price_cache_spy_hedge.pkl"
    if _is_fresh(path):
        return pd.read_pickle(path)
    import yfinance as yf
    df = yf.download("SPY", period="max", auto_adjust=True, interval="1d", progress=False)
    s = df["Close"]
    s = s.iloc[:, 0] if hasattr(s, "columns") else s
    s = s.dropna()
    os.makedirs("output", exist_ok=True)
    s.to_pickle(path)
    return s


def fetch_fred(series_id: str) -> pd.Series:
    path = f"output/fred_cache_{series_id}.pkl"
    if _is_fresh(path, max_stale_days=35):   # 금리류 월간 지표 — 일단위 신선도 기준 부적합
        return pd.read_pickle(path)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
    s = pd.to_numeric(df[series_id], errors="coerce").dropna()
    os.makedirs("output", exist_ok=True)
    s.to_pickle(path)
    _log(f"{series_id}: 신규 다운로드({s.index.min().date()}~{s.index.max().date()}, {len(s)}행)")
    return s


def build_calendar_frame(start=None, end=None) -> dict:
    """SPY 거래일을 기준 달력으로 SPY·KRW=X·한미 3개월 금리차(캐리)를 정렬(ffill)."""
    spy = fetch_spy()
    fx = RA.fetch("KRW=X", "output/regime_price_cache_fx.pkl")
    us_rate = fetch_fred("DTB3")              # % 단위, 일별
    kr_rate = fetch_fred("IR3TIB01KRM156N")   # % 단위, 월별

    if start:
        spy = spy[spy.index >= start]
    if end:
        spy = spy[spy.index <= end]
    cal = spy.index

    fx_aligned = fx.reindex(cal.union(fx.index)).sort_index().ffill().reindex(cal)
    us_aligned = us_rate.reindex(cal.union(us_rate.index)).sort_index().ffill().reindex(cal)
    kr_aligned = kr_rate.reindex(cal.union(kr_rate.index)).sort_index().ffill().reindex(cal)

    valid = spy.notna() & fx_aligned.notna() & us_aligned.notna() & kr_aligned.notna()
    cal = cal[valid]
    spy, fx_aligned, us_aligned, kr_aligned = (s[valid] for s in (spy, fx_aligned, us_aligned, kr_aligned))

    carry_daily = (kr_aligned - us_aligned) / 100.0 / TRADING_DAYS   # §2-1 근사
    return {"cal": cal, "spy": spy.to_numpy(), "fx": fx_aligned.to_numpy(),
            "carry": carry_daily.to_numpy(), "us_rate": us_aligned.to_numpy(),
            "kr_rate": kr_aligned.to_numpy()}


# ------------------------- 성과지표 -------------------------
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


def portfolio_nav(spy: np.ndarray, fx: np.ndarray, carry: np.ndarray, h: np.ndarray,
                  cost_bps: float) -> dict:
    """§0-2 포트폴리오 목적함수: r_t = r_SPY,t + h_{t-1}*r_FX,t + (1-h_{t-1})*carry_t − 비용.
    h는 t일 종가까지 정보로 결정 → t+1일 수익에 적용(1봉 지연, §2-2). 비용은
    |Δh|비례(§2-3, 앙상블 회전율 모델)."""
    r_spy = np.diff(spy) / spy[:-1]
    r_fx = np.diff(fx) / fx[:-1]
    carry_t = carry[1:]                      # t+1일에 적용되는 캐리(당일 금리차 그대로 근사)
    h_lag = h[:-1]                            # 1봉 지연
    r_port = r_spy + h_lag * r_fx + (1 - h_lag) * carry_t
    dh = np.diff(h, prepend=h[0])[:-1]        # h_t 변화(첫날은 비용 0 취급)
    cost = np.abs(dh) * (cost_bps / 10000.0)
    r_port = r_port - cost
    nav = np.cumprod(1 + r_port)
    return {"nav": nav, "ret": r_port, "cagr": _cagr(nav, len(nav)), "ulcer": _ulcer(nav),
           "mdd": _mdd(nav)}


# ------------------------- 앙상블(A안) h_t -------------------------
def build_ensemble_h(grid: dict = FX_GRID_WIDE, use_cache: bool = True) -> dict:
    """528조합 각각의 이진 레짐신호를 KRW=X 자체 달력에서 계산 → 동일가중 평균해 h_t.
    argmax 선택 없음(§1 A안). combo별 지속성 통계(p_i·평균ON/OFF지속일)도 함께 반환
    (Gate1 매칭 무작위 대조군 생성에 재사용). 528×5877 재계산이 무거워(수 분) 캐시."""
    cache_path = "output/fx_ensemble_h_cache.pkl"
    if use_cache and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    fx = RA.fetch("KRW=X", "output/regime_price_cache_fx.pkl")
    closes = fx.to_numpy()
    combos = [(tm, b, c) for tm in grid["trend_ma"] for b in grid["band"] for c in grid["confirm"]]
    exp_mat = np.full((len(combos), len(closes)), np.nan)
    for i, (tm, b, c) in enumerate(combos):
        exp_mat[i] = RA.regime_series(closes, tm, b, c)

    combo_stats = []
    for i in range(len(combos)):
        e = exp_mat[i]
        valid = ~np.isnan(e)
        e_valid = e[valid]
        if len(e_valid) < 30:
            combo_stats.append(None); continue
        p = float(e_valid.mean())
        runs, cur, cur_len = [], e_valid[0], 1
        for v in e_valid[1:]:
            if v == cur:
                cur_len += 1
            else:
                runs.append((cur, cur_len)); cur, cur_len = v, 1
        runs.append((cur, cur_len))
        on_runs = [l for v, l in runs if v == 1.0]
        off_runs = [l for v, l in runs if v == 0.0]
        l_on = float(np.mean(on_runs)) if on_runs else 1.0
        l_off = float(np.mean(off_runs)) if off_runs else 1.0
        combo_stats.append({"p": p, "l_on": l_on, "l_off": l_off,
                            "q_on_to_off": min(max(1.0 / l_on, 1e-4), 1.0),
                            "q_off_to_on": min(max(1.0 / l_off, 1e-4), 1.0)})

    # h_t = NaN(워밍업 구간) 무시하고 유효신호만 평균, 전부 NaN인 날은 ffill 대신 노출0(보수적)
    exp_filled = np.where(np.isnan(exp_mat), 0.0, exp_mat)
    n_valid = np.sum(~np.isnan(exp_mat), axis=0)
    h_t = np.divide(exp_filled.sum(axis=0), np.maximum(n_valid, 1))
    h_series = pd.Series(h_t, index=fx.index)
    result = {"h_series": h_series, "combo_stats": combo_stats, "n_combo": len(combos)}
    if use_cache:
        os.makedirs("output", exist_ok=True)
        pd.to_pickle(result, cache_path)
    return result


# ------------------------- Gate 1: 매칭 무작위 대조군 -------------------------
def simulate_markov_ensemble(combo_stats: list, T: int, n_rep: int, seed: int) -> np.ndarray:
    """combo_stats(조합별 p_i·평균ON/OFF지속일)와 동일한 마르코프 지속성을 갖는
    무작위 이진신호를 조합별로 생성해 평균낸 앙상블 경로를 n_rep회 반환. shape (T, n_rep)."""
    stats = [s for s in combo_stats if s is not None]
    n_combo = len(stats)
    q_on_to_off = np.array([s["q_on_to_off"] for s in stats])
    q_off_to_on = np.array([s["q_off_to_on"] for s in stats])
    p0 = q_off_to_on / (q_off_to_on + q_on_to_off)

    rng = np.random.default_rng(seed)
    state = rng.random((n_rep, n_combo)) < p0[None, :]
    h_rand = np.empty((T, n_rep))
    h_rand[0] = state.mean(axis=1)
    for t in range(1, T):
        r = rng.random((n_rep, n_combo))
        flip_off = state & (r < q_on_to_off[None, :])
        flip_on = (~state) & (r < q_off_to_on[None, :])
        state = np.where(flip_off, False, state)
        state = np.where(flip_on, True, state)
        h_rand[t] = state.mean(axis=1)
    return h_rand


def gate1_matched_random(frame: dict, h_real: np.ndarray, combo_stats: list,
                         n_rep: int = N_REP_DEFAULT, cost_bps: float = COST_BPS,
                         seed: int = 7) -> dict:
    """실제 앙상블과 동일한 (조합별 p_i·평균지속기간)을 갖는 마르코프 무작위 신호를
    조합별로 생성 → 528개 평균(=무작위 앙상블 1회) → 목적함수(ΔUlcer, 포트폴리오
    레벨) 분포를 n_rep회 확보 → 실제 필터가 몇 퍼센타일인지."""
    spy, fx, carry = frame["spy"], frame["fx"], frame["carry"]
    T = len(spy)

    base = portfolio_nav(spy, fx, carry, np.ones(T), cost_bps=0.0)   # 기준선: 항상 환노출, 비용0
    real = portfolio_nav(spy, fx, carry, h_real, cost_bps)
    delta_ulcer_real = base["ulcer"] - real["ulcer"]
    delta_cagr_real = real["cagr"] - base["cagr"]

    h_rand = simulate_markov_ensemble(combo_stats, T, n_rep, seed)

    r_spy = np.diff(spy) / spy[:-1]
    r_fx = np.diff(fx) / fx[:-1]
    carry_t = carry[1:]
    h_lag = h_rand[:-1, :]                                   # (T-1, n_rep), 1봉 지연
    r_port_rand = r_spy[:, None] + h_lag * r_fx[:, None] + (1 - h_lag) * carry_t[:, None]
    dh = np.diff(h_rand, axis=0, prepend=h_rand[0:1])[:-1, :]
    cost = np.abs(dh) * (cost_bps / 10000.0)
    r_port_rand = r_port_rand - cost
    nav_rand = np.cumprod(1 + r_port_rand, axis=0)
    cm_rand = np.maximum.accumulate(nav_rand, axis=0)
    dd_rand = (nav_rand / cm_rand - 1) * 100
    ulcer_rand = np.sqrt(np.mean(dd_rand ** 2, axis=0))
    cagr_rand = (nav_rand[-1] ** (1 / (len(r_port_rand) / TRADING_DAYS)) - 1) * 100

    delta_ulcer_rand = base["ulcer"] - ulcer_rand
    delta_cagr_rand = cagr_rand - base["cagr"]

    pctile_ulcer = float((delta_ulcer_rand < delta_ulcer_real).mean() * 100)
    pctile_cagr = float((delta_cagr_rand < delta_cagr_real).mean() * 100)

    n_combo = len([s for s in combo_stats if s is not None])
    return {
        "n_rep": n_rep, "n_combo_used": n_combo, "cost_bps": cost_bps,
        "baseline": {"cagr": round(base["cagr"], 3), "ulcer": round(base["ulcer"], 3), "mdd": round(base["mdd"], 2)},
        "real_filter": {"cagr": round(real["cagr"], 3), "ulcer": round(real["ulcer"], 3), "mdd": round(real["mdd"], 2),
                        "delta_ulcer": round(delta_ulcer_real, 4), "delta_cagr": round(delta_cagr_real, 4),
                        "avg_exposure": round(float(h_real.mean()), 4)},
        "random_control": {
            "delta_ulcer_mean": round(float(delta_ulcer_rand.mean()), 4),
            "delta_ulcer_p50": round(float(np.percentile(delta_ulcer_rand, 50)), 4),
            "delta_ulcer_p95": round(float(np.percentile(delta_ulcer_rand, 95)), 4),
            "delta_cagr_mean": round(float(delta_cagr_rand.mean()), 4),
            "delta_cagr_p50": round(float(np.percentile(delta_cagr_rand, 50)), 4),
        },
        "real_filter_percentile_vs_random": {"delta_ulcer": round(pctile_ulcer, 2),
                                             "delta_cagr": round(pctile_cagr, 2)},
        "verdict_ulcer": "통과(>=95퍼센타일)" if pctile_ulcer >= 95 else "미달 — §8 사전확약대로 전체 중단",
    }


# ------------------------- Gate 2: 정상성 블록부트스트랩(Politis-Romano) -------------------------
def stationary_bootstrap_indices(n: int, mean_block_len: float, n_rep: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap 인덱스 생성, n_rep개를 한 번에 벡터화(§4 Gate2).
    각 시점마다 확률 p=1/mean_block_len로 새 블록(무작위 시작점)을 열고, 아니면 직전
    인덱스+1(순환)로 블록을 이어간다 — 고정길이 블록이 아니라 기하분포 길이라 블록경계
    아티팩트가 완화된다. 반환 shape (n, n_rep)."""
    p = 1.0 / mean_block_len
    idx = np.empty((n, n_rep), dtype=np.int64)
    idx[0] = rng.integers(0, n, size=n_rep)
    new_block = rng.random((n, n_rep)) < p
    for t in range(1, n):
        starts = rng.integers(0, n, size=n_rep)
        idx[t] = np.where(new_block[t], starts, (idx[t - 1] + 1) % n)
    return idx


def gate2_bootstrap(frame: dict, h_real: np.ndarray, cost_bps: float = COST_BPS,
                    block_lens=(21, 63, 126), n_rep: int = 10000, seed: int = 11) -> dict:
    """§4 Gate2: 일별 (필터수익 · 기준선수익) 쌍을 정상성블록부트스트랩으로 함께 리샘플해
    Primary(ΔUlcer)·참고(ΔCAGR)의 95%CI를 블록길이 3종 각각 산출. 판정: CI가 0을 배제."""
    spy, fx, carry = frame["spy"], frame["fx"], frame["carry"]
    base = portfolio_nav(spy, fx, carry, np.ones(len(spy)), cost_bps=0.0)
    filt = portfolio_nav(spy, fx, carry, h_real, cost_bps)
    r_base, r_filt = base["ret"], filt["ret"]
    n = len(r_base)
    rng = np.random.default_rng(seed)

    out = {}
    for L in block_lens:
        idx = stationary_bootstrap_indices(n, L, n_rep, rng)
        rb = r_base[idx]     # (n, n_rep)
        rf = r_filt[idx]
        nav_b = np.cumprod(1 + rb, axis=0)
        nav_f = np.cumprod(1 + rf, axis=0)
        cm_b = np.maximum.accumulate(nav_b, axis=0)
        cm_f = np.maximum.accumulate(nav_f, axis=0)
        ulcer_b = np.sqrt(np.mean(((nav_b / cm_b - 1) * 100) ** 2, axis=0))
        ulcer_f = np.sqrt(np.mean(((nav_f / cm_f - 1) * 100) ** 2, axis=0))
        yrs = n / TRADING_DAYS
        cagr_b = (nav_b[-1] ** (1 / yrs) - 1) * 100
        cagr_f = (nav_f[-1] ** (1 / yrs) - 1) * 100
        d_ulcer = ulcer_b - ulcer_f      # 양수=필터가 Ulcer 개선
        d_cagr = cagr_f - cagr_b
        ci_u = (round(float(np.percentile(d_ulcer, 2.5)), 4), round(float(np.percentile(d_ulcer, 97.5)), 4))
        ci_c = (round(float(np.percentile(d_cagr, 2.5)), 4), round(float(np.percentile(d_cagr, 97.5)), 4))
        out[f"block{L}"] = {
            "delta_ulcer_ci95": ci_u, "delta_ulcer_excludes_zero": bool(ci_u[0] > 0 or ci_u[1] < 0),
            "delta_ulcer_mean": round(float(d_ulcer.mean()), 4),
            "delta_cagr_ci95": ci_c, "delta_cagr_excludes_zero": bool(ci_c[0] > 0 or ci_c[1] < 0),
            "delta_cagr_mean": round(float(d_cagr.mean()), 4),
        }
    all_exclude_zero_positive = all(out[f"block{L}"]["delta_ulcer_ci95"][0] > 0 for L in block_lens)
    any_flip = len({out[f"block{L}"]["delta_ulcer_excludes_zero"] for L in block_lens}) > 1
    out["n_rep"] = n_rep
    out["cost_bps"] = cost_bps
    out["point_estimate"] = {"delta_ulcer": round(float(base["ulcer"] - filt["ulcer"]), 4),
                             "delta_cagr": round(float(filt["cagr"] - base["cagr"]), 4)}
    out["verdict"] = ("통과(3개 블록길이 전부 CI가 0 배제·양의 방향)" if all_exclude_zero_positive
                      else "미달 — CI가 0을 포함하거나 블록길이에 따라 결과가 뒤집힘(그 자체가 경고, §4 Gate2)")
    out["block_length_sensitivity_flip"] = any_flip
    return out


# ------------------------- 실행 -------------------------
def run_gate1(n_rep: int = N_REP_DEFAULT) -> dict:
    ens = build_ensemble_h()
    frame = build_calendar_frame()
    h_real = ens["h_series"].reindex(frame["cal"].union(ens["h_series"].index)).sort_index() \
                            .ffill().reindex(frame["cal"]).fillna(0.0).to_numpy()
    _log(f"달력정렬 완료: {len(frame['cal'])}일({frame['cal'][0].date()}~{frame['cal'][-1].date()}), "
        f"앙상블 조합수 {ens['n_combo']}")
    result = gate1_matched_random(frame, h_real, ens["combo_stats"], n_rep=n_rep)
    result["date_range"] = [str(frame["cal"][0].date()), str(frame["cal"][-1].date())]
    result["n_days"] = len(frame["cal"])
    result["assumptions"] = {
        "equity_sleeve": "SPY(라이브 topn8 알고리즘 아님 — 환헤지 질문과 종목선정 알파를 분리)",
        "hedge_carry_proxy": "(KR 3M interbank − US 3M T-bill)/100/252, FRED IR3TIB01KRM156N·DTB3",
        "execution_lag": "1봉(신호 결정일 종가 → 익일 수익 적용)",
        "cost_model": "|Δh_t|×10bp",
    }
    return result


def run_gate2(n_rep: int = 10000) -> dict:
    ens = build_ensemble_h()
    frame = build_calendar_frame()
    h_real = ens["h_series"].reindex(frame["cal"].union(ens["h_series"].index)).sort_index() \
                            .ffill().reindex(frame["cal"]).fillna(0.0).to_numpy()
    result = gate2_bootstrap(frame, h_real, n_rep=n_rep)
    result["date_range"] = [str(frame["cal"][0].date()), str(frame["cal"][-1].date())]
    return result


def main():
    ap = argparse.ArgumentParser(description="원달러 환노출 필터 — 사전등록 프로토콜(포트폴리오 레벨)")
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--n-rep", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    if args.gate2:
        result = run_gate2(n_rep=args.n_rep or 10000)
        os.makedirs("output", exist_ok=True)
        with open("output/fx_hedge_gate2.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log("저장: output/fx_hedge_gate2.json")
        _log(f"기간: {result['date_range']}")
        _log(f"점추정치: {result['point_estimate']}")
        for L in (21, 63, 126):
            _log(f"블록{L}일: {result[f'block{L}']}")
        _log(f"블록길이 민감도(결과 뒤집힘 여부): {result['block_length_sensitivity_flip']}")
        _log(f"판정: {result['verdict']}")
        return

    result = run_gate1(n_rep=args.n_rep or N_REP_DEFAULT)
    os.makedirs("output", exist_ok=True)
    with open("output/fx_hedge_gate1.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log("저장: output/fx_hedge_gate1.json")
    _log(f"기간: {result['date_range']} ({result['n_days']}일)")
    _log(f"기준선(항상환노출): {result['baseline']}")
    _log(f"실제 앙상블 필터: {result['real_filter']}")
    _log(f"무작위 대조군: {result['random_control']}")
    _log(f"실제 필터의 퍼센타일(대조군 대비): {result['real_filter_percentile_vs_random']}")
    _log(f"판정(ΔUlcer 기준): {result['verdict_ulcer']}")


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 데이터 정렬·portfolio_nav·Gate1 배선을 합성 데이터로 확인")
    rng = np.random.default_rng(3)
    n = 1000
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    fx = 1200 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))
    carry = np.full(n, -0.00002)   # 약한 마이너스 캐리(원화금리<달러금리 가정)

    h_always = np.ones(n)
    h_never = np.zeros(n)
    m_always = portfolio_nav(spy, fx, carry, h_always, cost_bps=0.0)
    m_never = portfolio_nav(spy, fx, carry, h_never, cost_bps=0.0)
    assert np.isfinite(m_always["cagr"]) and np.isfinite(m_never["cagr"])
    # h=1(환노출)과 h=0(헤지)의 차이는 (fx수익-캐리)의 누적분이어야 함 — 부호 계산 검증
    diff_ret = (spy[1:] / spy[:-1] - 1 + (fx[1:] / fx[:-1] - 1)) - (spy[1:] / spy[:-1] - 1 + carry[1:])
    assert np.allclose(diff_ret, (fx[1:] / fx[:-1] - 1) - carry[1:])
    _log("[self-test] 통과: portfolio_nav h=1/h=0 차이가 (환수익-캐리)와 일치")

    # 비용 검증: h가 매일 요동치면(고회전) 비용>0이 CAGR을 깎아야 함
    h_noisy = (rng.random(n) < 0.5).astype(float)
    m_nocost = portfolio_nav(spy, fx, carry, h_noisy, cost_bps=0.0)
    m_cost = portfolio_nav(spy, fx, carry, h_noisy, cost_bps=50.0)
    assert m_cost["cagr"] < m_nocost["cagr"], "고회전 노출에 비용을 물리면 CAGR이 낮아져야 함"
    _log(f"[self-test] 통과: 비용모델 배선 정상(무비용 {m_nocost['cagr']:.2f} > 유비용 {m_cost['cagr']:.2f})")

    # Gate1: 실제 필터를 "완전한 사후 최적 신호"로 만들면(미래를 알고 하락 직전 회피) 매우 높은
    # 퍼센타일이 나와야 하고, 실제 필터를 무작위 신호 그 자체로 만들면 50퍼센타일 근방이어야 함.
    frame = {"spy": spy, "fx": fx, "carry": carry}
    combo_stats = [{"p": 0.7, "l_on": 20.0, "l_off": 8.0,
                   "q_on_to_off": 1 / 20.0, "q_off_to_on": 1 / 8.0} for _ in range(20)]

    # (a) "무작위 대조군과 완전히 같은 생성과정"에서 뽑은 신호를 실제 필터로 넣으면
    # 대략 50퍼센타일 근방이어야 함 — IID노이즈처럼 다른 지속성 구조를 넣으면 회전비용
    # 자체가 달라져 비교가 안 되므로, 반드시 동일한 마르코프 생성기(simulate_markov_ensemble)
    # 로 뽑은 한 경로를 써야 apples-to-apples 검증이 됨.
    h_rand_as_real = simulate_markov_ensemble(combo_stats, n, n_rep=1, seed=123)[:, 0]
    g = gate1_matched_random(frame, h_rand_as_real, combo_stats, n_rep=300, seed=1)
    pct = g["real_filter_percentile_vs_random"]["delta_ulcer"]
    assert 5 < pct < 95, f"같은 분포에서 뽑은 신호를 '실제'로 넣었는데 극단 퍼센타일이 나옴({pct}) — 배선 의심"
    _log(f"[self-test] 통과: 같은 분포에서 뽑은 신호를 실제필터로 넣으면 중간 퍼센타일 근방({pct:.1f})")

    # (b) 미래 하락을 완벽히 피하는 사후적 신호(look-ahead 커닝) → 매우 높은 퍼센타일이어야 함.
    # 정렬규약(portfolio_nav): h[t]는 h_lag=h[:-1]를 거쳐 r_fx[t](=day t→t+1 수익률)에 곱해진다
    # (RA.simulate과 동일하게 "당일 종가정보로 다음날 수익을 먹는다" 규약 — 시프트 아님).
    # 그러므로 오라클은 h_oracle[t] = 1{r_fx[t]>0}이어야 진짜 미래컨닝이 된다(하루 밀리면 안 됨).
    r_fx_all = np.diff(fx) / fx[:-1]
    h_oracle = np.concatenate([(r_fx_all > 0).astype(float), [1.0]])   # 마지막 값은 안 쓰임(h[:-1]로 잘림)
    g2 = gate1_matched_random(frame, h_oracle, combo_stats, n_rep=300, seed=1)
    pct2 = g2["real_filter_percentile_vs_random"]["delta_ulcer"]
    assert pct2 >= 95, f"완전한 사후 오라클 신호인데 퍼센타일이 낮음({pct2}) — 배선 오류"
    _log(f"[self-test] 통과: 사후 오라클 신호는 매우 높은 퍼센타일({pct2:.1f})")

    # Gate2: 필터=기준선(h=1 항상)이면 ΔUlcer가 0이어야 하니 CI도 0 근처(포함)여야 하고,
    # 필터가 오라클 신호(명백히 유리)면 CI가 0을 배제(양수)해야 함.
    g_null = gate2_bootstrap(frame, h_always, block_lens=(21,), n_rep=500, seed=2)
    ci_null = g_null["block21"]["delta_ulcer_ci95"]
    assert ci_null[0] - 1e-9 <= 0.0 <= ci_null[1] + 1e-9, f"h=1(기준선과 동일)인데 CI가 0을 포함 안 함: {ci_null}"
    _log(f"[self-test] 통과: 필터=기준선이면 Gate2 CI가 0을 포함({ci_null})")

    g_oracle2 = gate2_bootstrap(frame, h_oracle, block_lens=(21,), n_rep=500, seed=2)
    ci_oracle = g_oracle2["block21"]["delta_ulcer_ci95"]
    assert ci_oracle[0] > 0, f"오라클 신호인데 Gate2 CI가 0을 배제 안 함(양수 아님): {ci_oracle}"
    _log(f"[self-test] 통과: 오라클 신호는 Gate2 CI가 0을 배제(양수)({ci_oracle})")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
