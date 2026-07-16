#!/usr/bin/env python3
"""
us_core_satellite_ratio.py — 지호 님 질문(2026-07-17) 대응: "미국도 코어:새틀라이트
비율을 백테스트로 검증했나?" STRATEGY.md 확인 결과 §2(미국)에는 "코어" 언급이 전혀
없었음(§3/§3.5 코어-새틀라이트는 전부 한국 전용 — core_satellite_kr.py·
kr_topn_ratio_sweep.py). 지호 님이 기억한 "며칠 전 미국 비율 검증"은 별개 개념인
분할매수 비율(entry tranche, §2 line 148)이었을 가능성이 높음. 이 스크립트가 진짜
공백(포트폴리오 레벨 SPY 코어 : 커스텀종목 새틀라이트 비율)을 메운다.

방법(core_satellite_kr.py의 Stage 2 ratio와 동일 로직, 시장만 미국으로 교체):
  코어 = SPY × §1 레짐 타이밍(200일선 ±1% 히스테리시스·3일 확인)
  새틀라이트 = topn=8 커스텀 종목선정(현재 라이브 설정, §2 topN 재검증 champion) NAV
  코어비중 w = 1.0/0.8/0.65/0.5/0.35/0.0 스윕, SPY 매수후보유 대비 월간초과수익 PBO/DSR

실행: python us_core_satellite_ratio.py
결과: output/us_ratio_sweep.json · output/pbo_report_us_ratio.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

import backtest_costs as BC
import overfit_stats as OS
import backtest_portfolio as BP
import backtest_weights as BW
import core_satellite_kr as CS  # regime_series/timed_nav/mix_nav/stats — 시장 무관 범용 로직

RATIO_LIST = [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0]
TOPN = 8  # 현재 라이브 champion(STRATEGY.md §2 topN 재검증 v2)

# 2026-07-17 Fable 5 자문 반영: 샤프·MDD·서브기간 샤프만으로는 "분산=방어력" 직관과
# 어긋나는 결과(코어비중 낮을수록 단조 개선)의 원인을 못 가림 — (A)코어-새틀라이트
# 상관관계가 이미 높아 분산효과가 애초에 작은지, (B)이 특정 코어(레짐타이밍 SPY)가
# 급락구간에서 whipsaw로 오히려 못 버티는지 이벤트 단위로 확인 필요.
CRASH_WINDOWS = [("2018년 12월 급락", "2018-11-20", "2018-12-26"),
                  ("2020 코로나 급락", "2020-02-19", "2020-03-23"),
                  ("2022 약세장", "2022-01-01", "2022-10-14")]


def _log(m): print(f"[US코어새틀]  {m}", file=sys.stderr)


def _corr_diag(core: pd.Series, sat: pd.Series, spy_aligned: pd.Series) -> dict:
    """코어-새틀라이트 일간수익률 상관계수 — 전체 vs SPY 하락일 조건부(위기 시 상관관계
    붕괴 여부 확인)."""
    idx = core.index.intersection(sat.index)
    rc, rs = core.reindex(idx).pct_change().dropna(), sat.reindex(idx).pct_change().dropna()
    idx2 = rc.index.intersection(rs.index)
    rc, rs = rc.reindex(idx2), rs.reindex(idx2)
    corr_all = float(rc.corr(rs))
    r_spy = spy_aligned.reindex(idx2).pct_change()
    down_mask = r_spy < 0
    corr_down = float(rc[down_mask].corr(rs[down_mask])) if down_mask.sum() > 20 else None
    return {"corr_full": round(corr_all, 3),
            "corr_spy_down_days": round(corr_down, 3) if corr_down is not None else None,
            "n_down_days": int(down_mask.sum())}


def _crash_stats(navs: dict) -> dict:
    """고정 캘린더 구간별 최대낙폭(구간 내 peak-to-trough) — 서브기간 연 단위 평균이
    뭉개는 이벤트 단위 방어력을 직접 비교."""
    out = {}
    for label, a, b in CRASH_WINDOWS:
        row = {}
        for name, nav in navs.items():
            w = nav.loc[a:b]
            if len(w) < 5:
                row[name] = None
                continue
            row[name] = round(100 * float((w / w.cummax() - 1).min()), 1)
        out[label] = row
    return out


def _underwater_diag(nav: pd.Series) -> dict:
    """Ulcer Index(드로다운 크기+지속기간을 함께 벌점) + 최장 언더워터 일수."""
    dd = (nav / nav.cummax() - 1) * 100
    ulcer = round(float(np.sqrt((dd ** 2).mean())), 2)
    underwater = dd < -0.01
    max_run, cur = 0, 0
    for v in underwater:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    return {"ulcer_index": ulcer, "max_underwater_days": int(max_run)}


def _monthly_returns(nav: pd.Series) -> list:
    return [float(nav.iloc[t + BP.MONTH] / nav.iloc[t] - 1) for t in range(0, len(nav) - BP.MONTH, BP.MONTH)]


def _load_core_sat(core_ticker="SPY", years=10):
    """패널(S&P500 새틀라이트 종목 유니버스)+코어 지수 시세를 한 번만 로드해서 재사용
    (2026-07-17, 지호 님 요청 — 나스닥100(QQQ) 코어 비교 시 SPY와 같은 무거운 PIT 다운로드를
    반복하지 않도록). core_ticker만 바꾸면 코어 지수를 SPY 외 다른 걸로 교체 가능 — 새틀라이트
    (topn8 개별종목 유니버스·팩터)는 그대로, 코어 레짐타이밍 대상 지수만 바뀐다."""
    import sp500_daily_report as R
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    core_px = spy if core_ticker == "SPY" else R.download_histories([core_ticker], period=f"{years}y").get(core_ticker)
    if core_px is None:
        raise RuntimeError(f"{core_ticker} 시세 조회 실패")
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    ma200 = panel.rolling(200, min_periods=200).mean()
    decisions = BP.us_decisions(panel, funds, pit)
    sat_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False)
    if sat_nav is None:
        raise RuntimeError(f"topn={TOPN} 새틀라이트 NAV 산출 실패")
    core_aligned = core_px.reindex(panel.index).ffill()
    reg = CS.regime_series(core_aligned)
    core = CS.timed_nav(core_aligned, reg)
    sat = sat_nav / sat_nav.iloc[0]
    return panel, core, sat, core_aligned, cost


def _block_bootstrap_cagr_ci(monthly_rets: list, n_boot=5000, block=6, ci=0.95, seed=42) -> dict | None:
    """월간(비중첩) 수익률에서 블록 부트스트랩으로 CAGR 95% 신뢰구간 산출(2026-07-17, 지호
    님 요청 — 코어0%와 65% CAGR 차이가 통계적으로 유의한지). kr_topn_ratio_sweep.py의
    _block_bootstrap_sharpe_ci와 동일 방법론(block=6개월 단위 리샘플로 자기상관 보존),
    목적함수만 샤프 대신 CAGR로 교체. 리샘플된 12개월치 수익률의 기하평균을 연율화."""
    r = np.asarray(monthly_rets, dtype=float)
    n = len(r)
    if n < block * 2:
        return None
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block))
    cagrs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        compounded = np.prod(1 + sample)
        yrs = n / 12
        cagrs.append(float(compounded ** (1 / yrs) - 1) * 100)
    cagrs = np.array(cagrs)
    lo, mid, hi = np.percentile(cagrs, [(1 - ci) / 2 * 100, 50, (1 + ci) / 2 * 100])
    return {"lo": round(float(lo), 2), "median": round(float(mid), 2), "hi": round(float(hi), 2),
           "n_boot": n_boot, "block": block, "ci": ci, "n_months": n}


def _cagr_from_monthly(sample: np.ndarray) -> float:
    yrs = len(sample) / 12
    return float(np.prod(1 + sample) ** (1 / yrs) - 1) * 100


def run_paired_diff(w1=0.0, w2=0.65, n_boot=5000, block=6, seed=42, save=True,
                    core_ticker="SPY", loaded=None):
    """짝지은(paired) 블록부트스트랩: 매 회차 동일한 블록 시작 인덱스를 두 비중 모두에
    적용해 CAGR(w1)-CAGR(w2) 차이값 분포를 직접 산출(2026-07-17, 지호 님 요청). 두 CI를
    따로 구해 겹치는지 보는 것보다 검정력이 높다 — 두 시계열의 공통 변동(같은 시장 국면을
    같이 겪는 상관관계)을 상쇄해서 순수한 '비중 차이 자체의 효과'만 남기기 때문.
    loaded=(panel,core,sat,core_aligned)를 주면 재로드 생략(run_all에서 재사용)."""
    panel, core, sat, core_aligned, _cost = loaded or _load_core_sat(core_ticker)

    def _mixed(w):
        return core if w == 1 else (sat if w == 0 else CS.mix_nav(core, sat, w))

    r1 = np.asarray(_monthly_returns(_mixed(w1)), dtype=float)
    r2 = np.asarray(_monthly_returns(_mixed(w2)), dtype=float)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    if n < block * 2:
        raise RuntimeError("표본 부족")

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        cagr1 = _cagr_from_monthly(r1[idx])
        cagr2 = _cagr_from_monthly(r2[idx])
        diffs[i] = cagr1 - cagr2

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = float(diffs.mean())
    pct_positive = float((diffs > 0).mean()) * 100
    _log(f"CAGR({int(w1*100)}:{int((1-w1)*100)}) - CAGR({int(w2*100)}:{int((1-w2)*100)}) "
         f"차이값 분포: 평균 {mean_diff:+.2f}%p · 95% CI [{lo:+.2f}%p, {hi:+.2f}%p] · "
         f"{n_boot}회 중 {pct_positive:.1f}%가 양수({int(w1*100)}:{int((1-w1)*100)} 승리)")
    payload = {"core_ticker": core_ticker, "w1": w1, "w2": w2, "n_boot": n_boot,
              "block_months": block, "n_months": n,
              "mean_diff_pct": round(mean_diff, 2), "ci95_lo": round(float(lo), 2),
              "ci95_hi": round(float(hi), 2), "pct_positive": round(pct_positive, 1)}
    if save:
        os.makedirs("output", exist_ok=True)
        suffix = "" if core_ticker == "SPY" else f"_{core_ticker.lower()}"
        path = f"output/us_ratio_paired_diff{suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_bootstrap_ci(weights=(1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1, 0.0),
                      n_boot=5000, block=6, ci=0.95, save=True, core_ticker="SPY", loaded=None):
    """각 코어비중별 CAGR 블록부트스트랩 95% CI + 0%와 65% 겹침 여부 확인."""
    panel, core, sat, core_aligned, _cost = loaded or _load_core_sat(core_ticker)

    rows = []
    for w in weights:
        mixed = CS.mix_nav(core, sat, w) if 0 < w < 1 else (core if w == 1 else sat)
        mrets = _monthly_returns(mixed)
        boot = _block_bootstrap_cagr_ci(mrets, n_boot=n_boot, block=block, ci=ci)
        if boot is None:
            continue
        core_pct, sat_pct = int(round(w * 100)), int(round((1 - w) * 100))
        rows.append({"core_weight": w, "core_pct": core_pct, "sat_pct": sat_pct, **boot})
        _log(f"지수{core_pct}/새틀{sat_pct}: CAGR 95% CI [{boot['lo']:.1f}%, {boot['hi']:.1f}%] "
             f"(중앙값 {boot['median']:.1f}%)")

    by_core_pct = {r["core_pct"]: r for r in rows}
    r0, r65 = by_core_pct.get(0), by_core_pct.get(65)
    overlap = None
    if r0 and r65:
        overlap = not (r0["hi"] < r65["lo"] or r65["hi"] < r0["lo"])
        _log(f"0% CI [{r0['lo']:.1f}%, {r0['hi']:.1f}%] vs 65% CI [{r65['lo']:.1f}%, {r65['hi']:.1f}%] "
             f"→ {'겹침(통계적으로 구분 안 됨)' if overlap else '안 겹침(구분됨)'}")

    payload = {"as_of": panel.index[-1].date().isoformat(), "method": "block_bootstrap_cagr_ci",
              "core_ticker": core_ticker, "n_boot": n_boot, "block_months": block, "ci": ci,
              "rows": rows, "overlap_0_vs_65": overlap}
    if save:
        os.makedirs("output", exist_ok=True)
        suffix = "" if core_ticker == "SPY" else f"_{core_ticker.lower()}"
        path = f"output/us_ratio_cagr_ci{suffix}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run(years=10, save=True, core_ticker="SPY", loaded=None):
    panel, core, sat, core_aligned, cost = loaded or _load_core_sat(core_ticker, years)

    rows, matrix, dates0 = [], [], None
    subs_out = {}
    for w in RATIO_LIST:
        mixed = CS.mix_nav(core, sat, w) if 0 < w < 1 else (core if w == 1 else sat)
        subs = {tag: CS.stats(mixed, a, b) for tag, a, b in CS.SUBS}
        subs_out[w] = subs
        f = subs["full"]
        if f is None:
            continue
        rows.append({"core_weight": w, **f})
        d, r = BP.monthly_excess(mixed, core_aligned.reindex(mixed.index).ffill())
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        _log(f"core={w:.2f}: CAGR {f['cagr_pct']:6.2f}% 샤프 {f['sharpe']:5.2f} MDD {f['mdd_pct']:6.1f}% "
             f"· 서브기간 샤프 {subs['2018-2021'] and subs['2018-2021']['sharpe']}/"
             f"{subs['2022-2023'] and subs['2022-2023']['sharpe']}/{subs['2024+'] and subs['2024+']['sharpe']}")
    if len(rows) < 2:
        raise RuntimeError("ratio 결과 부족")

    corr = _corr_diag(core, sat, core_aligned)
    _log(f"코어-새틀라이트 상관계수: 전체 {corr['corr_full']} · {core_ticker}하락일 조건부 "
         f"{corr['corr_spy_down_days']} (n={corr['n_down_days']})")
    navs_for_crash = {f"core({core_ticker}레짐)": core, "satellite(topn8)": sat,
                       f"{core_ticker.lower()}_buyhold": core_aligned / core_aligned.iloc[0],
                       "mix0.65": CS.mix_nav(core, sat, 0.65)}
    crash = _crash_stats(navs_for_crash)
    for label, row in crash.items():
        _log(f"{label}: " + " · ".join(f"{k}={v}%" for k, v in row.items()))
    underwater = {name: _underwater_diag(nav) for name, nav in navs_for_crash.items()}
    for name, u in underwater.items():
        _log(f"{name}: Ulcer {u['ulcer_index']} · 최장 언더워터 {u['max_underwater_days']}일")

    n_ev = min(len(r) for r in matrix)
    trial_data = {"horizon": "us_ratio", "universe": "sp500_pit", "cost": cost.describe(),
                 "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev], "trials": [f"core{r['core_weight']:.2f}" for r in rows],
                 "excess_returns": [m[:n_ev] for m in matrix]}
    rpt = OS.analyze(trial_data, save=False)
    payload = {"as_of": panel.index[-1].date().isoformat(), "topn": TOPN, "satellite": "topn8_custom",
              "core_ticker": core_ticker,
              "judgment": f"{core_ticker} 매수후보유 대비 월간초과수익 PBO/DSR", "rows": rows,
              "subperiods": {str(w): s for w, s in subs_out.items()},
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "correlation": corr, "crash_windows": crash, "underwater": underwater}
    if save:
        os.makedirs("output", exist_ok=True)
        suffix = "" if core_ticker == "SPY" else f"_{core_ticker.lower()}"
        with open(f"output/us_ratio_sweep{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(f"output/pbo_report_us_ratio{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/us_ratio_sweep{suffix}.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


def run_all(core_ticker="SPY", years=10):
    """패널을 한 번만 로드해서 표(run)+CAGR CI(run_bootstrap_ci)+0%대65% 짝지은차이
    (run_paired_diff)를 전부 실행 — 지호 님 요청(2026-07-17, 나스닥100(QQQ) 코어 비교를
    백그라운드에서 한 번에)."""
    loaded = _load_core_sat(core_ticker, years)
    sweep = run(years, save=True, core_ticker=core_ticker, loaded=loaded)
    ci = run_bootstrap_ci(save=True, core_ticker=core_ticker, loaded=loaded)
    diff = run_paired_diff(save=True, core_ticker=core_ticker, loaded=loaded)
    return {"sweep": sweep, "ci": ci, "diff": diff}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-ci", action="store_true",
                    help="코어비중별 CAGR 블록부트스트랩 95% CI (지호 님 요청, 2026-07-17)")
    ap.add_argument("--paired-diff", action="store_true",
                    help="0%%와 65%% CAGR 차이값 짝지은부트스트랩 분포 (지호 님 요청, 2026-07-17)")
    ap.add_argument("--all", action="store_true", help="표+CI+짝지은차이 전부 (패널 1회 로드)")
    ap.add_argument("--core-ticker", default="SPY", help="코어 지수 티커 (기본 SPY, 예: QQQ)")
    args = ap.parse_args()
    if args.all:
        run_all(core_ticker=args.core_ticker)
    elif args.paired_diff:
        run_paired_diff(core_ticker=args.core_ticker)
    elif args.bootstrap_ci:
        run_bootstrap_ci(core_ticker=args.core_ticker)
    else:
        run(core_ticker=args.core_ticker)
