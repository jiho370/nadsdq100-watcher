#!/usr/bin/env python3
"""
us_spmo_blend_prereg.py — SPMO 70:30 블렌드 사전등록 단독 재검증 (2026-07-23, 지호 님 요청)

배경: §6-C(scratchpad 분석, 미커밋)에서 topn8 알고리즘과 SPMO(S&P500 모멘텀 ETF)를 여러
비율(100:0~0:100)로 스캔해 40:60~70:30 구간이 평평한 고원(절대샤프 1.39~1.43, 현행
100:0은 1.285)을 이루는 걸 확인했고, 65:35 동일비율 직접비교에서 t=1.95(유의기준 1.96에
근접)까지 봤다. 이 결과로 daily_ai_report.py 보유현황 차트에 "알고리즘70+SPMO30" 참고선을
추가했지만(§6-C "반영"), **어디까지나 여러 비율을 스캔해서 제일 좋아 보이는 지점 근방을
고른 사후 탐색**이라 데이터 스누핑 우려가 있었다 — STRATEGY.md §6-G 열린 실 ①:
"SPMO 70:30 블렌드 — 사전등록 단독 재검증 시 채택 여지".

이 스크립트는 그 사전등록을 실행한다: 그리드 재탐색 없이 **딱 하나의 비율(70:30, 이미
라이브 참고선으로 쓰이는 그 비율)만** 아래 판정규칙으로 검증한다.

⚠ 2026-07-23 수정: 최초 버전은 알고리즘 NAV를 `backtest_portfolio.us_decisions()`(=
`backtest_weights._z`, 전 팩터 ±3 클립)로 만들었는데, 실제 라이브 리포트(`export_data.
select_by_weights().z()`)는 §6-A 이후 shareholder_yield만 ±5로 완화돼 있어 미세하게
달랐다(지호 님 지적). `_select_basket_live_clip()`으로 그 클립 차이를 정확히 복제해
재실행 — "라이브 설정 그대로"라는 사전등록 전제를 문자 그대로 지키기 위함.

사전등록(실행 전 확정 — 결과를 보고 판정규칙을 바꾸지 않는다):
  가설: 알고리즘(topn8 라이브 설정) 70% + SPMO 30% 월간 리밸 블렌드가 순수 알고리즘
        (100:0, 현행 라이브)보다 위험조정 성과가 통계적으로 유의하게 낫다.
  비교 대상: 단 1개 지점(70:30) vs 기준(100:0) — 추가 비율 스윕 금지(스캔하면 이 검증
             자체가 다시 사후탐색이 됨).
  데이터: topn8 라이브설정(가중치는 best_weights.json 그대로 1:2:2, sector_cap=2,
          ma200_backup=False — §2/§6-A/§6-D 확정 설정) NAV. SPMO는 원시가격 NAV
          (레짐타이밍 미적용 — §6-C 원 분석의 "SPMO 절대샤프 1.193"과 동일 조건, 코어처럼
          200일선 타이밍을 SPMO에 걸지 않음). 상장(2015-10) 이후 최대 가용기간.
  방법: 월간(21거래일) 비중첩 수익률 페어드 t검정(§6-C의 "동일비율 직접비교 t=..."과 동일
        방식) + 짝지은 블록부트스트랩(6개월 블록, 5000회 — us_core_satellite_ratio.py
        run_paired_diff와 동일 방법론)로 CAGR·샤프 차이 분포 산출.
  판정규칙(셋 다 충족해야 "채택 후보" — 하나라도 미충족 시 "채택 보류/기각, 정보로만 기록"):
    ① CAGR 차이(블렌드-순수) 짝지은부트스트랩 95% CI 하한이 0보다 큼(=블렌드가 유의하게 우위)
    ② 페어드 t검정 t ≥ +1.96 (가설 방향, 양측 5% 상당)
    ③ 서브기간 두 구간(2018~2023 / 2024+) 모두 차이가 양수(블렌드 우위 방향 일관)
  주의: ①②③은 전부 "블렌드가 순수를 이긴다"는 가설 방향으로만 정의한다 — 유의하지만
  반대 방향(순수가 블렌드를 이김)이면 가설 기각이지 채택이 아니다.
  단일 사전등록 시행이라 PBO/DSR(다중검정 게이트)은 적용 대상 아님 — 그건 여러 후보를
  탐색할 때 사후탐색 편향을 보정하는 도구이고, 여기는 애초에 후보가 1개뿐이다.

실행: python us_spmo_blend_prereg.py
결과: output/us_spmo_blend_prereg.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import backtest_exec as BE
import overfit_stats as OS
import tech_factors as T
import core_satellite_kr as CS
import sp500_daily_report as R

TOPN = 8
W_ALGO = 0.70   # 사전등록: 70:30 이 지점 하나만 — 바꾸지 말 것
YEARS = 10
BLOCK = 6
N_BOOT = 5000
SEED = 42
SUBS = [("2018-2023", None, "2023-12-31"), ("2024+", "2024-01-01", None)]
RATIO_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]   # 탐색적 스윕 전용(§run_sweep)


def _log(m): print(f"[SPMO사전등록] {m}", file=sys.stderr)


def _sector_of_factory():
    sector_map = R.fetch_wikipedia_sectors()
    _log(f"위키 섹터맵 {len(sector_map)}종목 확보")
    return lambda date_s, sym: sector_map.get(sym)


def _z_live(col: pd.Series) -> pd.Series:
    """export_data.select_by_weights().z()와 동일 클립(shareholder_yield만 ±5, 나머지 ±3,
    §6-A) — backtest_weights._z(전 팩터 ±3 고정)와의 유일한 차이점."""
    sd = col.std()
    zz = (col - col.mean()) / sd if sd and not np.isnan(sd) else col * 0.0
    clip = (-5, 5) if col.name == "shareholder_yield" else (-3, 3)
    return zz.clip(*clip)


def _select_basket_live_clip(panel, p, funds, cross, pit, weights, topn):
    """backtest_exec._select_basket()과 동일하되 z-score만 _z_live로 교체."""
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
    z = raw[list(w)].apply(_z_live).fillna(0.0)
    score = (z * pd.Series(w)).sum(axis=1)
    return list(score.sort_values(ascending=False).index[:topn])


def _us_decisions_live_clip(panel, funds, pit, step=BP.MONTH):
    """backtest_portfolio.us_decisions()과 동일 구조, 클립만 라이브와 정확히 일치시킴."""
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    out = []
    for p in range(BW.LOOKBACK, len(panel) - 1, step):
        ranked = _select_basket_live_clip(panel, p, funds, cross, pit, weights, BP.POOL_SIZE)
        if ranked:
            out.append((p, ranked))
    _log(f"미장 결정 시점 {len(out)}개(라이브클립 ±5 shareholder_yield 반영)")
    return out


def _load(years=YEARS):
    """topn8 라이브설정 알고리즘 NAV + SPMO 원시가격 NAV(레짐타이밍 없음) + SPY 벤치마크,
    공통일자 정규화. years 지정 시 기본 10년 대신 다른 창 사용(예: §6-O-1의 11년=2016+)."""
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    ma200 = panel.rolling(200, min_periods=200).mean()
    decisions = _us_decisions_live_clip(panel, funds, pit)   # best_weights.json(1:2:2) + 라이브 클립(±5 sy)
    sector_of = _sector_of_factory()
    algo_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False,
                           sector_of=sector_of, sector_cap=2)
    if algo_nav is None:
        raise RuntimeError("topn=8 알고리즘 NAV 산출 실패")

    spmo_hist = R.download_histories(["SPMO"], period="max").get("SPMO")
    if spmo_hist is None or spmo_hist.empty:
        raise RuntimeError("SPMO 시세 조회 실패")

    idx = algo_nav.index.intersection(spmo_hist.reindex(algo_nav.index).ffill().dropna().index)
    if len(idx) < 60:
        raise RuntimeError(f"알고리즘-SPMO 공통 구간 부족(n={len(idx)})")
    algo_nav = algo_nav.reindex(idx)
    spmo_nav = spmo_hist.reindex(idx)
    spy_aligned = spy.reindex(idx).ffill()
    algo_nav = algo_nav / algo_nav.iloc[0]
    spmo_nav = spmo_nav / spmo_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")
    return algo_nav, spmo_nav, spy_aligned, cost


def _monthly_returns(nav: pd.Series) -> np.ndarray:
    return np.array([nav.iloc[t + BP.MONTH] / nav.iloc[t] - 1
                     for t in range(0, len(nav) - BP.MONTH, BP.MONTH)])


def _cagr_from_monthly(sample: np.ndarray) -> float:
    yrs = len(sample) / 12
    return float(np.prod(1 + sample) ** (1 / yrs) - 1) * 100


def _sharpe_from_monthly(sample: np.ndarray) -> float:
    return float(sample.mean() / sample.std() * np.sqrt(12)) if sample.std() else 0.0


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """scipy 미의존 페어드 t검정(project 관행 — score_calibration.py와 동일 이유).
    양측 p값은 정규근사(자유도 큰 표본에서 t분포≈정규분포)로 근사."""
    d = a - b
    n = len(d)
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = float(d.mean()) / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def run(save=True):
    algo_nav, spmo_nav, _spy, cost = _load()
    blend_nav = CS.mix_nav(algo_nav, spmo_nav, W_ALGO)   # 70% algo + 30% SPMO, 월간 리밸

    pure_full = CS.stats(algo_nav)
    blend_full = CS.stats(blend_nav)
    _log(f"순수(100:0, 현행): CAGR {pure_full['cagr_pct']}% 샤프 {pure_full['sharpe']} "
         f"MDD {pure_full['mdd_pct']}%")
    _log(f"블렌드(70:30): CAGR {blend_full['cagr_pct']}% 샤프 {blend_full['sharpe']} "
         f"MDD {blend_full['mdd_pct']}%")

    r_pure = _monthly_returns(algo_nav)
    r_blend = _monthly_returns(blend_nav)
    n = min(len(r_pure), len(r_blend))
    r_pure, r_blend = r_pure[:n], r_blend[:n]

    # ② 페어드 t검정(월간 수익률 차이, §6-C "동일비율 직접비교"와 동일 방식)
    tstat, pval = _paired_ttest(r_blend, r_pure)
    _log(f"페어드 t검정(월간수익률, n={n}): t={tstat:+.2f} p={pval:.3f}")

    # ① 짝지은 블록부트스트랩 CAGR·샤프 차이(us_core_satellite_ratio.run_paired_diff와 동일 방법론)
    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    cagr_diffs = np.empty(N_BOOT)
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = _cagr_from_monthly(r_blend[bidx]) - _cagr_from_monthly(r_pure[bidx])
        sharpe_diffs[i] = _sharpe_from_monthly(r_blend[bidx]) - _sharpe_from_monthly(r_pure[bidx])

    cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    cagr_mean = float(cagr_diffs.mean())
    cagr_pos = float((cagr_diffs > 0).mean()) * 100
    sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    sharpe_mean = float(sharpe_diffs.mean())

    _log(f"CAGR 차이(블렌드-순수) 95%CI: [{cagr_lo:+.2f}%p, {cagr_hi:+.2f}%p] (평균 {cagr_mean:+.2f}%p, "
         f"{N_BOOT}회 중 {cagr_pos:.1f}%가 양수)")
    _log(f"샤프 차이(블렌드-순수) 95%CI: [{sharpe_lo:+.3f}, {sharpe_hi:+.3f}] (평균 {sharpe_mean:+.3f})")

    # ③ 서브기간 방향 일치
    sub_rows = []
    signs = []
    for label, a, b in SUBS:
        sp_pure = CS.stats(algo_nav, a, b)
        sp_blend = CS.stats(blend_nav, a, b)
        if sp_pure is None or sp_blend is None:
            sub_rows.append({"period": label, "note": "표본 부족"})
            continue
        d_cagr = sp_blend["cagr_pct"] - sp_pure["cagr_pct"]
        d_sharpe = sp_blend["sharpe"] - sp_pure["sharpe"]
        signs.append(d_cagr > 0)
        sub_rows.append({"period": label, "pure_cagr": sp_pure["cagr_pct"],
                         "blend_cagr": sp_blend["cagr_pct"], "cagr_diff": round(d_cagr, 2),
                         "pure_sharpe": sp_pure["sharpe"], "blend_sharpe": sp_blend["sharpe"],
                         "sharpe_diff": round(d_sharpe, 3)})
        _log(f"{label}: CAGR차이 {d_cagr:+.2f}%p · 샤프차이 {d_sharpe:+.3f}")
    # 가설 방향(블렌드가 순수를 이긴다) 전부 양수여야 "방향 일관" — 전부 음수로 일관돼도
    # 그건 가설 기각의 일관성이지 채택 근거가 아니므로 all(signs)만 인정한다.
    subperiod_consistent = len(signs) >= 2 and all(signs)

    # 사전등록 판정규칙 적용(전부 "블렌드가 순수를 이긴다" 방향으로 정의 — §본문 주의 참고)
    gate1_ci_positive = cagr_lo > 0
    gate2_ttest_sig = tstat >= 1.96
    gate3_subperiod = subperiod_consistent
    passed = gate1_ci_positive and gate2_ttest_sig and gate3_subperiod
    direction = "블렌드 우위" if cagr_mean > 0 else "순수(현행) 우위"
    rejected_opposite = (not passed) and cagr_hi < 0   # CI 전체가 음수 = 유의하게 반대방향
    verdict = "채택 후보" if passed else ("가설 기각(유의하게 반대방향)" if rejected_opposite else "채택 보류")

    _log(f"판정 — ①CI하한>0:{gate1_ci_positive} ②t≥+1.96:{gate2_ttest_sig}({tstat:+.2f}) "
         f"③서브기간 블렌드우위 일관:{gate3_subperiod} → 방향:{direction} → 최종:{verdict}")

    payload = {
        "as_of": algo_nav.index[-1].date().isoformat(),
        "n_months": n,
        "cost": cost.describe(),
        "prereg": {
            "hypothesis": "algo(topn8 라이브)70% + SPMO30% 월간리밸 블렌드가 순수(100:0)보다 유의하게 낫다",
            "single_ratio_tested": W_ALGO,
            "decision_rule": "①CAGR차이 95%CI 하한>0 AND ②paired t>=+1.96 AND ③서브기간 둘 다 블렌드우위 — 셋 다 충족해야 채택후보(전부 가설방향 기준)",
        },
        "full_period": {"pure": pure_full, "blend": blend_full},
        "paired_ttest": {"t": round(float(tstat), 3), "p": round(float(pval), 4), "n": n},
        "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1),
                                "n_boot": N_BOOT, "block_months": BLOCK},
        "sharpe_diff_bootstrap": {"mean": round(sharpe_mean, 3), "ci95_lo": round(sharpe_lo, 3),
                                  "ci95_hi": round(sharpe_hi, 3), "n_boot": N_BOOT, "block_months": BLOCK},
        "subperiods": sub_rows,
        "gates": {"g1_ci_lo_positive": gate1_ci_positive,
                 "g2_ttest_significant_favorable": gate2_ttest_sig,
                 "g3_subperiod_consistent_favorable": gate3_subperiod},
        "direction": direction,
        "passed": passed,
        "verdict": verdict,
        "note": "단일 사전등록 시행(그리드 재탐색 없음) — PBO/DSR 다중검정 게이트 적용 대상 아님. "
                "판정 게이트는 전부 '블렌드가 순수를 이긴다'는 가설 방향으로 정의 — 유의하되 반대방향이면 채택이 아니라 기각.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_blend_prereg.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_sweep(save=True):
    """탐색적 스윕(사전등록 아님) — 2026-07-23, 지호 님이 "80:20은 어떨까"라고 물어서.
    70:30 하나만 기각됐다고 80:20을 또 따로 찍어서 테스트하면 §6-C가 저질렀던 것과
    똑같은 '안 되면 다른 비율 시도' 패턴이 된다. 대신 전체 비율(100:0~0:100)을 한 번에
    등록해 PBO/DSR로 다중검정을 정직하게 보정한다 — "80:20이 진짜 나은지"가 아니라
    "이 스윕에서 어느 지점이든 노이즈와 구분되는 진짜 우위가 있는지"를 묻는 질문으로
    바꾼 것. algo_nav(topn8 라이브+라이브클립)·spmo_nav는 run()과 동일 데이터 재사용."""
    algo_nav, spmo_nav, spy, cost = _load()
    rows, matrix, dates0 = [], [], None
    for w in RATIO_LIST:
        mixed = CS.mix_nav(algo_nav, spmo_nav, w) if 0 < w < 1 else (algo_nav if w == 1 else spmo_nav)
        s = CS.stats(mixed)
        algo_pct, spmo_pct = int(round(w * 100)), int(round((1 - w) * 100))
        rows.append({"algo_pct": algo_pct, "spmo_pct": spmo_pct, **s})
        d, r = BP.monthly_excess(mixed, spy)
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
        _log(f"algo{algo_pct}:spmo{spmo_pct}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}%")

    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]
    trial_data = {"horizon": "us_spmo_ratio_explore", "universe": "sp500_pit_topn8",
                 "cost": cost.describe(), "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev], "trials": [f"algo{r['algo_pct']}spmo{r['spmo_pct']}" for r in rows],
                 "excess_returns": matrix}
    rpt = OS.analyze(trial_data, save=False)
    by_pct = {r["algo_pct"]: r for r in rows}
    best = max(rows, key=lambda r: r["sharpe"])
    _log(f"샤프 최고점: algo{best['algo_pct']}:spmo{best['spmo_pct']} (샤프 {best['sharpe']}) · "
         f"PBO {rpt.get('pbo', {}).get('pbo')} · DSR {rpt.get('dsr', {}).get('dsr')} · "
         f"게이트 통과 {rpt.get('passed')}")

    payload = {"as_of": algo_nav.index[-1].date().isoformat(), "kind": "탐색적 스윕(사전등록 아님)",
              "rows": rows, "baseline": "algo100(현행 라이브, 순수)",
              "best_by_sharpe": {"algo_pct": best["algo_pct"], "spmo_pct": best["spmo_pct"],
                                 "sharpe": best["sharpe"]},
              "algo80_spmo20": by_pct.get(80),
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "note": "탐색적(사전등록 아님) — 여기서 뭘 발견해도 그 자체를 근거로 라이브에 "
                      "반영하면 안 되고, 발견한 지점을 다시 사전등록해 별도 표본으로 재검증해야 "
                      "함(§4 원칙과 동일). PBO/DSR 미통과면 '어느 비율이 나은지 이 표본으로는 "
                      "못 가린다'는 뜻."}
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_ratio_explore.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def _worst_drawdown_window(nav: pd.Series) -> tuple:
    """peak-to-trough 최대낙폭 구간의 (peak일, trough일) 반환."""
    cummax = nav.cummax()
    dd = nav / cummax - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    return peak, trough


def _rolling_return(nav: pd.Series, window_days: int) -> pd.Series:
    return nav / nav.shift(window_days) - 1


def run_tailrisk(save=True):
    """"과최적화된 알고리즘이 부진한 시기에 SPMO를 섞으면 완충되는가" — 2026-07-23, 지호
    님 지적("같은 과거 데이터로는 과최적화 헷지 효과 자체는 안 보일 것") 대응.
    평균 CAGR 비교(run/run_sweep)로는 이 질문에 답할 수 없다 — 그건 "블렌드가 과거
    수익을 깎는가"를 보는 것이지 "알고리즘이 무너질 때 블렌드가 덜 다치게 하는가"를
    보는 게 아니다. 이 함수는 후자에 최대한 근접한 대리지표 셋을 계산한다:
      ① 알고리즘 자체의 역사상 최대낙폭 구간(peak~trough)에서, 그 구간 동안 순수 vs
         70:30 vs 80:20 블렌드가 각각 얼마나 빠졌는가(같은 창을 그대로 재생, 사후적으로
         "그때 섞었으면" 시뮬레이션 — 실시간 헤지 효과가 아니라 과거 재생임에 유의)
      ② 롤링 12개월 수익률의 최저점(최악의 1년 성과) 비교
      ③ CVaR95(월간 수익률 하위 5% 평균, 꼬리위험)
    ⚠ 한계: 이것도 여전히 같은 과거 데이터 안에서 계산되므로 '진짜 미래 과최적화 붕괴를
    막아주는가'의 증거는 아니다 — 알고리즘의 실제 과거 최악의 구간에서 블렌드가 방석
    역할을 했는지를 보는 대리(proxy) 지표일 뿐. 진짜 답은 라이브 가중치(2026-07-18
    확정)의 실제 향후 성과로만 나온다."""
    algo_nav, spmo_nav, spy, cost = _load()
    blends = {"70:30": CS.mix_nav(algo_nav, spmo_nav, 0.70), "80:20": CS.mix_nav(algo_nav, spmo_nav, 0.80)}

    # ① 알고리즘 자체의 최대낙폭 구간에서 각 구성의 낙폭
    peak, trough = _worst_drawdown_window(algo_nav)
    _log(f"알고리즘 자체 최대낙폭 구간: {peak.date()} ~ {trough.date()}")
    dd_window = {}
    for name, nav in [("순수(100:0)", algo_nav)] + list(blends.items()):
        w = nav.loc[peak:trough]
        dd = float((w.iloc[-1] / w.iloc[0] - 1) * 100)
        dd_window[name] = round(dd, 1)
        _log(f"  {name}: 그 구간 수익률 {dd:+.1f}%")

    # ② 롤링 12개월 수익률 최저점
    worst_12m = {}
    for name, nav in [("순수(100:0)", algo_nav)] + list(blends.items()):
        r12 = _rolling_return(nav, 252).dropna()
        worst_12m[name] = round(float(r12.min() * 100), 1)
        _log(f"  {name} 최악의 롤링12개월: {worst_12m[name]:+.1f}%")

    # ③ CVaR95(월간, 하위 5% 평균)
    cvar95 = {}
    for name, nav in [("순수(100:0)", algo_nav)] + list(blends.items()):
        rm = _monthly_returns(nav)
        thresh = np.percentile(rm, 5)
        tail = rm[rm <= thresh]
        cvar95[name] = round(float(tail.mean() * 100), 2)
        _log(f"  {name} CVaR95(월간): {cvar95[name]:+.2f}%")

    payload = {
        "as_of": algo_nav.index[-1].date().isoformat(),
        "kind": "꼬리위험/최악구간 대리지표 — 과최적화 헷지 자체의 증명 아님(설명 참고)",
        "caveat": "같은 과거 데이터 안에서 '알고리즘이 이미 겪은 최악의 구간'을 재생한 것 — "
                  "미래에 알고리즘이 새로 겪을 미지의 부진(진짜 과최적화 리스크)을 막아주는지는 "
                  "라이브 실적이 쌓이기 전엔 증명 불가능. 방향성 참고용.",
        "algo_worst_drawdown_window": {"peak": peak.date().isoformat(), "trough": trough.date().isoformat()},
        "return_during_algo_worst_window_pct": dd_window,
        "worst_rolling_12m_return_pct": worst_12m,
        "cvar95_monthly_pct": cvar95,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_tailrisk.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_vs_brka(save=True):
    """알고리즘·SPY·SPMO를 버크셔 해서웨이 A주(BRK-A)와 비교(2026-07-23, 지호 님 요청).
    순수 서술 비교 — 가설검정·게이트 없음. algo_nav/spmo_nav/spy는 run()·run_sweep()과
    동일 파이프라인(라이브 클립 반영) 재사용, BRK-A만 신규 다운로드해 같은 공통구간에 정렬."""
    algo_nav, spmo_nav, spy, cost = _load()

    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")

    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    if len(idx) < 60:
        raise RuntimeError(f"공통구간 부족(n={len(idx)})")
    series = {
        "알고리즘(topn8)": algo_nav.reindex(idx),
        "SPY": spy.reindex(idx).ffill(),
        "SPMO": spmo_nav.reindex(idx),
        "BRK-A": brka_hist.reindex(idx),
    }
    series = {k: v / v.iloc[0] for k, v in series.items()}
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    full_rows = {}
    for name, nav in series.items():
        s = CS.stats(nav)
        full_rows[name] = s
        _log(f"{name}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}% "
             f"· 누적 {round((nav.iloc[-1]-1)*100, 1)}%")

    sub_rows = {}
    for label, a, b in SUBS:
        row = {}
        for name, nav in series.items():
            s = CS.stats(nav, a, b)
            row[name] = s
        sub_rows[label] = row
        _log(f"[{label}] " + " · ".join(f"{k}:CAGR{v['cagr_pct']}%/샤프{v['sharpe']}"
                                        for k, v in row.items() if v))

    payload = {
        "as_of": idx[-1].date().isoformat(), "n_days": len(idx),
        "kind": "서술 비교(가설검정 아님)",
        "full_period": full_rows,
        "subperiods": sub_rows,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_algo_vs_brka.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_brka_hedge_compare(save=True):
    """"SPMO 대신 BRK-A를 섞으면 꼬리위험 방어가 더 나은가" 검증(2026-07-23, 지호 님 요청
    — 직전 비교에서 나온 아이디어). SPMO는 모멘텀/성장 팩터 ETF라 알고리즘(팩터 기반
    종목선정)과 상관이 이미 꽤 높을 수 있고, BRK-A는 가치·보험·현금성자산 위주라 성격이
    다른 분산 효과를 낼 수 있다는 가설. run_tailrisk()와 같은 3개 대리지표 + 상관계수
    진단 + 평균 CAGR 비용(페어드 부트스트랩, 70:30 단일비율 사전등록)을 SPMO와 나란히
    비교한다. ⚠ 같은 한계: 같은 과거 데이터 재생이라 미래 과최적화 방어의 증거는 아님,
    '이 표본에서 어느 쪽이 분산효과가 더 컸는가'의 서술 비교."""
    algo_nav, spmo_nav, spy, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spmo_nav = spmo_nav.reindex(idx); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    _log(f"공통 구간(algo·SPMO·BRK-A 전부): {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    # ── 상관계수(전체 vs 알고리즘 하락일 조건부) — 낮을수록 진짜 분산 재료
    r_algo = algo_nav.pct_change().dropna()
    r_spmo = spmo_nav.pct_change().reindex(r_algo.index)
    r_brka = brka_nav.pct_change().reindex(r_algo.index)
    down_mask = r_algo < 0
    corr = {
        "spmo_full": round(float(r_algo.corr(r_spmo)), 3),
        "spmo_algo_down_days": round(float(r_algo[down_mask].corr(r_spmo[down_mask])), 3),
        "brka_full": round(float(r_algo.corr(r_brka)), 3),
        "brka_algo_down_days": round(float(r_algo[down_mask].corr(r_brka[down_mask])), 3),
    }
    _log(f"상관계수(알고리즘 대비): SPMO 전체 {corr['spmo_full']}/하락일 {corr['spmo_algo_down_days']} · "
         f"BRK-A 전체 {corr['brka_full']}/하락일 {corr['brka_algo_down_days']}")

    # ── 꼬리위험 대리지표 3종(run_tailrisk과 동일 방법론), SPMO·BRK-A 70:30·80:20 나란히
    variants = {
        "SPMO 70:30": CS.mix_nav(algo_nav, spmo_nav, 0.70),
        "SPMO 80:20": CS.mix_nav(algo_nav, spmo_nav, 0.80),
        "BRK-A 70:30": CS.mix_nav(algo_nav, brka_nav, 0.70),
        "BRK-A 80:20": CS.mix_nav(algo_nav, brka_nav, 0.80),
    }
    peak, trough = _worst_drawdown_window(algo_nav)
    _log(f"알고리즘 자체 최대낙폭 구간: {peak.date()} ~ {trough.date()}")

    tail = {}
    for name, nav in [("순수(100:0)", algo_nav)] + list(variants.items()):
        w = nav.loc[peak:trough]
        dd = float((w.iloc[-1] / w.iloc[0] - 1) * 100)
        r12 = _rolling_return(nav, 252).dropna()
        rm = _monthly_returns(nav)
        thresh = np.percentile(rm, 5)
        cvar = float(rm[rm <= thresh].mean() * 100)
        s = CS.stats(nav)
        tail[name] = {"cagr_pct": s["cagr_pct"], "sharpe": s["sharpe"],
                     "crash_window_return_pct": round(dd, 1),
                     "worst_rolling_12m_pct": round(float(r12.min() * 100), 1),
                     "cvar95_monthly_pct": round(cvar, 2)}
        _log(f"{name}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} · 위기구간 {dd:+.1f}% · "
             f"최악12개월 {tail[name]['worst_rolling_12m_pct']:+.1f}% · CVaR95 {cvar:+.2f}%")

    # ── 평균 CAGR 비용(70:30 단일비율, 페어드 블록부트스트랩 — SPMO 사전등록과 동일 방법론)
    def _cagr_cost(blend_nav):
        r_pure = _monthly_returns(algo_nav)
        r_blend = _monthly_returns(blend_nav)
        n = min(len(r_pure), len(r_blend))
        r_pure, r_blend = r_pure[:n], r_blend[:n]
        tstat, pval = _paired_ttest(r_blend, r_pure)
        rng = np.random.default_rng(SEED)
        n_blocks_needed = int(np.ceil(n / BLOCK))
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
            bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
            diffs[i] = _cagr_from_monthly(r_blend[bidx]) - _cagr_from_monthly(r_pure[bidx])
        lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
        return {"t": round(float(tstat), 3), "p": round(float(pval), 4),
               "cagr_diff_mean": round(float(diffs.mean()), 2), "ci95_lo": round(lo, 2), "ci95_hi": round(hi, 2)}

    cost70 = {"SPMO": _cagr_cost(variants["SPMO 70:30"]), "BRK-A": _cagr_cost(variants["BRK-A 70:30"])}
    _log(f"70:30 CAGR비용(블렌드-순수): SPMO 평균{cost70['SPMO']['cagr_diff_mean']:+.2f}%p "
         f"[{cost70['SPMO']['ci95_lo']:+.2f},{cost70['SPMO']['ci95_hi']:+.2f}] t={cost70['SPMO']['t']:+.2f} · "
         f"BRK-A 평균{cost70['BRK-A']['cagr_diff_mean']:+.2f}%p "
         f"[{cost70['BRK-A']['ci95_lo']:+.2f},{cost70['BRK-A']['ci95_hi']:+.2f}] t={cost70['BRK-A']['t']:+.2f}")

    payload = {
        "as_of": idx[-1].date().isoformat(), "kind": "서술비교 + 70:30 단일비율 CAGR비용 사전등록",
        "caveat": "꼬리위험 지표는 같은 과거 데이터의 알고리즘 자체 위기구간 재생 — 미래 "
                  "과최적화 방어의 증거 아님. 상관계수·CAGR비용은 통계적 근거 있음(부트스트랩).",
        "correlation_vs_algo": corr,
        "algo_worst_drawdown_window": {"peak": peak.date().isoformat(), "trough": trough.date().isoformat()},
        "tail_metrics": tail,
        "cagr_cost_70_30": cost70,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_brka_vs_spmo_hedge.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_brka_sharpe_prereg(save=True):
    """BRK-A 70:30 블렌드의 샤프비율이 순수 알고리즘(100:0)보다 유의하게 나은가 —
    사전등록 단일비율 검정(2026-07-23, 지호 님 요청). run_brka_hedge_compare()에서 점추정치
    샤프 1.15(블렌드) vs 1.13(순수)를 봤지만 유의성 검정 전이었다 — 이 대화에서 "그럴듯해
    보였지만 검정하니 유의하지 않았다"가 반복됐으므로(SPMO vs SPY 등) 같은 원칙 적용.

    사전등록(실행 전 확정):
      가설: BRK-A 70% + 알고리즘 70:30 블렌드의 샤프비율이 순수 알고리즘(100:0)보다 낫다.
      비율: 70:30 하나만(이미 앞서 쓴 비율 재사용 — 새 비율 스캔 안 함).
      방법: 짝지은 블록부트스트랩(6개월 블록, 5000회, run()의 SPMO 검정과 동일 방법론)으로
            샤프차이(블렌드-순수) 분포 산출. 샤프는 관측치별 페어드 t검정이 성립하지 않는
            비율 통계라 부트스트랩 CI만 판정 근거로 쓴다(CAGR처럼 t검정 병행 안 함).
      판정규칙(둘 다 충족해야 "채택 후보"):
        ① 샤프차이 95%CI 하한 > 0
        ② 서브기간(2018-2023 / 2024+) 둘 다 샤프차이 양수(방향 일관)
      단일 사전등록 시행 — PBO/DSR 대상 아님."""
    algo_nav, spmo_nav, spy, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    blend_nav = CS.mix_nav(algo_nav, brka_nav, 0.70)
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    pure_full, blend_full = CS.stats(algo_nav), CS.stats(blend_nav)
    _log(f"순수(100:0): 샤프 {pure_full['sharpe']} · BRK-A70:30: 샤프 {blend_full['sharpe']}")

    r_pure, r_blend = _monthly_returns(algo_nav), _monthly_returns(blend_nav)
    n = min(len(r_pure), len(r_blend))
    r_pure, r_blend = r_pure[:n], r_blend[:n]

    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        sharpe_diffs[i] = _sharpe_from_monthly(r_blend[bidx]) - _sharpe_from_monthly(r_pure[bidx])
    lo, hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    mean_diff = float(sharpe_diffs.mean())
    pct_pos = float((sharpe_diffs > 0).mean()) * 100
    _log(f"샤프차이(블렌드-순수) 95%CI: [{lo:+.3f}, {hi:+.3f}] (평균 {mean_diff:+.3f}, "
         f"{N_BOOT}회 중 {pct_pos:.1f}%가 양수)")

    sub_rows, signs = [], []
    for label, a, b in SUBS:
        sp_pure, sp_blend = CS.stats(algo_nav, a, b), CS.stats(blend_nav, a, b)
        if sp_pure is None or sp_blend is None:
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        d = sp_blend["sharpe"] - sp_pure["sharpe"]
        signs.append(d > 0)
        sub_rows.append({"period": label, "pure_sharpe": sp_pure["sharpe"],
                         "blend_sharpe": sp_blend["sharpe"], "sharpe_diff": round(d, 3)})
        _log(f"{label}: 샤프차이 {d:+.3f}")
    subperiod_consistent = len(signs) >= 2 and all(signs)

    gate1 = lo > 0
    gate2 = subperiod_consistent
    passed = gate1 and gate2
    direction = "블렌드 우위" if mean_diff > 0 else "순수 우위"
    rejected_opposite = (not passed) and hi < 0
    verdict = "채택 후보" if passed else ("가설 기각(유의하게 반대방향)" if rejected_opposite else "판정 보류(유의한 차이 없음)")
    _log(f"판정 — ①CI하한>0:{gate1} ②서브기간 일관:{gate2} → 방향:{direction} → 최종:{verdict}")

    payload = {
        "as_of": idx[-1].date().isoformat(), "n_months": n,
        "prereg": {"hypothesis": "BRK-A70:알고리즘30 블렌드 샤프가 순수(100:0)보다 유의하게 낫다",
                  "ratio_tested": 0.70,
                  "decision_rule": "①샤프차이 95%CI 하한>0 AND ②서브기간 둘 다 블렌드우위"},
        "full_period": {"pure_sharpe": pure_full["sharpe"], "blend_sharpe": blend_full["sharpe"]},
        "sharpe_diff_bootstrap": {"mean": round(mean_diff, 3), "ci95_lo": round(lo, 3),
                                  "ci95_hi": round(hi, 3), "pct_positive": round(pct_pos, 1),
                                  "n_boot": N_BOOT, "block_months": BLOCK},
        "subperiods": sub_rows,
        "gates": {"g1_ci_lo_positive": gate1, "g2_subperiod_consistent": gate2},
        "direction": direction, "passed": passed, "verdict": verdict,
        "note": "단일 사전등록 시행 — PBO/DSR 대상 아님.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_brka_sharpe_prereg.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


DEGRADE_LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]   # 1.0=실제 그대로, 0.0=alpha 완전소멸


def run_degradation_stress(save=True):
    """알고리즘 '엣지 열화' 스트레스 테스트(2026-07-24, 지호 님 지적 대응) — "SPMO는 그
    자체가 모멘텀 팩터 베팅이라 알고리즘과 같이 무너질 수 있고(하락일 상관 0.767이 이미
    그 증거), 애초 목적(과최적화 방어)에 필요한 건 낮은 상관(BRK-A 0.37)이다. 다만
    지금까지의 검정(페어드 부트스트랩)은 전부 '백테스트 수익률이 그대로 실현된다'는
    전제 위에 있어 진짜 질문('그 수익률 자체가 실현 안 될 수도 있다')엔 안 맞는다 —
    알고리즘 엣지를 인위적으로 깎은 가상 시나리오로 답해야 한다"는 지적을 그대로 구현.

    방법: 알고리즘 일별수익률에서 SPY 대비 평균 초과수익(alpha_bar = mean(r_algo-r_spy))을
    추출, 매일 alpha_bar*(1-s)만큼 깎은 가상 수익률로 알고리즘 NAV를 재구성(s=1.0 실제
    그대로 ~ s=0.0 alpha 완전소멸 — "내 8종목 팩터 선정이 사실 노이즈였다"의 극단).
    일별 변동성·타이밍·SPMO/BRK-A와의 상관구조(공분산)는 전혀 안 건드리고 평균만
    평행이동하므로, 순수한 '엣지 축소'만의 영향을 분리해서 본다. 각 s에서 순수(100:0
    열화algo)·SPMO70:30·BRK-A70:30·BRK-A90:10을 비교.

    ⚠ 한계(정직하게 명시): ①평균만 축소하고 변동성·꼬리형태는 그대로라, "진짜 모델
    붕괴"가 동반할 수 있는 변동성 급증·꼬리위험 확대까지는 반영 못 함(더 보수적으로
    보려면 후속 작업으로 변동성도 함께 부풀리는 버전 필요). ②SPMO 고유의 '모멘텀
    크래시'(2009년 반등장 -30%대 낙폭 같은 급락 이벤트)는 이 표본 기간에 실제로
    없었으므로 이 스트레스 테스트에도 안 잡힘 — 별도 시나리오(SPMO 경로에 인위적
    크래시 삽입) 필요시 후속 요청."""
    algo_nav, spmo_nav, spy, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spmo_nav = spmo_nav.reindex(idx); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    spy_nav = spy.reindex(idx).ffill(); spy_nav = spy_nav / spy_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    r_algo = algo_nav.pct_change().fillna(0)
    r_spy = spy_nav.pct_change().fillna(0)
    alpha_bar = float((r_algo - r_spy).mean())
    _log(f"알고리즘 일평균 초과수익(SPY대비): {alpha_bar*100:.4f}%/일 "
         f"(연율화 근사 {alpha_bar*252*100:+.1f}%p)")

    rows = []
    for s in DEGRADE_LEVELS:
        r_algo_deg = r_algo - (1 - s) * alpha_bar
        algo_deg_nav = (1 + r_algo_deg).cumprod()
        algo_deg_nav.iloc[0] = 1.0

        configs = {
            "순수(100:0)": algo_deg_nav,
            "SPMO 70:30": CS.mix_nav(algo_deg_nav, spmo_nav, 0.70),
            "BRK-A 70:30": CS.mix_nav(algo_deg_nav, brka_nav, 0.70),
            "BRK-A 90:10": CS.mix_nav(algo_deg_nav, brka_nav, 0.90),
        }
        row = {"degrade_s": s, "label": f"엣지 {int(s*100)}%"}
        for name, nav in configs.items():
            st = CS.stats(nav)
            row[name] = st
        rows.append(row)
        _log(f"[엣지 {int(s*100)}%] " + " · ".join(
            f"{k}:CAGR{v['cagr_pct']}%/샤프{v['sharpe']}/MDD{v['mdd_pct']}%"
            for k, v in row.items() if isinstance(v, dict)))

    # s=0(완전소멸) 구간에서 각 블렌드가 순수 대비 얼마나 방어했는지 요약
    zero = next(r for r in rows if r["degrade_s"] == 0.0)
    cushion = {}
    for name in ["SPMO 70:30", "BRK-A 70:30", "BRK-A 90:10"]:
        cushion[name] = {
            "sharpe_diff_vs_pure": round(zero[name]["sharpe"] - zero["순수(100:0)"]["sharpe"], 3),
            "mdd_diff_vs_pure_pp": round(zero[name]["mdd_pct"] - zero["순수(100:0)"]["mdd_pct"], 1),
            "cagr_diff_vs_pure_pp": round(zero[name]["cagr_pct"] - zero["순수(100:0)"]["cagr_pct"], 2),
        }
        _log(f"[엣지 0%, 완전소멸] {name} vs 순수: 샤프차이 {cushion[name]['sharpe_diff_vs_pure']:+.3f} · "
             f"MDD차이 {cushion[name]['mdd_diff_vs_pure_pp']:+.1f}%p · CAGR차이 {cushion[name]['cagr_diff_vs_pure_pp']:+.2f}%p")

    payload = {
        "as_of": idx[-1].date().isoformat(),
        "kind": "가상 시나리오 스트레스 테스트(사전등록 아님) — 알고리즘 alpha 인위적 축소",
        "method": "r_algo_degraded(s) = r_algo - (1-s)*mean(r_algo-r_spy); 변동성·상관구조 불변, 평균만 이동",
        "limitations": ["변동성/꼬리위험 확대는 미반영(평균만 축소)",
                        "SPMO 고유 모멘텀크래시 이벤트는 표본 기간에 미발생·미반영"],
        "alpha_bar_daily_pct": round(alpha_bar * 100, 4),
        "alpha_bar_annualized_approx_pct": round(alpha_bar * 252 * 100, 1),
        "rows": rows,
        "cushion_at_full_degradation": cushion,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_algo_degradation_stress.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_joint_degradation_stress(save=True):
    """"공통 팩터리스크 동반붕괴" 시나리오(2026-07-24, 지호 님 요청 — run_degradation_stress
    의 한계 보완). 앞선 열화 테스트는 SPMO 경로를 실제 그대로 두고 알고리즘만 깎았는데,
    이건 지호 님의 핵심 우려("알고리즘=퀄리티/밸류 팩터·SPMO=모멘텀 팩터, 둘 다 '팩터
    레짐이 살아있을 때 작동하는 베팅'이라 구조가 같다 — 하락일 상관 0.767이 그 증거,
    레짐이 무너지면 SPMO도 같이 흔들릴 것")를 반영 못 했다. 여기서는 알고리즘과 SPMO
    둘 다의 alpha(각자 SPY 대비 평균초과수익)를 **동시에** 같은 비율로 깎는다(공통
    팩터리스크가 실현되면 두 팩터베팅 다 타격받는다는 가정을 직접 구현) — BRK-A는
    상관이 낮아(하락일 0.37) 이 공통리스크에서 열외라는 가정으로 실제 경로 그대로 둔다.

    이 시나리오에서 "SPMO가 자기 절대수익으로 방어해준다"는 이점(run_degradation_stress
    의 반전 원인)이 SPMO도 같이 깎이므로 사라진다 — 지호 님 우려가 맞다면 여기서
    BRK-A가 SPMO를 역전해야 한다."""
    algo_nav, spmo_nav, spy, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spmo_nav = spmo_nav.reindex(idx); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    spy_nav = spy.reindex(idx).ffill(); spy_nav = spy_nav / spy_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    r_algo = algo_nav.pct_change().fillna(0)
    r_spmo = spmo_nav.pct_change().fillna(0)
    r_spy = spy_nav.pct_change().fillna(0)
    alpha_algo = float((r_algo - r_spy).mean())
    alpha_spmo = float((r_spmo - r_spy).mean())
    _log(f"algo alpha(연율화 근사) {alpha_algo*252*100:+.1f}%p · SPMO alpha(연율화 근사) "
         f"{alpha_spmo*252*100:+.1f}%p — 둘 다 동시에 (1-s)만큼 깎음, BRK-A는 그대로")

    rows = []
    for s in DEGRADE_LEVELS:
        r_algo_deg = r_algo - (1 - s) * alpha_algo
        r_spmo_deg = r_spmo - (1 - s) * alpha_spmo
        algo_deg_nav = (1 + r_algo_deg).cumprod(); algo_deg_nav.iloc[0] = 1.0
        spmo_deg_nav = (1 + r_spmo_deg).cumprod(); spmo_deg_nav.iloc[0] = 1.0

        configs = {
            "순수(100:0)": algo_deg_nav,
            "SPMO 70:30(동반열화)": CS.mix_nav(algo_deg_nav, spmo_deg_nav, 0.70),
            "BRK-A 70:30(그대로)": CS.mix_nav(algo_deg_nav, brka_nav, 0.70),
            "BRK-A 90:10(그대로)": CS.mix_nav(algo_deg_nav, brka_nav, 0.90),
        }
        row = {"degrade_s": s, "label": f"엣지 {int(s*100)}%"}
        for name, nav in configs.items():
            row[name] = CS.stats(nav)
        rows.append(row)
        _log(f"[엣지 {int(s*100)}%] " + " · ".join(
            f"{k}:CAGR{v['cagr_pct']}%/샤프{v['sharpe']}/MDD{v['mdd_pct']}%"
            for k, v in row.items() if isinstance(v, dict)))

    zero = next(r for r in rows if r["degrade_s"] == 0.0)
    cushion = {}
    for name in ["SPMO 70:30(동반열화)", "BRK-A 70:30(그대로)", "BRK-A 90:10(그대로)"]:
        cushion[name] = {
            "sharpe_diff_vs_pure": round(zero[name]["sharpe"] - zero["순수(100:0)"]["sharpe"], 3),
            "mdd_diff_vs_pure_pp": round(zero[name]["mdd_pct"] - zero["순수(100:0)"]["mdd_pct"], 1),
            "cagr_diff_vs_pure_pp": round(zero[name]["cagr_pct"] - zero["순수(100:0)"]["cagr_pct"], 2),
        }
        _log(f"[엣지 0%] {name} vs 순수: 샤프차이 {cushion[name]['sharpe_diff_vs_pure']:+.3f} · "
             f"MDD차이 {cushion[name]['mdd_diff_vs_pure_pp']:+.1f}%p · CAGR차이 {cushion[name]['cagr_diff_vs_pure_pp']:+.2f}%p")

    payload = {
        "as_of": idx[-1].date().isoformat(),
        "kind": "가상 시나리오 스트레스 테스트(사전등록 아님) — 알고리즘+SPMO 동반 alpha 축소, BRK-A는 실제경로 유지",
        "assumption": "알고리즘·SPMO는 둘 다 '팩터 레짐 의존' 베팅이라 공통 팩터리스크 실현 시 "
                      "같이 타격받는다고 가정(하락일 상관 0.767 근거). BRK-A는 상관 낮아(0.37) 열외.",
        "alpha_algo_annualized_approx_pct": round(alpha_algo * 252 * 100, 1),
        "alpha_spmo_annualized_approx_pct": round(alpha_spmo * 252 * 100, 1),
        "rows": rows,
        "cushion_at_full_degradation": cushion,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_algo_joint_degradation_stress.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


CRASH_MAGNITUDE = -0.30   # 2009년 모멘텀 롱숏 크래시 규모(지호 님 인용) 근사
CRASH_DAYS = 21           # "한 달" 근사(거래일)


def _inject_crash(nav: pd.Series, start_idx: int, magnitude: float, days: int) -> pd.Series:
    """nav의 start_idx부터 days거래일에 걸쳐 총 magnitude(예: -0.30)만큼 균등 복리로
    추가 하락시킨 경로. 그 구간 동안의 실제 일별 수익률 위에 매일 동일한 승수를 곱해서
    얹는 방식(실제 시장 변동 + 인위적 크래시 드래그가 같이 반영됨), 구간 밖은 실제
    그대로. 구간 이후는 이미 낮아진 레벨에서 실제 이후 수익률이 그대로 복리 적용됨
    (크래시가 영구적 손실로 남는다는 뜻 — 모멘텀 크래시 이후 회복은 이 시나리오에 포함
    안 함, 보수적 가정)."""
    r = nav.pct_change().fillna(0).copy()
    daily_shock = (1 + magnitude) ** (1 / days) - 1
    end_idx = min(start_idx + days, len(r))
    for i in range(start_idx, end_idx):
        r.iloc[i] = (1 + r.iloc[i]) * (1 + daily_shock) - 1
    out = (1 + r).cumprod()
    out.iloc[0] = 1.0
    return out


def run_spmo_crash_injection(save=True):
    """SPMO 고유 모멘텀크래시 급락 이벤트 직접 주입(2026-07-24, 지호 님 요청 — §6-K-3
    한계 보완: "alpha 완만한 평균축소로는 2009년 같은 급격한 단일 이벤트 근사가 약함").
    2009년 모멘텀 롱숏 팩터가 한 달 새 -30%대 낙폭을 낸 사례(지호 님 인용)를 SPMO
    경로에 직접 주입 — 21거래일(~1개월)에 걸쳐 총 -30% 복리 하락을 추가로 얹는다.
    BRK-A는 모멘텀 팩터가 아니므로 이 크래시 메커니즘 자체가 해당 안 돼 실제 경로
    그대로 둔다(크래시 면역 가정). BRK-A70:30 외에 BRK-A90:10(소량비중)도 같은 크래시
    환경에서 나란히 비교(2026-07-24, 지호 님 요청 — 비중 낮춰도 방어력이 얼마나
    남는지 확인).

    두 시나리오:
      A. 고립 크래시 — 표본 중간의 평온한 구간(다른 사건과 안 겹침)에 크래시만 주입,
         "SPMO 크래시 리스크 자체"의 순수한 영향만 격리.
      B. 동시 크래시 — 알고리즘 자체의 역사상 최대낙폭 구간(2020-02-19 코로나, §6-H-5·
         §6-J-2와 동일 창) 시작점에 크래시 주입 — "공통 팩터리스크가 실현돼 알고리즘도
         부진한데 SPMO도 동시에 크래시나는" 지호 님이 우려한 최악의 동시발생 시나리오.

    ⚠ 한계: 크래시 이후 회복(모멘텀 팩터는 역사적으로 크래시 후 빠르게 반등하기도 함)은
    반영 안 함 — 크래시분이 영구 손실로 남는 보수적 가정. 크래시 규모(-30%/21거래일)는
    지호 님이 인용한 2009년 롱숏 팩터 사례를 그대로 가져온 것이라, SPMO(롱온리 ETF라
    이론상 완충됨)엔 다소 과하게 보수적일 수 있음."""
    algo_nav, spmo_nav, spy, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spmo_nav = spmo_nav.reindex(idx); spmo_nav = spmo_nav / spmo_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    peak, trough = _worst_drawdown_window(algo_nav)
    isolated_idx = len(idx) // 2   # 표본 중간 지점(고립 시나리오)
    joint_idx = idx.get_loc(peak)  # 알고리즘 최대낙폭 시작점(동시 시나리오)
    _log(f"고립 크래시 시작일: {idx[isolated_idx].date()} · 동시 크래시 시작일(알고리즘 "
         f"최대낙폭 시작): {idx[joint_idx].date()}")

    scenarios = {}
    for label, start in [("A_고립", isolated_idx), ("B_동시(알고리즘최대낙폭과겹침)", joint_idx)]:
        spmo_crashed = _inject_crash(spmo_nav, start, CRASH_MAGNITUDE, CRASH_DAYS)
        pure = algo_nav
        spmo7030_normal = CS.mix_nav(algo_nav, spmo_nav, 0.70)      # 크래시 없는 기준
        spmo7030_crashed = CS.mix_nav(algo_nav, spmo_crashed, 0.70)  # 크래시 주입
        brka7030 = CS.mix_nav(algo_nav, brka_nav, 0.70)             # 크래시 면역(비교군)
        brka9010 = CS.mix_nav(algo_nav, brka_nav, 0.90)             # 크래시 면역(비교군, 소량비중)

        crash_end = min(start + CRASH_DAYS, len(idx) - 1)
        configs_for_window = {
            "순수(100:0)": pure, "SPMO70:30(크래시없음)": spmo7030_normal,
            "SPMO70:30(크래시주입)": spmo7030_crashed, "BRK-A70:30": brka7030, "BRK-A90:10": brka9010,
        }
        window_ret = {k: round(float(v.iloc[crash_end] / v.iloc[start] - 1) * 100, 1)
                      for k, v in configs_for_window.items()}
        full = {k: CS.stats(v) for k, v in configs_for_window.items()}
        scenarios[label] = {"crash_start": idx[start].date().isoformat(),
                            "window_return_pct": window_ret, "full_period": full}
        _log(f"[{label}] 크래시구간({idx[start].date()}~{idx[crash_end].date()}) 수익률: " +
             " · ".join(f"{k}:{v:+.1f}%" for k, v in window_ret.items()))
        _log(f"[{label}] 전체기간: " + " · ".join(
            f"{k}:CAGR{v['cagr_pct']}%/샤프{v['sharpe']}/MDD{v['mdd_pct']}%" for k, v in full.items()))

    payload = {
        "as_of": idx[-1].date().isoformat(),
        "kind": "가상 시나리오(사전등록 아님) — SPMO 경로에 모멘텀크래시 직접 주입",
        "crash_assumption": f"{int(CRASH_MAGNITUDE*100)}% over {CRASH_DAYS}거래일, 복리 균등분배, "
                            "회복 없음(영구손실 가정), BRK-A는 크래시 면역(모멘텀 팩터 아님)",
        "scenarios": scenarios,
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_spmo_crash_injection.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def _mix3_nav(a: pd.Series, b: pd.Series, c: pd.Series, wa: float, wb: float, wc: float,
              rebal=BP.MONTH) -> pd.Series:
    """3자산 월간 목표비중 리밸런싱 혼합 NAV(core_satellite_kr.mix_nav의 2자산 버전을
    3자산으로 확장, 동일 로직)."""
    idx = a.index
    ra, rb, rc = a.pct_change().fillna(0), b.pct_change().fillna(0), c.pct_change().fillna(0)
    nav = []
    va, vb, vcc = wa, wb, wc
    for i in range(len(idx)):
        va *= 1 + ra.iloc[i]; vb *= 1 + rb.iloc[i]; vcc *= 1 + rc.iloc[i]
        tot = va + vb + vcc
        nav.append(tot)
        if i % rebal == 0:
            va, vb, vcc = tot * wa, tot * wb, tot * wc
    return pd.Series(nav, index=idx)


def run_3way_sweep(step=0.1, save=True):
    """알고리즘:SPY:BRK-A 3자산 혼합 최적점 탐색(2026-07-24, 지호 님 요청) — 지금까지
    2자산(algo:SPMO, algo:BRK-A)만 봤는데, SPY 자체도 넣어 3원 배분에서 최적점을
    찾는다. step(기본10%) 간격 삼각격자 전수 스윕 — 알고리즘 하나만 찍어보는 게 아니라
    전체 격자를 PBO/DSR 다중검정 보정 걸고 스캔(§6-H-4·§6-J 스윕과 동일 원칙 —
    "80:20 하나만 찍어보지 말고 전체를 보라"는 이 세션의 반복된 교훈 적용).

    ⚠ 탐색적 스윕(사전등록 아님) — "최적"이라 불러도 사후탐색 편향 대상이라 PBO/DSR
    게이트 미통과면 "이 격자에서 최고점이 진짜 최고인지 확정 못 함"으로 읽어야 한다."""
    algo_nav, spmo_nav, spy_nav, cost = _load()
    brka_hist = R.download_histories(["BRK-A"], period="max").get("BRK-A")
    if brka_hist is None or brka_hist.empty:
        raise RuntimeError("BRK-A 시세 조회 실패")
    idx = algo_nav.index.intersection(brka_hist.reindex(algo_nav.index).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    spy_nav = spy_nav.reindex(idx).ffill(); spy_nav = spy_nav / spy_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일)")

    n_steps = int(round(1 / step))
    rows, matrix, dates0 = [], [], None
    for i in range(n_steps + 1):
        wa = i * step
        for j in range(n_steps + 1 - i):
            wspy = j * step
            wbrka = round(1 - wa - wspy, 6)
            if wbrka < -1e-9:
                continue
            wbrka = max(wbrka, 0.0)
            nav = _mix3_nav(algo_nav, spy_nav, brka_nav, wa, wspy, wbrka)
            s = CS.stats(nav)
            rows.append({"algo": round(wa, 2), "spy": round(wspy, 2), "brka": round(wbrka, 2), **s})
            d, r = BP.monthly_excess(nav, spy_nav)
            if dates0 is None:
                dates0 = d
            matrix.append(r[:len(dates0)])
    _log(f"격자 {len(rows)}개(step {int(step*100)}%) 스윕 완료")

    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]
    trial_data = {"horizon": "us_3way_algo_spy_brka", "universe": "sp500_pit_topn8",
                 "cost": cost.describe(), "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev],
                 "trials": [f"a{r['algo']}s{r['spy']}b{r['brka']}" for r in rows],
                 "excess_returns": matrix}
    rpt = OS.analyze(trial_data, save=False)

    best_sharpe = max(rows, key=lambda r: r["sharpe"])
    best_cagr = max(rows, key=lambda r: r["cagr_pct"])
    best_mdd = max(rows, key=lambda r: r["mdd_pct"])   # MDD는 음수라 max가 "가장 덜 빠짐"
    calmar = lambda r: r["cagr_pct"] / abs(r["mdd_pct"]) if r["mdd_pct"] else 0
    best_calmar = max(rows, key=calmar)
    pure = next(r for r in rows if r["algo"] == 1.0)

    _log(f"샤프 최고점: algo{best_sharpe['algo']}:spy{best_sharpe['spy']}:brka{best_sharpe['brka']} "
         f"(샤프{best_sharpe['sharpe']}, CAGR{best_sharpe['cagr_pct']}%, MDD{best_sharpe['mdd_pct']}%)")
    _log(f"CAGR 최고점: algo{best_cagr['algo']}:spy{best_cagr['spy']}:brka{best_cagr['brka']} "
         f"(CAGR{best_cagr['cagr_pct']}%)")
    _log(f"MDD 최소점: algo{best_mdd['algo']}:spy{best_mdd['spy']}:brka{best_mdd['brka']} "
         f"(MDD{best_mdd['mdd_pct']}%)")
    _log(f"Calmar(CAGR/|MDD|) 최고점: algo{best_calmar['algo']}:spy{best_calmar['spy']}:"
         f"brka{best_calmar['brka']} (Calmar{calmar(best_calmar):.2f})")
    _log(f"순수(100:0:0) 대조군: 샤프{pure['sharpe']} CAGR{pure['cagr_pct']}% MDD{pure['mdd_pct']}%")
    _log(f"PBO {rpt.get('pbo', {}).get('pbo')} · DSR {rpt.get('dsr', {}).get('dsr')} · "
         f"게이트 통과 {rpt.get('passed')}")

    payload = {
        "as_of": idx[-1].date().isoformat(), "kind": "탐색적 3자산 격자 스윕(사전등록 아님)",
        "grid_step": step, "n_combos": len(rows), "rows": rows,
        "baseline": {"algo": 1.0, "spy": 0.0, "brka": 0.0, **pure},
        "best_by_sharpe": best_sharpe, "best_by_cagr": best_cagr,
        "best_by_mdd": best_mdd, "best_by_calmar": best_calmar,
        "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
        "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
        "passed": rpt.get("passed", False),
        "note": "탐색적 스윕 — PBO/DSR 미통과면 이 격자의 '최고점'이 노이즈와 구분 안 됨. "
                "채택하려면 발견한 지점을 별도 사전등록으로 재검증 필요.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_3way_algo_spy_brka_sweep.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_long_window_check(years=15, save=True):
    """topn8 챔피언을 더 긴 창(기본 15년)으로 재확인(2026-07-25, 지호 님 요청 —
    EDGAR 20년 재수집으로 커버리지 확인 결과 20년은 유니버스가 지나치게 얇아지고
    (2006년 기준 25.5%), 15년(~2011년부터, 커버리지 약 62~66%)이 현실적 상한으로
    판단됨). SPMO 의존 없이 알고리즘 단독만 본다(SPMO 자체가 2015-10 상장이라 15년
    창과 안 맞음 — 이 체크의 목적과도 무관).

    프로젝트의 반복된 교훈("8년 표본의 1등이 13년에서 사라짐" — topn 재검증·손절%
    재검증 등, STRATEGY.md §3 Stage 3.1·6.1.1)과 같은 패턴이 미국 topn8+라이브가중치
    조합에도 있는지 확인 — 이 세션 내내 써온 9~10년 창의 CAGR~32%·샤프~1.13 수치가
    더 긴 창에서도 비슷한지, 아니면 최근 소수 연도(특히 2024+ AI랠리)에 크게 좌우된
    수치인지가 핵심 질문."""
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    ma200 = panel.rolling(200, min_periods=200).mean()
    decisions = _us_decisions_live_clip(panel, funds, pit)
    sector_of = _sector_of_factory()
    algo_nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False,
                           sector_of=sector_of, sector_cap=2)
    if algo_nav is None:
        raise RuntimeError("topn=8 알고리즘 NAV 산출 실패")
    algo_nav = algo_nav / algo_nav.iloc[0]
    spy_aligned = spy.reindex(algo_nav.index).ffill()
    _log(f"공통 구간: {algo_nav.index[0].date()} ~ {algo_nav.index[-1].date()} "
         f"({len(algo_nav)}거래일, 요청 {years}년)")

    full = CS.stats(algo_nav)
    bench = CS.stats(spy_aligned)
    _log(f"알고리즘({years}년): CAGR {full['cagr_pct']}% 샤프 {full['sharpe']} MDD {full['mdd_pct']}% "
         f"vs SPY: CAGR {bench['cagr_pct']}% 샤프 {bench['sharpe']} MDD {bench['mdd_pct']}%")

    # 연도별 CAGR 분해 — 특정 연도(예: 2024+) 쏠림 여부 확인
    yearly = []
    for y in sorted(set(algo_nav.index.year)):
        mask = algo_nav.index.year == y
        w = algo_nav[mask]
        if len(w) < 5:
            continue
        r = float(w.iloc[-1] / w.iloc[0] - 1) * 100
        yearly.append({"year": int(y), "return_pct": round(r, 1)})
        _log(f"  {y}: {r:+.1f}%")

    payload = {
        "as_of": algo_nav.index[-1].date().isoformat(), "years_requested": years,
        "n_days": len(algo_nav),
        "algo_full_period": full, "spy_benchmark": bench,
        "yearly_returns": yearly,
        "note": f"EDGAR {years}년 커버리지가 100%가 아니므로(직전 세션 커버리지 분석 참고) "
                "결측 종목은 해당 팩터 z=0(중립) 처리됨 — 완전한 PIT 표본은 아님.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/us_algo_{years}y_champion_check.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def _mixN_nav(navs: list, weights: list, rebal=BP.MONTH) -> pd.Series:
    """N자산 월간 목표비중 리밸런싱 혼합 NAV(_mix3_nav의 일반화)."""
    idx = navs[0].index
    rets = [n.pct_change().fillna(0) for n in navs]
    nav_out = []
    vals = list(weights)
    for i in range(len(idx)):
        vals = [v * (1 + r.iloc[i]) for v, r in zip(vals, rets)]
        tot = sum(vals)
        nav_out.append(tot)
        if i % rebal == 0:
            vals = [tot * w for w in weights]
    return pd.Series(nav_out, index=idx)


def _weight_compositions(n_assets: int, step: float) -> list:
    """합이 1이 되는 n_assets개 비중 조합(step 단위) 전부 생성 — 격자 스윕용."""
    n_units = int(round(1 / step))
    out = []
    def rec(remaining_units, remaining_assets, acc):
        if remaining_assets == 1:
            out.append(acc + [remaining_units * step]); return
        for u in range(remaining_units + 1):
            rec(remaining_units - u, remaining_assets - 1, acc + [u * step])
    rec(n_units, n_assets, [])
    return out


def run_5way_sweep(step=0.2, bond_ticker="IEF", years=YEARS, save=True):
    """알고리즘:SPY:BRK-A:금(GLD):채권(IEF) 5자산 최적 조합 탐색(2026-07-26, 지호 님
    요청 — 실제 포트폴리오에 이미 금·채권이 편입돼 있어 "처음부터 전체를 같이 보자").
    §6-J~O에서 산발적으로 검증한 개별 페어(algo:SPMO, algo:BRK-A, algo:SPY:BRK-A)를
    한 번에 통합 — 금·채권까지 포함해 어느 조합이 실제로 유리한지 정직하게 재확인한다.
    탐색적 스윕(사전등록 아님) — PBO/DSR로 다중검정 보정.
    years: 기본 9~10년(2017+) 대신 §6-O-1에서 확인한 "데이터 신뢰 가능한" 11년(2016+)
    등으로 창을 늘려 재확인 가능(2026-07-26, 지호 님 요청 — 촘촘한 격자+더 긴 창)."""
    algo_nav, spmo_nav, spy_nav, cost = _load(years=years)
    hist = R.download_histories(["BRK-A", "GLD", bond_ticker], period="max")
    brka_hist, gld_hist, bond_hist = hist.get("BRK-A"), hist.get("GLD"), hist.get(bond_ticker)
    if brka_hist is None or gld_hist is None or bond_hist is None:
        raise RuntimeError("BRK-A/GLD/채권 시세 조회 실패")

    idx = algo_nav.index
    for s in (brka_hist, gld_hist, bond_hist):
        idx = idx.intersection(s.reindex(idx).ffill().dropna().index)
    if len(idx) < 60:
        raise RuntimeError(f"공통구간 부족(n={len(idx)})")
    series = {"algo": algo_nav, "spy": spy_nav, "brka": brka_hist, "gld": gld_hist, "bond": bond_hist}
    series = {k: (v.reindex(idx).ffill() if k != "algo" else v.reindex(idx)) for k, v in series.items()}
    series = {k: v / v.iloc[0] for k, v in series.items()}
    names = ["algo", "spy", "brka", "gld", "bond"]
    navs = [series[n] for n in names]
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일) · 채권={bond_ticker}")

    combos = _weight_compositions(len(names), step)
    _log(f"조합 {len(combos)}개(step {int(step*100)}%) 스윕 시작")
    rows, matrix, dates0 = [], [], None
    for w in combos:
        nav = _mixN_nav(navs, w)
        s = CS.stats(nav)
        row = {names[i]: round(w[i], 2) for i in range(len(names))}
        row.update(s)
        rows.append(row)
        d, r = BP.monthly_excess(nav, series["spy"])
        if dates0 is None:
            dates0 = d
        matrix.append(r[:len(dates0)])
    _log(f"스윕 완료: {len(rows)}개")

    n_ev = min(len(r) for r in matrix)
    matrix = [r[:n_ev] for r in matrix]
    trial_data = {"horizon": "us_5way_algo_spy_brka_gld_bond", "universe": "sp500_pit_topn8",
                 "cost": cost.describe(), "rebal_days": BP.MONTH, "hold_days": BP.MONTH,
                 "dates": dates0[:n_ev],
                 "trials": [f"a{r['algo']}s{r['spy']}b{r['brka']}g{r['gld']}d{r['bond']}" for r in rows],
                 "excess_returns": matrix}
    rpt = OS.analyze(trial_data, save=False)

    def _fmt(r):
        return f"algo{r['algo']}:spy{r['spy']}:brka{r['brka']}:gld{r['gld']}:{bond_ticker.lower()}{r['bond']}"
    best_sharpe = max(rows, key=lambda r: r["sharpe"])
    best_cagr = max(rows, key=lambda r: r["cagr_pct"])
    best_mdd = max(rows, key=lambda r: r["mdd_pct"])
    calmar = lambda r: r["cagr_pct"] / abs(r["mdd_pct"]) if r["mdd_pct"] else 0
    best_calmar = max(rows, key=calmar)
    pure = next(r for r in rows if r["algo"] == 1.0)
    top10_sharpe = sorted(rows, key=lambda r: -r["sharpe"])[:10]

    _log(f"샤프 최고: {_fmt(best_sharpe)} (샤프{best_sharpe['sharpe']}·CAGR{best_sharpe['cagr_pct']}%·MDD{best_sharpe['mdd_pct']}%)")
    _log(f"CAGR 최고: {_fmt(best_cagr)} (CAGR{best_cagr['cagr_pct']}%)")
    _log(f"MDD 최소: {_fmt(best_mdd)} (MDD{best_mdd['mdd_pct']}%)")
    _log(f"Calmar 최고: {_fmt(best_calmar)} (Calmar{calmar(best_calmar):.2f})")
    _log(f"순수(100:0:0:0:0): 샤프{pure['sharpe']} CAGR{pure['cagr_pct']}% MDD{pure['mdd_pct']}%")
    _log(f"PBO {rpt.get('pbo', {}).get('pbo')} · DSR {rpt.get('dsr', {}).get('dsr')} · 게이트 통과 {rpt.get('passed')}")
    for r in top10_sharpe:
        _log(f"  상위: {_fmt(r)} 샤프{r['sharpe']} CAGR{r['cagr_pct']}% MDD{r['mdd_pct']}%")

    payload = {
        "as_of": idx[-1].date().isoformat(), "kind": "탐색적 5자산 격자 스윕(사전등록 아님)",
        "years_requested": years, "bond_ticker": bond_ticker, "grid_step": step,
        "n_combos": len(rows), "rows": rows,
        "baseline": pure, "best_by_sharpe": best_sharpe, "best_by_cagr": best_cagr,
        "best_by_mdd": best_mdd, "best_by_calmar": best_calmar, "top10_by_sharpe": top10_sharpe,
        "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
        "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
        "passed": rpt.get("passed", False),
        "note": "탐색적 스윕 — PBO/DSR 미통과면 '최고점'이 노이즈와 구분 안 됨. 채택하려면 "
                "발견한 지점을 별도 사전등록으로 재검증 필요.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/us_5way_algo_spy_brka_gld_bond_sweep_{years}y_step{int(step*100)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_algo_brka_gld_prereg(wa=0.8, wb=0.1, wc=0.1, years=11, save=True):
    """algo:BRK-A:금 특정 비율(기본 80:10:10, §6-P-2 Calmar 최고점)의 샤프·CAGR·MDD
    유의성 사전등록 검정(2026-07-26, 지호 님 요청). §6-P-3에서 이미 "촘촘한 격자에서
    찾은 점을 그대로 채택하면 안 된다"고 경고했으므로, 그 경고를 실제로 지켜 이 단일
    지점만 사전등록해서 별도 검정한다(격자 재탐색 아님). 방법은 run_brka_sharpe_prereg
    와 동일(짝지은 블록부트스트랩, 6개월 블록, 5000회) — 여기선 3자산 블렌드로 확장,
    Calmar 후보답게 샤프뿐 아니라 CAGR·MDD도 같이 검정.

    사전등록(실행 전 확정):
      가설: algo{wa}:brka{wb}:gld{wc} 블렌드가 순수 알고리즘(100:0:0)보다 샤프가
            유의하게 낫다(주 가설). 부가: CAGR 손실이 통계적으로 유의한 손해인지,
            MDD가 유의하게 개선되는지도 같이 보고(참고용, 채택 게이트는 샤프 기준만).
      데이터: §6-P-2와 동일 11년(2016+) 창, algo 라이브설정 NAV + BRK-A·GLD 원시가격.
      판정규칙(샤프 게이트, 둘 다 충족해야 "채택 후보"):
        ① 샤프차이 95%CI 하한 > 0
        ② 서브기간(SUBS) 전부 블렌드 우위 방향 일관
      단일 사전등록 시행 — PBO/DSR 대상 아님(격자 재탐색이 아니므로)."""
    algo_nav, spmo_nav, spy, cost = _load(years=years)
    hist = R.download_histories(["BRK-A", "GLD"], period="max")
    brka_hist, gld_hist = hist.get("BRK-A"), hist.get("GLD")
    if brka_hist is None or gld_hist is None:
        raise RuntimeError("BRK-A/GLD 시세 조회 실패")
    idx = algo_nav.index
    for s in (brka_hist, gld_hist):
        idx = idx.intersection(s.reindex(idx).ffill().dropna().index)
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    brka_nav = brka_hist.reindex(idx); brka_nav = brka_nav / brka_nav.iloc[0]
    gld_nav = gld_hist.reindex(idx); gld_nav = gld_nav / gld_nav.iloc[0]
    blend_nav = _mixN_nav([algo_nav, brka_nav, gld_nav], [wa, wb, wc])
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일) · "
         f"비율 algo{wa}:brka{wb}:gld{wc}")

    pure_full, blend_full = CS.stats(algo_nav), CS.stats(blend_nav)
    _log(f"순수: CAGR{pure_full['cagr_pct']}% 샤프{pure_full['sharpe']} MDD{pure_full['mdd_pct']}% · "
         f"블렌드: CAGR{blend_full['cagr_pct']}% 샤프{blend_full['sharpe']} MDD{blend_full['mdd_pct']}%")

    r_pure, r_blend = _monthly_returns(algo_nav), _monthly_returns(blend_nav)
    n = min(len(r_pure), len(r_blend))
    r_pure, r_blend = r_pure[:n], r_blend[:n]

    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    sharpe_diffs, cagr_diffs = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        sharpe_diffs[i] = _sharpe_from_monthly(r_blend[bidx]) - _sharpe_from_monthly(r_pure[bidx])
        cagr_diffs[i] = _cagr_from_monthly(r_blend[bidx]) - _cagr_from_monthly(r_pure[bidx])
    s_lo, s_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    s_mean, s_pos = float(sharpe_diffs.mean()), float((sharpe_diffs > 0).mean()) * 100
    c_lo, c_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    c_mean, c_pos = float(cagr_diffs.mean()), float((cagr_diffs > 0).mean()) * 100
    _log(f"샤프차이 95%CI: [{s_lo:+.3f},{s_hi:+.3f}] 평균{s_mean:+.3f} ({N_BOOT}회 중 {s_pos:.1f}%양수)")
    _log(f"CAGR차이 95%CI: [{c_lo:+.2f}%p,{c_hi:+.2f}%p] 평균{c_mean:+.2f}%p ({N_BOOT}회 중 {c_pos:.1f}%양수)")

    sub_rows, signs = [], []
    for label, a, b in SUBS:
        sp_pure, sp_blend = CS.stats(algo_nav, a, b), CS.stats(blend_nav, a, b)
        if sp_pure is None or sp_blend is None:
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        d = sp_blend["sharpe"] - sp_pure["sharpe"]
        signs.append(d > 0)
        sub_rows.append({"period": label, "pure_sharpe": sp_pure["sharpe"],
                         "blend_sharpe": sp_blend["sharpe"], "sharpe_diff": round(d, 3),
                         "pure_mdd": sp_pure["mdd_pct"], "blend_mdd": sp_blend["mdd_pct"]})
        _log(f"{label}: 샤프차이 {d:+.3f} · MDD 순수{sp_pure['mdd_pct']}% 블렌드{sp_blend['mdd_pct']}%")
    subperiod_consistent = len(signs) >= 2 and all(signs)

    gate1 = s_lo > 0
    gate2 = subperiod_consistent
    passed = gate1 and gate2
    direction = "블렌드 우위" if s_mean > 0 else "순수 우위"
    rejected_opposite = (not passed) and s_hi < 0
    verdict = "채택 후보" if passed else ("가설 기각(유의하게 반대방향)" if rejected_opposite else "판정 보류(유의한 차이 없음)")
    _log(f"판정 — ①CI하한>0:{gate1} ②서브기간일관:{gate2} → 방향:{direction} → 최종:{verdict}")

    payload = {
        "as_of": idx[-1].date().isoformat(), "n_months": n, "years": years,
        "ratio": {"algo": wa, "brka": wb, "gld": wc},
        "prereg": {"hypothesis": f"algo{wa}:brka{wb}:gld{wc} 블렌드 샤프가 순수(100:0:0)보다 유의하게 낫다",
                  "decision_rule": "①샤프차이 95%CI 하한>0 AND ②서브기간 전부 블렌드우위"},
        "full_period": {"pure": pure_full, "blend": blend_full},
        "sharpe_diff_bootstrap": {"mean": round(s_mean, 3), "ci95_lo": round(s_lo, 3),
                                  "ci95_hi": round(s_hi, 3), "pct_positive": round(s_pos, 1)},
        "cagr_diff_bootstrap": {"mean": round(c_mean, 2), "ci95_lo": round(c_lo, 2),
                                "ci95_hi": round(c_hi, 2), "pct_positive": round(c_pos, 1)},
        "subperiods": sub_rows,
        "gates": {"g1_sharpe_ci_lo_positive": gate1, "g2_subperiod_consistent": gate2},
        "direction": direction, "passed": passed, "verdict": verdict,
        "note": "단일 사전등록 시행(격자 재탐색 아님) — PBO/DSR 대상 아님.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/us_algo{int(wa*100)}_brka{int(wb*100)}_gld{int(wc*100)}_prereg.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_candidate_screen(tickers=("DBMF", "XLP", "XLU", "EFA", "JNJ"), save=True):
    """버크셔 외 헷지 후보 빠른 스크리닝(2026-07-26, 지호 님 요청). 전체 스트레스테스트
    (§6-J~K급) 전에 저비용으로 먼저 걸러낸다 — 각 후보의 단독 성과 + 알고리즘과의
    상관계수(전체·알고리즘 하락일 조건부, §6-J-2와 동일 방법론)만 계산. 상관 낮고
    단독 성과도 있는 후보만 이후 본격 검증(꼬리위험·동반열화 등) 대상으로 추천."""
    algo_nav, spmo_nav, spy_nav, cost = _load()
    hist = R.download_histories(list(tickers), period="max")
    r_algo = algo_nav.pct_change().fillna(0)
    down_mask = r_algo < 0

    rows = []
    for t in tickers:
        h = hist.get(t)
        if h is None or h.empty:
            _log(f"{t}: 시세 조회 실패, 스킵"); continue
        idx = algo_nav.index.intersection(h.reindex(algo_nav.index).ffill().dropna().index)
        if len(idx) < 60:
            _log(f"{t}: 공통구간 부족(n={len(idx)}), 스킵"); continue
        nav = h.reindex(idx); nav = nav / nav.iloc[0]
        s = CS.stats(nav)
        r_t = nav.pct_change().reindex(r_algo.index)
        corr_full = float(r_algo.corr(r_t))
        corr_down = float(r_algo[down_mask].corr(r_t[down_mask]))
        row = {"ticker": t, "n_days": len(idx), "start": idx[0].date().isoformat(),
              **s, "corr_vs_algo_full": round(corr_full, 3), "corr_vs_algo_down_days": round(corr_down, 3)}
        rows.append(row)
        _log(f"{t}(시작 {idx[0].date()}, {len(idx)}거래일): CAGR{s['cagr_pct']}% 샤프{s['sharpe']} "
             f"MDD{s['mdd_pct']}% · 상관 전체{corr_full:.3f}/하락일{corr_down:.3f}")

    rows_sorted = sorted(rows, key=lambda r: r["corr_vs_algo_down_days"])
    _log("하락일 상관 낮은 순(유망 후보 상단): " +
         " · ".join(f"{r['ticker']}({r['corr_vs_algo_down_days']:.3f})" for r in rows_sorted))

    payload = {"as_of": algo_nav.index[-1].date().isoformat(), "kind": "후보 스크리닝(서술, 가설검정 아님)",
              "brka_reference": "corr_vs_algo_full 0.445 · down_days 0.37 (§6-J-2 기준점)",
              "gold_reference": "gld는 §6-P에서 이미 상위권 확인됨(비교 스크리닝 대상 아님)",
              "rows": rows_sorted}
    if save:
        os.makedirs("output", exist_ok=True)
        path = "output/us_hedge_candidate_screen.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


def run_candidate_deepdive(ticker="JNJ", ratio=0.7, years=None, save=True):
    """개별 헷지 후보(스크리닝 통과자) 본격 검증(2026-07-26, 지호 님 요청 — "JNJ 단독부터
    가자"). run_brka_hedge_compare()+run_brka_sharpe_prereg()를 일반화해 한 번에 수행:
    ①상관계수(전체·알고리즘 하락일) ②꼬리위험 대리지표 3종(위기구간·최악12개월·CVaR95)
    ③CAGR 비용 ④샤프 개선 사전등록 유의성 검정(파라미터화된 ratio, 기본 70:30 —
    §6-J의 BRK-A 검증과 동일 비율로 비교 가능성 유지). 단일 사전등록 시행."""
    years = years or YEARS
    algo_nav, spmo_nav, spy, cost = _load(years=years)
    hist = R.download_histories([ticker], period="max").get(ticker)
    if hist is None or hist.empty:
        raise RuntimeError(f"{ticker} 시세 조회 실패")
    idx = algo_nav.index.intersection(hist.reindex(algo_nav.index).ffill().dropna().index)
    if len(idx) < 60:
        raise RuntimeError(f"공통구간 부족(n={len(idx)})")
    algo_nav = algo_nav.reindex(idx); algo_nav = algo_nav / algo_nav.iloc[0]
    cand_nav = hist.reindex(idx); cand_nav = cand_nav / cand_nav.iloc[0]
    blend_nav = CS.mix_nav(algo_nav, cand_nav, ratio)
    _log(f"공통 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}거래일) · "
         f"{ticker} 비율 {int(ratio*100)}:{int((1-ratio)*100)}")

    # ① 상관계수
    r_algo = algo_nav.pct_change().dropna()
    r_cand = cand_nav.pct_change().reindex(r_algo.index)
    down_mask = r_algo < 0
    corr_full = float(r_algo.corr(r_cand))
    corr_down = float(r_algo[down_mask].corr(r_cand[down_mask]))
    _log(f"상관계수: 전체 {corr_full:.3f} · 알고리즘 하락일 {corr_down:.3f}")

    # ② 꼬리위험 대리지표
    peak, trough = _worst_drawdown_window(algo_nav)
    w = blend_nav.loc[peak:trough]
    crash_ret = float((w.iloc[-1] / w.iloc[0] - 1) * 100)
    r12 = _rolling_return(blend_nav, 252).dropna()
    worst12m = float(r12.min() * 100)
    rm = _monthly_returns(blend_nav)
    cvar = float(rm[rm <= np.percentile(rm, 5)].mean() * 100)
    pure_full, blend_full = CS.stats(algo_nav), CS.stats(blend_nav)
    _log(f"순수: CAGR{pure_full['cagr_pct']}% 샤프{pure_full['sharpe']} MDD{pure_full['mdd_pct']}% · "
         f"블렌드: CAGR{blend_full['cagr_pct']}% 샤프{blend_full['sharpe']} MDD{blend_full['mdd_pct']}%")
    _log(f"꼬리위험(블렌드): 위기구간({peak.date()}~{trough.date()}) {crash_ret:+.1f}% · "
         f"최악12개월 {worst12m:+.1f}% · CVaR95 {cvar:+.2f}%")

    # ③④ CAGR 비용 + 샤프 유의성(짝지은 블록부트스트랩)
    r_pure, r_blend = _monthly_returns(algo_nav), _monthly_returns(blend_nav)
    n = min(len(r_pure), len(r_blend))
    r_pure, r_blend = r_pure[:n], r_blend[:n]
    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    cagr_diffs, sharpe_diffs = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = _cagr_from_monthly(r_blend[bidx]) - _cagr_from_monthly(r_pure[bidx])
        sharpe_diffs[i] = _sharpe_from_monthly(r_blend[bidx]) - _sharpe_from_monthly(r_pure[bidx])
    c_lo, c_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    c_mean = float(cagr_diffs.mean())
    s_lo, s_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    s_mean, s_pos = float(sharpe_diffs.mean()), float((sharpe_diffs > 0).mean()) * 100
    _log(f"CAGR차이 95%CI: [{c_lo:+.2f}%p,{c_hi:+.2f}%p] 평균{c_mean:+.2f}%p")
    _log(f"샤프차이 95%CI: [{s_lo:+.3f},{s_hi:+.3f}] 평균{s_mean:+.3f} ({N_BOOT}회 중 {s_pos:.1f}%양수)")

    sub_rows, signs = [], []
    for label, a, b in SUBS:
        sp_pure, sp_blend = CS.stats(algo_nav, a, b), CS.stats(blend_nav, a, b)
        if sp_pure is None or sp_blend is None:
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        d = sp_blend["sharpe"] - sp_pure["sharpe"]
        signs.append(d > 0)
        sub_rows.append({"period": label, "pure_sharpe": sp_pure["sharpe"],
                         "blend_sharpe": sp_blend["sharpe"], "sharpe_diff": round(d, 3)})
        _log(f"{label}: 샤프차이 {d:+.3f}")
    subperiod_consistent = len(signs) >= 2 and all(signs)

    gate1 = s_lo > 0
    gate2 = subperiod_consistent
    passed = gate1 and gate2
    direction = "블렌드 우위" if s_mean > 0 else "순수 우위"
    rejected_opposite = (not passed) and s_hi < 0
    verdict = "채택 후보" if passed else ("가설 기각(유의하게 반대방향)" if rejected_opposite else "판정 보류(유의한 차이 없음)")
    _log(f"판정 — ①샤프CI하한>0:{gate1} ②서브기간일관:{gate2} → 방향:{direction} → 최종:{verdict}")

    payload = {
        "as_of": idx[-1].date().isoformat(), "ticker": ticker, "ratio": ratio, "years": years,
        "n_months": n,
        "prereg": {"hypothesis": f"algo{int(ratio*100)}:{ticker}{int((1-ratio)*100)} 블렌드 샤프가 "
                                 "순수(100:0)보다 유의하게 낫다",
                  "decision_rule": "①샤프차이 95%CI 하한>0 AND ②서브기간 전부 블렌드우위"},
        "correlation_vs_algo": {"full": round(corr_full, 3), "algo_down_days": round(corr_down, 3)},
        "full_period": {"pure": pure_full, "blend": blend_full},
        "tail_metrics": {"algo_worst_drawdown_window": {"peak": peak.date().isoformat(),
                                                         "trough": trough.date().isoformat()},
                        "crash_window_return_pct": round(crash_ret, 1),
                        "worst_rolling_12m_pct": round(worst12m, 1), "cvar95_monthly_pct": round(cvar, 2)},
        "cagr_diff_bootstrap": {"mean": round(c_mean, 2), "ci95_lo": round(c_lo, 2), "ci95_hi": round(c_hi, 2)},
        "sharpe_diff_bootstrap": {"mean": round(s_mean, 3), "ci95_lo": round(s_lo, 3),
                                  "ci95_hi": round(s_hi, 3), "pct_positive": round(s_pos, 1)},
        "subperiods": sub_rows,
        "gates": {"g1_sharpe_ci_lo_positive": gate1, "g2_subperiod_consistent": gate2},
        "direction": direction, "passed": passed, "verdict": verdict,
        "note": "단일 사전등록 시행 — PBO/DSR 대상 아님. 꼬리위험 지표는 같은 과거데이터 재생(§6-H-5 한계 동일).",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        path = f"output/us_{ticker.lower().replace('-', '')}_deepdive.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {path}")
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="탐색적 비율 스윕(PBO/DSR 포함, 사전등록 아님)")
    ap.add_argument("--tailrisk", action="store_true", help="최악구간/꼬리위험 대리지표(과최적화 헷지 논의 대응)")
    ap.add_argument("--vs-brka", action="store_true", help="알고리즘·SPY·SPMO vs 버크셔 A주 비교")
    ap.add_argument("--brka-hedge", action="store_true", help="BRK-A 블렌드 vs SPMO 블렌드 헷지효과 비교")
    ap.add_argument("--brka-sharpe", action="store_true", help="BRK-A 70:30 블렌드 샤프 유의성 사전등록 검정")
    ap.add_argument("--degrade-stress", action="store_true", help="알고리즘 엣지 열화 시나리오 스트레스 테스트")
    ap.add_argument("--joint-degrade-stress", action="store_true",
                    help="알고리즘+SPMO 동반열화 시나리오(공통 팩터리스크 붕괴 가정, BRK-A는 그대로)")
    ap.add_argument("--crash-injection", action="store_true",
                    help="SPMO 경로에 모멘텀크래시(-30%/1개월) 직접 주입 시나리오")
    ap.add_argument("--3way-sweep", dest="three_way_sweep", action="store_true",
                    help="알고리즘:SPY:BRK-A 3자산 격자 스윕(탐색적, PBO/DSR 포함)")
    ap.add_argument("--long-window", type=int, default=None, metavar="YEARS",
                    help="topn8 챔피언을 지정 연수(예: 15)로 재확인(EDGAR 커버리지 한계 감안)")
    ap.add_argument("--5way-sweep", dest="five_way_sweep", action="store_true",
                    help="알고리즘:SPY:BRK-A:금(GLD):채권 5자산 격자 스윕(탐색적, PBO/DSR 포함)")
    ap.add_argument("--bond-ticker", default="IEF", help="5way-sweep에서 쓸 채권 티커(기본 IEF)")
    ap.add_argument("--grid-step", type=float, default=0.2, help="5way-sweep 격자 간격(기본 0.2=20%%)")
    ap.add_argument("--sweep-years", type=float, default=YEARS,
                    help="5way-sweep에 쓸 연수(기본 10년≈실사용 9년, §6-O-1의 11년=2016+ 권장)")
    ap.add_argument("--algo-brka-gld-prereg", action="store_true",
                    help="algo:BRK-A:금 특정 비율 샤프/CAGR 유의성 사전등록 검정(기본 80:10:10)")
    ap.add_argument("--ratio", default="0.8,0.1,0.1", help="algo,brka,gld 비율(콤마구분, algo-brka-gld-prereg용)")
    ap.add_argument("--candidate-screen", action="store_true", help="버크셔 외 헷지 후보 상관계수·성과 빠른 스크리닝")
    ap.add_argument("--candidates", default="DBMF,XLP,XLU,EFA,JNJ", help="candidate-screen용 티커(콤마구분)")
    ap.add_argument("--deepdive", default=None, metavar="TICKER", help="개별 후보 본격 검증(상관·꼬리위험·CAGR비용·샤프유의성)")
    ap.add_argument("--deepdive-ratio", type=float, default=0.7, help="deepdive에서 쓸 algo 비중(기본 0.7=70:30)")
    args = ap.parse_args()
    if args.sweep:
        run_sweep()
    elif args.three_way_sweep:
        run_3way_sweep()
    elif args.five_way_sweep:
        run_5way_sweep(step=args.grid_step, bond_ticker=args.bond_ticker, years=args.sweep_years)
    elif args.algo_brka_gld_prereg:
        wa, wb, wc = (float(x) for x in args.ratio.split(","))
        run_algo_brka_gld_prereg(wa=wa, wb=wb, wc=wc, years=args.sweep_years)
    elif args.candidate_screen:
        run_candidate_screen(tickers=tuple(t.strip() for t in args.candidates.split(",")))
    elif args.deepdive:
        run_candidate_deepdive(ticker=args.deepdive, ratio=args.deepdive_ratio, years=args.sweep_years)
    elif args.long_window:
        run_long_window_check(years=args.long_window)
    elif args.degrade_stress:
        run_degradation_stress()
    elif args.joint_degrade_stress:
        run_joint_degradation_stress()
    elif args.crash_injection:
        run_spmo_crash_injection()
    elif args.tailrisk:
        run_tailrisk()
    elif args.vs_brka:
        run_vs_brka()
    elif args.brka_hedge:
        run_brka_hedge_compare()
    elif args.brka_sharpe:
        run_brka_sharpe_prereg()
    else:
        run()
