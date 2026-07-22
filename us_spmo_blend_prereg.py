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


def _load():
    """topn8 라이브설정 알고리즘 NAV + SPMO 원시가격 NAV(레짐타이밍 없음) + SPY 벤치마크,
    공통일자 정규화."""
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(YEARS, pit)
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="탐색적 비율 스윕(PBO/DSR 포함, 사전등록 아님)")
    ap.add_argument("--tailrisk", action="store_true", help="최악구간/꼬리위험 대리지표(과최적화 헷지 논의 대응)")
    args = ap.parse_args()
    if args.sweep:
        run_sweep()
    elif args.tailrisk:
        run_tailrisk()
    else:
        run()
