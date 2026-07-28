#!/usr/bin/env python3
"""
us_open_vs_close_return_backtest.py — "장 개장 직후(시가) 기준으로 리포트를 계산하면
실제 수익률(CAGR·샤프·MDD)이 종가 기준(현행)과 통계적으로 다른가?" (지호 님 요청, 2026-07-28)

배경: us_intraday_timing_sensitivity[_longterm].py가 "신호값이 얼마나 흔들리는가"(gap_trend·
RSI·레짐뒤집힘)를 봤다면, 이건 그 다음 질문 — "그래서 실제로 포트폴리오를 그렇게 운용하면
수익이 달라지는가"다. topn8 라이브 챔피언(가중치 1:2:2, 섹터캡2, ma200_backup=False —
§6-A/§6-D/§6-O 확정 설정, us_spmo_blend_prereg._load()와 동일 레시피)을 그대로 재사용하되,
매매체결가·200일선 계산에 쓰는 가격 시계열만 종가(현행) vs 시가로 바꿔 비교한다.

⚠ 스코프 결정(정직하게 명시): 종목 "선정"(팩터 랭킹)은 **종가 기준 그대로 고정**했다 —
PER/EPS 등 펀더멘탈은 애초에 일중 스냅샷과 무관하고, 가격의존 팩터(rd_mktcap의 시가총액)까지
전부 시가로 다시 뽑으면 "체결가가 다르다"는 질문과 "랭킹 자체가 다르다"는 질문이 섞인다.
여기서 답하는 건 후자를 뺀 전자만 — "어떤 종목을 살지는 그대로 두고, 그 종목을 언제 가격에
사고팔고 200일선을 어디에 대볼지"만 시가로 바꿨을 때의 효과.

⚠ 데이터 한계: 이 비교는 시가(Open)만 가능하다 — 야후 일봉은 Open을 티커 역사 전체로 주므로
9년 전체를 그대로 쓸 수 있지만, +30분/+1시간/+2시간은 인트라데이 봉이 필요해 무료 데이터로는
2년(60분봉)·60일(30분봉) 한도를 못 넘는다(us_intraday_timing_*.py 참고) — 그 자산군에서
포트폴리오 규모(PIT 유니버스 수백 종목) 다년간 수익률 백테스트는 데이터 자체가 없어 불가능.
신호값 오차가 시가→+30분→+1시간→+2시간 순으로 줄어드는 게 이미 확인됐으므로(별도 스크립트),
시가의 수익률 영향이 작다면 +30분/+1시간/+2시간의 영향은 그보다 더 작을 것으로 추정할 수는
있으나 직접 실측은 아니다.

방법: us_spmo_blend_prereg의 종가기준 topn8 챔피언 레시피를 그대로 쓰되, 매매체결가·200일선
계산용 패널만 시가(Open)로 교체한 두 번째 NAV를 만들어 페어드 비교(월간 비중첩 수익률
t검정 + 6개월블록 부트스트랩 5000회 — us_spmo_blend_prereg.py와 동일 방법론 재사용).

실행: python us_open_vs_close_return_backtest.py [--years 9]
결과: output/us_open_vs_close_return_backtest.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_portfolio as BP
import backtest_weights as BW
import us_spmo_blend_prereg as SP

YEARS = 9
TOPN = 8
BLOCK = SP.BLOCK
N_BOOT = SP.N_BOOT
SEED = SP.SEED
SUBS = SP.SUBS


def _log(m): print(f"[OPEN-vs-CLOSE] {m}", file=sys.stderr)


def _load(years=YEARS):
    pit = BC.load_pit()
    panel, spy, opens = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    decisions = SP._us_decisions_live_clip(panel, funds, pit)   # 종목 선정 = 종가 기준 고정(스코프 결정)
    sector_of = SP._sector_of_factory()

    ma200_close = panel.rolling(200, min_periods=200).mean()
    nav_close = BP.simulate(panel, ma200_close, decisions, TOPN, cost, ma200_backup=False,
                            sector_of=sector_of, sector_cap=2)
    if nav_close is None:
        raise RuntimeError("종가기준 NAV 산출 실패")

    if opens is None:
        raise RuntimeError("시가 데이터 확보 실패(build_panel_pit이 opens=None 반환)")
    opens_aligned = opens.reindex(columns=panel.columns)
    n_missing = int(opens_aligned.isna().all().sum())
    if n_missing:
        _log(f"시가 결측 {n_missing}종목 → 그날 종가로 대체(두 변형이 그 종목만 동일하게 취급됨)")
    opens_aligned = opens_aligned.fillna(panel)   # 시가 못 구한 종목은 종가로 폴백(보수적 — 차이를 과소평가하는 쪽)
    ma200_open = opens_aligned.rolling(200, min_periods=200).mean()
    nav_open = BP.simulate(opens_aligned, ma200_open, decisions, TOPN, cost, ma200_backup=False,
                           sector_of=sector_of, sector_cap=2)
    if nav_open is None:
        raise RuntimeError("시가기준 NAV 산출 실패")
    return nav_close, nav_open, spy, cost


def run(years=YEARS, save=True):
    nav_close, nav_open, spy, cost = _load(years)
    close_full = SP.CS.stats(nav_close)
    open_full = SP.CS.stats(nav_open)
    _log(f"종가기준(현행): CAGR {close_full['cagr_pct']}% 샤프 {close_full['sharpe']} MDD {close_full['mdd_pct']}%")
    _log(f"시가기준: CAGR {open_full['cagr_pct']}% 샤프 {open_full['sharpe']} MDD {open_full['mdd_pct']}%")

    r_close = SP._monthly_returns(nav_close)
    r_open = SP._monthly_returns(nav_open)
    n = min(len(r_close), len(r_open))
    r_close, r_open = r_close[:n], r_open[:n]

    tstat, pval = SP._paired_ttest(r_open, r_close)
    _log(f"페어드 t검정(월간수익률, n={n}): t={tstat:+.2f} p={pval:.3f}")

    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    cagr_diffs = np.empty(N_BOOT)
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = SP._cagr_from_monthly(r_open[bidx]) - SP._cagr_from_monthly(r_close[bidx])
        sharpe_diffs[i] = SP._sharpe_from_monthly(r_open[bidx]) - SP._sharpe_from_monthly(r_close[bidx])

    cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    cagr_mean = float(cagr_diffs.mean())
    cagr_pos = float((cagr_diffs > 0).mean()) * 100
    sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    sharpe_mean = float(sharpe_diffs.mean())
    _log(f"CAGR차이(시가-종가) 95%CI: [{cagr_lo:+.2f}%p,{cagr_hi:+.2f}%p] 평균{cagr_mean:+.2f}%p "
         f"({N_BOOT}회 중 {cagr_pos:.1f}%가 양수)")
    _log(f"샤프차이(시가-종가) 95%CI: [{sharpe_lo:+.3f},{sharpe_hi:+.3f}] 평균{sharpe_mean:+.3f}")

    sub_rows = []
    for label, a, b in SUBS:
        sc = SP.CS.stats(nav_close, a, b)
        so = SP.CS.stats(nav_open, a, b)
        if sc is None or so is None:
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        d_cagr = so["cagr_pct"] - sc["cagr_pct"]
        sub_rows.append({"period": label, "close_cagr": sc["cagr_pct"], "open_cagr": so["cagr_pct"],
                         "cagr_diff": round(d_cagr, 2)})
        _log(f"{label}: 종가CAGR {sc['cagr_pct']}% vs 시가CAGR {so['cagr_pct']}% (차이 {d_cagr:+.2f}%p)")

    significant = cagr_lo > 0 or cagr_hi < 0   # 방향 무관 유의성(이건 가설검정이 아니라 순수 측정)
    payload = {
        "as_of": nav_close.index[-1].date().isoformat(), "years": years, "n_months": n,
        "cost": cost.describe(),
        "scope_note": "종목 선정(팩터 랭킹)은 종가 기준 고정 — 매매체결가·200일선 계산만 시가로 교체",
        "full_period": {"close_baseline": close_full, "open_variant": open_full},
        "paired_ttest": {"t": round(float(tstat), 3), "p": round(float(pval), 4), "n": n},
        "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1),
                                "n_boot": N_BOOT, "block_months": BLOCK},
        "sharpe_diff_bootstrap": {"mean": round(sharpe_mean, 3), "ci95_lo": round(sharpe_lo, 3),
                                  "ci95_hi": round(sharpe_hi, 3), "n_boot": N_BOOT, "block_months": BLOCK},
        "subperiods": sub_rows,
        "significant_difference": significant,
        "note": "+30분/+1시간/+2시간은 야후 인트라데이 한도(60일/2년)로 이 규모(PIT 유니버스 "
                "다년간)의 수익률 백테스트가 데이터상 불가능 — us_intraday_timing_*.py의 "
                "신호값 민감도(시가>=30분후>=1시간후>=2시간후 오차 크기)로 방향만 추정.",
    }
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_open_vs_close_return_backtest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_open_vs_close_return_backtest.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=YEARS)
    a = ap.parse_args()
    result = run(a.years)
    print(json.dumps(result, ensure_ascii=False, indent=2))
