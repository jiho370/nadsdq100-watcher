#!/usr/bin/env python3
"""
us_algo_blend_spmo_voo_rsp.py — 알고리즘(topn8 라이브) 70% + [SPMO/VOO/RSP] 30% 월간리밸
블렌드 3종 비교 (지호 님 요청, 2026-07-29 — us_spmo_blend_prereg.py의 §6-H SPMO 단독
검증을 VOO(S&P500 시총가중)·RSP(S&P500 동일가중)까지 확장한 3자 비교).

배경: §6-H가 algo+SPMO 70:30을 사전등록 검증해 기각(CAGR차이 95%CI가 완전히 음수)했다.
이번엔 "그 블렌드 파트너를 SPMO 대신 VOO나 RSP로 하면 다른가"를 같은 방법론으로 확인한다
— VOO는 순수 시장 노출(알고리즘과 팩터리스크 공유 안 함, §6-P에서 이미 "SPY류는 상위권에
못 든다"고 나온 자산군), RSP는 대형주 쏠림이 덜한 시총 동일가중이라 분산 성격이 SPMO·VOO
와는 또 다르다.

방법: topn8 라이브 챔피언 NAV(가중치 1:2:2·섹터캡2·ma200_backup=False, us_spmo_blend_
prereg 레시피 재사용) 9년 구축. SPMO·VOO·RSP 각각과 core_satellite_kr.mix_nav()로 70:30
월간리밸 블렌드 생성. 순수(100:0) 대비 3블렌드 + 블렌드 상호비교, 페어드 t검정 + 6개월
블록부트스트랩(5000회, us_spmo_blend_prereg와 동일 방법론)으로 유의성 확인.

실행: python us_algo_blend_spmo_voo_rsp.py [--years 9] [--ratio 0.7]
결과: output/us_algo_blend_spmo_voo_rsp.json
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
W_ALGO = 0.7
BLOCK = SP.BLOCK
N_BOOT = SP.N_BOOT
SEED = SP.SEED
SUBS = SP.SUBS
PARTNERS = ["SPMO", "VOO", "RSP"]


def _log(m): print(f"[BLEND3] {m}", file=sys.stderr)


def _algo_nav(years=YEARS):
    pit = BC.load_pit()
    panel, spy, _opens = BC.build_panel_pit(years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    decisions = SP._us_decisions_live_clip(panel, funds, pit)
    sector_of = SP._sector_of_factory()
    ma200 = panel.rolling(200, min_periods=200).mean()
    nav = BP.simulate(panel, ma200, decisions, TOPN, cost, ma200_backup=False,
                      sector_of=sector_of, sector_cap=2)
    if nav is None:
        raise RuntimeError("알고리즘 NAV 산출 실패")
    return nav


def _spy_regime_tags(spy_close: pd.Series, n_points: int, trail_days: int = 252,
                     bull_ret: float = 0.20, bear_ret: float = -0.10, bear_dd: float = -0.15,
                     min_days: int | None = None) -> list[str]:
    """SPY 가격만으로 95개 월간 시점 각각을 상승/하락/횡보장으로 분류(2026-07-29, 지호 님
    요청 — "장세별로도 봐야, 지금은 횡보장이니까"). 그 시점 '이전'(자기 자신 포함) 데이터만
    써서 미래참조 없이 분류한다(BP.MONTH=21거래일 간격, _monthly_returns()과 동일 인덱싱).
    trail_days로 장기(기본 252일=12개월)·단기(예: 63일=3개월) 둘 다 만들 수 있게 파라미터화
    (지호 님 지적 — 12개월 기준으론 "상승장"이 나오는데 최근 1~3개월만 보면 확실히 둔화돼
    있어 체감상 "횡보"와 다른 답이 나왔음, 두 창을 나란히 보기 위함).
    기준(단순·투명성 우선, 정교한 다상태 HMM 등은 아님):
      · 하락장: trail_days 고점 대비 낙폭 <=bear_dd 또는 trail_days 수익률 <=bear_ret
      · 상승장: trail_days 수익률 >=bull_ret 이고 낙폭 >bear_dd/2(고점 근처 유지)
      · 횡보장: 위 둘 다 아님(추세가 강하지 않은 나머지 — 디폴트 버킷)."""
    min_days = min_days if min_days is not None else trail_days
    s = spy_close.reset_index(drop=True)
    tags = []
    for i in range(n_points):
        t = i * BP.MONTH
        if t < min_days or t >= len(s):
            tags.append("unknown"); continue
        window = s.iloc[max(0, t - trail_days + 1):t + 1]
        trail_ret = float(s.iloc[t] / s.iloc[t - trail_days] - 1)
        dd = float(s.iloc[t] / window.max() - 1)
        if dd <= bear_dd or trail_ret <= bear_ret:
            tags.append("bear")
        elif trail_ret >= bull_ret and dd > bear_dd / 2:
            tags.append("bull")
        else:
            tags.append("sideways")
    return tags


def _by_regime(diffs: np.ndarray, tags: list[str]) -> dict:
    out = {}
    for regime in ("bull", "bear", "sideways", "unknown"):
        idx = [i for i, t in enumerate(tags) if t == regime and i < len(diffs)]
        if not idx:
            continue
        vals = diffs[idx]
        out[regime] = {"n_months": len(idx), "mean_excess_monthly_pct": round(float(vals.mean()) * 100, 3),
                      "pct_positive": round(float((vals > 0).mean()) * 100, 1)}
    return out


def _paired(nav_a, nav_b, label, regime_sets=None):
    r_a, r_b = SP._monthly_returns(nav_a), SP._monthly_returns(nav_b)
    n = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:n], r_b[:n]
    regime_breakdowns = None
    if regime_sets is not None:
        regime_breakdowns = {}
        for set_name, tags in regime_sets.items():
            bd = _by_regime(r_a - r_b, tags[:n])
            regime_breakdowns[set_name] = bd
            for regime, stats in bd.items():
                _log(f"  [{label}][{set_name}/{regime}] n={stats['n_months']}개월 · 월평균 초과수익 "
                     f"{stats['mean_excess_monthly_pct']:+.3f}%p · {stats['pct_positive']}% 양수")
    tstat, pval = SP._paired_ttest(r_a, r_b)
    rng = np.random.default_rng(SEED)
    n_blocks_needed = int(np.ceil(n / BLOCK))
    cagr_diffs = np.empty(N_BOOT)
    sharpe_diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=n_blocks_needed)
        bidx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        cagr_diffs[i] = SP._cagr_from_monthly(r_a[bidx]) - SP._cagr_from_monthly(r_b[bidx])
        sharpe_diffs[i] = SP._sharpe_from_monthly(r_a[bidx]) - SP._sharpe_from_monthly(r_b[bidx])
    cagr_lo, cagr_hi = (float(v) for v in np.percentile(cagr_diffs, [2.5, 97.5]))
    cagr_mean, cagr_pos = float(cagr_diffs.mean()), float((cagr_diffs > 0).mean()) * 100
    sharpe_lo, sharpe_hi = (float(v) for v in np.percentile(sharpe_diffs, [2.5, 97.5]))
    _log(f"[{label}] t={tstat:+.2f} p={pval:.3f} · CAGR차이 95%CI [{cagr_lo:+.2f}%p,{cagr_hi:+.2f}%p] "
         f"평균{cagr_mean:+.2f}%p({cagr_pos:.1f}%양수) · 샤프차이95%CI [{sharpe_lo:+.3f},{sharpe_hi:+.3f}]")
    out = {"n": n, "paired_ttest": {"t": round(float(tstat), 3), "p": round(float(pval), 4)},
          "cagr_diff_bootstrap": {"mean": round(cagr_mean, 2), "ci95_lo": round(cagr_lo, 2),
                                  "ci95_hi": round(cagr_hi, 2), "pct_positive": round(cagr_pos, 1)},
          "sharpe_diff_bootstrap": {"mean": round(float(sharpe_diffs.mean()), 3),
                                   "ci95_lo": round(sharpe_lo, 3), "ci95_hi": round(sharpe_hi, 3)}}
    if regime_breakdowns is not None:
        out["by_regime"] = regime_breakdowns
    return out


def run(years=YEARS, ratio=W_ALGO, save=True):
    algo_nav = _algo_nav(years)
    _log(f"알고리즘(topn8) 구축 완료: {algo_nav.index[0].date()}~{algo_nav.index[-1].date()}")

    partner_navs = {}
    for t in PARTNERS:
        hist = SP.R.download_histories([t], period="max").get(t)
        if hist is None or hist.empty:
            raise RuntimeError(f"{t} 시세 조회 실패")
        idx = algo_nav.index.intersection(hist.index)
        if len(idx) < 60:
            raise RuntimeError(f"{t} 공통구간 부족(n={len(idx)})")
        partner_navs[t] = hist

    # 공통 구간(셋 다 존재하는 날짜)으로 알고리즘 NAV까지 통일 정렬
    common_idx = algo_nav.index
    for t, s in partner_navs.items():
        common_idx = common_idx.intersection(s.index)
    if len(common_idx) < 60:
        raise RuntimeError(f"3자산 공통구간 부족(n={len(common_idx)})")
    algo_nav = algo_nav.reindex(common_idx)
    algo_nav = algo_nav / algo_nav.iloc[0]

    blends = {}
    for t in PARTNERS:
        s = partner_navs[t].reindex(common_idx).ffill()
        s = s / s.iloc[0]
        blends[t] = SP.CS.mix_nav(algo_nav, s, ratio)

    # 장세 분류(2026-07-29, 지호 님 요청 — "횡보/상승/하락장 구분해서도 보자, 지금은 횡보장").
    # 12개월 기준(장기)만 봤더니 SPY가 고점대비 -2.2%(거의 신고가)·추이 +17.6%로 "상승장"
    # 판정이 나와 지호 님 체감("지금 횡보장")과 어긋났다 — 최근 1~3개월 페이스는 뚜렷이
    # 둔화돼 있어(3개월 +3.9%) 단기 창을 따로 만들어 나란히 비교(지호 님 재요청).
    spy_hist = SP.R.download_histories(["SPY"], period="max").get("SPY")
    spy_aligned = spy_hist.reindex(common_idx).ffill()
    n_months = len(range(0, len(common_idx) - BP.MONTH, BP.MONTH))   # _monthly_returns()와 동일 인덱싱
    regime_sets = {
        "long_12m": _spy_regime_tags(spy_aligned, n_months, trail_days=252,
                                     bull_ret=0.20, bear_ret=-0.10, bear_dd=-0.15),
        "short_3m": _spy_regime_tags(spy_aligned, n_months, trail_days=63,
                                     bull_ret=0.08, bear_ret=-0.08, bear_dd=-0.10, min_days=252),
    }
    for set_name, tags in regime_sets.items():
        counts = {r: tags.count(r) for r in ("bull", "bear", "sideways", "unknown")}
        _log(f"장세 분류[{set_name}]: {counts} · 가장 최근 시점 = {tags[-1] if tags else 'unknown'}")

    pure_full = SP.CS.stats(algo_nav)
    _log(f"순수 알고리즘(100:0): CAGR {pure_full['cagr_pct']}% 샤프 {pure_full['sharpe']} MDD {pure_full['mdd_pct']}%")
    blend_full = {}
    for t in PARTNERS:
        s = SP.CS.stats(blends[t])
        blend_full[t] = s
        _log(f"알고리즘{int(ratio*100)}+{t}{int((1-ratio)*100)}: CAGR {s['cagr_pct']}% 샤프 {s['sharpe']} MDD {s['mdd_pct']}%")

    vs_pure = {t: _paired(blends[t], algo_nav, f"{t}블렌드-vs-순수", regime_sets=regime_sets) for t in PARTNERS}
    cross = {}
    for i, a in enumerate(PARTNERS):
        for b in PARTNERS[i + 1:]:
            cross[f"{a}_vs_{b}"] = _paired(blends[a], blends[b], f"{a}블렌드-vs-{b}블렌드")

    sub_rows = []
    for label, a, b in SUBS:
        row = {"period": label}
        sp_pure = SP.CS.stats(algo_nav, a, b)
        if sp_pure is None:
            sub_rows.append({"period": label, "note": "표본 부족"}); continue
        row["pure_cagr"] = sp_pure["cagr_pct"]; row["pure_sharpe"] = sp_pure["sharpe"]
        for t in PARTNERS:
            sb = SP.CS.stats(blends[t], a, b)
            if sb:
                row[f"{t}_cagr"] = sb["cagr_pct"]; row[f"{t}_sharpe"] = sb["sharpe"]
        sub_rows.append(row)
        _log(f"{label}: 순수 {row.get('pure_cagr')}%/{row.get('pure_sharpe')} · " +
             " · ".join(f"{t} {row.get(t+'_cagr')}%/{row.get(t+'_sharpe')}" for t in PARTNERS))

    payload = {"as_of": algo_nav.index[-1].date().isoformat(), "years": years, "ratio": ratio,
              "full_period": {"pure": pure_full, **{t: blend_full[t] for t in PARTNERS}},
              "vs_pure": vs_pure, "cross_blend": cross, "subperiods": sub_rows,
              "regime_classification": {
                  "long_12m": {"method": "SPY 12개월 추이수익률+고점대비낙폭(하락:낙폭<=-15%"
                              "또는 추이<=-10% / 상승:추이>=+20%이고 낙폭>-7.5% / 횡보:나머지)",
                              "counts": {r: regime_sets["long_12m"].count(r) for r in
                                        ("bull", "bear", "sideways", "unknown")},
                              "current_regime": regime_sets["long_12m"][-1] if regime_sets["long_12m"] else "unknown"},
                  "short_3m": {"method": "SPY 3개월 추이수익률+고점대비낙폭(하락:낙폭<=-10%"
                              "또는 추이<=-8% / 상승:추이>=+8%이고 낙폭>-5% / 횡보:나머지)",
                              "counts": {r: regime_sets["short_3m"].count(r) for r in
                                        ("bull", "bear", "sideways", "unknown")},
                              "current_regime": regime_sets["short_3m"][-1] if regime_sets["short_3m"] else "unknown"},
              },
              "note": "SPMO=S&P500모멘텀ETF, VOO=S&P500시총가중ETF, RSP=S&P500동일가중ETF. "
                      "블렌드는 월간리밸(core_satellite_kr.mix_nav), 알고리즘=topn8 라이브 "
                      "(1:2:2·섹터캡2·ma200_backup=False)."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_algo_blend_spmo_voo_rsp.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log("저장: output/us_algo_blend_spmo_voo_rsp.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=YEARS)
    ap.add_argument("--ratio", type=float, default=W_ALGO)
    a = ap.parse_args()
    run(a.years, a.ratio)
