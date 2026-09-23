#!/usr/bin/env python3
"""
kr_sector_rotation.py — "섹터 랠리(건설·철강 등)에 올라탈 로테이션 알고리즘이
가능한가"를 실제로 검증(2026-09-15, 지호 님 요청).

설계(신규 탐색 아님 — 기존 라이브 KOSPI/KOSDAQ 추세신호와 동일한 사전등록 파라미터
그대로 사용 — market_signals.PARAMS["equity"]: 200일선·밴드1%·확인3일):
  1) 섹터별: KRX가 공식 발행하는 17개 산업 지수(건설/철강/반도체/헬스케어 등,
     2015-01~현재)에 그 추세신호를 각각 적용 — 매수후보유 대비 유의한지 페어드
     블록부트스트랩으로 확인.
  2) 로테이션 포트폴리오: 매일 "그 시점 ON 상태인 섹터들"에 동일가중 배분(OFF만
     있으면 현금), 노출 변경 시 비용 반영. vs (a) 코스피 매수후보유, (b) 17개 섹터
     동일가중 매수후보유(=로테이션의 "타이밍" 효과만 따로 분리해서 보기 위한 대조군).

실행: python -m research.kr.kr_sector_rotation
결과: output/kr_sector_rotation.json
"""
from __future__ import annotations
import sys, json
import numpy as np
from pykrx import stock as K

from research.regime.backtest_regime_assets import regime_series, simulate, _cagr, _mdd, _ulcer
from research.regime.ma_trend_strategies import bootstrap_vs_bh, _sharpe

SECTORS = {
    "건설": "5052", "철강": "5049", "반도체": "5044", "헬스케어": "5045",
    "은행": "5046", "보험": "5056", "증권": "5054", "자동차": "5043",
    "에너지화학": "5048", "운송": "5057", "유틸리티": "5065", "방송통신": "5051",
    "정보기술": "5064", "필수소비재": "5062", "경기소비재": "5061",
    "기계장비": "5055", "K콘텐츠": "5063",
}
KOSPI_CODE = "1001"
START, END = "20150101", "20260915"
TREND_MA, BAND, CONFIRM = 200, 0.01, 3
COST_BPS = 20


def _log(m):
    print(f"[섹터로테이션] {m}", file=sys.stderr)


def fetch_index(code):
    df = K.get_index_ohlcv_by_date(START, END, code)
    close_col = df.columns[3]
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    closes = df[close_col].astype(float).to_numpy()
    return dates, closes


def main():
    _log("데이터 수집 시작")
    sector_data = {}
    for name, code in SECTORS.items():
        dates, closes = fetch_index(code)
        sector_data[name] = {"dates": dates, "closes": closes}
        _log(f"{name}({code}) {len(closes)}일")
    k_dates, k_closes = fetch_index(KOSPI_CODE)

    # 공통 날짜(전부 KRX 지수라 기본적으로 동일 거래일이어야 하지만 방어적으로 교집합)
    common = set(k_dates)
    for d in sector_data.values():
        common &= set(d["dates"])
    common = sorted(common)
    _log(f"공통 거래일 {len(common)}일")

    def _reindex(dates, closes, master):
        idx = {d: i for i, d in enumerate(dates)}
        return np.array([closes[idx[d]] for d in master])

    k_aligned = _reindex(k_dates, k_closes, common)
    for name, d in sector_data.items():
        d["aligned"] = _reindex(d["dates"], d["closes"], common)

    # 1) 섹터별 개별 추세추종 vs 매수후보유
    per_sector = {}
    exposures = {}
    for name, d in sector_data.items():
        closes = d["aligned"]
        exp = regime_series(closes, TREND_MA, BAND, CONFIRM)
        exposures[name] = exp
        res = bootstrap_vs_bh(closes, exp, COST_BPS)
        per_sector[name] = res
        _log(f"{name}: CAGR {res.get('cagr')}% (매수후보유 {res.get('bh_cagr')}%) · "
             f"승률 {res.get('prob_beats_buyhold_pct')}%")

    # 2) 로테이션 포트폴리오: 매일 ON인 섹터에 동일가중(전일 노출 기준, 미래참조 없음)
    names = list(SECTORS.keys())
    exp_matrix = np.vstack([np.nan_to_num(exposures[n], nan=0.0) for n in names])  # (n_sectors, T)
    rets = np.vstack([np.diff(sector_data[n]["aligned"]) / sector_data[n]["aligned"][:-1] for n in names])
    exp_lag = exp_matrix[:, :-1]  # t-1 노출로 t일 수익 반영
    n_on = exp_lag.sum(axis=0)
    weights = np.divide(exp_lag, n_on, out=np.zeros_like(exp_lag), where=n_on > 0)
    port_ret_gross = (weights * rets).sum(axis=0)
    turnover = np.abs(np.diff(weights, axis=1, prepend=weights[:, :1])).sum(axis=0)
    cost = turnover * (COST_BPS / 10000.0)
    port_ret = port_ret_gross - cost
    port_nav = np.cumprod(1 + port_ret)

    kospi_ret = np.diff(k_aligned) / k_aligned[:-1]
    kospi_nav = np.cumprod(1 + kospi_ret)

    ew_all_ret = rets.mean(axis=0)  # 17개 섹터 동일가중 매수후보유(리밸런싱 없음 근사 — 일별 리밸 가정)
    ew_all_nav = np.cumprod(1 + ew_all_ret)

    n_days = len(port_ret)

    def _block_bootstrap_diff(ra, rb, block=63, n_boot=3000, seed=7):
        n = min(len(ra), len(rb))
        ra, rb = ra[:n], rb[:n]
        n_blocks = n // block
        rng = np.random.default_rng(seed)
        d_cagr = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_blocks, n_blocks)
            sel = np.concatenate([np.arange(j * block, (j + 1) * block) for j in idx])
            nav_a = np.cumprod(1 + ra[sel]); nav_b = np.cumprod(1 + rb[sel])
            d_cagr[i] = _cagr(nav_a, len(sel)) - _cagr(nav_b, len(sel))
        return {"delta_cagr_ci90": (round(float(np.percentile(d_cagr, 5)), 3),
                                    round(float(np.percentile(d_cagr, 95)), 3)),
                "prob_better_pct": round(float((d_cagr > 0).mean()) * 100, 1)}

    vs_kospi = _block_bootstrap_diff(port_ret, kospi_ret)
    vs_ew_all = _block_bootstrap_diff(port_ret, ew_all_ret)

    summary = {
        "period": f"{common[0]}~{common[-1]}", "n_days": n_days,
        "params": {"trend_ma": TREND_MA, "band": BAND, "confirm": CONFIRM, "cost_bps": COST_BPS},
        "rotation_portfolio": {
            "cagr_pct": round(_cagr(port_nav, n_days), 2),
            "sharpe": round(_sharpe(port_ret), 2),
            "mdd_pct": round(_mdd(port_nav), 1),
            "ulcer": round(_ulcer(port_nav), 2),
        },
        "kospi_buyhold": {
            "cagr_pct": round(_cagr(kospi_nav, n_days), 2),
            "sharpe": round(_sharpe(kospi_ret), 2),
            "mdd_pct": round(_mdd(kospi_nav), 1),
            "ulcer": round(_ulcer(kospi_nav), 2),
        },
        "sector_ew_buyhold_all17": {
            "cagr_pct": round(_cagr(ew_all_nav, n_days), 2),
            "sharpe": round(_sharpe(ew_all_ret), 2),
            "mdd_pct": round(_mdd(ew_all_nav), 1),
            "ulcer": round(_ulcer(ew_all_nav), 2),
        },
        "rotation_vs_kospi_bootstrap": vs_kospi,
        "rotation_vs_ew_all17_bootstrap": vs_ew_all,
        "per_sector_trend_vs_buyhold": per_sector,
    }
    with open("output/kr_sector_rotation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sector_trend_vs_buyhold"},
                     ensure_ascii=False, indent=2))
    _log("저장: output/kr_sector_rotation.json")


if __name__ == "__main__":
    main()
