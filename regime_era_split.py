#!/usr/bin/env python3
"""
regime_era_split.py — BTC·ETH 라이브 레짐 신호의 시대별(era) 강건성 + 데이터 결측 점검
(2026-08-22).

배경: 2026-08-07~09 세션(코드가 보존되지 않고 output/us_bitcoin_signal_robustness.png·
us_signal_binary_switch_allassets.png 차트 PNG만 남음)에서 "BTC 라이브 레짐 신호가
2014-2018·2018-2022 구간엔 매수후보유를 이겼지만 2022-2026 구간엔 밑돌았다(~19% vs
~21% CAGR)"는 결과가 있었다는데 재현 가능한 코드가 없었다. 이 스크립트는 그 결측을
채운다 — 동일한 차트를 그대로 재현하진 않지만 동일 질문(시대별로 쪼개도 신호가 이기는가)에
`backtest_regime_assets.simulate()`로 직접 답한다. ETH에도 동일 방법론을 처음 적용한다.

"라이브 레짐 규칙"의 정의: market_signals.PARAMS["crypto"] / backtest_regime_assets.BTC_CURRENT
(120일선·±3%밴드·확인3일). 모멘텀 필터(3개월 절대모멘텀)는 포함하지 않음 — STRATEGY.md §1의
"필터 자체 효과" 헤드라인 수치(MDD -83.4%→-60.0%, CAGR 33.5%→45.0% 등)와
output/regime_backtest_btc.json의 stage1.current가 정확히 이 정의(추세선·밴드·확인일수만,
모멘텀 AND게이트 제외)로 계산된 것과 일치함을 확인 후 사용했다(모멘텀 AND게이트는 Stage2의
별도 조건부 실험일 뿐, 이 프로젝트가 "필터 효과"라 부르는 수치의 정의가 아니다).

ETH는 독립 라이브 파라미터가 없으므로 BTC_CURRENT를 그대로 적용한다(기존
eth_confirmatory_check와 동일한 접근 — ETH 자체 그리드 결과는 run_eth_grid.py 참고).

시대 구간: BTC는 2014-2018/2018-2022/2022-2026(오차드 차트와 동일한 3구간, 연도 경계).
ETH는 2017-11 시작이라 3구간 중 첫 구간이 2개월뿐이라 무의미 — 과제 지시대로 2구간
(2017-2022/2022-2026)만 사용.

신호 노출은 전체 이력에 대해 한 번만 계산(연속 신호 — 라이브 시스템이 실제로 돌아가는
방식과 동일)한 뒤, 시대 경계로 수익률을 잘라 시대 "내에서" 복리 계산한다. 시대마다 처음부터
MA를 다시 쌓게 하면(구간별 재계산) 실제로 없던 워밍업 손실을 인위적으로 만들어낸다.

실행: python regime_era_split.py
결과: output/regime_era_split.json + output/regime_data_gap_check.json (콘솔에도 요약 출력)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from backtest_regime_assets import (
    fetch, regime_series, momentum_ok, simulate, _cagr, BTC_CURRENT, BTC_MOM_CURRENT, COST_BPS, _log
)

BTC_ERAS = [("2014-2018", "2014-01-01", "2018-01-01"),
            ("2018-2022", "2018-01-01", "2022-01-01"),
            ("2022-2026", "2022-01-01", "2027-01-01")]
ETH_ERAS = [("2017-2022", "2017-01-01", "2022-01-01"),
            ("2022-2026", "2022-01-01", "2027-01-01")]


def _regime_only_exposure(closes: np.ndarray, params: dict) -> np.ndarray:
    return regime_series(closes, **params)


def _regime_and_momentum_exposure(closes: np.ndarray, params: dict) -> np.ndarray:
    """market_signals.PARAMS의 '라이브 규칙'을 더 넓게 해석한 버전 — 레짐(추세선) ON뿐
    아니라 절대모멘텀(3개월)까지 AND로 요구(Stage2 정의와 동일). regime-only 버전과
    나란히 병기하는 이유: 정정 없는 원 STRATEGY.md §1 헤드라인 수치는 regime-only로
    확인되지만(별도 검증됨), 분실된 2026-08 세션 차트가 실제로 어느 정의를 썼는지는
    코드가 없어 알 수 없다 — 두 정의 모두로 시대분할을 확인해 결론이 정의에 좌우되는지
    점검한다."""
    regime = regime_series(closes, **params)
    mok = momentum_ok(closes, BTC_MOM_CURRENT)
    return np.where((regime == 1.0) & (mok == 1.0), 1.0,
                     np.where(np.isnan(regime) | np.isnan(mok), np.nan, 0.0))


def era_performance(closes_s: pd.Series, params: dict, cost_bps: float, eras: list,
                     exposure_fn=_regime_only_exposure) -> list:
    closes = closes_s.to_numpy()
    dates = closes_s.index.to_numpy()
    exp = exposure_fn(closes, params)
    m = simulate(closes, exp, cost_bps)
    strat_ret = m["strat_ret"]
    bh_ret = np.diff(closes) / closes[:-1]
    ret_dates = dates[1:]          # strat_ret[t]/bh_ret[t]는 ret_dates[t] 날짜에 확정되는 수익
    out = []
    for label, start, end in eras:
        mask = (ret_dates >= np.datetime64(start)) & (ret_dates < np.datetime64(end))
        n = int(mask.sum())
        if n < 20:
            out.append({"era": label, "n_days": n, "note": "표본 부족(20일 미만) — 생략"})
            continue
        sr, br = strat_ret[mask], bh_ret[mask]
        nav_s, nav_b = np.cumprod(1 + sr), np.cumprod(1 + br)
        sig_cagr, bh_cagr = _cagr(nav_s, n), _cagr(nav_b, n)
        out.append({"era": label, "n_days": n,
                     "start": str(pd.Timestamp(ret_dates[mask][0]).date()),
                     "end": str(pd.Timestamp(ret_dates[mask][-1]).date()),
                     "signal_cagr": round(sig_cagr, 2), "buyhold_cagr": round(bh_cagr, 2),
                     "excess_cagr": round(sig_cagr - bh_cagr, 2),
                     "signal_beats_bh": bool(sig_cagr > bh_cagr)})
    return out


def gap_check(closes_s: pd.Series, name: str, max_gap_days: int = 3) -> dict:
    """캘린더일 기준 결측 구간 탐지 — 코인은 매일(주말 포함) 거래되므로 max_gap_days(기본 3일)를
    넘는 공백은 주말이 아니라 데이터 문제일 가능성이 있어 의심 대상으로 표시."""
    idx = closes_s.index
    day_diffs = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    gap_pos = np.where(day_diffs > max_gap_days)[0]
    gaps = [{"after": str(pd.Timestamp(idx[i]).date()), "before": str(pd.Timestamp(idx[i + 1]).date()),
             "gap_days": int(day_diffs[i])} for i in gap_pos]
    return {"asset": name, "n_rows": len(idx), "start": str(idx.min().date()), "end": str(idx.max().date()),
            "max_gap_days_threshold": max_gap_days, "n_gaps": len(gaps), "gaps": gaps}


def main():
    btc_s = fetch("BTC-USD", "output/regime_price_cache_btc.pkl")
    eth_s = fetch("ETH-USD", "output/regime_price_cache_eth.pkl")

    era_result = {
        "live_rule_definition": BTC_CURRENT,
        "note": "두 정의 병기 — regime_only: STRATEGY.md §1 '필터 효과' 헤드라인 수치와 동일 "
                "정의(추세선·밴드·확인일수만). regime_and_momentum: 3개월 절대모멘텀까지 "
                "AND(Stage2 정의). 분실된 2026-08 세션 차트가 어느 쪽을 썼는지 코드가 없어 "
                "확인 불가 — 결론이 정의에 좌우되는지 점검하기 위해 둘 다 계산.",
        "regime_only": {
            "btc": {"eras_used": [e[0] for e in BTC_ERAS],
                    "by_era": era_performance(btc_s, BTC_CURRENT, COST_BPS["btc"], BTC_ERAS,
                                               _regime_only_exposure)},
            "eth": {"eras_used": [e[0] for e in ETH_ERAS],
                    "note": "ETH 독립 라이브 파라미터 없음 — BTC_CURRENT 그대로 적용",
                    "by_era": era_performance(eth_s, BTC_CURRENT, COST_BPS["btc"], ETH_ERAS,
                                               _regime_only_exposure)},
        },
        "regime_and_momentum": {
            "btc": {"eras_used": [e[0] for e in BTC_ERAS],
                    "by_era": era_performance(btc_s, BTC_CURRENT, COST_BPS["btc"], BTC_ERAS,
                                               _regime_and_momentum_exposure)},
            "eth": {"eras_used": [e[0] for e in ETH_ERAS],
                    "note": "ETH 독립 라이브 파라미터 없음 — BTC_CURRENT/BTC_MOM_CURRENT 그대로 적용",
                    "by_era": era_performance(eth_s, BTC_CURRENT, COST_BPS["btc"], ETH_ERAS,
                                               _regime_and_momentum_exposure)},
        },
    }
    with open("output/regime_era_split.json", "w", encoding="utf-8") as f:
        json.dump(era_result, f, ensure_ascii=False, indent=2)
    _log("저장: output/regime_era_split.json")
    for defn in ("regime_only", "regime_and_momentum"):
        for asset in ("btc", "eth"):
            for row in era_result[defn][asset]["by_era"]:
                _log(f"  [{defn}/{asset}] {row}")

    gaps = {"btc": gap_check(btc_s, "btc"), "eth": gap_check(eth_s, "eth")}
    with open("output/regime_data_gap_check.json", "w", encoding="utf-8") as f:
        json.dump(gaps, f, ensure_ascii=False, indent=2)
    _log("저장: output/regime_data_gap_check.json")
    for asset in ("btc", "eth"):
        g = gaps[asset]
        _log(f"  [{asset}] {g['n_rows']}행 {g['start']}~{g['end']} · 3일 초과 공백 {g['n_gaps']}건")


if __name__ == "__main__":
    main()
