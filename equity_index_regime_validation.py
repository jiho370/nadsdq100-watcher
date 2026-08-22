#!/usr/bin/env python3
"""
equity_index_regime_validation.py — S&P500(^GSPC)·코스피(^KS11) 추세 "매수신호" 최초 정식
검증 (2026-08-22, 지호 님 재질문 대응).

배경: market_signals.PARAMS["equity"](200일선·±1%·확인3일·12-1모멘텀)는 Zakamulin·Faber(2007)
등 일반 문헌 근거로 채택됐을 뿐(§0), 이 프로젝트 자체 데이터로 금·비트코인처럼
Stage1(그리드)+Stage2(모멘텀)+PBO/DSR+쌍대부트스트랩을 거친 적이 **단 한 번도 없다**
(`backtest_regime_assets.py`는 원래 금·비트코인 전용으로만 만들어졌음, `market_signals.py`·
`daily_ai_report.py`·`weekly_report.py`·`us_intraday_timing_sensitivity.py`에 GSPC/KS11
문자열이 있지만 전부 "표시"용이지 "검증"용이 아님 — grep으로 확인). 이 스크립트가 그
공백을 처음 메운다. 금(GOLD_GRID)과 동일한 그리드를 재사용(금도 원래 '주식'류 파라미터를
그대로 물려받은 자산이라 같은 그리드가 공정한 비교 기준).

실행: python equity_index_regime_validation.py
결과: output/regime_backtest_{spx,kospi}.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from backtest_regime_assets import run_asset, GOLD_GRID, GOLD_MOM_GRID

EQUITY_CURRENT = {"trend_ma": 200, "band": 0.01, "confirm": 3}
EQUITY_MOM_CURRENT = "12_1"
COST_BPS = 5


def _log(m): print(f"[지수레짐검증] {m}", file=sys.stderr)


def main():
    os.makedirs("output", exist_ok=True)
    for name, ticker in [("spx", "^GSPC"), ("kospi", "^KS11")]:
        result = run_asset(name, ticker, EQUITY_CURRENT, GOLD_GRID, EQUITY_MOM_CURRENT,
                           GOLD_MOM_GRID, COST_BPS, do_bootstrap=True)
        path = f"output/regime_backtest_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        s1 = result["stage1"]
        _log(f"[{name}] 저장: {path}")
        _log(f"[{name}] Stage1 최우수: {s1['best']}")
        _log(f"[{name}] Stage1 현행(200/1%/3): {s1['current']}")
        _log(f"[{name}] 고원여부: {s1['plateau_ok']}")
        if "bootstrap_best_vs_current" in result:
            _log(f"[{name}] 최우수 vs 현행 부트스트랩: {result['bootstrap_best_vs_current']}")
        pbo = result.get("pbo_gate") or {}
        _log(f"[{name}] PBO={pbo.get('pbo', {}).get('pbo')} DSR={pbo.get('dsr', {}).get('dsr')} "
            f"passed={pbo.get('passed')}")


if __name__ == "__main__":
    main()
