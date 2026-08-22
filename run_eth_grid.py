#!/usr/bin/env python3
"""
run_eth_grid.py — ETH-USD 전용 독립 Stage1(+Stage2) 그리드 검증 (2026-08-22).

배경: backtest_regime_assets.py의 기존 eth_confirmatory_check(main() 내부)는 BTC의
"최우수 후보" 파라미터를 ETH 데이터에 그대로 적용해 방향성만 확인하는 1줄짜리 확인일 뿐,
ETH 전용 그리드 탐색이 아니었다. 이 스크립트는 동일한 run_asset() 파이프라인(Stage1 →
Stage2 → 쌍대 블록부트스트랩 → PBO/DSR 게이트)을 ETH-USD에 대해 처음부터 독립적으로
돌린다 — 재구현 없이 backtest_regime_assets.py의 함수를 그대로 재사용.

BTC_CURRENT/BTC_GRID/BTC_MOM_CURRENT/BTC_MOM_GRID를 ETH의 "현행 파라미터"·그리드 축으로
쓰는 이유: ETH는 아직 독립적으로 채택된 라이브 파라미터가 없다(금·비트코인도 처음엔 주식류
기본값을 최초 앵커로 썼던 것과 동일한 상황 — market_signals.py는 코인 전체(BTC·ETH 공통
카테고리)에 PARAMS["crypto"]를 적용하는 구조이므로, "ETH가 BTC 검증으로 정해진 규칙을
그대로 써도 되는가"를 묻는 게 이 그리드의 목적이다). 그리드 자체가 BTC용으로 설계됐다는
한계는 있음(ETH 고유 변동성 특성에 최적화된 그리드가 아닐 수 있음) — 결과 해석 시 명시.

실행: python run_eth_grid.py
결과: output/regime_backtest_eth.json
"""
from __future__ import annotations
import json

from backtest_regime_assets import (
    run_asset, BTC_CURRENT, BTC_GRID, BTC_MOM_CURRENT, BTC_MOM_GRID, COST_BPS, _log
)


def main():
    eth = run_asset("eth", "ETH-USD", BTC_CURRENT, BTC_GRID, BTC_MOM_CURRENT, BTC_MOM_GRID,
                     COST_BPS["btc"], do_bootstrap=True)
    with open("output/regime_backtest_eth.json", "w", encoding="utf-8") as f:
        json.dump(eth, f, ensure_ascii=False, indent=2)
    _log("저장: output/regime_backtest_eth.json")


if __name__ == "__main__":
    main()
