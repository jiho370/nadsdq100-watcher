#!/usr/bin/env python3
"""
backtest_regime_fx.py — 원달러 환율(KRW=X) 레짐타이밍 검증 (2026-07-29).

배경(지호 님 지시): "환율 백테스트해보자 — 환율 자체를 매매 대상으로." market_signals.py의
6자산 레짐엔진(§STRATEGY.md §1)엔 원래 환율이 없다. `regime_kr.py`가 USDKRW 20일 변화
부호(won_weak/won_strong)를 한국 팩터 IC 진단용으로만 쓰고 있을 뿐, 환율 자체를 추세추종
매매(또는 미국주식 보유의 동적 환헤지) 대상으로 백테스트한 적은 이번이 처음이다.

`backtest_regime_assets.py`(금·비트코인 검증, 2026-07-16)와 완전히 동일한 방법론을
그대로 재사용한다(로직 재구현 없이 import) — Stage1(추세선×밴드×확인일수 그리드,
Ulcer 중심 composite score+CAGR손실예산+고원조건) → Stage2(모멘텀 조건부 스윕) →
PBO/DSR 게이트 → 쌍대 블록부트스트랩. 유일한 차이는 거래비용: 금(5bp)·비트코인(30bp)은
그 시장의 실제 스프레드가 명확하지만, 원달러는 실제로 어떤 수단으로 노출을 조절하느냐에
따라 비용이 크게 갈린다(KRX 달러선물 수 bp ~ 은행 현찰환전 100bp+) — 그래서 이 스크립트는
단일 비용을 가정하지 않고 비용 스윕(0/5/10/20/50bp)으로 결론의 민감도를 직접 보고한다.

해석: exposure=1(ON)="달러 보유"(원화 대비 환노출), exposure=0(OFF)="원화로 헤지"
(그 구간 환손익 0). 즉 이 백테스트가 답하는 질문은 "원달러 환율 자체를 사고파는 게
의미있나"이자 곧바로 "미국주식 보유분의 환헤지를 추세로 타이밍해야 하나"와 동일한 질문
(현재 라이브: 어느 쪽도 안 함 — 미국주식은 항상 환노출 그대로).

"현행" 앵커: 라이브에 이미 정해진 파라미터가 없으므로(원래 6자산에 없던 자산), 금이
market_signals.py에 처음 편입될 때 '주식'류 파라미터를 그냥 물려받았던 것과 동일한 방식
으로 지수(equity) 기본값(200일선·±1%·확인3일·12-1모멘텀)을 "검증 없이 그대로 갖다 쓰면
이렇게 된다"는 비교 앵커로 사용한다.

실행: python backtest_regime_fx.py --stage1
      python backtest_regime_fx.py --all             # 전부 + 비용스윕 + 부트스트랩
      python backtest_regime_fx.py --self-test
결과: output/regime_backtest_fx.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

import backtest_regime_assets as RA

# ------------------------- 사전등록 그리드 -------------------------
# 트렌드/밴드 범위는 금(GLD)과 동일 격자를 재사용 — 원달러도 연변동성(~14%)이 GLD와
# 비슷한 저변동 매크로자산이라 같은 격자가 합리적 출발점(암묵적 가정, 아래 한계에 명시).
FX_CURRENT = {"trend_ma": 200, "band": 0.01, "confirm": 3}   # 앵커: 지수 기본값 그대로 물려받은 경우
FX_MOM_CURRENT = "12_1"
FX_GRID = {"trend_ma": [100, 150, 200, 250, 300], "band": [0.0, 0.005, 0.01, 0.02], "confirm": [1, 3, 5]}
FX_MOM_GRID = ["6m", "9m", "12_1", "12m"]

COST_BPS_SWEEP = [0, 5, 10, 20, 50]   # 편도. 실제값 불확실 → 스윕으로 민감도 직접 보고
COST_BPS_DEFAULT = 10                  # 메인 리포트용 대표값(달러선물/환헤지ETF 스프레드 근사)

# 2026-07-30 지호 님 지시("대대적으로 다양한 걸 백테스트해보고 젤 나은걸 확인") — 60조합
# 기본 격자는 금(GLD)과 동일 범위를 그냥 재사용한 것이었으므로, 원달러 자체 특성(연변동성
# ~14%, 금·지수 중간)을 반영해 훨씬 넓고 촘촘한 격자로 재탐색. 528조합(11×8×6).
# ⚠ §6-P-3 교훈: 격자를 넓힐수록 "우연히 좋아 보이는 조합"도 늘어 PBO/DSR은 대체로
# 악화되는 경향이 있다 — 결론은 "방향"에서 취하고 "정확한 최적점"은 취하지 말 것.
FX_GRID_WIDE = {
    "trend_ma": [20, 50, 75, 100, 120, 150, 175, 200, 250, 300, 350],
    "band": [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03],
    "confirm": [1, 2, 3, 5, 7, 10],
}
FX_MOM_GRID_WIDE = ["1m", "3m", "6m", "9m", "12_1", "12m"]


def _log(m): print(f"[환율레짐백테스트] {m}", file=sys.stderr)


def run_cost_sweep(closes: np.ndarray, params: dict) -> list:
    """동일 파라미터(Stage1 최우수)를 비용만 바꿔가며 재실행 — 원달러는 실제 매매
    수단(달러선물/환헤지ETF/외화예금)마다 비용이 크게 달라 이 민감도가 핵심 결론이다."""
    rows = []
    exp = RA.regime_series(closes, params["trend_ma"], params["band"], params["confirm"])
    for bps in COST_BPS_SWEEP:
        m = RA.simulate(closes, exp, bps)
        rows.append({"cost_bps": bps, "cagr": round(m["cagr"], 2), "bh_cagr": round(m["bh_cagr"], 2),
                     "ulcer": round(m["ulcer"], 2), "bh_ulcer": round(m["bh_ulcer"], 2),
                     "mdd": round(m["mdd"], 1), "bh_mdd": round(m["bh_mdd"], 1),
                     "score": RA.composite_score(m)})
    return rows


def run_decade_split(closes: np.ndarray, index, params: dict, cost_bps: float) -> dict:
    """반기 분할 대신 원달러 특유의 두 국면(GFC~2010년대 저변동 vs 2020년대 고변동 강달러기)
    으로 분리해 강건성 확인 — split 기준 2020-01-01(BTC 검증과 동일 관행)."""
    import pandas as pd
    s = pd.Series(closes, index=index)
    split_date = "2020-01-01"
    half1 = s[s.index < split_date].to_numpy()
    half2 = s[s.index >= split_date].to_numpy()
    out = {}
    for label, arr in [("~2019", half1), ("2020~", half2)]:
        if len(arr) < params["trend_ma"] + 30:
            out[label] = None
            continue
        exp = RA.regime_series(arr, params["trend_ma"], params["band"], params["confirm"])
        m = RA.simulate(arr, exp, cost_bps)
        out[label] = {"score": RA.composite_score(m), "cagr": round(m["cagr"], 2),
                      "ulcer": round(m["ulcer"], 2), "mdd": round(m["mdd"], 1),
                      "off_episodes": m["off_episodes"], "n_days": len(arr)}
    return out


def main():
    ap = argparse.ArgumentParser(description="원달러 환율(KRW=X) 레짐타이밍 검증")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="528조합(11×8×6) 확장 격자로 재탐색, output/regime_backtest_fx_wide.json에 저장")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    grid = FX_GRID_WIDE if args.wide else FX_GRID
    mom_grid = FX_MOM_GRID_WIDE if args.wide else FX_MOM_GRID
    out_path = "output/regime_backtest_fx_wide.json" if args.wide else "output/regime_backtest_fx.json"

    os.makedirs("output", exist_ok=True)
    price = RA.fetch("KRW=X", "output/regime_price_cache_fx.pkl")
    closes = price.to_numpy()

    payload = RA.run_asset("fx", "KRW=X", FX_CURRENT, grid, FX_MOM_CURRENT, mom_grid,
                           COST_BPS_DEFAULT, do_bootstrap=True)
    payload["date_range"] = [str(price.index.min().date()), str(price.index.max().date())]
    payload["grid_size"] = len(grid["trend_ma"]) * len(grid["band"]) * len(grid["confirm"])
    payload["note"] = ("exposure=1(ON)=달러보유(원화대비 환노출)/exposure=0(OFF)=원화헤지. "
                       "cost_bps는 실제 수단(달러선물/환헤지ETF/외화예금)에 따라 크게 달라 "
                       "고정값 대신 cost_sensitivity로 별도 보고.")

    best = payload["stage1"]["best"]
    if best:
        best_params = {"trend_ma": best["trend_ma"], "band": best["band"], "confirm": best["confirm"]}
        payload["cost_sensitivity"] = run_cost_sweep(closes, best_params)
        payload["decade_split_check"] = run_decade_split(closes, price.index, best_params, COST_BPS_DEFAULT)
        # 현행 앵커(FX_CURRENT)에 대해서도 동일 비용스윕 — "검증 없이 그냥 지수 기본값
        # 갖다 썼으면" 시나리오의 비용 민감도도 같이 보고
        payload["cost_sensitivity_anchor"] = run_cost_sweep(closes, FX_CURRENT)
        # 상위 15개 조합(전체 격자에서) — "정확히 이 지점" 대신 상위권 분포를 같이 보고
        rankable = [r for r in payload["stage1"]["rows"] if r["rankable"] and r["score"] != float("-inf")]
        rankable.sort(key=lambda r: r["score"], reverse=True)
        payload["top15"] = rankable[:15]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {out_path} (격자 {payload['grid_size']}조합)")

    # 요약 출력
    _log(f"데이터: {payload['date_range'][0]} ~ {payload['date_range'][1]} ({payload['n_days']}일)")
    _log(f"Stage1 최우수: {best}")
    _log(f"Stage1 앵커(지수기본값 그대로): {payload['stage1']['current']}")
    _log(f"고원 여부: {payload['stage1']['plateau_ok']}")
    if "bootstrap_best_vs_current" in payload:
        _log(f"쌍대부트스트랩(최우수 vs 앵커): {payload['bootstrap_best_vs_current']}")
    if "pbo_gate" in payload and payload["pbo_gate"]:
        g = payload["pbo_gate"]
        _log(f"PBO/DSR 게이트: PBO={g.get('pbo', {}).get('pbo')} DSR판정={g.get('dsr_verdict')}")
    if "cost_sensitivity" in payload:
        _log("비용 민감도(최우수 파라미터):")
        for r in payload["cost_sensitivity"]:
            _log(f"  {r['cost_bps']}bp: score={r['score']:.3f} CAGR={r['cagr']} vs B&H={r['bh_cagr']} "
                f"Ulcer={r['ulcer']} vs B&H={r['bh_ulcer']} MDD={r['mdd']}")


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] backtest_regime_assets 재사용 배선 확인 + FX 자체 로직(비용스윕·기간분할)")
    rng = np.random.default_rng(11)
    n = 2000
    # 완만한 약세(달러강세) 추세 + 급격한 반전(원화 급락→강세 전환) + 횡보를 합성
    trend = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.006, 900)))
    reversal = trend[-1] * np.exp(np.cumsum(rng.normal(-0.0008, 0.008, 400)))
    chop = reversal[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.005, 700)))
    closes = np.concatenate([trend, reversal, chop])
    import pandas as pd
    idx = pd.bdate_range("2010-01-01", periods=len(closes))

    params = {"trend_ma": 200, "band": 0.01, "confirm": 3}
    sweep = run_cost_sweep(closes, params)
    assert len(sweep) == len(COST_BPS_SWEEP)
    assert sweep[0]["cost_bps"] == 0
    # 비용이 늘수록 전략 CAGR은 단조 비증가여야 함(빈번한 전환이 있다는 전제하에 최소 같거나 감소)
    cagrs = [r["cagr"] for r in sweep]
    assert all(cagrs[i] >= cagrs[i + 1] - 1e-9 for i in range(len(cagrs) - 1)), \
        f"비용 증가 시 CAGR이 단조 감소해야 함: {cagrs}"
    _log(f"[self-test] 통과: run_cost_sweep 배선 정상(비용↑→CAGR 단조감소: {cagrs})")

    split = run_decade_split(closes, idx, params, 10)
    assert "~2019" in split and "2020~" in split
    _log(f"[self-test] 통과: run_decade_split 배선 정상({list(split.keys())})")

    # RA 재사용 함수들이 그대로 동작하는지(회귀 방지)
    exp = RA.regime_series(closes, 200, 0.01, 3)
    m = RA.simulate(closes, exp, 10)
    assert np.isfinite(m["cagr"])
    _log("[self-test] 통과: backtest_regime_assets 함수 재사용 정상")

    _log("[self-test] 전부 통과")


if __name__ == "__main__":
    main()
