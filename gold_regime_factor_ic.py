#!/usr/bin/env python3
"""
gold_regime_factor_ic.py — 금 배분타이밍 레짐신호 보완1: 팩터별 개별 유의성 검증.

지호 님 피드백(2026-08-26): 조합 로직(gold_regime_signal.classify)으로 바로 가기
전에 "이 팩터들이 개별적으로 정말 금의 향후 수익률을 예측하는가"를 먼저 넓게
검증해야 했는데 빠졌음. 이어서 지호 님이 이 스크립트 실행 전에 방법론 4가지를
지적해서(2026-08-26) 반영했다:

  1) 블록부트스트랩 블록 길이는 horizon별로 분리(12주 타겟엔 최소 12주 이상
     블록 필요 — 겹치는 선행수익률이 만드는 MA(h-1) 자기상관을 커버해야 함).
     BLOCK_BY_HORIZON = {"4w": 8, "12w": 16} (각 horizon의 2배).
  2) 다중검정 문제 — 팩터를 3개 룩백(3/6/12m) 각각 따로 테스트하지 않고
     classify()가 실제로 쓰는 것과 똑같이 "3룩백 부호합"(-3~+3, gold_regime_signal
     ._direction의 내부 스코어와 동일 공식) 하나로 합쳐서 테스트 — 팩터 수를
     13개에서 4개(금·DXY·실질금리 합의방향 + IEF동조)로 줄였다. 그리고 "채택
     기준"을 명시: 전체기간+2022이전 구간에서 4주·12주 두 horizon 모두 유의
     (90%CI가 0 제외)하고 부호가 같아야 "검증됨"으로 채택. 2022이후는 표본이
     작고(가설상 약화가 예상되는 구간이라) 채택 기준에서 제외, 참고로만 보고.
  3) 상관 살아있음/죽음 구간 분류는 gold_regime_data.build_weekly_features가
     이미 만든 60일 롤링상관 피처(gold_dxy_corr60/gold_realrate_corr60)를 그대로
     쓴다 — 이 피처는 t 시점까지의 데이터만으로 계산된 인과적(causal) 값이라
     look-ahead 없음. 전체구간 상관강도를 보고 사후적으로 구간을 나누는 것과는
     다르다(사후분할이면 미래정보 유출).
  4) (보완2에 반영) 게이트 켠 DXY/실질금리 vs 게이트 없는 단독 버전을 나란히
     비교하는 arm을 gold_regime_strategies.py에 추가 — 이 파일이 아니라 그
     파일에서 classify()의 gate_dxy/gate_realrate 옵션으로 구현.

방법론:
  - IC = 스피어만 순위상관(비선형에도 강건). scipy 미설치 환경이라 rank+pearson
    으로 직접 구현.
  - 유의성: backtest_regime_assets.paired_block_bootstrap과 동일한 블록리샘플
    관행으로 IC의 90%CI 계산(HAC 회귀 같은 새 방법론 안 들여옴).
  - 예측대상: 금의 향후 4주·12주 수익률.
  - "상관강도 자체"의 유의성: DXY·실질금리 합의방향의 IC를 상관 살아있음/죽음
    구간으로 나눠 비교(12주 타겟 기준) — 상관 게이팅이라는 프로젝트 핵심 전제를
    직접 검증. 이 조건부 부분집합은 non-contiguous(듬성듬성)라 블록부트스트랩
    대신 단순(iid) 부트스트랩을 씀 — 잔여 자기상관을 다소 과소평가할 수 있음
    (한계로 명시).

실행: python gold_regime_factor_ic.py            # output/gold_regime_factor_ic.json
      python gold_regime_factor_ic.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

from gold_regime_data import load_or_build
from gold_regime_signal import DEFAULT_CORR_THRESHOLD

OUTPUT_PATH = "output/gold_regime_factor_ic.json"
BLOCK_BY_HORIZON = {"4w": 8, "12w": 16}   # 각 horizon의 2배 — 겹치는 선행수익률 자기상관 커버
N_BOOT = 2000
N_BOOT_ERA = 1000  # 시대분할·조건부 표본이 작아서 반복횟수만 줄임(방법론은 동일)
HORIZONS = {"4w": 4, "12w": 12}
ERA_SPLIT = "2022-01-01"
MIN_N = 30


def _log(m): print(f"[금레짐팩터IC] {m}", file=sys.stderr)


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """scipy 없이 순위변환 후 피어슨상관으로 스피어만 IC 계산."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _block_bootstrap_ic_ci(x: np.ndarray, y: np.ndarray, block: int,
                            n_boot: int = N_BOOT, seed: int = 7) -> tuple[float, float]:
    """x,y 쌍을 블록 단위로 함께 리샘플해서 스피어만 IC의 90%CI 계산
    (backtest_regime_assets.paired_block_bootstrap과 동일한 블록리샘플 관행).
    block은 호출부에서 horizon에 맞게(BLOCK_BY_HORIZON) 넘겨야 한다."""
    n = len(x)
    n_blocks = n // block
    if n_blocks < 4:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ics = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        sel = np.concatenate([np.arange(j * block, (j + 1) * block) for j in idx])
        ics[i] = _spearman_ic(x[sel], y[sel])
    return (float(np.nanpercentile(ics, 5)), float(np.nanpercentile(ics, 95)))


def _iid_bootstrap_ic_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT_ERA,
                          seed: int = 7) -> tuple[float, float]:
    """조건부(상관생사) 부분집합처럼 non-contiguous한 표본용 — 블록 대신 단순
    복원추출. 잔여 자기상관을 다소 과소평가할 수 있음(한계로 명시)."""
    n = len(x)
    if n < MIN_N:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ics = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n, n)
        ics[i] = _spearman_ic(x[sel], y[sel])
    return (float(np.nanpercentile(ics, 5)), float(np.nanpercentile(ics, 95)))


def forward_return(weekly_gold: pd.Series, weeks: int) -> pd.Series:
    """t 시점 기준 향후 `weeks`주 금 수익률(팩터 값은 t 시점, 라벨은 t~t+weeks
    미래 수익률이라 순수 예측력 검정용 — look-ahead 아님, 미래를 예측하는 게
    바로 목적)."""
    return weekly_gold.shift(-weeks) / weekly_gold - 1


def _consensus_score(mom_3m: pd.Series, mom_6m: pd.Series, mom_12m: pd.Series) -> pd.Series:
    """gold_regime_signal._direction()의 내부 부호합과 동일 공식(-3~+3) —
    classify()가 실제로 소비하는 것과 같은 granularity로 IC를 검정하기 위해
    3개 룩백을 따로 테스트하지 않고 이걸 단일 팩터로 씀(다중검정 축소)."""
    return np.sign(mom_3m) + np.sign(mom_6m) + np.sign(mom_12m)


def _build_factor_series(features: pd.DataFrame) -> dict[str, pd.Series]:
    gold_s = _consensus_score(features["gold_mom_3m"], features["gold_mom_6m"], features["gold_mom_12m"])
    ief_s = _consensus_score(features["ief_mom_3m"], features["ief_mom_6m"], features["ief_mom_12m"])
    gold_label = pd.cut(gold_s, bins=[-4, -1.5, 1.5, 4], labels=["DOWN", "MIXED", "UP"])
    ief_label = pd.cut(ief_s, bins=[-4, -1.5, 1.5, 4], labels=["DOWN", "MIXED", "UP"])
    ief_sync = pd.Series(np.where((gold_label == "UP") & (ief_label == "UP"), 1,
                          np.where((gold_label == "DOWN") & (ief_label == "DOWN"), -1, 0)),
                         index=features.index, dtype=float)
    return {
        "gold_consensus": gold_s,
        "dxy_consensus": _consensus_score(features["dxy_mom_3m"], features["dxy_mom_6m"], features["dxy_mom_12m"]),
        "realrate_consensus": _consensus_score(features["real_rate_mom_3m"], features["real_rate_mom_6m"],
                                               features["real_rate_mom_12m"]),
        "ief_sync": ief_sync,
    }


def factor_ic_table(features: pd.DataFrame, weekly_gold: pd.Series) -> list[dict]:
    """각 팩터 × 각 예측기간의 IC + 90%CI(+시대분할). horizon별로 다른 블록길이 사용."""
    factors = _build_factor_series(features)
    rows = []
    for horizon_label, h in HORIZONS.items():
        block = BLOCK_BY_HORIZON[horizon_label]
        fwd = forward_return(weekly_gold, h)
        for factor_name, series in factors.items():
            valid = series.notna() & fwd.notna()
            x_all = series[valid].to_numpy(dtype=float)
            y_all = fwd[valid].to_numpy(dtype=float)
            dates_all = series.index[valid]
            if len(x_all) < MIN_N:
                continue
            ic = _spearman_ic(x_all, y_all)
            ci = _block_bootstrap_ic_ci(x_all, y_all, block=block)
            rows.append({"factor": factor_name, "horizon": horizon_label, "era": "전체",
                        "n": len(x_all), "ic": round(ic, 4),
                        "ic_ci90": (round(ci[0], 4), round(ci[1], 4)) if not np.isnan(ci[0]) else None,
                        "significant": bool(not np.isnan(ci[0]) and (ci[0] > 0 or ci[1] < 0))})
            for era_label, era_mask in (("2022이전", dates_all < pd.Timestamp(ERA_SPLIT)),
                                        ("2022이후", dates_all >= pd.Timestamp(ERA_SPLIT))):
                xe, ye = x_all[era_mask], y_all[era_mask]
                if len(xe) < MIN_N:
                    rows.append({"factor": factor_name, "horizon": horizon_label, "era": era_label,
                                "n": int(era_mask.sum()), "ic": None, "ic_ci90": None,
                                "significant": False, "note": "표본 부족"})
                    continue
                ice = _spearman_ic(xe, ye)
                cie = _block_bootstrap_ic_ci(xe, ye, block=block, n_boot=N_BOOT_ERA)
                rows.append({"factor": factor_name, "horizon": horizon_label, "era": era_label,
                            "n": len(xe), "ic": round(ice, 4),
                            "ic_ci90": (round(cie[0], 4), round(cie[1], 4)) if not np.isnan(cie[0]) else None,
                            "significant": bool(not np.isnan(cie[0]) and (cie[0] > 0 or cie[1] < 0))})
    return rows


def adoption_verdict(table: list[dict]) -> list[dict]:
    """채택 기준(다중검정 대응): 전체기간+2022이전 구간에서 4주·12주 두 horizon
    모두 유의(90%CI가 0 제외)하고 부호가 같아야 '검증됨'. 2022이후는 표본이
    작고 가설상 약화가 예상되는 구간이라 채택 기준에서 제외, 참고로만 별도 보고."""
    out = []
    factors = sorted({r["factor"] for r in table})
    for factor in factors:
        required = [(factor, h, era) for h in HORIZONS for era in ("전체", "2022이전")]
        rows = {(r["factor"], r["horizon"], r["era"]): r for r in table if r["factor"] == factor}
        checks = []
        for key in required:
            r = rows.get(key)
            ok = bool(r and r.get("significant"))
            checks.append({"horizon": key[1], "era": key[2], "significant": ok,
                          "ic": r["ic"] if r else None})
        signs = [np.sign(c["ic"]) for c in checks if c["ic"] is not None and c["significant"]]
        consistent_sign = len(signs) == len(checks) and len(set(signs)) == 1
        validated = all(c["significant"] for c in checks) and consistent_sign
        post2022 = [r for (f, h, era), r in rows.items() if era == "2022이후"]
        out.append({"factor": factor, "validated": bool(validated), "checks": checks,
                    "post2022_reference": [{"horizon": r["horizon"], "ic": r["ic"],
                                            "significant": r["significant"]} for r in post2022]})
    return out


def corr_gate_conditional_ic(features: pd.DataFrame, weekly_gold: pd.Series,
                              corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> list[dict]:
    """DXY·실질금리 합의방향의 IC(12주 타겟)를 '상관 살아있음/죽음' 구간으로 나눠
    비교 — 상관 게이팅 자체가 실제로 의미있는 조건분기인지 직접 검증.
    상관 살아있음/죽음 분류는 gold_regime_data가 만든 60일 롤링상관 피처
    (gold_dxy_corr60/gold_realrate_corr60, 인과적·look-ahead 없음)를 그대로
    쓴다 — 별도로 전체구간 상관강도를 계산해서 사후분할하지 않는다."""
    factors = _build_factor_series(features)
    fwd12 = forward_return(weekly_gold, 12)
    checks = [("dxy_consensus", "gold_dxy_corr60"), ("realrate_consensus", "gold_realrate_corr60")]
    out = []
    for factor_name, corr_col in checks:
        series = factors[factor_name]
        alive = features[corr_col].abs() >= corr_threshold
        for state_label, mask in (("살아있음", alive), ("죽음", ~alive)):
            valid = series.notna() & fwd12.notna() & mask
            x = series[valid].to_numpy(dtype=float)
            y = fwd12[valid].to_numpy(dtype=float)
            if len(x) < MIN_N:
                out.append({"factor": factor_name, "corr_state": state_label,
                            "n": int(valid.sum()), "ic": None, "ic_ci90": None,
                            "significant": False, "note": "표본 부족"})
                continue
            ic = _spearman_ic(x, y)
            ci = _iid_bootstrap_ic_ci(x, y)
            out.append({"factor": factor_name, "corr_state": state_label, "n": len(x),
                        "ic": round(ic, 4),
                        "ic_ci90": (round(ci[0], 4), round(ci[1], 4)) if not np.isnan(ci[0]) else None,
                        "significant": bool(not np.isnan(ci[0]) and (ci[0] > 0 or ci[1] < 0)),
                        "note": "단순(iid) 부트스트랩 — non-contiguous 부분집합이라 "
                                "잔여 자기상관 과소평가 가능성 있음"})
    return out


def run(save: bool = True) -> dict:
    features = load_or_build()
    # 주간 금 가격 시리즈가 필요하지만 features에는 모멘텀만 있음 — gold_regime_data의
    # 원 데이터를 다시 불러 주간 리샘플해서 얻는다(라이브 코드 경로와 동일한 W-FRI).
    from gold_regime_data import fetch_all
    daily = fetch_all()
    weekly_gold = daily["gold"].resample("W-FRI").last().reindex(features.index)

    table = factor_ic_table(features, weekly_gold)
    verdict = adoption_verdict(table)
    conditional = corr_gate_conditional_ic(features, weekly_gold)
    result = {"factor_ic": table, "adoption_verdict": verdict,
              "corr_gate_conditional_ic": conditional,
              "horizons": HORIZONS, "block_by_horizon": BLOCK_BY_HORIZON,
              "era_split": ERA_SPLIT, "n_boot": N_BOOT,
              "adoption_criterion": "전체기간+2022이전 구간에서 4주·12주 모두 유의(90%CI가 0 "
                                    "제외)+부호일관 — 2022이후는 참고용, 채택기준 아님"}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUTPUT_PATH}")
        for v in verdict:
            _log(f"  [{v['factor']}] 채택={v['validated']}")
    return result


def self_test():
    # 합성 데이터: gold_consensus가 향후수익률과 강한 양의 상관을 갖도록 구성
    # (추세지속), dxy_consensus는 무관한 노이즈로 구성 — IC가 방향성 있게 나오는지
    # 배선 확인.
    idx = pd.date_range("2003-01-03", periods=300, freq="W-FRI")
    rng = np.random.default_rng(11)
    gold_mom_3m = rng.normal(0, 0.05, 300)
    gold_mom_6m = gold_mom_3m + rng.normal(0, 0.01, 300)  # 상관관계 있게(합의방향 만들기 위해)
    gold_mom_12m = gold_mom_3m + rng.normal(0, 0.01, 300)
    signal = np.sign(gold_mom_3m) + np.sign(gold_mom_6m) + np.sign(gold_mom_12m)

    prices = [100.0]
    for i in range(1, 300):
        prices.append(prices[-1] * (1 + 0.003 * signal[i - 1] / 12))  # 합의방향에 연동된 완만한 추세
    weekly_gold = pd.Series(prices, index=idx)

    features = pd.DataFrame({
        "gold_mom_3m": gold_mom_3m, "gold_mom_6m": gold_mom_6m, "gold_mom_12m": gold_mom_12m,
        "dxy_mom_3m": rng.normal(0, 0.02, 300), "dxy_mom_6m": rng.normal(0, 0.02, 300),
        "dxy_mom_12m": rng.normal(0, 0.02, 300),
        "real_rate_mom_3m": rng.normal(0, 0.3, 300), "real_rate_mom_6m": rng.normal(0, 0.3, 300),
        "real_rate_mom_12m": rng.normal(0, 0.3, 300),
        "ief_mom_3m": rng.normal(0, 0.02, 300), "ief_mom_6m": rng.normal(0, 0.02, 300),
        "ief_mom_12m": rng.normal(0, 0.02, 300),
        "gold_dxy_corr60": rng.uniform(-0.6, -0.1, 300),
        "gold_realrate_corr60": rng.uniform(-0.6, -0.1, 300),
    }, index=idx)

    table = factor_ic_table(features, weekly_gold)
    gold_rows = [r for r in table if r["factor"] == "gold_consensus" and r["era"] == "전체"]
    assert len(gold_rows) == 2, gold_rows  # 4w, 12w 각 1행
    for r in gold_rows:
        assert r["ic"] > 0.1, f"gold_consensus가 향후수익률과 연동되도록 만들었으니 IC가 양수여야 함: {r}"

    dxy_rows = [r for r in table if r["factor"] == "dxy_consensus" and r["era"] == "전체"]
    for r in dxy_rows:
        assert abs(r["ic"]) < 0.3, f"dxy_consensus는 무관한 노이즈로 만들었으니 IC가 크지 않아야 함: {r}"

    # 채택기준 배선: gold_consensus는 전체+2022이전 두 horizon 모두 유의+동일부호일 가능성이
    # 높게 구성했으므로 validated=True 나오는 경로를 확인(합성데이터라 표본크기에 따라
    # 항상 보장되진 않지만, checks 필드 자체가 4개 항목(2 era × 2 horizon)으로 구성되는지는
    # 항상 검증 가능).
    verdict = adoption_verdict(table)
    gold_v = next(v for v in verdict if v["factor"] == "gold_consensus")
    assert len(gold_v["checks"]) == 4, gold_v  # (전체,4w)(전체,12w)(2022이전,4w)(2022이전,12w)
    assert all(c["era"] != "2022이후" for c in gold_v["checks"]), "채택기준 checks에 2022이후가 섞이면 안 됨"

    cond = corr_gate_conditional_ic(features, weekly_gold, corr_threshold=0.35)
    assert len(cond) == 4, cond  # dxy/realrate × 살아있음/죽음
    for row in cond:
        assert row["factor"] in ("dxy_consensus", "realrate_consensus")
        assert row["corr_state"] in ("살아있음", "죽음")

    # 블록길이가 horizon별로 다르게 쓰였는지 확인(4주=8, 12주=16)
    assert BLOCK_BY_HORIZON["4w"] == 8 and BLOCK_BY_HORIZON["12w"] == 16

    _log("통과: factor_ic_table/adoption_verdict/corr_gate_conditional_ic 배선 정상 "
        "(gold_consensus IC 양수, dxy_consensus IC 노이즈 수준, horizon별 블록길이 분리)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
