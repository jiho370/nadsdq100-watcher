# 금 배분 타이밍 레짐 신호 시스템 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DXY·실질금리·국채(IEF) 모멘텀과 금-DXY/금-실질금리 상관관계 생사 여부를 결합해
"확대(ADD)/관망(HOLD)/축소(REDUCE)" 3단계 + 권고 노출비율을 산출하는 레짐 신호 엔진을
만들고, 국면분리 백테스트·walk-forward·DSR/PBO로 검증한 뒤 현재 시점 판정을 출력한다.

**Architecture:** 4개 신규 스크립트(`gold_regime_data.py` → `gold_regime_signal.py` →
`gold_regime_overfit_gate.py` → `gold_regime_report.py`)가 파이프라인으로 이어진다.
기존 `backtest_regime_assets.py`(fetch/`_cagr`/`_ulcer`/`_mdd`), `fx_hedge_validation.py`
(`fetch_fred`), `overfit_stats.py`(`analyze` — DSR+PBO)를 그대로 재사용한다.

**Tech Stack:** Python, pandas, numpy, yfinance(가격), FRED CSV(금리), 저장소 기존
`--self-test` CLI 플래그 + `assert` 패턴(pytest 미사용 — 저장소 컨벤션).

## Global Constraints

- 스펙 문서: `docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md` (모든 태스크의
  요구사항은 이 문서를 따른다)
- 공통 백테스트 구간: 2003-01~현재 (DFII10/T10YIE 시작일 제약)
- 판단 주기: 일간 스위칭 금지. 주간을 기본, 월간을 파라미터로 스윕
- 포지션 모델: 노출비율 0.6~1.4배(1.0=중립), 레버리지/현금 조달비용 0% 가정
- 파라미터는 정확히 3개: `CORR_THRESHOLDS = (0.25, 0.35, 0.45)`,
  `REBAL_FREQS = ("weekly", "monthly")`,
  `WEIGHT_SCALES = {"base": {"strong": 0.20, "mild": 0.10}, "wide": {"strong": 0.30, "mild": 0.15}}`
  — 총 3×2×2=12조합, 어떤 태스크도 이 그리드를 벗어나 임의 파라미터를 추가하지 않는다
- 비용 가정: 기본 5bp 편도(`COST_BPS = 5.0`), 4단계에서 15bp 스트레스 케이스 병기
- 각 모듈은 `--self-test` CLI 플래그로 합성 데이터 기반 배선 확인(네트워크 호출 없음)을
  제공한다 — 이 저장소는 pytest를 쓰지 않고 `def self_test(): assert ...` 패턴을 쓴다
  (예: `overfit_stats.py`, `backtest_regime_assets.py` 참고)
- 저장 경로: `output/gold_regime_dataset.pkl`, `output/gold_regime_overfit_gate.json`,
  `output/gold_regime_summary.json`, `output/gold_regime_report.md`

---

## Task 1: `gold_regime_data.py` — 데이터 수집(`fetch_all`)

**Files:**
- Create: `gold_regime_data.py`
- Test: 동일 파일 내 `self_test()` 함수(합성 데이터, 네트워크 없음) + 수동 실행 확인

**Interfaces:**
- Consumes: `backtest_regime_assets.fetch(ticker, cache_path)` (yfinance 캐시),
  `fx_hedge_validation.fetch_fred(series_id)` (FRED 캐시)
- Produces: `fetch_all(max_stale_days: int = 5) -> pd.DataFrame` — 컬럼
  `["gold", "dxy", "ief", "real_rate", "breakeven"]`, 인덱스는 Gold(GC=F) 거래일
  DatetimeIndex, 나머지 시리즈는 그 위에 ffill로 정렬됨. 결측(정렬 전 시작일 이전)은
  dropna로 제거.

- [ ] **Step 1: 합성 데이터로 정렬 로직을 검증할 self-test 뼈대 작성(실패 확인용)**

```python
# gold_regime_data.py 맨 아래
def self_test():
    """합성 시리즈로 fetch_all의 정렬(ffill) 로직만 별도 함수로 분리해 검증
    (네트워크 호출 없이). _align_calendar가 아직 없으므로 이 단계는 실패해야 정상."""
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", periods=10)
    gold = pd.Series(range(10), index=idx, dtype=float)
    dxy = pd.Series([100.0, 101.0], index=[idx[0], idx[5]])  # 듬성듬성 — ffill 확인용
    out = _align_calendar(gold, {"dxy": dxy})
    assert list(out.columns) == ["gold", "dxy"]
    assert out["dxy"].iloc[3] == 100.0, "ffill이 안 됐음"
    assert out["dxy"].iloc[7] == 101.0
    print("[self-test] 통과: _align_calendar 정렬/ffill 정상", file=__import__("sys").stderr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        df = fetch_all()
        print(df.tail())
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_data.py --self-test`
Expected: `NameError: name '_align_calendar' is not defined` (또는 `AttributeError`)

- [ ] **Step 3: `fetch_all`과 `_align_calendar` 구현**

```python
#!/usr/bin/env python3
"""
gold_regime_data.py — 금 배분타이밍 레짐신호 1단계: 데이터 수집 + 주간 피처.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md 참고.

실행: python gold_regime_data.py            # 데이터셋 캐시 갱신 후 tail 출력
      python gold_regime_data.py --self-test
결과: output/gold_regime_dataset.pkl
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

from backtest_regime_assets import fetch as fetch_yf
from fx_hedge_validation import fetch_fred

GOLD_TICKER = "GC=F"
DXY_TICKER = "DX-Y.NYB"
IEF_TICKER = "IEF"
FRED_REAL_RATE = "DFII10"
FRED_BREAKEVEN = "T10YIE"
DATASET_PATH = "output/gold_regime_dataset.pkl"
CORR_WINDOW = 60


def _log(m): print(f"[금레짐데이터] {m}", file=sys.stderr)


def _align_calendar(base: pd.Series, others: dict[str, pd.Series]) -> pd.DataFrame:
    """base(Gold) 거래일 캘린더 위에 others를 ffill로 정렬. 정렬 전 시작일 이전(전부
    NaN)인 앞부분은 제거."""
    cal = base.index
    out = pd.DataFrame({"gold": base})
    for name, s in others.items():
        aligned = s.reindex(cal.union(s.index)).sort_index().ffill().reindex(cal)
        out[name] = aligned
    return out.dropna()


def fetch_all(max_stale_days: int = 5) -> pd.DataFrame:
    gold = fetch_yf(GOLD_TICKER, "output/regime_price_cache_gold_dxy.pkl", max_stale_days)
    dxy = fetch_yf(DXY_TICKER, "output/regime_price_cache_dxy.pkl", max_stale_days)
    ief = fetch_yf(IEF_TICKER, "output/regime_price_cache_ief_dxy.pkl", max_stale_days)
    real_rate = fetch_fred(FRED_REAL_RATE)
    breakeven = fetch_fred(FRED_BREAKEVEN)
    df = _align_calendar(gold, {"dxy": dxy, "ief": ief, "real_rate": real_rate,
                                 "breakeven": breakeven})
    _log(f"정렬 완료: {df.index.min().date()}~{df.index.max().date()}, {len(df)}행")
    return df
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_data.py --self-test`
Expected: `[self-test] 통과: _align_calendar 정렬/ffill 정상` 출력, 종료코드 0

- [ ] **Step 5: 실제 네트워크 호출로 데이터 범위 수동 확인**

Run: `python gold_regime_data.py`
Expected: 콘솔에 "정렬 완료: 2003-0X-XX~2026-08-XX, N행" 형태 로그 (2003년 시작 —
DFII10/T10YIE 제약), 마지막 5행이 정상 숫자로 출력됨(NaN 없음)

- [ ] **Step 6: 커밋**

```bash
git add gold_regime_data.py
git commit -m "feat: 금 레짐신호 1단계 — Gold/DXY/IEF/실질금리/기대인플레 정렬 fetch_all"
```

---

## Task 2: `gold_regime_data.py` — 주간 피처(`build_weekly_features`, `load_or_build`)

**Files:**
- Modify: `gold_regime_data.py`

**Interfaces:**
- Consumes: `fetch_all()` 반환 DataFrame(Task 1)
- Produces: `build_weekly_features(df: pd.DataFrame) -> pd.DataFrame` — 주간(`W-FRI`)
  인덱스, 컬럼: `gold_mom_3m/6m/12m`, `dxy_mom_3m/6m/12m`, `ief_mom_3m/6m/12m`,
  `real_rate_mom_3m/6m/12m`(부호 아닌 원값 — pct_change 또는 diff, 부호는 `classify`가
  계산), `gold_dxy_corr60`, `gold_realrate_corr60`.
  `load_or_build(force: bool = False) -> pd.DataFrame` — 캐시(`DATASET_PATH`) 있으면
  로드, 없거나 `force=True`면 `fetch_all()`+`build_weekly_features()` 실행 후 저장.

**중요**: 실질금리(`real_rate`)는 음수를 가질 수 있는 레벨(%) 값이라 `pct_change`가
아니라 `diff`(레벨 변화량)로 모멘텀을 계산한다. Gold/DXY/IEF는 가격이라 `pct_change`.
상관계수도 단위를 맞춰 gold 일간수익률 vs real_rate 일간 diff로 계산한다.

- [ ] **Step 1: self-test에 주간 리샘플/모멘텀 부호/상관 검증 추가(실패 확인용)**

```python
def self_test_features():
    """합성 20주치 일간 데이터(추세 있는 gold, 반대추세 dxy, 하락하는 real_rate)로
    build_weekly_features 배선 확인."""
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", periods=300)  # ~60주
    gold = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.002))), index=idx)  # 꾸준 상승
    dxy = pd.Series(100 * np.exp(np.cumsum(np.full(300, -0.001))), index=idx)  # 꾸준 하락
    ief = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.0005))), index=idx)
    real_rate = pd.Series(2.0 - np.cumsum(np.full(300, 0.002)), index=idx)      # 꾸준 하락
    breakeven = pd.Series(np.full(300, 2.0), index=idx)
    df = pd.DataFrame({"gold": gold, "dxy": dxy, "ief": ief, "real_rate": real_rate,
                       "breakeven": breakeven})
    feat = build_weekly_features(df)
    assert feat.index.freqstr in ("W-FRI",) or feat.index.freq is not None or len(feat) < len(df)
    last = feat.iloc[-1]
    assert last["gold_mom_12m"] > 0, "금 12개월 모멘텀은 양수여야 함(꾸준 상승 합성데이터)"
    assert last["dxy_mom_12m"] < 0, "DXY 12개월 모멘텀은 음수여야 함"
    assert last["real_rate_mom_12m"] < 0, "실질금리 12개월 모멘텀(레벨변화)은 음수여야 함"
    assert last["gold_dxy_corr60"] < 0, "반대추세로 만들었으니 상관은 음수여야 함"
    print("[self-test] 통과: build_weekly_features 배선 정상", file=sys.stderr)


def self_test():
    ... # Step 1의 _align_calendar 테스트
    self_test_features()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_data.py --self-test`
Expected: `NameError: name 'build_weekly_features' is not defined`

- [ ] **Step 3: `build_weekly_features`, `load_or_build` 구현**

```python
MOM_WEEKS = {"3m": 13, "6m": 26, "12m": 52}


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    gold_ret = df["gold"].pct_change()
    dxy_ret = df["dxy"].pct_change()
    real_rate_chg = df["real_rate"].diff()
    corr_dxy = gold_ret.rolling(CORR_WINDOW).corr(dxy_ret)
    corr_rr = gold_ret.rolling(CORR_WINDOW).corr(real_rate_chg)

    weekly = df.resample("W-FRI").last()
    corr_dxy_w = corr_dxy.resample("W-FRI").last()
    corr_rr_w = corr_rr.resample("W-FRI").last()

    out = pd.DataFrame(index=weekly.index)
    for col in ("gold", "dxy", "ief"):
        for label, w in MOM_WEEKS.items():
            out[f"{col}_mom_{label}"] = weekly[col].pct_change(w)
    for label, w in MOM_WEEKS.items():
        out[f"real_rate_mom_{label}"] = weekly["real_rate"].diff(w)
    out["gold_dxy_corr60"] = corr_dxy_w
    out["gold_realrate_corr60"] = corr_rr_w
    return out.dropna()


def load_or_build(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(DATASET_PATH):
        return pd.read_pickle(DATASET_PATH)
    df = fetch_all()
    feat = build_weekly_features(df)
    os.makedirs("output", exist_ok=True)
    feat.to_pickle(DATASET_PATH)
    _log(f"저장: {DATASET_PATH} ({len(feat)}주, {feat.index.min().date()}~{feat.index.max().date()})")
    return feat
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_data.py --self-test`
Expected: 두 self-test 함수 모두 통과 로그, 종료코드 0

- [ ] **Step 5: 실제 데이터로 `load_or_build` 수동 확인**

Run: `python -c "from gold_regime_data import load_or_build; f = load_or_build(force=True); print(f.tail()); print(f.isna().sum().sum())"`
Expected: 마지막 행에 NaN 없음(`isna().sum().sum() == 0`), `output/gold_regime_dataset.pkl` 생성됨

- [ ] **Step 6: 커밋**

```bash
git add gold_regime_data.py
git commit -m "feat: 금 레짐신호 1단계 — 주간 모멘텀/상관 피처 build_weekly_features"
```

---

## Task 3: `gold_regime_signal.py` — 분류 로직(`classify`, `target_exposure`)

**Files:**
- Create: `gold_regime_signal.py`

**Interfaces:**
- Consumes: `gold_regime_data.load_or_build()` (주간 피처 DataFrame, 컬럼명은 Task 2와
  동일: `gold_mom_3m/6m/12m` 등, `gold_dxy_corr60`, `gold_realrate_corr60`)
- Produces:
  - `classify(row: pd.Series, corr_threshold: float) -> dict` — 반환 키:
    `score(int)`, `verdict("ADD"|"HOLD"|"REDUCE")`, `strength("strong"|"mild"|None)`,
    `gold_direction("UP"|"DOWN"|"MIXED")`, `dxy_alive(bool)`,
    `dxy_direction("UP"|"DOWN"|"MIXED")`, `realrate_alive(bool)`, `realrate_direction`,
    `ief_sync(-1|0|1)`, `unexplained(bool)`, `confidence("normal"|"low")`
  - `target_exposure(verdict: str, strength: str | None, weight_scale: str) -> float`
  - 상수: `CORR_THRESHOLDS = (0.25, 0.35, 0.45)`, `REBAL_FREQS = ("weekly", "monthly")`,
    `WEIGHT_SCALES = {"base": {...}, "wide": {...}}`,
    `DEFAULT_CORR_THRESHOLD = 0.35`, `DEFAULT_REBAL_FREQ = "weekly"`,
    `DEFAULT_WEIGHT_SCALE = "base"`, `COST_BPS = 5.0`

- [ ] **Step 1: self-test 뼈대 작성 — 4가지 시나리오(정상 ADD강/HOLD/설명안됨/REDUCE)**

```python
#!/usr/bin/env python3
"""
gold_regime_signal.py — 금 배분타이밍 레짐신호 2·3단계: 분류 로직 + 백테스트.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §레짐 분류 로직 참고.

실행: python gold_regime_signal.py --self-test
"""
from __future__ import annotations
import sys, argparse
import numpy as np
import pandas as pd

CORR_THRESHOLDS = (0.25, 0.35, 0.45)
REBAL_FREQS = ("weekly", "monthly")
WEIGHT_SCALES = {"base": {"strong": 0.20, "mild": 0.10},
                 "wide": {"strong": 0.30, "mild": 0.15}}
DEFAULT_CORR_THRESHOLD = 0.35
DEFAULT_REBAL_FREQ = "weekly"
DEFAULT_WEIGHT_SCALE = "base"
COST_BPS = 5.0


def _log(m): print(f"[금레짐신호] {m}", file=sys.stderr)


def _mk_row(**kw) -> pd.Series:
    base = {"gold_mom_3m": 0.0, "gold_mom_6m": 0.0, "gold_mom_12m": 0.0,
            "dxy_mom_3m": 0.0, "dxy_mom_6m": 0.0, "dxy_mom_12m": 0.0,
            "real_rate_mom_3m": 0.0, "real_rate_mom_6m": 0.0, "real_rate_mom_12m": 0.0,
            "ief_mom_3m": 0.0, "ief_mom_6m": 0.0, "ief_mom_12m": 0.0,
            "gold_dxy_corr60": 0.0, "gold_realrate_corr60": 0.0}
    base.update(kw)
    return pd.Series(base)


def self_test():
    # 1) 강한 ADD: 금 UP 다수결 + DXY 살아있고 DOWN + 실질금리 살아있고 DOWN + IEF 동조
    row = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                  dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                  real_rate_mom_3m=-0.3, real_rate_mom_6m=-0.5, real_rate_mom_12m=-0.4,
                  ief_mom_3m=0.02, ief_mom_6m=0.01, ief_mom_12m=0.03,
                  gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c = classify(row, 0.35)
    assert c["verdict"] == "ADD" and c["strength"] == "strong", c
    assert c["score"] == 2 + 1 + 1 + 1, c  # 금+DXY확인+실질금리확인+IEF동조
    assert target_exposure(c["verdict"], c["strength"], "base") == 1.20

    # 2) HOLD: 모든 신호 혼조
    row2 = _mk_row(gold_mom_3m=0.01, gold_mom_6m=-0.01, gold_mom_12m=0.005,
                   gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c2 = classify(row2, 0.35)
    assert c2["verdict"] == "HOLD" and c2["score"] == 0, c2
    assert target_exposure("HOLD", None, "base") == 1.0

    # 3) 설명 안 되는 구간: 상관 둘 다 죽음 → 금 방향만 사용, 강한 등급도 mild로 다운캡
    row3 = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                   dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                   ief_mom_3m=0.02, ief_mom_6m=0.01, ief_mom_12m=0.03,
                   gold_dxy_corr60=-0.1, gold_realrate_corr60=0.05)  # 둘 다 임계값 미만
    c3 = classify(row3, 0.35)
    assert c3["unexplained"] is True and c3["confidence"] == "low", c3
    assert c3["verdict"] == "ADD" and c3["strength"] == "mild", c3  # 금만으로는 score=2→mild
    assert c3["score"] == 2, c3  # DXY/실질금리/IEF 기여 전부 0

    # 4) REDUCE 강: 금 DOWN + DXY 살아있고 UP(약세컨펌) + 실질금리 살아있고 UP + IEF 동조(둘다하락)
    row4 = _mk_row(gold_mom_3m=-0.05, gold_mom_6m=-0.08, gold_mom_12m=-0.10,
                   dxy_mom_3m=0.02, dxy_mom_6m=0.03, dxy_mom_12m=0.01,
                   real_rate_mom_3m=0.3, real_rate_mom_6m=0.5, real_rate_mom_12m=0.4,
                   ief_mom_3m=-0.02, ief_mom_6m=-0.01, ief_mom_12m=-0.03,
                   gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    c4 = classify(row4, 0.35)
    assert c4["verdict"] == "REDUCE" and c4["strength"] == "strong", c4
    assert target_exposure(c4["verdict"], c4["strength"], "base") == 0.80

    _log("통과: classify/target_exposure 4개 시나리오(ADD강/HOLD/설명안됨/REDUCE강) 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: `NameError: name 'classify' is not defined`

- [ ] **Step 3: `classify`/`target_exposure` 구현 (파일 상단, `_mk_row` 위쪽에 삽입)**

```python
def _direction(mom_3m: float, mom_6m: float, mom_12m: float) -> str:
    s = int(np.sign(mom_3m)) + int(np.sign(mom_6m)) + int(np.sign(mom_12m))
    if s >= 2:
        return "UP"
    if s <= -2:
        return "DOWN"
    return "MIXED"


def _score_to_verdict(total: int) -> tuple[str, str | None]:
    if total >= 3:
        return "ADD", "strong"
    if total >= 1:
        return "ADD", "mild"
    if total == 0:
        return "HOLD", None
    if total >= -2:
        return "REDUCE", "mild"
    return "REDUCE", "strong"


def classify(row: pd.Series, corr_threshold: float) -> dict:
    gold_dir = _direction(row["gold_mom_3m"], row["gold_mom_6m"], row["gold_mom_12m"])
    base_score = {"UP": 2, "DOWN": -2, "MIXED": 0}[gold_dir]

    dxy_dir = _direction(row["dxy_mom_3m"], row["dxy_mom_6m"], row["dxy_mom_12m"])
    dxy_alive = abs(row["gold_dxy_corr60"]) >= corr_threshold
    dxy_confirm = (1 if dxy_dir == "DOWN" else -1 if dxy_dir == "UP" else 0) if dxy_alive else 0

    rr_dir = _direction(row["real_rate_mom_3m"], row["real_rate_mom_6m"], row["real_rate_mom_12m"])
    rr_alive = abs(row["gold_realrate_corr60"]) >= corr_threshold
    rr_confirm = (1 if rr_dir == "DOWN" else -1 if rr_dir == "UP" else 0) if rr_alive else 0

    ief_dir = _direction(row["ief_mom_3m"], row["ief_mom_6m"], row["ief_mom_12m"])
    ief_sync = 1 if (gold_dir == "UP" and ief_dir == "UP") else \
               -1 if (gold_dir == "DOWN" and ief_dir == "DOWN") else 0

    unexplained = (not dxy_alive) and (not rr_alive)
    total = base_score if unexplained else base_score + dxy_confirm + rr_confirm + ief_sync

    verdict, strength = _score_to_verdict(total)
    if unexplained and strength == "strong":
        strength = "mild"

    return {"score": total, "verdict": verdict, "strength": strength,
            "gold_direction": gold_dir,
            "dxy_alive": bool(dxy_alive), "dxy_direction": dxy_dir,
            "realrate_alive": bool(rr_alive), "realrate_direction": rr_dir,
            "ief_sync": ief_sync, "unexplained": bool(unexplained),
            "confidence": "low" if unexplained else "normal"}


def target_exposure(verdict: str, strength: str | None, weight_scale: str) -> float:
    if verdict == "HOLD":
        return 1.0
    delta = WEIGHT_SCALES[weight_scale][strength]
    sign = 1.0 if verdict == "ADD" else -1.0
    return round(1.0 + sign * delta, 4)
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: `통과: classify/target_exposure 4개 시나리오(ADD강/HOLD/설명안됨/REDUCE강) 정상`

- [ ] **Step 5: 커밋**

```bash
git add gold_regime_signal.py
git commit -m "feat: 금 레짐신호 2단계 — classify/target_exposure 점수제 분류 로직"
```

---

## Task 4: `gold_regime_signal.py` — 리밸런싱 노출 생성(`build_regime_series`)

**Files:**
- Modify: `gold_regime_signal.py`

**Interfaces:**
- Consumes: `classify()`, `target_exposure()` (Task 3), 주간 피처 DataFrame(Task 2 스키마)
- Produces: `build_regime_series(features: pd.DataFrame, corr_threshold: float,
  rebal_freq: str, weight_scale: str) -> pd.DataFrame` — `features`와 동일한 주간
  인덱스, 컬럼: `exposure(float)`, `verdict`, `strength`, `confidence`, `unexplained(bool)`.
  `rebal_freq="weekly"`면 매주, `"monthly"`면 각 월의 마지막 주에만 재판정하고 나머지
  주는 직전 판정을 유지(ffill).

- [ ] **Step 1: self-test 추가 — weekly는 매주 값이 바뀔 수 있고 monthly는 월 내 동일한지 확인**

```python
def self_test_regime_series():
    idx = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    rng = np.random.default_rng(3)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 30), "gold_mom_6m": rng.normal(0, 0.05, 30),
        "gold_mom_12m": rng.normal(0, 0.05, 30),
        "dxy_mom_3m": rng.normal(0, 0.02, 30), "dxy_mom_6m": rng.normal(0, 0.02, 30),
        "dxy_mom_12m": rng.normal(0, 0.02, 30),
        "real_rate_mom_3m": rng.normal(0, 0.3, 30), "real_rate_mom_6m": rng.normal(0, 0.3, 30),
        "real_rate_mom_12m": rng.normal(0, 0.3, 30),
        "ief_mom_3m": rng.normal(0, 0.02, 30), "ief_mom_6m": rng.normal(0, 0.02, 30),
        "ief_mom_12m": rng.normal(0, 0.02, 30),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.3, 30),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.3, 30),
    }, index=idx)

    weekly = build_regime_series(feat, 0.35, "weekly", "base")
    assert list(weekly.index) == list(feat.index)
    assert weekly["exposure"].nunique() > 1, "weekly는 주마다 다른 판정이 나올 수 있어야 함"

    monthly = build_regime_series(feat, 0.35, "monthly", "base")
    assert list(monthly.index) == list(feat.index)
    for _, grp in monthly.groupby(monthly.index.to_period("M")):
        assert grp["exposure"].nunique() == 1, "monthly는 같은 달 안에서 노출이 고정돼야 함"

    _log("통과: build_regime_series weekly/monthly 배선 정상")


def self_test():
    ...  # Task 3의 4개 시나리오
    self_test_regime_series()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: `NameError: name 'build_regime_series' is not defined`

- [ ] **Step 3: 구현**

```python
def build_regime_series(features: pd.DataFrame, corr_threshold: float, rebal_freq: str,
                         weight_scale: str) -> pd.DataFrame:
    if rebal_freq == "weekly":
        decision_idx = features.index
    elif rebal_freq == "monthly":
        decision_idx = features.groupby(features.index.to_period("M")).tail(1).index
    else:
        raise ValueError(f"알 수 없는 rebal_freq: {rebal_freq}")

    rows = []
    for ts in decision_idx:
        c = classify(features.loc[ts], corr_threshold)
        exp = target_exposure(c["verdict"], c["strength"], weight_scale)
        rows.append({"date": ts, "exposure": exp, **c})
    decisions = pd.DataFrame(rows).set_index("date")
    return decisions.reindex(features.index).ffill()
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: 5개 self-test 함수 전부 통과 로그

- [ ] **Step 5: 커밋**

```bash
git add gold_regime_signal.py
git commit -m "feat: 금 레짐신호 2단계 — build_regime_series 주간/월간 리밸런싱 노출 생성"
```

---

## Task 5: `gold_regime_signal.py` — 백테스트 시뮬레이션(`simulate_exposure`, `fixed_weight_benchmark`)

**Files:**
- Modify: `gold_regime_signal.py`

**Interfaces:**
- Consumes: `backtest_regime_assets._cagr/_ulcer/_mdd`, `build_regime_series()` 반환
  (Task 4), 일간 금 가격 시리즈(`gold_regime_data.fetch_all()["gold"]`)
- Produces:
  - `simulate_exposure(gold_daily: pd.Series, regime: pd.DataFrame, cost_bps: float) -> dict`
    — 반환 키: `nav(np.ndarray)`, `cagr(float)`, `ulcer(float)`, `mdd(float)`,
    `strat_ret(np.ndarray)`. `regime["exposure"]`를 일간 캘린더에 ffill 후 1일 지연
    적용, 노출 변경 시 턴오버 비용(`|Δexposure|×cost_bps`) 차감.
  - `fixed_weight_benchmark(regime: pd.DataFrame) -> float` — `regime["exposure"]`의
    단순평균(시간가중평균, 각 주 동일 가중).

- [ ] **Step 1: self-test 추가 — 상시 100% 노출은 buy-hold와 일치, 노출 하향 구간은 buy-hold보다 낮은 변동성**

```python
def self_test_simulate():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(9)
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 500))), index=idx)

    weekly_idx = pd.date_range(idx[0], idx[-1], freq="W-FRI")
    always_on = pd.DataFrame({"exposure": 1.0, "verdict": "HOLD", "strength": None,
                              "confidence": "normal", "unexplained": False}, index=weekly_idx)
    sim_bh = simulate_exposure(gold_daily, always_on, 0.0)
    raw_cagr = (gold_daily.iloc[-1] / gold_daily.iloc[0]) ** (252 / len(gold_daily)) - 1
    assert abs(sim_bh["cagr"] - raw_cagr * 100) < 1.0, (sim_bh["cagr"], raw_cagr * 100)

    half_off = always_on.copy()
    half_off.loc[half_off.index[len(half_off) // 2:], "exposure"] = 0.5
    sim_half = simulate_exposure(gold_daily, half_off, 5.0)
    assert sim_half["ulcer"] < sim_bh["ulcer"], "노출을 줄인 구간이 있으면 Ulcer가 더 낮아야 함"

    fw = fixed_weight_benchmark(half_off)
    assert 0.5 < fw < 1.0, fw

    _log("통과: simulate_exposure/fixed_weight_benchmark 배선 정상")


def self_test():
    ...  # 이전 self-test들
    self_test_simulate()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: `NameError: name 'simulate_exposure' is not defined`

- [ ] **Step 3: 구현 (파일 상단 import에 `from backtest_regime_assets import _cagr, _ulcer, _mdd` 추가)**

```python
from backtest_regime_assets import _cagr, _ulcer, _mdd


def simulate_exposure(gold_daily: pd.Series, regime: pd.DataFrame, cost_bps: float) -> dict:
    exposure_daily = regime["exposure"].reindex(
        gold_daily.index.union(regime.index)).sort_index().ffill().reindex(gold_daily.index)
    exposure_daily = exposure_daily.ffill().fillna(1.0)
    ret = gold_daily.pct_change().to_numpy()
    exp_arr = exposure_daily.to_numpy()
    exp_lag = np.roll(exp_arr, 1)
    strat_ret = (exp_lag * ret)[1:]
    turnover = np.abs(np.diff(exp_arr))
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = np.nan_to_num(strat_ret - cost, nan=0.0)
    nav = np.cumprod(1 + strat_ret)
    n = len(nav)
    return {"nav": nav, "cagr": _cagr(nav, n), "ulcer": _ulcer(nav), "mdd": _mdd(nav),
            "strat_ret": strat_ret}


def fixed_weight_benchmark(regime: pd.DataFrame) -> float:
    return float(regime["exposure"].mean())
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: 6개 self-test 함수 전부 통과

- [ ] **Step 5: 커밋**

```bash
git add gold_regime_signal.py
git commit -m "feat: 금 레짐신호 3단계 — simulate_exposure 턴오버비용 백테스트 + 고정비중 벤치마크"
```

---

## Task 6: `gold_regime_signal.py` — 국면분리(`era_performance`)

**Files:**
- Modify: `gold_regime_signal.py`

**Interfaces:**
- Consumes: `simulate_exposure()`, `fixed_weight_benchmark()` (Task 5)
- Produces: `ERAS`(리스트 상수), `era_performance(gold_daily: pd.Series,
  regime: pd.DataFrame, cost_bps: float, eras: list = ERAS) -> list[dict]` — 각 dict:
  `era, n_days, signal_cagr, buyhold_cagr, fixed_weight_cagr, signal_ulcer,
  buyhold_ulcer, fixed_weight_ulcer, unexplained_pct`. 표본 20일 미만 구간은
  `{"era":..., "n_days":..., "note": "표본 부족(20일 미만) — 생략"}`만 반환.

- [ ] **Step 1: self-test 추가**

```python
def self_test_era():
    idx = pd.bdate_range("2000-06-01", periods=1800)  # ~7년, 2000년대 강세장 구간 포함
    rng = np.random.default_rng(1)
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 1800))), index=idx)
    weekly_idx = pd.date_range(idx[0], idx[-1], freq="W-FRI")
    regime = pd.DataFrame({"exposure": 1.0, "verdict": "HOLD", "strength": None,
                           "confidence": "normal", "unexplained": False}, index=weekly_idx)
    result = era_performance(gold_daily, regime, 5.0, eras=[("전체구간", "2000-01-01", "2008-01-01")])
    assert len(result) == 1
    assert "signal_cagr" in result[0] and "note" not in result[0]

    tiny = era_performance(gold_daily, regime, 5.0, eras=[("표본부족", "2000-06-01", "2000-06-05")])
    assert tiny[0]["note"].startswith("표본 부족")

    _log("통과: era_performance 배선 정상")


def self_test():
    ...
    self_test_era()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: `NameError: name 'era_performance' is not defined`

- [ ] **Step 3: 구현**

```python
ERAS = [
    ("2000년대 강세장", "2001-01-01", "2011-08-01"),
    ("2013-2015 약세장", "2013-01-01", "2016-01-01"),
    ("2022 금리인상기", "2022-03-01", "2023-07-01"),
    ("2022년 이후", "2022-01-01", "2027-01-01"),
]


def era_performance(gold_daily: pd.Series, regime: pd.DataFrame, cost_bps: float,
                    eras: list = ERAS) -> list[dict]:
    sim = simulate_exposure(gold_daily, regime, cost_bps)
    bh = regime.copy(); bh["exposure"] = 1.0
    sim_bh = simulate_exposure(gold_daily, bh, 0.0)
    fw_level = fixed_weight_benchmark(regime)
    fw = regime.copy(); fw["exposure"] = fw_level
    sim_fw = simulate_exposure(gold_daily, fw, cost_bps)

    dates = gold_daily.index[1:]
    unexplained_daily = regime["unexplained"].reindex(gold_daily.index).ffill().fillna(False).to_numpy()[1:]

    out = []
    for label, start, end in eras:
        mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
        n = int(mask.sum())
        if n < 20:
            out.append({"era": label, "n_days": n, "note": "표본 부족(20일 미만) — 생략"})
            continue
        nav_s = np.cumprod(1 + sim["strat_ret"][mask])
        nav_bh = np.cumprod(1 + sim_bh["strat_ret"][mask])
        nav_fw = np.cumprod(1 + sim_fw["strat_ret"][mask])
        out.append({
            "era": label, "n_days": n,
            "signal_cagr": round(_cagr(nav_s, n), 2),
            "buyhold_cagr": round(_cagr(nav_bh, n), 2),
            "fixed_weight_cagr": round(_cagr(nav_fw, n), 2),
            "signal_ulcer": round(_ulcer(nav_s), 2),
            "buyhold_ulcer": round(_ulcer(nav_bh), 2),
            "fixed_weight_ulcer": round(_ulcer(nav_fw), 2),
            "unexplained_pct": round(float(unexplained_daily[mask].mean() * 100), 1),
        })
    return out
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_signal.py --self-test`
Expected: 7개 self-test 함수 전부 통과

- [ ] **Step 5: 실데이터로 수동 확인**

Run:
```bash
python -c "
from gold_regime_data import load_or_build, fetch_all
from gold_regime_signal import build_regime_series, era_performance, DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE, COST_BPS
feat = load_or_build()
gold = fetch_all()['gold']
regime = build_regime_series(feat, DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE)
for r in era_performance(gold, regime, COST_BPS):
    print(r)
"
```
Expected: 4개 국면 각각 signal_cagr/buyhold_cagr 등이 출력됨(2000년대 강세장 구간은
2003년부터만 유효 데이터라 n_days가 예상보다 적을 수 있음 — 그 자체가 정상, 리포트에
반영됨)

- [ ] **Step 6: 커밋**

```bash
git add gold_regime_signal.py
git commit -m "feat: 금 레짐신호 3단계 — era_performance 4구간 국면분리 리포트"
```

---

## Task 7: `gold_regime_overfit_gate.py` — DSR/PBO 그리드(`grid_trials`, `run_dsr_pbo`)

**Files:**
- Create: `gold_regime_overfit_gate.py`

**Interfaces:**
- Consumes: `overfit_stats.analyze()`(기존, DSR+PBO 동시 산출), `gold_regime_signal`의
  `build_regime_series/simulate_exposure/fixed_weight_benchmark/CORR_THRESHOLDS/
  REBAL_FREQS/WEIGHT_SCALES/COST_BPS`
- Produces: `grid_trials(features: pd.DataFrame, gold_daily: pd.Series,
  cost_bps: float = COST_BPS) -> dict` — `overfit_stats.analyze()`에 그대로 넣을 수 있는
  `trial_data` dict(`horizon, universe, cost, rebal_days, hold_days, dates, trials,
  excess_returns`). `run_dsr_pbo(features, gold_daily, cost_bps=COST_BPS) -> dict` —
  `overfit_stats.analyze(trial_data, n_blocks=12, save=False)` 반환 그대로.

- [ ] **Step 1: self-test 뼈대 작성**

```python
#!/usr/bin/env python3
"""
gold_regime_overfit_gate.py — 금 배분타이밍 레짐신호 4단계: walk-forward + DSR/PBO.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §4단계 참고.

실행: python gold_regime_overfit_gate.py            # output/gold_regime_overfit_gate.json
      python gold_regime_overfit_gate.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

import overfit_stats as OS
from gold_regime_data import load_or_build, fetch_all
from gold_regime_signal import (
    build_regime_series, simulate_exposure, fixed_weight_benchmark,
    CORR_THRESHOLDS, REBAL_FREQS, WEIGHT_SCALES,
    DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE, COST_BPS,
)

OUTPUT_PATH = "output/gold_regime_overfit_gate.json"
N_PARAM_COMBOS = len(CORR_THRESHOLDS) * len(REBAL_FREQS) * len(WEIGHT_SCALES)


def _log(m): print(f"[금레짐과최적화]{m}", file=sys.stderr)


def self_test():
    idx = pd.date_range("2003-01-03", periods=400, freq="W-FRI")  # ~7.7년
    rng = np.random.default_rng(11)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 400), "gold_mom_6m": rng.normal(0, 0.05, 400),
        "gold_mom_12m": rng.normal(0, 0.05, 400),
        "dxy_mom_3m": rng.normal(0, 0.02, 400), "dxy_mom_6m": rng.normal(0, 0.02, 400),
        "dxy_mom_12m": rng.normal(0, 0.02, 400),
        "real_rate_mom_3m": rng.normal(0, 0.3, 400), "real_rate_mom_6m": rng.normal(0, 0.3, 400),
        "real_rate_mom_12m": rng.normal(0, 0.3, 400),
        "ief_mom_3m": rng.normal(0, 0.02, 400), "ief_mom_6m": rng.normal(0, 0.02, 400),
        "ief_mom_12m": rng.normal(0, 0.02, 400),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.1, 400),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.1, 400),
    }, index=idx)
    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=7))
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays)))), index=bdays)

    trial_data = grid_trials(feat, gold_daily)
    assert len(trial_data["trials"]) == N_PARAM_COMBOS, trial_data["trials"]
    assert len(trial_data["excess_returns"]) == N_PARAM_COMBOS

    report = OS.analyze(trial_data, n_blocks=8, save=False)
    assert "dsr" in report and "pbo" in report
    _log("통과: grid_trials/run_dsr_pbo 배선 정상")
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_overfit_gate.py --self-test`
Expected: `NameError: name 'grid_trials' is not defined`

- [ ] **Step 3: `grid_trials`, `run_dsr_pbo` 구현**

```python
def grid_trials(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS) -> dict:
    trials, matrix, dates0 = [], [], None
    block_weeks = 4  # 월간 근사(주간 인덱스 4개 = 1블록)
    for ct in CORR_THRESHOLDS:
        for freq in REBAL_FREQS:
            for scale in WEIGHT_SCALES:
                regime = build_regime_series(features, ct, freq, scale)
                sim = simulate_exposure(gold_daily, regime, cost_bps)
                fw_level = fixed_weight_benchmark(regime)
                fw_regime = regime.copy(); fw_regime["exposure"] = fw_level
                sim_fw = simulate_exposure(gold_daily, fw_regime, cost_bps)
                r, b = sim["strat_ret"], sim_fw["strat_ret"]
                n = min(len(r), len(b))
                excess = r[:n] - b[:n]
                step = block_weeks * 5  # ~21거래일
                d, ex = [], []
                for t in range(0, n - step, step):
                    d.append(str(t)); ex.append(round(float(excess[t:t + step].sum()), 6))
                if dates0 is None or len(d) < len(dates0):
                    dates0 = d
                matrix.append(ex)
                trials.append(f"ct{ct}_{freq}_{scale}")
    matrix = [row[:len(dates0)] for row in matrix]
    return {"horizon": "monthly_approx", "universe": "gold_regime", "cost": f"{cost_bps}bp",
            "rebal_days": 21, "hold_days": 21, "dates": dates0,
            "trials": trials, "excess_returns": matrix}


def run_dsr_pbo(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS) -> dict:
    trial_data = grid_trials(features, gold_daily, cost_bps)
    return OS.analyze(trial_data, n_blocks=12, save=False)
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_overfit_gate.py --self-test`
Expected: `통과: grid_trials/run_dsr_pbo 배선 정상`

- [ ] **Step 5: 커밋**

```bash
git add gold_regime_overfit_gate.py
git commit -m "feat: 금 레짐신호 4단계 — 12조합 그리드 DSR/PBO (overfit_stats.analyze 재사용)"
```

---

## Task 8: `gold_regime_overfit_gate.py` — walk-forward + 비용민감도 + `run()`

**Files:**
- Modify: `gold_regime_overfit_gate.py`

**Interfaces:**
- Consumes: `grid_trials`, `run_dsr_pbo` (Task 7)
- Produces:
  - `walk_forward(features, gold_daily, cost_bps=COST_BPS, train_years=5, test_years=1) -> dict`
    — 반환: `n_folds(int), folds(list[dict]), oos_cagr_mean(float|None),
    n_param_combos_tried(int)`. 각 fold dict: `train_end, test_end, chosen_params
    ({"ct":,"freq":,"scale":}), oos_cagr, oos_ulcer`.
  - `cost_sensitivity(features, gold_daily, params: dict, cost_levels=(5.0, 15.0)) -> dict`
    — `{"5.0bp": {"cagr":, "ulcer":}, "15.0bp": {...}}`
  - `run(save: bool = True) -> dict` — `{"dsr_pbo":, "walk_forward":, "cost_sensitivity":,
    "n_param_combos": N_PARAM_COMBOS}`, `save=True`면 `OUTPUT_PATH`에 JSON 저장.

- [ ] **Step 1: self-test 추가**

```python
def self_test_walk_forward():
    idx = pd.date_range("2003-01-03", periods=600, freq="W-FRI")  # ~11.5년(5+1년 fold 최소 2개)
    rng = np.random.default_rng(13)
    feat = pd.DataFrame({
        "gold_mom_3m": rng.normal(0, 0.05, 600), "gold_mom_6m": rng.normal(0, 0.05, 600),
        "gold_mom_12m": rng.normal(0, 0.05, 600),
        "dxy_mom_3m": rng.normal(0, 0.02, 600), "dxy_mom_6m": rng.normal(0, 0.02, 600),
        "dxy_mom_12m": rng.normal(0, 0.02, 600),
        "real_rate_mom_3m": rng.normal(0, 0.3, 600), "real_rate_mom_6m": rng.normal(0, 0.3, 600),
        "real_rate_mom_12m": rng.normal(0, 0.3, 600),
        "ief_mom_3m": rng.normal(0, 0.02, 600), "ief_mom_6m": rng.normal(0, 0.02, 600),
        "ief_mom_12m": rng.normal(0, 0.02, 600),
        "gold_dxy_corr60": rng.uniform(-0.7, -0.1, 600),
        "gold_realrate_corr60": rng.uniform(-0.7, -0.1, 600),
    }, index=idx)
    bdays = pd.bdate_range(idx[0], idx[-1] + pd.Timedelta(days=7))
    gold_daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays)))), index=bdays)

    wf = walk_forward(feat, gold_daily, train_years=5, test_years=1)
    assert wf["n_folds"] >= 2, wf["n_folds"]
    assert wf["n_param_combos_tried"] == N_PARAM_COMBOS
    for f in wf["folds"]:
        assert set(f["chosen_params"]) == {"ct", "freq", "scale"}

    default_params = {"ct": DEFAULT_CORR_THRESHOLD, "freq": DEFAULT_REBAL_FREQ, "scale": DEFAULT_WEIGHT_SCALE}
    cs = cost_sensitivity(feat, gold_daily, default_params)
    assert set(cs) == {"5.0bp", "15.0bp"}
    assert cs["15.0bp"]["cagr"] <= cs["5.0bp"]["cagr"] + 1e-9, "비용이 높으면 CAGR이 같거나 낮아야 함"

    _log("통과: walk_forward/cost_sensitivity 배선 정상")


def self_test():
    ...  # Task 7의 self_test 본문
    self_test_walk_forward()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_overfit_gate.py --self-test`
Expected: `NameError: name 'walk_forward' is not defined`

- [ ] **Step 3: 구현**

```python
def walk_forward(features: pd.DataFrame, gold_daily: pd.Series, cost_bps: float = COST_BPS,
                 train_years: int = 5, test_years: int = 1) -> dict:
    train_weeks, test_weeks = train_years * 52, test_years * 52
    n = len(features)
    folds, start = [], 0
    while start + train_weeks + test_weeks <= n:
        train = features.iloc[start:start + train_weeks]
        test = features.iloc[start + train_weeks:start + train_weeks + test_weeks]
        train_gold = gold_daily[gold_daily.index <= train.index[-1]]
        best = None
        for ct in CORR_THRESHOLDS:
            for freq in REBAL_FREQS:
                for scale in WEIGHT_SCALES:
                    regime = build_regime_series(train, ct, freq, scale)
                    sim = simulate_exposure(train_gold, regime, cost_bps)
                    fw_level = fixed_weight_benchmark(regime)
                    fw_regime = regime.copy(); fw_regime["exposure"] = fw_level
                    sim_fw = simulate_exposure(train_gold, fw_regime, cost_bps)
                    score = ((sim_fw["ulcer"] - sim["ulcer"]) / sim_fw["ulcer"]
                            if sim_fw["ulcer"] > 0 else float("-inf"))
                    if best is None or score > best["score"]:
                        best = {"ct": ct, "freq": freq, "scale": scale, "score": score}
        test_regime = build_regime_series(test, best["ct"], best["freq"], best["scale"])
        test_gold = gold_daily[(gold_daily.index >= test.index[0]) & (gold_daily.index <= test.index[-1])]
        sim_test = simulate_exposure(test_gold, test_regime, cost_bps)
        folds.append({"train_end": str(train.index[-1].date()), "test_end": str(test.index[-1].date()),
                      "chosen_params": {k: best[k] for k in ("ct", "freq", "scale")},
                      "oos_cagr": round(sim_test["cagr"], 2), "oos_ulcer": round(sim_test["ulcer"], 2)})
        start += test_weeks
    return {"n_folds": len(folds), "folds": folds,
            "oos_cagr_mean": round(float(np.mean([f["oos_cagr"] for f in folds])), 2) if folds else None,
            "n_param_combos_tried": N_PARAM_COMBOS}


def cost_sensitivity(features: pd.DataFrame, gold_daily: pd.Series, params: dict,
                     cost_levels: tuple = (5.0, 15.0)) -> dict:
    out = {}
    for cb in cost_levels:
        regime = build_regime_series(features, params["ct"], params["freq"], params["scale"])
        sim = simulate_exposure(gold_daily, regime, cb)
        out[f"{cb}bp"] = {"cagr": round(sim["cagr"], 2), "ulcer": round(sim["ulcer"], 2)}
    return out


def run(save: bool = True) -> dict:
    features = load_or_build()
    gold_daily = fetch_all()["gold"]
    dsr_pbo = run_dsr_pbo(features, gold_daily)
    wf = walk_forward(features, gold_daily)
    default_params = {"ct": DEFAULT_CORR_THRESHOLD, "freq": DEFAULT_REBAL_FREQ, "scale": DEFAULT_WEIGHT_SCALE}
    cost_sens = cost_sensitivity(features, gold_daily, default_params)
    result = {"dsr_pbo": dsr_pbo, "walk_forward": wf, "cost_sensitivity": cost_sens,
             "n_param_combos": N_PARAM_COMBOS}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_overfit_gate.py --self-test`
Expected: 2개 self-test 함수 전부 통과

- [ ] **Step 5: 실데이터로 전체 게이트 실행(수 분 소요 가능 — walk-forward가 12조합 × fold 수만큼 반복)**

Run: `python gold_regime_overfit_gate.py`
Expected: `output/gold_regime_overfit_gate.json` 생성, `dsr_pbo.dsr`/`dsr_pbo.pbo` 존재,
`walk_forward.n_folds >= 1`

- [ ] **Step 6: 커밋**

```bash
git add gold_regime_overfit_gate.py
git commit -m "feat: 금 레짐신호 4단계 — walk-forward + 비용민감도 + run() 종합"
```

---

## Task 9: `gold_regime_report.py` — 현재 판정 + 최종 리포트

**Files:**
- Create: `gold_regime_report.py`

**Interfaces:**
- Consumes: `gold_regime_data.load_or_build/fetch_all`, `gold_regime_signal.classify/
  target_exposure/build_regime_series/era_performance` 및 기본값 상수,
  `gold_regime_overfit_gate.run`
- Produces:
  - `current_judgment(features: pd.DataFrame, corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> dict`
    — `{"as_of": "YYYY-MM-DD", **classify()반환, "target_exposure": float}`
  - `build_report(save: bool = True) -> dict` — `{"eras": [...], "overfit_gate": {...},
    "current_judgment": {...}}`, `save=True`면 `output/gold_regime_summary.json`(JSON)과
    `output/gold_regime_report.md`(서술형) 저장.

- [ ] **Step 1: self-test 뼈대 작성**

```python
#!/usr/bin/env python3
"""
gold_regime_report.py — 금 배분타이밍 레짐신호 5단계: 최종 리포트 + 현재 시점 판정.
docs/superpowers/specs/2026-08-25-gold-dxy-trading-design.md §5단계 참고.

실행: python gold_regime_report.py
      python gold_regime_report.py --self-test
결과: output/gold_regime_summary.json, output/gold_regime_report.md
"""
from __future__ import annotations
import os, sys, json, argparse
import pandas as pd

from gold_regime_data import load_or_build, fetch_all
from gold_regime_signal import (
    classify, target_exposure, build_regime_series, era_performance,
    DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE, COST_BPS,
)
import gold_regime_overfit_gate as OG

SUMMARY_PATH = "output/gold_regime_summary.json"
REPORT_PATH = "output/gold_regime_report.md"


def _log(m): print(f"[금레짐리포트] {m}", file=sys.stderr)


def self_test():
    from gold_regime_signal import _mk_row  # 재사용 — 합성 row
    row = _mk_row(gold_mom_3m=0.05, gold_mom_6m=0.08, gold_mom_12m=0.10,
                  dxy_mom_3m=-0.02, dxy_mom_6m=-0.03, dxy_mom_12m=-0.01,
                  gold_dxy_corr60=-0.6, gold_realrate_corr60=-0.5)
    feat = pd.DataFrame([row], index=[pd.Timestamp("2026-08-21")])
    j = current_judgment(feat)
    assert j["as_of"] == "2026-08-21"
    assert j["verdict"] in ("ADD", "HOLD", "REDUCE")
    assert "target_exposure" in j
    _log("통과: current_judgment 배선 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        build_report()
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python gold_regime_report.py --self-test`
Expected: `NameError: name 'current_judgment' is not defined`

- [ ] **Step 3: `current_judgment`, `_render_markdown`, `build_report` 구현**

```python
def current_judgment(features: pd.DataFrame, corr_threshold: float = DEFAULT_CORR_THRESHOLD) -> dict:
    last_ts = features.index[-1]
    c = classify(features.loc[last_ts], corr_threshold)
    return {"as_of": str(last_ts.date()), **c,
            "target_exposure": target_exposure(c["verdict"], c["strength"], DEFAULT_WEIGHT_SCALE)}


def _render_markdown(summary: dict) -> str:
    lines = ["# 금 배분타이밍 레짐신호 — 백테스트 리포트\n"]
    cj = summary["current_judgment"]
    lines.append(f"## 현재 판정 ({cj['as_of']} 기준)\n")
    lines.append(f"- **{cj['verdict']}"
                 f"{'(' + cj['strength'] + ')' if cj['strength'] else ''}** "
                 f"— 권고 노출 {cj['target_exposure']*100:.0f}%")
    lines.append(f"- 금 방향: {cj['gold_direction']} · "
                 f"DXY {'살아있음' if cj['dxy_alive'] else '죽음'}({cj['dxy_direction']}) · "
                 f"실질금리 {'살아있음' if cj['realrate_alive'] else '죽음'}({cj['realrate_direction']}) · "
                 f"IEF동조 {cj['ief_sync']}")
    lines.append(f"- 신뢰도: {cj['confidence']}"
                 f"{' (상관관계 붕괴 — 금 추세만으로 보수적 판단)' if cj['unexplained'] else ''}\n")

    lines.append("## 국면별 성과\n")
    lines.append("| 국면 | 일수 | 신호CAGR | Buy&Hold CAGR | 고정비중CAGR | 신호Ulcer | "
                 "BH Ulcer | 설명안됨% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in summary["eras"]:
        if "note" in e:
            lines.append(f"| {e['era']} | {e['n_days']} | {e['note']} | | | | | |")
            continue
        lines.append(f"| {e['era']} | {e['n_days']} | {e['signal_cagr']}% | {e['buyhold_cagr']}% | "
                     f"{e['fixed_weight_cagr']}% | {e['signal_ulcer']} | {e['buyhold_ulcer']} | "
                     f"{e['unexplained_pct']}% |")

    gate = summary["overfit_gate"]
    dsr = gate["dsr_pbo"]["dsr"]
    lines.append(f"\n## 과최적화 검증\n")
    lines.append(f"- 시도한 파라미터 조합: {gate['n_param_combos']}개")
    lines.append(f"- DSR: {dsr.get('dsr')} (95% 통과 기준 0.95) — {gate['dsr_pbo']['dsr_verdict']}")
    lines.append(f"- PBO(참고): {gate['dsr_pbo']['pbo']['pbo']} — {gate['dsr_pbo']['pbo_verdict']}")
    lines.append(f"- Walk-forward: {gate['walk_forward']['n_folds']}개 폴드, "
                 f"OOS 평균 CAGR {gate['walk_forward']['oos_cagr_mean']}%")
    lines.append(f"- 비용 민감도: {gate['cost_sensitivity']}")
    lines.append("\n## 리스크\n")
    lines.append("- DXY/실질금리-금 상관관계가 앞으로도 붕괴 상태를 유지하거나 다시 강화될지 불확실.")
    lines.append("- 2022년 이후 구조변화(중앙은행 매입 급증)가 지속될지는 이 백테스트로 확정할 수 없음.")
    return "\n".join(lines) + "\n"


def build_report(save: bool = True) -> dict:
    features = load_or_build()
    gold_daily = fetch_all()["gold"]
    regime = build_regime_series(features, DEFAULT_CORR_THRESHOLD, DEFAULT_REBAL_FREQ, DEFAULT_WEIGHT_SCALE)
    eras = era_performance(gold_daily, regime, COST_BPS)
    gate = OG.run(save=False)
    current = current_judgment(features)
    summary = {"eras": eras, "overfit_gate": gate, "current_judgment": current}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(_render_markdown(summary))
        _log(f"저장: {SUMMARY_PATH}, {REPORT_PATH}")
    return summary
```

- [ ] **Step 4: 실행해서 통과 확인**

Run: `python gold_regime_report.py --self-test`
Expected: `통과: current_judgment 배선 정상`

- [ ] **Step 5: 커밋**

```bash
git add gold_regime_report.py
git commit -m "feat: 금 레짐신호 5단계 — 현재 판정 + 최종 마크다운/JSON 리포트"
```

---

## Task 10: 전체 파이프라인 end-to-end 실행 + 최종 검증

**Files:** 없음(기존 4개 스크립트를 순서대로 실행만)

**Interfaces:** 없음 — 전 태스크 산출물 통합 확인

- [ ] **Step 1: 전체 self-test 일괄 실행**

Run:
```bash
python gold_regime_data.py --self-test && \
python gold_regime_signal.py --self-test && \
python gold_regime_overfit_gate.py --self-test && \
python gold_regime_report.py --self-test
```
Expected: 4개 스크립트 전부 종료코드 0, 에러 없음

- [ ] **Step 2: 실데이터로 파이프라인 순서대로 실행**

Run:
```bash
python gold_regime_data.py
python gold_regime_report.py
```
(`gold_regime_report.build_report()`가 내부에서 `gold_regime_overfit_gate.run(save=False)`를
호출하므로 overfit_gate를 별도로 먼저 실행할 필요는 없지만, 진행상황 확인을 위해 먼저
`python gold_regime_overfit_gate.py`를 실행해 `output/gold_regime_overfit_gate.json`을
따로 남겨도 됨)

Expected: `output/gold_regime_dataset.pkl`, `output/gold_regime_summary.json`,
`output/gold_regime_report.md` 전부 생성

- [ ] **Step 3: 리포트 내용 육안 확인**

Run: `python -c "print(open('output/gold_regime_report.md', encoding='utf-8').read())"`
Expected: "현재 판정" 섹션에 ADD/HOLD/REDUCE 중 하나 + 근거, "국면별 성과" 표에 4개 행,
"과최적화 검증"에 DSR 숫자와 판정 문구, "리스크" 섹션 존재. 국면별 표에서 2003년 이전
구간(2000년대 강세장 앞부분)이 데이터 부족으로 일수가 예상보다 적게 나올 수 있음 —
스펙의 명시된 데이터 제약이므로 정상.

- [ ] **Step 4: `output/gold_regime_overfit_gate.json`에서 DSR 값 확인 및 기록**

Run: `python -c "import json; d=json.load(open('output/gold_regime_overfit_gate.json', encoding='utf-8')); print(d['dsr_pbo']['dsr_verdict']); print(d['n_param_combos'])"`
Expected: `n_param_combos == 12`, `dsr_verdict`가 문자열로 출력됨(통과/미통과 어느 쪽이든
정상 — 이 태스크는 파이프라인이 끝까지 도는지 확인하는 것이지 결과가 반드시 유의해야
한다는 뜻은 아님)

- [ ] **Step 5: 최종 커밋(output 산출물 포함 여부는 저장소 컨벤션 확인 후 — 기존 output/*.json이
  커밋돼 있으므로 동일하게 포함)**

```bash
git add output/gold_regime_dataset.pkl output/gold_regime_overfit_gate.json \
        output/gold_regime_summary.json output/gold_regime_report.md \
        output/regime_price_cache_dxy.pkl output/regime_price_cache_ief_dxy.pkl \
        output/regime_price_cache_gold_dxy.pkl output/fred_cache_DFII10.pkl \
        output/fred_cache_T10YIE.pkl
git commit -m "chore: 금 레짐신호 파이프라인 최초 실행 산출물"
```

---

## Self-Review 결과 (계획 작성자 확인)

- **스펙 커버리지**: 0단계(배경)는 코드 요구사항 없음(설계 근거). 1단계(데이터/피처)
  → Task 1-2. 2단계(분류 로직)·3단계(백테스트) → Task 3-6. 4단계(과최적화 방지)
  → Task 7-8. 5단계(리포트+현재판정) → Task 9. 파이프라인 통합 확인 → Task 10.
  스펙의 모든 섹션에 대응하는 태스크가 있음.
- **플레이스홀더 스캔**: "TBD"/"이후 구현" 등 없음. 모든 코드 스텝에 실제 코드 포함.
- **타입/시그니처 일관성**: `classify()`가 반환하는 키(`verdict/strength/...`)를
  `target_exposure()`·`build_regime_series()`·`current_judgment()`가 동일하게 소비.
  `simulate_exposure()`의 반환 키(`nav/cagr/ulcer/mdd/strat_ret`)를 `era_performance()`·
  `walk_forward()`·`cost_sensitivity()`가 동일하게 사용. `WEIGHT_SCALES`/`CORR_THRESHOLDS`/
  `REBAL_FREQS` 상수는 `gold_regime_signal.py`에 한 번만 정의하고 `overfit_gate.py`가
  import해서 재사용(중복 정의 없음).
