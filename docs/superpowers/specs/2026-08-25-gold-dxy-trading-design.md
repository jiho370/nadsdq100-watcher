# DXY-Gold 트레이딩 알고리즘 설계·백테스트 — 설계

날짜: 2026-08-25
상태: 승인됨 (지호 님, 2026-08-25)

## 배경 / 가설

DXY(달러인덱스)-금은 강한 음의 상관관계를 가지지만 고정 상수가 아니라 시변(time-varying)
관계다. 특히 2022년 이후 중앙은행 금매입 급증(연 1,000톤+)과 지정학 리스크 프리미엄으로
전통적인 달러/실질금리-금 관계가 약화되었다는 것이 최근 Fed·WGC·학술 페이퍼의 공통된
발견이다. 이 프로젝트는 이 가설을 전제로, "DXY/실질금리 필터가 금 모멘텀 신호를 개선하는가,
그리고 그 개선 효과가 2022년 이후 구간에서 약화되는가"를 실증적으로 검증한다.

## 목표

금 트렌드추종 전략에 DXY·실질금리·국채모멘텀 필터를 결합한 6가지 변형(A~F)을 설계하고,
국면별(4구간) 성과 차이와 과최적화 통계(walk-forward, PBO, DSR)로 통계적 유의성을 검증해
최종 비교 리포트를 만든다.

## 범위

- **포함**: 데이터/피처 파이프라인, 전략 A~F 구현·백테스트, 4구간 국면분리 검증,
  walk-forward + PBO/DSR 과최적화 게이트, 거래비용 반영, 최종 비교 리포트.
- **제외**: 실계좌 연동/자동매매, 실시간 신호 발송(기존 daily_ai_report.py 파이프라인에
  통합하지 않음 — 이번엔 순수 리서치/백테스트만), 옵션/선물 롤오버 최적화.

## 데이터 제약 (실측 확인됨, 2026-08-25)

| 시리즈 | 소스 | 시작일 | 비고 |
|---|---|---|---|
| Gold | GC=F (yfinance) | 2000-08-30 | 선물 연속계약, 자동조정 |
| DXY | DX-Y.NYB (yfinance) | 1971-01-04 | 결측/이상치 없음 확인 |
| IEF | yfinance | 2002-07-30 | |
| VIX | ^VIX (yfinance) | 1990-01-02 | |
| 실질금리 DFII10 | FRED | 2003-01-02 | |
| 기대인플레 T10YIE | FRED | 2003-01-02 | |
| S&P500 (SPY) | yfinance | 1993-01-29 | 벤치마크용 |
| UUP (DXY 프록시) | yfinance | 2007-03-01 | F 페어트레이드 전용 |

**공통 구간은 2003-01~현재.** 실질금리·기대인플레(C 전략의 필수 입력)가 2003-01 이전에
없어 A/B/D/E처럼 금·DXY·IEF만 쓰는 전략은 각자 2000/2002년대부터 더 길게 돌릴 수 있지만,
전체 6개 전략을 동일 기간으로 비교하는 헤드라인 결과는 2003-01~현재 기준으로 낸다.
개별 전략의 최대가용기간 결과는 부록으로 병기한다.

## 기존 인프라 재사용

- `backtest_regime_assets.fetch()` — yfinance 캐시(stale 5일 재다운로드)
- `fx_hedge_validation.fetch_fred()` — FRED CSV 캐시(stale 35일)
- `backtest_regime_assets._cagr/_ulcer/_mdd/paired_block_bootstrap`
- `overfit_stats.pbo_cscv`(purging+embargo 내장 — CPCV 요건 충족) + `deflated_sharpe` —
  4단계 CPCV+PBO+DSR을 새로 구현하지 않고 그대로 재사용
- `regime_era_split.py`의 국면분할 패턴(신호는 전체 이력에 대해 한 번만 계산 후 구간을
  잘라 구간 내 복리 계산 — 구간마다 워밍업을 다시 만들지 않음)

## 설계 결정 (assumption)

1. **롱/현금(0~1) 전용, 숏 없음** — F(페어트레이드)만 예외적으로 DXY 프록시(UUP) 숏.
   기존 저장소 전체가 금/주식을 숏하지 않는 컨벤션이라 그대로 따름.
2. **변동성 스케일링은 피처로만 생성**, A~F 전략에 자동 적용하지 않음. 1단계 스펙에서
   "피처 생성" 항목으로 명시돼 있어 신호 로직과 분리해 해석. 최종 채택 후보 전략에 한해
   vol-target 오버레이 버전을 비교표에 추가로 병기.
3. **파라미터는 3개로 제한**: 모멘텀 룩백(3/6/12개월), 신호 확인일수(1/3/5일, 휩쏘 방지),
   상관 임계값(E 전용, -0.2/-0.4/-0.6). B/C/D의 필터 자산 모멘텀 룩백은 금과 동일 룩백을
   재사용(4번째 파라미터를 만들지 않음).
4. **비용 가정**: 5bp 편도(기존 GOLD COST_BPS와 동일) 기본 + 15bp 스트레스 케이스 병기.
5. **국면 구간**: 2000년대 강세장=2001-01~2011-08, 2013-2015 약세장=2013-01~2015-12,
   2022 금리인상기=2022-03~2023-07, 2022년 이후=2022-01~현재(겹침 허용 — 사용자 스펙이
   둘 다 요구).
6. **exposure/cost 모델**: 모든 전략은 턴오버 기반 비용(`|Δexposure| × cost_bps`)을 쓰는
   `simulate_exposure()`로 통일한다 — 이진(0/1) 노출에서는 flip 시 정확히 1×cost_bps가
   부과되어 기존 `backtest_regime_assets.simulate()`의 flip 비용과 동일하게 축소된다.
   연속 노출(D의 0.5 단계, vol-target 오버레이 등)에서는 부분 리밸런싱 비용까지 반영.

## 모듈 설계

### 1. `gold_dxy_data.py` (1단계: 데이터·피처)

- `fetch_all()`: Gold(GC=F)/DXY(DX-Y.NYB)/IEF/VIX/SPY(yfinance, `RA.fetch` 재사용) +
  DFII10/T10YIE(FRED, `fetch_fred` 재사용)를 Gold 거래일 캘린더에 정렬(ffill).
- `build_features(df)`:
  - 금/DXY/실질금리/IEF 각각의 3·6·12개월 모멘텀(방향+강도 = 수익률 그 자체)
  - 금-DXY, 금-실질금리 60일 롤링 상관계수
  - vol-scaled 피처: 63일 realized vol 기준 ex-ante vol targeting 스칼라
    (`target_vol / realized_vol`, cap 있음) — 참고용 컬럼으로만 저장
- 캐시: `output/gold_dxy_dataset.pkl` (파케이 대신 pickle, 저장소 기존 컨벤션과 동일)

### 2. `gold_dxy_strategies.py` (2단계: 신호 A~F)

- 공용: `simulate_exposure(returns, exposure, cost_bps)` (턴오버 비용), `simulate_pair()`
  (금 롱 / UUP 숏 2-leg)
- A: 금 모멘텀(룩백 P1) 부호 → 노출 0/1 (확인일수 P2로 휩쏘 방지)
- B: A ∧ (DXY 모멘텀(P1) < 0)
- C: A ∧ (실질금리 모멘텀(P1) < 0) — B와 반드시 병렬 비교 리포트
- D: 금 모멘텀·IEF 모멘텀 동조 레짐 — 둘 다 양(+)이면 노출 1.0, 금만 양이면 0.5, 그 외 0
- E: 상관 게이트 — 60일 롤링 corr(금,DXY) < 임계값(P3)일 때만 B 활성화, 붕괴 시 A로 대체
- F: 금 롱 100% / UUP 숏 100% 스프레드 (상시, 필터 없음 — 페어 자체의 구조적 수익성 확인용)
- 벤치마크: 금 buy-hold(GC=F), S&P500 buy-hold(SPY)

### 3. `gold_dxy_regime_split.py` (3단계: 국면분리)

- `regime_era_split.py` 패턴 그대로: 전체 이력에서 신호 1회 계산 → 4개 구간으로 수익률
  절단 후 구간 내 복리. 구간별 CAGR/Ulcer/MDD/승률 표.
- 2022년 이후 구간에서 B(DXY필터)·C(실질금리필터)의 초과성과가 이전 구간 대비 축소/역전
  되는지 별도 비교 행 추가.

### 4. `gold_dxy_overfit_gate.py` (4단계: 과최적화 방지)

- Walk-forward: expanding window, 최소 학습 5년 → 1년씩 테스트, 매 스텝 학습구간에서
  composite score(Ulcer 개선 vs CAGR 손실예산 — `backtest_regime_assets.composite_score`
  재사용) 최고 파라미터 선택 → 다음 1년 OOS 적용, 체이닝.
- CPCV/PBO/DSR: 전체 파라미터 그리드(P1×P2×P3, 최대 3×3×3=27조합 x 전략 6개)의 월간
  비중첩 초과수익 행렬을 `overfit_stats.analyze()`에 투입 (embargo=purging 자동 적용).
  시도한 조합 수를 결과 JSON에 그대로 기록.
- 비용 민감도: 5bp/15bp 두 코스트로 각 후보 재실행, 델타 보고.

### 5. `gold_dxy_report.py` (5단계: 최종 리포트)

- 전략별 비교표(CAGR/Sharpe/MDD/승률/벤치마크 대비 상관), 국면별 성과, DSR/PBO 결과,
  다중검정 조합 수, 실전 리스크(상관관계 붕괴 가능성, 구조변화 지속 불확실성) 정리.
- 산출물: `output/gold_dxy_report.md` (서술형) + `output/gold_dxy_summary.json` (표 데이터).

## 검증 계획

- 각 모듈에 `--self-test` (합성 데이터로 배선 확인, 저장소 기존 컨벤션과 동일)
- 최종 실행 순서: data → strategies → regime_split → overfit_gate → report,
  각 단계 산출물이 다음 단계 입력이 되므로 순서대로 실행
