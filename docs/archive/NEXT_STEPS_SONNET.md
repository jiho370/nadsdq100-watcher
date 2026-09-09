# NEXT_STEPS_SONNET.md — v2 백테스트 결과 이후 작업 지시문 (Sonnet용)

> 이 문서는 Fable 5가 작성했다. 지호 님이 PC에서 v2 검증 파이프라인을 실행한 뒤,
> 그 **결과 JSON을 근거로** 아래 분기 중 하나를 수행하라.

## ★ 판정 확정 (2026-07-12, Fable 5) — 분기 선택 완료
- **G1 통과**: PBO 15.0%(purged CSCV) · DSR(T_eff=16) 0.9711 · WF OOS +3.58%p.
  채택 가중치 `int_gp_assets1·rd_mktcap2·shareholder_yield2`, 회전율 12.5%,
  6M 순초과 +5.4%p·12M +12.5%p·승률 87.5% (전기간·비용 반영·PIT).
- **G2 실패 (3개 호라이즌 전부)**: 스냅샷 Spearman — 6m ρ̄=0.026 t=0.83,
  3m ρ̄=0.015 t=0.71, 12m 시도 기록 있음(동일하게 미통과). 분위표는 1~10분위가
  평평(상위 1~2분위만 근소 우위). 다중검정 예산 소진 — **추가 호라이즌 시도 금지**.
- **결론: 분기 C-3 수행** — 종목별 0~10 점수는 도입 보류. 아래 "기대값 리포트"로 전환.

### 분기 C-3 상세: 기대값(캘리브레이션) 리포트 스펙
점수 대신, 검증된 사실만 리포트에 추가한다(발송 코드 수정은 최소 침습 1곳 원칙 유지):
1. 전략 레벨 기대값 박스(주 1회 또는 월 1회 표기):
   "이 추천 방식의 과거 10년 실측(비용·생존편향 반영): 6개월 보유 시 평균 +12.4%
   (S&P500 대비 +5.4%p), 승률 87.5%, 최악 -14% · 통계 검증: PBO 15%·DSR 0.97"
   — 숫자는 `backtest_costs_compare.json`의 pit_best에서 읽고, 하드코딩 금지.
2. 종목별로는 순위 사실만: "오늘 팩터 랭킹 N위 / 후보 M종목" (0~10 점수·기대수익
   문구 금지 — 근거 없는 정밀함을 팔지 않는다).
3. `score_calibration.load_calibration()`이 None을 반환하는 현 상태를 그대로 이용:
   점수 표시 코드는 만들되 게이트 뒤에 두면, 향후 분기 재검증에서 G2가 통과될 때
   자동으로 점수가 켜진다(수동 개입 불필요).
4. 분기 재검증(분기~반기)마다 1~3단계 파이프라인 재실행을 스케줄에 반영할 것.

## 0. 시작하기 전에 반드시 읽을 것
1. `SCORE_MODEL_DESIGN.md` — v1 실패 원인(D1~D5)과 v2 설계 근거. 이 문서와 충돌하는
   구현을 하지 말 것.
2. `STRATEGY_UPGRADE_PROPOSAL.md` 5~6장, `VALIDATION_PIPELINE.md`
3. 실행 결과: `output/backtest_costs_compare.json`, `output/pbo_report.json`,
   `output/score_calibration.json`
4. 코드: `score_calibration.py`(v2), `overfit_stats.py`, `backtest_costs.py`,
   `fundamentals_edgar.py`(신규 팩터 5종 추가됨: droe, debt_issuance, rd_mktcap,
   int_gp_assets, int_value)

## 1. 절대 제약 (협상 불가)
- `daily_ai_report.py`·`pregen.py`·`ai_report.py` 등 발송 파이프라인은 **점수 표시
  1곳 추가 외에는 수정 금지**. 그 1곳도 반드시 `score_calibration.load_calibration()`이
  None이면 아무것도 표시하지 않는 가드 뒤에 둘 것.
- 모든 새 통계 주장은 PBO/DSR(T_eff 보정)/스냅샷 Spearman 수치로 뒷받침할 것.
  "될 것 같다"는 서술 금지.
- 새 외부 의존성 추가 금지(현 requirements 범위 내).
- 모든 수정 후 해당 스크립트의 `--self-test` 통과를 확인하고 결과를 보고에 포함할 것.

## 2. 판정 기준 (결과 JSON에서 읽기)
```
G1 = pbo_report.json:  passed == true          (PBO<50% & DSR(T_eff)≥0.95)
G2 = score_calibration.json:  display_allowed == true
     (스냅샷 ρ̄>0 & t≥2 & D10−D1 > 왕복비용)
G3 = score_calibration.json:  recent_5y_deciles 의 상위분위 우위가 전체 기간과
     방향 일치 (10분위 평균 > 1분위 평균)
```

## 3. 분기별 작업

### 분기 A — G1·G2·G3 모두 참: 점수를 리포트에 통합
1. `score_live.py` 신규 작성(발송 로직과 분리된 순수 계산 모듈):
   - 입력: 당일 후보 종목들의 팩터값(`export_data.py`가 이미 `factor_values`를 호출함),
     `score_calibration.json`의 `latest_weights`.
   - 계산: 후보 유니버스 내 z→가중합→백분위→`score_from_percentile()` (0~10).
   - AI 검증 결과에 따른 가감은 **±1점 상한**(설계 §2.1-3), 근거 문자열 포함.
2. 리포트 표시(발송 코드 최소 침습 1곳):
   - `load_calibration()`이 None → 표시 생략(코드 경로 자체가 이렇게 짜여 있어야 함).
   - 표시 형식: `점수 N — 과거 동일 분위(6개월): 평균 +X.X% · 승률 XX% · 손익비 X.X`
     숫자는 전부 `deciles[N-1]` 실측치. 임의 문구로 기대수익을 서술하지 말 것.
3. `output/score_log.json`에 날짜별 (종목, 점수, 분위) 기록 — 향후 라이브 성과 추적용.

### 분기 B — G1 거짓 (PBO/DSR 미통과): 점수 작업 중단
1. 점수 통합 작업을 하지 말 것. 지호 님께 수치와 함께 보고:
   PBO, DSR(T_eff), dsr_uncorrected, 워크포워드 OOS.
2. 허용된 후속 조사(각각 결과를 `attempted_horizons`처럼 기록):
   - `backtest_costs.py --rebal-days 126` (리밸=보유 → 중첩 제거 후 재판정)
   - 팩터 후보를 군집(IR-가중)만으로 좁혀 재실행 — 이미 v2가 이 방식이므로
     `score_calibration.py --force`로 **연구용** 캘리브레이션만 뽑아 ρ̄를 참고 보고.
3. 위로도 미통과면 결론은 "점수 도입 보류"로 문서화(설계 §2.5). 우회 금지.

### 분기 C — G1 참, G2 또는 G3 거짓: 호라이즌 한정 재시도
1. `score_calibration.py --horizon 3m`, 그다음 `--horizon 12m` (최대 2회 추가,
   `attempted_horizons`에 자동 기록됨 — 그 이상 시도 금지: 다중검정 예산).
2. 어느 호라이즌이든 G2·G3 통과 → 그 호라이즌으로 분기 A 수행(표시 문구의
   "6개월"을 해당 기간으로).
3. 전부 실패 → 분기 B-3와 동일하게 보류 결론. 단, 분위표 자체는 "전략 기대값
   캘리브레이션 리포트"로 지호 님께 전달(점수 없이).

## 4. 병행 트랙 — 한국 팩터 (미국 결과와 무관하게 진행 가능)
1. 지호 님이 `kr_factor_ic.py --collect` → 본 실행을 마치면 `output/kr_ic_report.json`
   확인. `significant_factors`(IC>0 & t≥3)가 있으면:
   - `kr_stocks.py`의 랭킹에 `추세 z + 펀더멘털 z` 형태로 편입하는 패치를 **별도
     브랜치/파일**로 제안(기존 select() 시그니처 유지, 환경변수로 on/off).
   - 없으면 현행 유지 + "한국 펀더멘털 IC 무유의" 기록.
2. 주의: 미국에서 mom12_1 IC가 음수였다(SCORE_MODEL_DESIGN.md §4-3). 한국 랭킹의
   모멘텀 0.6 가중도 kr_ic_report의 mom12_1 행으로 교차 확인해 보고할 것.

## 5. 마무리 하우스키핑 (어느 분기든 공통)
- `STRATEGY.md`의 "+20%p·승률 89.5%·MDD -1.8%" 등 재현 안 된 수치를 실측치
  (`backtest_costs_compare.json` 기준)로 교체하고, 각주로 "2026-07 재검증" 명시.
- `VALIDATION_PIPELINE.md`에 v2 변경(6m 캘리브레이션·T_eff·embargo) 반영.
- 분기 결과와 무관하게: 시도한 것·실패한 것·수치를 전부 보고서에 남길 것.
  실패 기록이 다음 재검증(분기~반기)의 다중검정 보정 입력이 된다.

## 6. 트랙 C — 지지·저항 실행규칙 백테스트 (분기 판정과 병행 가능)
`SCORE_MODEL_DESIGN.md` 부록 A 스펙대로:
1. `backtest_exec.py` 신규 작성 — A1 신호 계산 + A2-(b) 진입·청산 규칙 비교.
   `backtest_costs.CostModel`·PIT 유니버스 재사용. 규칙 조합 수를 기록하고
   trial_returns 포맷으로 저장해 `overfit_stats.py`로 동일 판정.
2. A2-(a): `SR_CANDIDATES` 7종을 `score_calibration.py`에 별도 후보 리스트로 추가
   (`--candidates sr` 옵션). 기본 실행에는 포함하지 말 것(예산 분리).
3. 채택 기준: 청산 규칙은 "현행 -20% 전량 대비 net 수익 개선 & 손절빈도 감소"가
   T_eff 보정 후에도 유지될 때만 entry_plan/holdings 패치 제안.

## 7. 트랙 D — 산업/테마 모멘텀 (부록 B 스펙)
1. `sector_trend.py` 신규 — B2 규칙(200MA & 6m 모멘텀 상위 K, 히스테리시스 3일).
   먼저 **정보 신호로만** 리포트에 추가(자동 매수 없음). 리포트 통합은 분기 A와
   동일하게 1곳·가드 뒤에.
2. ETF 로테이션 백테스트(월간, 비용 반영, SPY 벤치마크) → PBO/DSR 통과 시에만
   매수신호 승격을 지호 님께 제안.
3. `ind_mom6`·`rel_ind_mom`을 v2 후보로 추가(비PIT 산업분류 한계를 결과에 명기).
4. 한국: pykrx 업종지수로 동일 신호. 테마 매핑은 `theme_map_kr.json` 옵션(B3).

## 부록 — 지호 님 PC 실행 명령 (Sonnet 착수 전 선행)
```
py fundamentals_edgar.py                    # 신규 항목(rnd·sga)만 증분 수집(수 분)
py backtest_costs.py --self-test
py overfit_stats.py --self-test
py score_calibration.py --self-test
py backtest_costs.py --years 10 --oos 0.4
py overfit_stats.py                         # G1 판정 (T_eff·purging 반영)
py score_calibration.py --years 10          # G2·G3 판정 (게이트 통과 시에만 실행됨)
```
