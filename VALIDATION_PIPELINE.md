# 검증 파이프라인 실행 가이드 (STRATEGY_UPGRADE_PROPOSAL.md 6장 로드맵 1~5단계)

기존 발송 로직(`daily_ai_report.py`, `pregen.py` 등)은 전혀 건드리지 않았다.
아래 4개 스크립트는 전부 독립 실행형이며, 검증을 통과한 뒤에만 리포트에 통합한다.

| 스크립트 | 로드맵 | 출력 |
|---|---|---|
| `backtest_costs.py` | 1~2단계: 거래비용·세금 + 생존편향 제거(PIT) | `output/backtest_costs_compare.json`, `output/trial_returns.json` |
| `overfit_stats.py` | 3단계: PBO(purged CSCV, embargo) / Deflated Sharpe Ratio(T_eff 보정 병기) | `output/pbo_report.json` |
| `score_calibration.py` | 4단계(v2): 6m 기본 호라이즌·워크포워드 IR-가중 군집 점수 + 스냅샷 단위 Spearman | `output/score_calibration.json` |
| `expectancy_report.py` | 4단계 실패 시(분기 C-3): 점수 대신 검증된 기대값(캘리브레이션)만 리포트에 노출 | 리포트 HTML(전략 박스 + 종목 순위 사실) |
| `kr_factor_ic.py` | 5단계: DART 기반 한국 팩터 IC 검증 | `output/kr_fundamentals_dart.json`, `output/kr_ic_report.json` |

## 실행 순서와 체크포인트 (반드시 이 순서대로)

```
0) 준비 (1회)
   pip install -r requirements.txt
   python fundamentals_edgar.py                       # 미국 펀더멘탈 수집/증분
   # (선택·권장) 과거 편입 종목까지 펀더멘탈 보강:
   #   python backtest_costs.py --export-universe 로 티커 목록을 뽑아
   #   python fundamentals_edgar.py --tickers <목록> 실행

1) 거래비용 + PIT 백테스트
   python backtest_costs.py --years 10 --topn 30 --oos 0.4
   ── 체크포인트 1→2 ──────────────────────────────────
   비교표에서 확인할 것:
   · "6M수익 gross" 대비 "6M수익 net"이 얼마나 깎였는가
   · pit(기존가중치)·pit(재탐색) 열에서 순초과수익(excess net)이 여전히 +인가
   · PIT 커버리지(평균/최저 %) — 낮으면 잔존 생존편향이 큼
   ── 체크포인트 2→3 ──────────────────────────────────
   · output/backtest_costs_compare.json 의 ic.legacy vs ic.pit 비교
     → 생존편향 제거 후 팩터 IC가 크게 바뀌면 기존 가중치 재검토 필요

2) 과최적화 통계 검증 (v2: 2026-07 재검증 — embargo·T_eff 반영)
   python overfit_stats.py
   ── 체크포인트 3→4 (자동 게이트) ─────────────────────
   · PBO < 50%(purged CSCV, embargo=⌈보유일/리밸일⌉ 이벤트) 그리고
     DSR(T_eff = T×rebal_days/hold_days 보정) ≥ 0.95 → pbo_report.json 에 passed=true
   · T=이벤트수(중첩 무시) 기준 DSR은 참고용(dsr_uncorrected)일 뿐 판정에 쓰지 않는다
     (중첩 표본이 게이트를 관대하게 만드는 것을 방지 — SCORE_MODEL_DESIGN.md D3)
   · passed=false 면 3단계(점수)는 실행 자체가 차단됨 — 여기서 멈추고 결과 검토
   · passed=true 면 라이브 선정에 반영: python backtest_costs.py --publish-weights
     → pit_best 가중치를 output/best_weights.json 으로 발행(export_data.select_pool 이 사용).
     이 파일이 없으면 발송 로직은 12-1 모멘텀 폴백으로 동작하니(2026-07 발견: 그간 실제로
     폴백이었음), 재검증 통과 때마다 발행을 잊지 말 것.

3) 0~10 점수 캘리브레이션 (v2: 6m 기본, passed=true 일 때만 실행 가능)
   python score_calibration.py --years 10           # 기본 horizon=6m(D1: 팩터는 느린 신호)
   ── 체크포인트 4→5 (자동 게이트) ─────────────────────
   · 스냅샷 단위 Spearman ρ̄>0 & t≥2, D10−D1 스프레드 > 왕복비용 — 전부 통과해야 display_allowed=true
   · 실패 시 --horizon 3m, --horizon 12m 순으로 최대 2회 추가 시도 가능
     (attempted_horizons 에 자동 기록 — 다중검정 예산 관리, 그 이상 시도 금지)
   · 리포트 통합 시 score_calibration.load_calibration() 이 None 이면
     점수를 표시하지 않는 구조 — 통합 코드는 검증 통과 후 별도 작업
   · 전 호라이즌 실패(2026-07 재검증 결과: 6m·3m·12m 전부 미통과) 시 점수 도입은 보류하고,
     `expectancy_report.py`가 검증된 사실(전략 레벨 기대값 박스 + 종목 순위)만 리포트에
     노출한다(분기 C-3, NEXT_STEPS_SONNET.md). 이 경로도 load_calibration() 게이트를 그대로
     재사용하므로, 다음 재검증에서 G2가 통과하면 수동 개입 없이 점수 표시가 자동으로 켜진다.

4) 한국 팩터 IC (독립 트랙 — 1~3과 병행 가능)
   # 사전: opendart.fss.or.kr 무료 API 키 → 환경변수 DART_API_KEY
   python kr_factor_ic.py --collect --years 10        # DART 수집(재개 가능)
   python kr_factor_ic.py --years 10                  # IC 검증
   · IC>0 & t≥3.0(Harvey-Liu-Zhu) 통과 팩터만 kr_stocks.py 랭킹 편입 후보

5) (후순위) 개인화 시스템(7장)은 위 1~4가 전부 통과된 뒤에만 시작
```

## 설계 노트

- **비용 모델**: 한국 매도 거래세 0.20%(코스피=거래세0.05+농특세0.15, 코스닥=거래세0.20),
  미국 SEC fee 0.00278%(매도), 공통 `--commission-bps`·`--slippage-bps`(기본 편도 5bp).
  이벤트 기반 백테스트("바스켓 매수 후 h개월 보유")이므로 이벤트당 왕복 1회가 정확한 반영.
- **PIT 유니버스**: fja05680/sp500 공개 CSV(월별 구성종목, 1996~). 최초 실행 시 자동
  다운로드 후 `output/sp500_pit.csv` 캐시. 상장폐지 종목은 야후에 시세가 없어 완전 제거는
  불가 — 커버리지 %로 잔존 편향 크기를 정직하게 보고.
- **시점정합성**: 미국은 기존 fundamentals_edgar의 filed(공시일) asof 조회를 그대로 사용.
  한국(DART)은 접수일(rcept_no) +1일 지연을 저장 단계에서 적용.
- **self-test**: 다섯 스크립트(위 표 4개 + `expectancy_report.py`) 모두 `--self-test` 지원
  (합성 데이터로 로직 검증, 네트워크 불필요).
- `output/` 의 결과 JSON은 실행 시점 데이터 기준으로 재생성됨. 분기 재검증 때마다
  1→2→3 순서로 다시 돌리고 pbo_report.json 을 로그로 보관할 것(제안서 4.2절).
- **v2 변경(2026-07 재검증)**: 점수 캘리브레이션 기본 호라이즌 1m→6m(D1), 임의 고정가중 폐지 →
  워크포워드 IR-가중 팩터 군집(D2·D4), PBO에 purged/embargo 추가·DSR에 T_eff 병기(D3),
  전체기간·최근5년 분위표 분리(D5). 근거: `SCORE_MODEL_DESIGN.md`.
