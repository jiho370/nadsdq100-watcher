# 검증 파이프라인 실행 가이드 (STRATEGY_UPGRADE_PROPOSAL.md 6장 로드맵 1~5단계)

기존 발송 로직(`daily_ai_report.py`, `pregen.py` 등)은 전혀 건드리지 않았다.
아래 4개 스크립트는 전부 독립 실행형이며, 검증을 통과한 뒤에만 리포트에 통합한다.

| 스크립트 | 로드맵 | 출력 |
|---|---|---|
| `backtest_costs.py` | 1~2단계: 거래비용·세금 + 생존편향 제거(PIT) | `output/backtest_costs_compare.json`, `output/trial_returns.json` |
| `overfit_stats.py` | 3단계: PBO / Deflated Sharpe Ratio | `output/pbo_report.json` |
| `score_calibration.py` | 4단계: 0~10 점수 캘리브레이션 + 단조성 검증 | `output/score_calibration.json` |
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

2) 과최적화 통계 검증
   python overfit_stats.py
   ── 체크포인트 3→4 (자동 게이트) ─────────────────────
   · PBO < 50% 그리고 DSR ≥ 0.95 → pbo_report.json 에 passed=true
   · passed=false 면 3단계(점수)는 실행 자체가 차단됨 — 여기서 멈추고 결과 검토

3) 0~10 점수 캘리브레이션  (passed=true 일 때만 실행 가능)
   python score_calibration.py --years 10 --horizon 1m
   ── 체크포인트 4→5 (자동 게이트) ─────────────────────
   · Spearman ρ>0 & p<0.05 통과 시에만 display_allowed=true
   · 리포트 통합 시 score_calibration.load_calibration() 이 None 이면
     점수를 표시하지 않는 구조 — 통합 코드는 검증 통과 후 별도 작업

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
- **self-test**: 네 스크립트 모두 `--self-test` 지원(합성 데이터로 로직 검증, 네트워크 불필요).
- `output/` 의 결과 JSON은 실행 시점 데이터 기준으로 재생성됨. 분기 재검증 때마다
  1→2→3 순서로 다시 돌리고 pbo_report.json 을 로그로 보관할 것(제안서 4.2절).
