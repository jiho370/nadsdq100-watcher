# STRATEGY.md §3.5 초안 (미반영) — 팩터 ETF 대조군 실험 (2026-07-14)

**상태: 결과 미채움 — 아래 표는 전부 `TBD`.** §5 지시대로 자동 반영하지 않았다. 이 파일은
로컬 실행 후 실제 수치가 나오면 STRATEGY.md §3 뒤에 §3.5로 삽입할 초안 텍스트다.

## 왜 결과가 비어 있는가 (환경 제약, 설계 문제 아님)

이 세션(원격 샌드박스)은 조직 프록시 정책으로 `data.krx.co.kr`·Yahoo Finance
(`query1/2.finance.yahoo.com`, `fc.yahoo.com`, `stooq.com` 등)에 대한 아웃바운드 접속이
전부 `403`으로 차단된다(`curl -sS "$HTTPS_PROXY/__agentproxy/status"`로 확인 —
"gateway answered 403 to CONNECT (policy denial)"). 프록시 안내서 자체가 "이런 차단은
재시도하거나 우회하지 말고 보고하라"고 명시한다. 이 저장소의 기존 스크립트들이 전부
"실행(PC):"로 헤더를 다는 이유와 같다 — 실제 데이터 수집은 지호 님 로컬 PC(코스피200
캐시·`kr_bt_cache.json` 등이 이미 그렇게 생성됨, HANDOFF.md 참고)에서 이뤄지고, 이 세션은
로직을 작성·자체검증(self-test)까지만 한다.

**직접 확인된 것**: `output/kospi200_cache.json`·`output/kr_bt_cache.json`·
`output/kr_strategy_navs.json`(코어 B1·커스텀 새틀라이트 valuediv_flow — 둘 다 로컬에서 이미
받아둔 캐시)는 이 세션에서도 읽을 수 있어 **프레임 A(현행)·프레임 C(KODEX200 단독)는 새
데이터 없이 재현 가능**했고, `backtest_etf_control.py --market kr`로 실행해 STRATEGY.md
§3의 기존 수치(코어65:새틀35 → CAGR 21.7%·샤프 1.22·MDD -17.1%)와 **소수점까지 일치**함을
확인했다(엔진 재사용이 올바르게 배선됐다는 검증).

**막힌 것**: 프레임 B의 신규 후보 4종(161510·211900·211560·279530)·미국 6종
(SYLD·PKW·QUAL·COWZ·SCHD·QUAL50+SYLD50)·SPY·코리아밸류업 지수는 전부 이 세션에서
새로 받아야 하는 시계열이라 전부 미확보. §6 "가격 시계열로 고배당 ETF를 평가하지 말 것"
지시에 따라, TR(총수익) 지수를 자동 탐색해서 못 찾은 후보는 **가격만으로라도 평가하지 않고
`blocked_need_tr`로만 표시**했다(임의 수치 날조 방지).

## 완료된 것 — `backtest_etf_control.py` (신규, 커밋됨)

기존 `backtest_portfolio.py`(NAV 프레임·비용모델)·`core_satellite_kr.py`(레짐·혼합·서브기간·
`SUBS`)·`overfit_stats.py`(PBO/DSR)를 그대로 재사용, 중복 구현 없음. 구현 내용:

- **프레임 A/B/C/D** 전부 배선(§2). B는 KR 비중그리드 9단계(100~0)·US 축약그리드 4단계.
- **상관 기반 재분류**(§4): 후보 vs KODEX200 일별수익률 상관 ≥0.95면 프레임 D로만 해석 —
  자체 검증 데이터로 통과 확인(합성 근사복제 상관 0.998 → 정확히 재분류됨).
- **비용 비대칭**(§3-3): ETF는 연 총보수(근사치, 로컬 실행 시 실제 공시값으로 교체 필요)를
  일별 산술 drag로 반영·증권거래세 없음. 커스텀 새틀라이트는 `kr_strategy_navs.json` 자체가
  이미 `CostModel("kospi")` net(거래세 0.20% 포함)이라 이중 반영하지 않도록 명시적으로 스킵.
- **TR 우선 원칙**(§3-1): pykrx `get_index_ticker_list`/`get_index_ticker_name`으로 후보 지수
  이름에 "TR"·"총수익"이 포함된 지수를 자동 탐색 → 찾으면 그 TR 지수로 대체 평가, 못 찾으면
  평가 자체를 보류(`blocked_need_tr`)하고 로그에 사유 남김.
- **밸류업 지수 look-ahead 분리**(§3-2): `--valueup-csv`로 수동 CSV를 넣으면 발표일
  (2024-09-24, 재확인 필요) 기준 전/후 서브 통계를 별도 컬럼(`pre_announcement_lookahead_caution`
  / `post_announcement`)으로 분리 저장 — 24+ 단독 성과로 채택 결론 내지 않도록 주석 강제 포함.
- **PBO/DSR**: 시장별로 각각 CSCV(서로 다른 캘린더·유니버스라 행렬 합산 불가 — `pbo_report_etf_control.json`에
  `kr`/`us` 따로, `total_trials_registered`로 누적 시행 수만 별도 기록).
- **`--self-test`**: 네트워크 없이 비용drag 방향·상관분류 임계값·가중치그리드 경계값(코어=1.0/0.0이
  순수 시리즈와 일치)·PBO 배선을 검증. 통과 확인.

## 다음 단계 (지호 님 액션 필요)

1. **로컬 PC에서 실행**: `python backtest_etf_control.py --market kr` /
   `python backtest_etf_control.py --market us` — 기존 스크립트들과 동일하게 pykrx(로컬
   `KRX_ID`/`KRX_PW` 또는 캐시)·yfinance가 정상 동작하는 환경에서. 결과 JSON을 커밋/공유해주시면
   이 초안에 실제 수치를 채워 STRATEGY.md §3.5 반영 여부를 다시 논의합니다.
2. **코리아밸류업 지수 백데이터**: KRX 정보데이터시스템에서 수동 다운로드(자동 수집 불가,
   §5에서 이미 예견된 사항) → `date,value` 컬럼의 CSV로 저장 → `--valueup-csv` 인자로 전달.
3. TR 지수 자동탐색이 로컬에서도 실패하는 후보가 있으면(지수명이 예상과 다르거나 KRX가 해당
   지수를 별도 코드로 관리하지 않는 경우), 그 후보만 분배금 이력 수동 요청으로 좁혀서 알려주시면
   됩니다 — 나머지 후보 파이프라인은 그대로 유효.

## §3.5 삽입 예정 텍스트 골격 (수치는 실행 후 채움)

```
### 3.5 팩터 ETF 대조군 실험 (H1/H2, backtest_etf_control.py)

가설: 새틀라이트를 상장 ETF로 대체해도 위험조정 성과가 통계적으로 구분되지 않으면
(§0 "구분 안 되면 단순한 쪽") ETF가 이긴다.

프레임 B(한국, 비중그리드 100~0) 샤프·MDD 곡선: TBD
프레임 B(미국, 축약그리드) 샤프·MDD: TBD
상관 진단(vs KODEX200/SPY): TBD — 0.95↑ 후보는 프레임 D로 재분류
프레임 D(밸류업 코어스왑, H2) 서브기간(18-21/22-23/24+) 일관성: TBD

PBO/DSR: TBD

결정: [B가 A와 구분 안 됨 → ETF 채택 권고] / [A가 B를 유의하게 이김 → 개별종목 유지 근거] /
      [D가 서브기간 일관 우위 → H2 채택 검토] 중 해당하는 것으로 교체.
```
