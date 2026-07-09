# S&P500 자동 리포트 시스템 — 인수인계 (Cowork용)

아래 내용을 새 Cowork 채팅 첫 메시지로 붙여넣으면 됩니다. 폴더는
`C:\Users\JH\Documents\stock` 를 그대로 허용하세요.

---

## 붙여넣을 프롬프트

이 폴더(`C:\Users\JH\Documents\stock`)에 이미 완성된 **S&P500 일일 주식 추천 이메일 시스템**이 있어. 아래 맥락을 숙지하고, 앞으로 유지보수/개선을 이어서 해줘. 새로 만들지 말고 기존 코드를 이어받는 거야.

### 이 시스템이 하는 일
매일 오후 5시(미국장 마감 후) 내 PC에서 Windows 작업 스케줄러가 `run_daily.bat`을 실행 → S&P500 종목을 분석해 HTML 이메일을 아버지(choej7432@gmail.com)와 나에게 발송. **전부 무료 자원**: AI 해설은 Claude 구독 CLI(`claude -p`), 발송은 Gmail 앱 비밀번호(SMTP), 데이터는 Yahoo Finance + SEC EDGAR.

### 리포트 구조 (3분류)
- 🟢 **지금 매수** 5종목 — 상승 지속 중 (과열이면 "과열" 태그)
- 🟡 **관찰(눌림목 대기)** 5종목 — 좋은 종목인데 조정 중, 내려오면 매수
- 🔴 **매도 검토** — 보유 종목 자동 추적, 추격손절 −25% 또는 200일선 −3% 이탈 시
- 분류는 **코드가 확정**(점수+진입타이밍), AI는 각 종목 설명 텍스트만 작성. 뉴스 헤드라인은 야후에서 미리 받아 AI에 주입(웹검색 타임아웃 방지).

### 종목 선정 알고리즘 (백테스트로 확정됨)
데이터 기반(IC = 팩터값과 미래수익의 순위상관)으로 팩터를 선별하고 가중치를 그리드서치. 최신 확정 결과(`output/best_weights.json`):
- **승리 가중치**: gross_margin×2 + accruals×2 + mom6×2 + mom12_1×1 + shareholder_yield×1
- 성과: 12개월 초과수익 +20%p, 승률 89.5%, 최악낙폭 −1.8%, 샤프 1.6, 권장보유 1개월
- **워크포워드 검증 통과**: 학습 6M +9.73%p → 표본외 +9.88%p (과최적화 아님)
- 검증 완료로 **탈락한 팩터**(넣었다가 데이터가 걸러냄): 골든크로스류(IC~0), 저변동성, 단기 반전, 오버나이트 모멘텀(IC −0.017, 대형주 역효과). 잔차 모멘텀은 약함(IC 0.004).
- 살아있는 좋은 팩터: gp_assets(0.088)·gross_margin(0.075)·shareholder_yield·accruals·roa·mom6·cop(0.029).

### 파일 지도
- `daily_ai_report.py` — 엔트리포인트. 데이터수집→종목선정→분류→보유업데이트→뉴스→리포트→차트→발송.
- `export_data.py` — 가중치 기반 종목 랭킹(`select_pool`, `select_by_weights`), 진입타이밍 분류(`split_by_entry`: 상승지속 vs 눌림목).
- `ai_report.py` — AI 해설 생성(구독 CLI 호출) + HTML 렌더. AI 실패 시 `deterministic_report`(지표만으로) 자동 대체 → 발송 절대 안 거름.
- `fundamentals_edgar.py` — SEC EDGAR 재무 팩터(value, roe, roa, margins, fcf, accruals, shareholder_yield, cop 등). `python fundamentals_edgar.py`로 갱신.
- `tech_factors.py` — 이동평균 크로스, 잔차 모멘텀.
- `backtest_weights.py` — 장기 팩터/가중치 백테스트. `python backtest_weights.py --years 10 --keep 8 --oos 0.4`
- `holdings.py` — 보유종목 상태/손절 추적(`output/ai_holdings.json`).
- `run_daily.bat` — 스케줄러가 실행하는 배치. 환경변수(AI_BACKEND=cli, AI_TIMEOUT=1200, REPORT_WEB=1, SMTP 계정, EMAIL_TO) 설정 후 파이썬 실행.
- `backtest_alloc.py` — **주간 자산배분 규칙의 백테스트·그리드서치** (리서치 명세 축소 구현). 후보: 기준배분 4종(growth/sixty40/equal/allweather) × 추세창(SMA 100/150/200/250) × 절대모멘텀(없음/6M/12M/멀티) × 컷 강도 3종 × 도피처 9종(한국주식 컷·달러자산 컷 각각 독립적으로 미국채/달러현금/원화현금 — 매도 후 환전·KRX 상장 미국채 ETF 매수 등 자유로우므로) = 1,728조합(일반 PC 1분 내외). 조합 수가 늘어난 만큼 다중검정 위험도 커지므로 게이트(OOS·원화 기준·벤치마크 상대평가)를 통과한 것만 채택. 생존 게이트는 **원화 기준 포함**(정적 대비 원화 MDD −2%p·원화 CAGR −1.5%p 이내 + USD MDD 3%p 개선 + OOS 붕괴 없음) — 환율 자연헤지(위기 시 원달러 급등이 달러자산 손실 상쇄)를 이겨야만 동적 규칙 채택. 통제: 월말 신호→다음달 적용(look-ahead 방지), 월간 리밸런스(리포트≠매매 주기 분리), 비용 편도 0.25%, USD·원화 양 기준, IS 60%/OOS 40% 워크포워드, 파라미터 평탄성, 위기구간(2008·2020·2022) 방어력. **생존 조합 없으면 정적 배분 폴백**. 실행: `python backtest_alloc.py` (몇 분). 출력: `output/best_alloc.json`(승자 규칙 — weekly_report 가 자동으로 읽음), `output/ALLOC_RESULT.md`(요약). 매크로/상관관계는 매매 신호에서 제외(리포트 설명용만) — 리서치 문서 권고. **분기~반기마다 재실행 권장.**
- `weekly_report.py` — **일요일 주간 자산배분 리포트**(자동 분기). `output/best_alloc.json` 이 있으면 그 규칙(기준배분·추세창·모멘텀·컷)을 사용하고, 없으면 기본 규칙으로 동작. 자산군 6종(미국주식 SPY·한국주식 ^KS11·글로벌 VXUS·채권 IEF·금 GLD·리츠 VNQ) + 참고지역(유럽 VGK·일본 EWJ·중국 MCHI, 200일선 위 & 3개월 +5% 이상이면 '매수 신호' 표시) + 환율. 기준 배분(주식60=미35·한10·글15 / 채권25 / 금10 / 리츠5)에서 틸트 규칙(200일선 아래 & 6개월 음수 → 비중 절반 컷, 하나만 해당 → 25% 컷, 컷분은 채권·현금으로)으로 권장 비중을 코드가 확정. AI는 해설만(실패 시 지표 기반 폴백). 테스트: `python weekly_report.py --no-email` → `output/weekly_report.html`.
- 문서: `AI_SETUP.md`, `HANDOFF.md`(이 파일).

### 운영 규칙 / 하드폰 주의점 (중요)
- Claude CLI 프롬프트는 **argv 아니라 stdin**으로 넘김(Windows 파일명 길이 오류 회피). 출력은 **UTF-8**로 읽기. `--allowedTools "WebSearch,WebFetch"`는 **콤마 한 인자**로.
- 웹검색은 켜두되(REPORT_WEB=1) 타임아웃을 1200초로 크게 잡아 실패 방지.
- 한자 사용 금지, "MA20" 대신 "20일선" 표기, 중립 종목은 아래에 아무것도 안 씀.
- PowerShell에 명령 붙일 때 뒤에 한글 설명(괄호) 붙이면 오류남 — 명령어만.

### 스케줄 동작 (2026-07-09 재개편 — 메일 2통 분리)
- **한국장 메일**: KST **월~금 08:00**(장전, cron UTC `0 23 * * 0-4`) — `--kr`.
  전날 한국장 마감 기준 코스피200 매수/관찰/매도 + 밤사이 미국 마감 포함 세계표·지수신호(코드 생성).
  AI 검증은 전날 저녁 PC pregen(`output/pregen_kr.json`) 재사용, 없으면 API 폴백.
- **미국장 메일**: KST **화~토 16:40 실행 → 17시경 도착**(cron UTC `40 7 * * 2-6`) — `--us`.
  그날 새벽 마감된 미국장 분석 + S&P500 매수/관찰/매도. 당일 아침 PC pregen(`pregen_us.json`) 재사용.
- **주간 배분**: KST **일 07:30**(cron UTC `30 22 * * 6`) — `--weekly` (weekly_report).
- 어느 cron 이 깨웠는지(`github.event.schedule`)로 모드 분기. Actions cron 은 수 분 지연 가능.
- 휴장일에도 발송은 계속 — 상단 배너로 안내. 중복 가드는 `output/last_sent.json` 의
  `sent_kr_kst`/`sent_us_kst` 분리 키(발송 성공 시에만 기록).
- 수동 옵션: `--kr` `--us` `--weekly` `--daily`(둘 다) `--force` `--no-email`.

### 전략 개편 (2026-07-07 — 근거는 STRATEGY.md 필독)
- **신규 `market_signals.py`**: 지수·코인 6자산(나스닥100·S&P500·코스피·코스닥·BTC·ETH) 신호 엔진.
  주식=200일선 ±1% 히스테리시스(3일 확인)+12-1 모멘텀, 코인=120일선 ±3%+3개월 모멘텀.
  상태 5단계(적극매수/눌림목분할매수/보유/축소/위험회피). 스테이트리스(상태파일 불필요).
- **신규 `kr_stocks.py`**: 코스피200 선별(펀더멘탈 EPS>0·ROE≥8%·PER≤40 + 200일선 위 + z(12-1)×0.6+z(52주고점근접)×0.4). 매수3+관찰2. 보유추적 `output/kr_holdings.json`.
- **미국 진입 필터 추가**(`export_data.split_by_entry`): 지금매수는 200일선 위 & 52주고점 -25% 이내.
- **매도 강화**(`holdings.py`): 트레일링 -25%→**-20%**(연구 최적 15~20%), 200일선 -3%는 유지.
- **주간 리포트 개편**(`weekly_report.py`): 배분을 통념 부합으로 교체(안정형 미30/한10/코인2/채권40/금10/현금8 · 공격형 미50/한15/코인5/채권15/금10/현금5 — 기존 금 25~30%는 과최적화로 폐기). '1주 ±5% 차익실현/저점매수' 폐기 → **리밸런싱 밴드(목표의 1.2/0.8배) + 방어 컷(레짐 OFF+12개월 음수→절반)**.

### AI 검증 레이어 (2026-07-08 추가 — STRATEGY.md §4.5)
- 규칙이 후보 '풀'을 뽑고(미국 매수7/관찰7, 한국 4/3) → **AI가 웹검색으로 각 후보를 검증**해
  `매수유지/관찰강등/제외` verdict 부여 → 코드가 최종 확정(미국 5/5, 한국 3/2).
- AI는 종목 추가 불가(할루시네이션 차단), 줄이는 권한만. 매수 3개 미만이면 강등분 복원.
- 종목별 근거는 3축 강제: ①추세 ②펀더멘털 ③뉴스/촉매. 제외 종목+사유는 리포트에 표기.
- 보유목록(`ai_holdings.json`/`kr_holdings.json`)은 AI 검증 통과한 '최종 매수'만 편입.
- 구 백테스트 스크립트(alloc/allweather/goldencross/models/short)·AI_SETUP.md·run_daily.bat·
  daily-per-report.yml 은 삭제됨. `backtest_weights.py`만 유지(분기 재검증용). 사용법은 `USAGE.md`.

### 상세화 + 비용 최소화 개편 (2026-07-09)
회당 ~$0.5 → pregen 있는 날 ~$0.02·없는 날 ~$0.12 목표. 핵심: "일관된 패턴은 하드코딩, 변하는 것만 AI".
- **신규 `entry_plan.py`(하드코딩, $0)**: 분할매수 계획(1~3차 가격·비율 — 과열 30/30/40, 평시 50/50),
  손절선(매수가 -20% vs 200일선 -3% 중 높은 쪽), 관찰→매수 전환 조건(구체 가격), 매도 처분 계획
  (-15% 초과 손실=전량 / 그 외 50% 즉시+50% 20일선 반등 대기)을 지표로 계산. **AI는 이 숫자 못 바꿈**.
- **`ai_report.py` 2단계 분리**: ①검증=sonnet-5+웹검색(≤4회, 묶음 검색, 출력 초압축 JSON)
  ②서술=haiku-4.5(검색 없음, '최종 확정 종목만' 대상 → 토큰 대부분 저가 모델). opus는 코드에서 차단.
  points 4축(추세/펀더멘털/뉴스/촉매)·catalyst 필드·매수계획 표 렌더. 미국 최종 5/5→**4/4**(풀 7→6).
- **프로필 캐시(`gen_profiles.py`)**: 종목 사업 설명은 매일 안 바뀜 → 전 종목 1회 생성
  (`sp500_profiles.json` detail + `kospi200_profiles.json`), 데일리는 재사용. 분기 1회
  `python gen_profiles.py --refresh`. **2026-07-09 개편**: 로컬 claude CLI(Pro 구독)가 있으면
  순차 호출로 $0(분기 1회·~900종목이라 배치 병렬성 불필요) — CLI 없을 때만 API Batch(50%
  할인, ANTHROPIC_API_KEY 필요, ~$0.1)로 폴백. 실행 전 실험: `--limit N`(앞 N종목만) +
  `--dry-run`(파일 저장 없이 콘솔 출력만)으로 소규모 검증 후 전체 실행 권장
  (예: `python gen_profiles.py --limit 3 --dry-run` → 확인 후 `--refresh`).
- **사전 검증(`pregen.py` — Pro 구독, $0)**: 로컬 PC 작업 스케줄러 2개(놓치면 다음 부팅 시 실행,
  StartWhenAvailable) — `StockPregenKR`(`--kr`: 한국장 마감 확정 데이터 검증 → `pregen_kr.json`,
  다음날 08:00 메일용)과 `StockPregenUS`(`--us`: 새벽 마감된 미국장 검증(풀 버퍼 +3) →
  `pregen_us.json`, 당일 17:00 메일용). 유효 시간 창은 pregen.py 가 스스로 판단
  (KR: 16시~다음날 8시 / US: 6~16시, 창 밖이면 스킵). Actions 는 for_kst 가 발송일과 일치할 때만
  사용해 **검증 단계 생략(웹검색 0회)**. 등록: 관리자 PowerShell `.\register_pregen_task.ps1` 1회.

### 완전 사전생성 + 재시도 트리거 개편 (2026-07-09 추가 — 메일 2통 분리로 생긴 여유시간 활용)
메일이 국장/미장으로 분리되면서 pregen 시점(KR 19:00 · US 09:30)에 이미 해당 세션 데이터가
마감 확정된다는 점을 반영해 API 개입을 더 줄임. 목표: pregen 있는 날 발송 시점 API 호출 **0회**
(KR은 시황 4문장만 예외).
- **`pregen.py`가 `write_stage`까지 실행**(`_write_ahead`): verify_stage 성공 뒤 `_apply_verdicts`로
  최종 목록을 재현하고 종목별 서술(summary/points/comment)까지 미리 만들어 `pregen_{kr,us}.json`에
  `written`(종목별)·`sells_written`·(US만) `market_written`으로 저장. 서술 생성이 실패해도 verify
  캐시는 그대로 저장(예외 흡수) — 검증 생략 효과는 유지.
  - US(09:30)는 세션이 이미 마감 확정이라 시황 4문장(`market_written`)까지 전부 사전생성.
  - KR(19:00)은 미국장이 아직 개장 전이라 시황만 예외 — 발송 시점(08:00)에 신설된 경량 함수
    `ai_report.write_market_stage()`(종목 JSON 없이 시황 4문장만, haiku)로 저비용 보충.
- **`ai_report.build_report`가 `written` 캐시를 활용**: 캐시가 있으면 `write_stage` 호출 자체를
  생략, 캐시에 없는 심볼(후보풀 변동 등 드문 경우)만 신설 `_auto_fields()`(지표+프로필 기반 무료
  대체, `deterministic_report`와 공유)로 채운다 — 이 경로에서는 API 호출이 전혀 발생하지 않는다.
  pregen에 `written`이 없고 `by_sym`만 있으면(서술 생성만 실패한 날) 기존처럼 `write_stage` 1회만
  호출(검증은 여전히 생략). pregen 자체가 없으면(PC 꺼짐) 기존 전체 폴백(검증+서술 2회) 그대로.
- **재시도 트리거**(`register_pregen_task.ps1`): 시간 여유가 커진 만큼(KR 19:00→08:00=13시간,
  US 09:30→16:40=7시간10분) 코드 변경 없이 트리거만 하나씩 추가 — `StockPregenKR` 19:00+22:00,
  `StockPregenUS` 09:30+12:30. `run_pregen.ps1`이 성공 시마다 최신본으로 덮어써 커밋하므로 재시도는
  멱등(부작용 없음) — CLI 일시 오류로 인한 API 폴백 발생 빈도를 낮춘다.
- 예상 비용: pregen 완전 성공 날 사실상 $0(KR은 haiku 경량 콜 1회 · 수백 토큰), 서술만 실패한 날
  기존과 동일(haiku 1회), PC 꺼진 날만 기존 폴백(~$0.12).
- 검증: `/tmp` 오프라인 mock 테스트로 완전캐시(API 0회)·부분캐시(무료 대체)·검증만 캐시(1회)·
  캐시 없음(2회)·`_write_ahead` 성공/실패 경로를 모두 확인(수동 삭제된 스크립트 — 재현 필요시 이
  섹션 참고해 재작성).
- 매도 카드에 처분 계획 표기, 한국 매도는 `groups["kr_sells"]`로 합류(report["sells"]에 통합).
- workflow env: `REPORT_MODEL_VERIFY=claude-sonnet-5`, `REPORT_MODEL_WRITE=claude-haiku-4-5`, `REPORT_WEB_USES=4`.
- **버그 수정 — CLI 경로에 `--model` 미전달**: `ai_report._call_cli()`에 `model` 인자 자체가 없어서
  로컬 `claude -p`(pregen.py·gen_profiles.py·weekly_report.py가 사용) 호출이 전부 CLI 기본 모델로
  갔다 — sonnet(검증)/haiku(서술) 분리가 **API 경로에만** 적용되고 있었다(구독 한도 소모가 컸던
  원인). `_call_cli(..., model=...)`로 확장해 verify_stage=`MODEL_VERIFY`, write_stage·
  write_market_stage·gen_profiles=`MODEL_WRITE`를 `--model` 플래그로 명시 전달하도록 수정.
  **주의**: `claude` CLI가 실제로 `--model <모델ID>` 형식을 그대로 받는지는 로컬에서 직접 검증
  필요(`claude -p --model claude-haiku-4-5 --output-format json` 스모크 테스트 권장) — 버전에 따라
  플래그명·값 포맷(별칭 `haiku`/`sonnet` 등)이 다를 수 있음.
- **`test_pregen.ps1`(신규, 검토용)**: 실제 로컬 CLI로 pregen+리포트를 돌려 발송 없이 HTML
  미리보기(`output/kr_report.html`/`us_report.html`)를 만들고 브라우저로 연다. `.\test_pregen.ps1`
  (kr+us) / `-Mode kr` / `-Mode us`. git 커밋·이메일 발송 없음 — 순수 확인용. BOM 추가해 Windows
  PowerShell 5.1 콘솔 한글 깨짐도 수정(비-ASCII .ps1은 BOM 없으면 한글이 깨짐).
- **외부 변경 — KRX가 로그인 필수로 전환(2025-12-27)**: 한국거래소 정보데이터시스템이 회원제
  'KRX Data Marketplace'로 개편되며 pykrx(코스피200 구성종목·PER·ROE)가 `KRX_ID`/`KRX_PW`
  환경변수 없이는 작동하지 않는다(무료 가입 필요, data.krx.co.kr, 네이버/카카오 연동 가능).
  로컬은 세션 변수(`$env:KRX_ID=...`)가 아니라 `[Environment]::SetEnvironmentVariable(...,"User")`로
  영구 등록해야 스케줄 작업(19:00 등)에도 적용됨. Actions는 Secrets `KRX_ID`/`KRX_PW` 추가 필요
  (`report.yml`에 이미 배선함, GITHUB_SETUP.md 참고). 없어도 에러 없이 한국 섹션만 빈 채로 발송됨
  (기존 캐시 폴백 로직 그대로 — 최초 1회 로그인 성공 시 캐시가 생겨 이후엔 미설정 PC에서도 동작).
  `kr_stocks.select()`가 원인을 로그에 명시하도록 수정함.

### 지금 상태
S&P500 팩터 가중치(`best_weights.json`)는 기존 워크포워드 검증 결과 유지(러너에 파일 없으면 폴백 모델로 동작 — 가능하면 백테스트 1회 돌려 재생성 권장). 남은 일: GITHUB_SETUP.md 대로 저장소 푸시 + Secrets 등록(ANTHROPIC_API_KEY·KRX_ID·KRX_PW 포함) + `.\register_pregen_task.ps1` 등록 + `python gen_profiles.py` 1회.

### 내가 앞으로 부탁할 만한 것
- 매년/분기 백테스트 재검증(팩터 IC는 시간이 지나면 감쇠함) + 프로필 캐시 갱신(gen_profiles --refresh).
- 리포트 디자인·문구 개선, 발송 오류 디버깅.
- 새 팩터/아이디어 IC 검증 후 채택 여부 판단.

우선 위 내용 이해했는지 확인하고, 폴더의 실제 파일들을 훑어서 최신 상태 파악한 다음 대기해줘.
