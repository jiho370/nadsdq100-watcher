# 사용법 가이드 (USAGE.md)

매일 아침 자동으로 시장을 점검하고 종목을 추천하는 메일 시스템의 사용 설명서입니다.
전략의 근거·수치는 `STRATEGY.md`, 자동화 설정은 `GITHUB_SETUP.md`, 유지보수 맥락은 `HANDOFF.md`를 함께 보세요.

---

## 1. 이 시스템이 하는 일

| 요일 (KST) | 발송 내용 |
|---|---|
| 화~토 07:30 | **일일 리포트** — 전일 미국·한국·코인·세계 시황, 지수/코인 6종 추세 신호, S&P500 매수 5·관찰 5, 코스피200 매수 3·관찰 2, 매도 검토 |
| 일 07:30 | **주간 자산배분 리포트** — 표준 배분(안정형/공격형), 방어 컷, 리밸런싱 밴드 안내, 자산군별 흐름 |
| 월 | 발송 없음 (주말 — 새 거래일 없음) |

### 작동 원리 (3단계 — 규칙과 AI의 역할 분담)
1. **규칙(코드)이 후보 발굴** — 백테스트로 검증된 팩터(매출총이익률·발생액·모멘텀·주주환원)와
   추세 필터(200일선 위 + 52주 고점 -25% 이내)로 후보 풀을 뽑음 (미국 매수 7·관찰 7, 한국 4·3)
2. **AI(Claude)가 검증** — 각 후보를 웹검색으로 점검(실적 쇼크·소송·가이던스 하향·급락 촉매)해
   `매수유지 / 관찰강등 / 제외` 판정 + 종목별 3축 근거(①추세 ②펀더멘털 ③뉴스) 작성
3. **코드가 최종 확정** — AI 판정 반영해 최종 목록 확정(미국 5/5, 한국 3/2).
   AI는 종목을 **추가할 수 없고**(할루시네이션 차단) 줄이는 권한만 가짐. 과도 제외 시 최소 3개 복원.

철학: **큰 손실을 피하고, 상승장에는 시장만큼 참여하고, 하락장에는 덜 잃는다.**

---

## 2. 처음 설정 (1회, 약 10분)

`GITHUB_SETUP.md`의 절차를 따르세요. 요약:
1. 기존 저장소(`jiho370/nadsdq100-watcher`)를 **Private으로 전환** 후, 이 폴더의 파일을 업로드
2. 기존 `.github/workflows/` 내 옛 yml 삭제 → `report.yml` 생성
3. Secrets 확인/추가: `SMTP_USER` `SMTP_PASS` `EMAIL_TO` (기존) + **`ANTHROPIC_API_KEY`** (AI 검증용)
4. Actions 탭 → Run workflow → mode `daily`로 테스트 발송

> ANTHROPIC_API_KEY가 없어도 발송은 됩니다(지표 기반 폴백). 단 AI 검증·해설이 빠집니다.
> 비용: Sonnet + 웹검색 8회 기준 하루 수십 원 수준.

---

## 3. 리포트 읽는 법

### 일일 리포트
- **🌐 전일 시장 요약**: 핵심 6자산 + 다우·닛케이·DAX·FTSE·항셍·상해 + 환율의 전일/1주/1개월 등락표
- **🧭 지수·코인 추세 신호** (5단계, 규칙 기반):

| 신호 | 의미 | 행동 |
|---|---|---|
| 🟢 적극 매수 | 상승 레짐 + 모멘텀 + | 정기 적립 계속, 신규 매수 가능 |
| 🔵 눌림목 분할 매수 | 상승 레짐 속 단기 조정 | **기대값 높은 진입 구간** — 2~3회 분할 |
| 🟡 보유 | 레짐 ON이나 모멘텀 약함 | 신규 보류 |
| 🟠 축소 검토 | 레짐 OFF 전환 | 신규 중단, 반등 시 일부 축소 |
| 🔴 위험 회피 | 하락 레짐 + 모멘텀 − | 비중 절반 이상 축소 권고 |

- **⭐ 지금 매수 / 👀 관찰**: 종목 카드의 배지 의미 —
  `⚠️과열·분할`=RSI 72↑ 또는 50일선 +15%↑(한 번에 사지 말 것), `🤖 AI 강등`=규칙은 뽑았지만 AI가 뉴스 검증에서 관찰로 내림,
  `호재/악재/중립`=최근 뉴스 성격
- **🤖 AI 검증에서 제외된 후보**: 규칙 통과했으나 명백한 악재로 탈락한 종목과 사유 — "왜 빠졌는지"도 판단 재료
- **🔴 매도 검토**: 과거 추천 종목이 트레일링 -20% 또는 200일선 -3% 이탈 시 자동 표시

### 주간 리포트 (일요일)
- **표준 배분**: 안정형(미30/한10/코인2/채권40/금10/현금8) · 공격형(미50/한15/코인5/채권15/금10/현금5)
- **차익실현/저점매수** = 리밸런싱 밴드: 내 실제 비중이 목표의 **1.2배 초과 → 초과분 매도**, **0.8배 미만 → 매수**
- **방어 컷**: 자산이 레짐 OFF + 12개월 음수면 목표 비중의 절반만 유지(컷분은 현금성)

---

## 4. 수동 실행 (PC에서 테스트할 때)

```bat
cd C:\Users\JH\Documents\stock
pip install -r requirements.txt

python daily_ai_report.py --daily --no-email     :: 일일 미리보기 → output/ai_report.html
python weekly_report.py --no-email               :: 주간 미리보기 → output/weekly_report.html
python daily_ai_report.py --daily --force        :: 실제 발송(중복 체크 무시)
```
PC에서 AI까지 쓰려면(구독 CLI, 무료): `set AI_BACKEND=cli` 후 실행 (`claude --version` 되는 상태).
발송하려면 `SMTP_USER` `SMTP_PASS` `EMAIL_TO` 환경변수 필요.

---

## 5. 조정 가능한 파라미터 (환경변수 — report.yml 의 env 에서 변경)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `REPORT_POOL` | 7 | 미국 후보 풀 크기(매수/관찰 각각) |
| `REPORT_FINAL_BUY` / `REPORT_FINAL_WATCH` | 5 / 5 | 미국 최종 채택 수 |
| `KR_POOL_BUY` / `KR_POOL_WATCH` | 4 / 3 | 한국 후보 풀 크기 |
| `KR_FINAL_BUY` / `KR_FINAL_WATCH` | 3 / 2 | 한국 최종 채택 수 |
| `KR_ROE_MIN` / `KR_PER_MAX` | 0.08 / 40 | 한국 펀더멘털 필터 |
| `SELL_TRAIL` | 0.20 | 트레일링 스톱(고점 대비 -20%) |
| `SELL_MA_BUFFER` | 0.03 | 200일선 이탈 버퍼(-3%) |
| `REPORT_WEB` / `REPORT_WEB_USES` | 1 / 8 | AI 웹검색 on/off·횟수(비용 절약은 0) |
| `REPORT_MODEL` | claude-sonnet-4-6 | AI 모델 |
| `AI_BACKEND` | cli(PC) / api(클라우드) | AI 경로 |

지수 신호 파라미터(200일선·히스테리시스 등)는 `market_signals.py`의 `PARAMS`, 주간 배분은 `weekly_report.py`의 `*_WEIGHTS`.
규칙을 바꿀 땐 **STRATEGY.md를 먼저 수정**하고 코드에 반영하세요.

---

## 6. 유지보수

- **분기~반기마다**: `python backtest_weights.py --years 10 --keep 8 --oos 0.4` 로 팩터 가중치 재검증
  → `output/best_weights.json` 갱신(없으면 12-1 모멘텀 폴백으로 동작).
- **발송이 안 올 때**: GitHub 저장소 → Actions 탭에서 실패 로그 확인. 발송 성공 시에만
  `output/last_sent.json`이 기록되므로, 실패한 날은 다음 실행에서 자동 재시도됩니다.
- **같은 날 두 번 실행돼도** 중복 발송되지 않습니다(KST 날짜 기준 가드). 강제 재발송은 mode `daily`.
- **미국/한국 휴장일**: 발송은 계속되며(코인·세계는 매일 새로움) 상단 배너로 "휴장 — 직전 거래일 기준" 안내.

## 7. 파일 지도

| 파일 | 역할 |
|---|---|
| `daily_ai_report.py` | 엔트리포인트(요일 분기·파이프라인·발송) |
| `market_signals.py` | 지수·코인 6종 추세 신호 엔진 + 세계시장 요약 |
| `kr_stocks.py` | 코스피200 선별(pykrx+yfinance) + 보유 추적 |
| `export_data.py` | S&P500 후보 선별(팩터 가중치·진입 필터) |
| `ai_report.py` | AI 검증(verdict)·해설 생성 + HTML 렌더 |
| `holdings.py` | 매도 규칙(-20% 트레일링/200일선 -3%) |
| `weekly_report.py` | 일요일 주간 자산배분 리포트 |
| `sp500_daily_report.py` | 데이터 수집·지표 계산·메일 발송 엔진 |
| `fundamentals_edgar.py` / `tech_factors.py` / `ai_commentary.py` | 팩터·뉴스 보조 모듈 |
| `backtest_weights.py` | 팩터 가중치 백테스트(분기 재검증용) |
| `.github/workflows/report.yml` | 자동 실행 스케줄(GitHub Actions) |
| `STRATEGY.md` / `GITHUB_SETUP.md` / `HANDOFF.md` | 전략 근거 / 설정법 / 인수인계 |

⚠️ 이 시스템은 정보 제공용이며 투자 권유가 아닙니다. 최종 판단과 책임은 본인에게 있습니다.
