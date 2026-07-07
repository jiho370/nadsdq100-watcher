# S&P500 AI 종목추천 보고서 — 무료(구독) + 자동 발송 셋업

매일 아침 아빠에게 **AI 종목추천 보고서**(추천 종목 + 매수전략 + 시황 + 환율·글로벌 + 차트)를 자동 발송한다.
비용 구조: **AI = Claude 구독(무료)**, **발송 = 기존 Gmail(무료)**. Anthropic API 요금 없음.

핵심 제약(중요): **Claude 구독은 당신 PC에서만 쓸 수 있다**(클라우드 GitHub에선 구독 인증 불가). 그리고 메일 발송도 네트워크가 되는 곳이어야 한다. 그래서 **하루 작업 전체를 PC에서** 돌린다.

---

## 무엇이 실행되나 — `daily_ai_report.py` 하나

PC에서 이 한 줄이면 끝(데이터→AI 보고서→차트→메일):
```bat
set AI_BACKEND=cli
set SMTP_USER=<내 gmail 주소>
set SMTP_PASS=<gmail 앱 비밀번호>       & rem 기존에 쓰던 그 값
set EMAIL_TO=<아빠 이메일>              & rem 여러 명은 쉼표로
python daily_ai_report.py
```
- `AI_BACKEND=cli` → 로컬 `claude`(구독)로 보고서 생성. **API 키 불필요, 무료.**
- 데이터 수집(야후), 보고서(구독 claude), 발송(gmail) 모두 PC 네트워크로 처리.
- 보고서 생성이 안 되면(claude 미설치 등) 자동으로 기존 규칙기반 메일로 폴백 → 안 망가짐.

미리보기만: `python daily_ai_report.py --no-email` → `output/ai_report.html` 열어 확인.

---

## PC 1회 셋업

1. **Python + 패키지**: `pip install -r requirements.txt`
2. **Claude Code CLI 로그인 확인**: 터미널에서 `claude --version` 이 나오는지 확인.
   - 안 나오면 Claude Code를 설치하고 구독 계정으로 로그인(무료 사용). `claude` 가 PATH에 없으면 `set CLAUDE_BIN=C:\경로\claude.exe`.
3. **Gmail 앱 비밀번호**: 이미 GitHub에서 쓰던 값을 그대로 환경변수로. (새 키 필요 없음)
4. **아빠 이메일**을 `EMAIL_TO` 에.
5. **자동 실행 등록(Windows 작업 스케줄러)**:
   - 위 명령을 담은 `run.bat` 를 만들고, 작업 스케줄러에서 매일 아침(예: 07:30) 실행 등록.
   - "예약 시작 시간을 놓친 경우 가능한 한 빨리 시작" 옵션 체크 → PC가 그 시간에 꺼져 있었어도 켜지면 실행.
   - 미국장 마감(≈KST 새벽)이라 새벽 정시 발송은 PC가 켜져 있어야 하므로, 아침 시간대 권장.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `daily_ai_report.py` | **실행 진입점**. 데이터→AI보고서→차트→메일. 이것만 돌리면 됨 |
| `ai_report.py` | 보고서 생성(구독 claude 호출·웹검색·HTML). `AI_BACKEND=cli`=무료 |
| `export_data.py` | 500종목 지표를 후보풀/시황으로 컴팩트화(러너가 내부 사용) |
| `sp500_daily_report.py` | 기존 계산 엔진 + 규칙기반 메일(폴백) |

---

## 참고

- **환경변수 요약**: `AI_BACKEND=cli`(무료) / `CLAUDE_BIN`(claude 경로) / `REPORT_PICKS`(추천 개수, 기본 6) / `SMTP_USER`·`SMTP_PASS`·`EMAIL_TO`.
- **유료 자동화가 필요하면**(PC 안 켜도 클라우드에서 완전 무인): `AI_BACKEND=api` + `ANTHROPIC_API_KEY` 로 두면 GitHub Actions에서 `daily_ai_report.py` 를 돌릴 수 있다(하루 몇 센트). 무료 원칙과 배치되므로 기본은 cli.
- 앞서 만든 Cowork 스케줄 태스크 `sp500-ai-commentary` 는 이 PC 방식으로 대체되므로, PC 방식으로 가면 사이드바 "Scheduled"에서 삭제해도 된다.
