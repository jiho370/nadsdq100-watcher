# 자동 발송 설정 가이드 (GitHub Actions) — 10분 1회 설정

이걸 마치면 **PC를 꺼둬도** 화~토 07:30(전일 시장 점검+종목추천), 일 07:30(주간 자산배분)에 메일이 온다.
월요일은 발송 없음. 기존 Windows 작업 스케줄러/run_daily.bat 는 더 이상 필요 없다(수동 테스트용으로만 남김).

## 0. 기존 저장소 재활용 (jiho370/nadsdq100-watcher) — 권장 경로

이미 SMTP_USER·SMTP_PASS·EMAIL_TO 시크릿이 등록된 기존 저장소를 그대로 쓴다. 절차:
1. (권장) Settings → General → Danger Zone → **Change visibility → Private** (현재 Public — 보유종목 상태파일이 커밋되므로 비공개 권장)
2. (선택) 같은 화면 Repository name 에서 이름 변경(예: `market-report`) — 코드가 이름에 의존하지 않으므로 안 바꿔도 동작
3. `.github/workflows/` 안의 **기존 워크플로 yml 삭제** (이중 발송 방지)
4. 아래 "업로드 파일 목록"의 파일을 업로드 (Add file → Upload files)
5. `.github/workflows/report.yml` 생성 (Add file → Create new file → 파일명에 경로 포함 입력)
6. Secrets 에 `ANTHROPIC_API_KEY` 추가 (AI 해설용 — 없어도 발송은 됨). FMP_API_KEY 는 새 코드에서 안 쓰지만 둬도 무해
7. Actions 탭 → Run workflow → mode `daily` 로 테스트

### 업로드 파일 목록 (전부 저장소 루트에, C:\Users\JH\Documents\stock 에서)
필수 12개: daily_ai_report.py · ai_report.py · export_data.py · market_signals.py · kr_stocks.py ·
holdings.py · weekly_report.py · tech_factors.py · fundamentals_edgar.py · ai_commentary.py(덮어쓰기) ·
sp500_daily_report.py(덮어쓰기) · requirements.txt(덮어쓰기)
선택: STRATEGY.md · HANDOFF.md · GITHUB_SETUP.md · .gitignore
업로드 금지: **run_daily.bat**(비밀번호 평문!) · __pycache__ · output 폴더 · daily-per-report.yml(옛 참고파일)
그대로 두기: sp500_profiles.json · state_prev_list.json (저장소에 이미 있음)

---

## 1. GitHub 저장소 만들기 (새로 만들 경우만)
1. github.com 로그인 → New repository → 이름 예: `stock-report` → **Private** 선택 → Create.
2. PC에서 이 폴더(`C:\Users\JH\Documents\stock`)를 푸시:
```bat
cd C:\Users\JH\Documents\stock
git init
git add .
git commit -m "market report system"
git branch -M main
git remote add origin https://github.com/<내계정>/stock-report.git
git push -u origin main
```
(이미 저장소가 있으면 `git add . && git commit -m "update" && git push` 만.)

⚠️ **중요**: `run_daily.bat` 안에 Gmail 앱 비밀번호가 평문으로 있다. Private 저장소라도 올리지 않는 게 좋다.
푸시 전에 `run_daily.bat`의 `set SMTP_PASS=...` 줄을 지우거나, `.gitignore`에 `run_daily.bat` 추가 권장.

## 2. Secrets 등록 (메일·AI 인증)
저장소 페이지 → **Settings → Secrets and variables → Actions → New repository secret** 로 4개 등록:

| 이름 | 값 |
|---|---|
| `SMTP_USER` | Gmail 주소 (기존 run_daily.bat 의 값) |
| `SMTP_PASS` | Gmail 앱 비밀번호 (기존 값 그대로) |
| `EMAIL_TO` | 수신자. 여러 명이면 쉼표: `choej7432@gmail.com, rametal.choi@gmail.com` |
| `ANTHROPIC_API_KEY` | AI 해설용 API 키 — console.anthropic.com 에서 발급. **없어도 발송은 됨**(지표 기반 해설로 폴백) |

AI 해설 비용: Sonnet + 웹검색 6회 기준 하루 수십 원, 월 2~5천 원 수준.
아끼려면 `.github/workflows/report.yml` 에서 `REPORT_WEB: "0"` 으로.

## 3. 워크플로 활성화 확인
1. 저장소 → **Actions** 탭 → "Daily & Weekly Market Report" 가 보이면 OK.
2. 첫 테스트: **Run workflow** 버튼 → mode 에 `daily` 입력 → Run.
   몇 분 뒤 메일 도착 + Artifacts 에서 `report-preview`(html) 확인 가능.
3. 주간 리포트 테스트: mode 에 `weekly`.
4. 발송 없이 미리보기만: mode 에 `preview`.

## 4. 이후 운영
- 자동: 매일 UTC 22:30(=KST 07:30) 실행. 요일 분기는 코드가 KST 기준으로 자동.
  (GitHub cron 은 5~20분 지연될 수 있음 → 실제 도착 07:30~08:00.)
- 발송 실패 시: Actions 탭에서 실패 로그 확인. 발송 성공 시에만 `output/last_sent.json` 이 기록돼 이중 발송이 방지된다.
- 보유목록(매도 추적)은 `output/ai_holdings.json`(미국)·`output/kr_holdings.json`(한국)에 자동 커밋된다.
- 전략 파라미터 변경: `STRATEGY.md` 규칙 수정 후 해당 코드(주로 `market_signals.py` PARAMS)와 워크플로 env.
- 분기~반기마다 `backtest_weights.py` 재검증 권장(팩터 감쇠).

## 5. 예전 방식과의 관계
- 기존 `daily-per-report.yml`(구 워크플로 참고 파일)은 사용하지 않는다. 저장소의 `.github/workflows/` 에
  옛 워크플로가 이미 있다면 삭제할 것(이중 발송 방지).
- PC에서 수동 테스트: `run_daily.bat` 또는 `python daily_ai_report.py --daily --no-email` → `output/ai_report.html`.
