# stock — 미국·한국 주식 / 자산배분 자동 리포트

팩터 기반으로 종목을 선별하고, AI 검증을 거쳐 **매일 이메일로 리포트를 보내는** 개인용 자동화
시스템. 실행은 전부 GitHub Actions가 담당하고, PC는 보조 트리거로만 쓴다.

> 참고: 개인 연구·기록용 프로젝트다. 투자 자문이 아니며 실주문 기능은 없다.

---

## 알고리즘 한눈에 보기 (외부 독자용)

숫자의 단일 출처는 코드다 — 아래는 요약이고 최신 값·근거는 [docs/strategy/LIVE.md](docs/strategy/LIVE.md)에.

**미국 개별주(S&P500)**
1. 팩터 점수 = `int_gp_assets×1 + rd_mktcap×2 + shareholder_yield×2`(z-score 가중합 — 자산대비수익성·연구개발집약도·주주환원 3팩터, `output/best_weights.json`)
2. 필터: 점수하한 3.25 미달 제외, 100일 이동평균 아래 제외 (둘 다 백테스트로 채택 — `research/us/`)
3. 최종 매수 상위 10종목, 섹터캡 없음
4. 매수 집행: 평시 2분할(현재가 50%+20일선 부근 50%), 과열(RSI≥72 등)이면 3분할(30/30/40) — `entry_plan.py`
5. 매도: 가격 개입 없이 6개월 정기 재평가만(그때 팩터 후보풀 밖이면 정리). 손절선·트레일링스톱·목표가 익절은 전부 백테스트 후 폐기 — 공통 이유는 "승자를 일찍 잘라내서 기대값을 깎는다"
6. AI(sonnet)는 이미 뽑힌 후보의 최신 뉴스·리스크만 검토(카드에 유의사항으로 표시) — 종목 추가·순위 변경 불가, 최종 결정은 항상 코드

**한국 개별주(코스피200)**: EPS>0·ROE>0 필터 → `z(1/PER)+z(1/PBR)+z(배당수익률)` 1:1:1 → 상위 5종목. 매도는 3개월 정기 재평가(미국과 동일 철학).

**검증 방법론**: 모든 채택 결정에 PBO(과최적화 확률, CSCV)·DSR(다중검정 보정 샤프비율)을 게이트로 씀 — `overfit_stats.py`, 기준은 [docs/VALIDATION_PIPELINE.md](docs/VALIDATION_PIPELINE.md). 시행착오·기각된 대안은 전부 [docs/strategy/HISTORY.md](docs/strategy/HISTORY.md)에 날짜별로 기록.

**알려진 한계(정직하게 공개)**
- 팩터 가중치(1:2:2)의 과거 백테스트 성과 자체는 좋다(9년 백테스트 CAGR 29.43%·샤프 1.02, HISTORY.md §7) — 이건 실측 사실. 다만 이게 "여러 대안 중 최적"이라고 증명하는 건 애초에 목표가 아니었고, 실제 한계는 이 좋은 성과가 665개 가중치 조합을 탐색하는 과정에서 생긴 과최적화가 아니라고 통계적으로 완전히 배제하진 못한다는 점이다(PBO 22%·DSR 0.88, 다중검정 보정 기준 0.95 미달).
- floor+100일선 필터까지 포함한 최종 선정+분할매수+청산규칙 전체를 하나로 합쳐 검증하면 표본이 부족하다. topn을 10→8→6으로 낮춰(필터 통과 조건 완화) 이벤트 수 18→23→24건, DSR 0.7657→0.8418→0.8704로 기준(0.95)에 조금씩 가까워지지만 **셋 다 아직 미통과**(PBO도 topn별로 60%/66.3%/20.6%로 들쭉날쭉). 방향(분할매수+6개월재평가가 최고시행)은 topn 무관하게 일관되나, 통계적으로 확정하려면 표본이 더 쌓여야 한다(`research/us/us_full_stack_exec_validation.py`, 2026-09-24).

---

## 빠른 시작

```bash
pip install -r requirements.txt
```

메일 없이 리포트만 만들어 보기(로컬 점검용):

```bash
python daily_ai_report.py --us --no-email --force
```

리서치 스크립트는 **리포 루트에서 `-m`으로** 돌린다:

```bash
python -m research.us.us_factor_formula_sweep
```

---

## 폴더 구조

```
.
├─ daily_ai_report.py     ← 유일한 실행 진입점 (GitHub Actions가 이것만 호출)
├─ *.py                   ← 운영 모듈 25개 (아래 "운영 코드" 참고)
├─ research/              ← 일회성 검증·백테스트 스크립트 79개 (주제별)
│   ├─ us/ kr/ crypto/ fx/ bonds/ gold/ regime/ common/
├─ scripts/               ← Windows 작업 스케줄러용 .ps1
├─ state/                 ← CI가 commit-back 하는 상태파일 (추적됨)
├─ output/                ← 실행 산출물·백테스트 결과 (대부분 gitignore)
├─ docs/                  ← 설계·전략 문서
│   ├─ strategy/          ← HISTORY.md 등 전략 근거
│   ├─ research/          ← 리서치 노트·스펙
│   └─ archive/           ← 완료된 핸드오프 기록
└─ data/                  ← 수동 다운로드 원본 (gitignore)
```

### 운영 코드 (루트)

`daily_ai_report.py` 하나에서 뻗어나가는 import 그래프가 곧 운영 경로다.
루트의 `.py`는 전부 여기에 속하거나, 스케줄러가 직접 부르는 진입점이다.

| 묶음 | 파일 |
|---|---|
| 파이프라인 | `daily_ai_report` `weekly_report` `pregen` `export_data` |
| 데이터 수집 | `sp500_daily_report` `kr_stocks` `fundamentals_edgar` `market_signals` |
| 신호·점수 | `tech_factors` `score_calibration` `kr_factor_ic` `entry_plan` `expectancy_report` |
| 보유 추적 | `holdings` |
| 백테스트 코어 | `backtest_costs` `backtest_weights` `backtest_exec` `backtest_kr` `overfit_stats` |
| 리포트 생성 | `ai_report` `ai_commentary` `ai_verdict_log` |
| 상시 운영 | `upbit_crash_check` `realtime_circuit_breaker_paper` |
| 유지보수 | `gen_profiles` (분기 1회 — 종목 프로필 캐시 재생성, claude CLI 필요) |

### 리서치 코드 (`research/`)

한 번 돌려 결론을 내고 결과를 `output/`에 남기는 검증 스크립트다. 운영 경로에서
import 되지 않는다. 서로를 import 하므로 **반드시 `python -m research.<주제>.<이름>`
형태로 리포 루트에서 실행**한다 (`python research/us/foo.py`는 동작하지 않는다).

---

## 파일 추적 정책

`output/`은 "재생성 가능한 것"과 "재생성 불가능한 것"이 섞이기 쉬워 규칙을 정해 뒀다.

| 위치 | 내용 | git |
|---|---|---|
| `state/` | 보유종목·발송기록·KRX 캐시 — CI가 매 실행 후 commit-back | **추적** |
| `output/*.json` | 백테스트 결과 (검증 근거로 남김) | **추적** |
| `output/fundamentals_cache.json` | EDGAR 펀더멘탈 — 런타임이 읽는 데이터 | **추적** |
| `output/*.html *.png *.log` | 리포트 산출물·로그 | 무시 |
| `output/kr_bt_cache.json` 등 | 네트워크 캐시 (재생성 가능, 용량 큼) | 무시 |
| `data/` | 수동 다운로드 원본 | 무시 |

⚠️ **`output/fundamentals_cache.json`은 이름과 달리 캐시가 아니다.**
`export_data.py`가 이 파일을 읽고, 없으면 조용히 `None`을 반환해 **종목 선정에서
펀더멘탈 팩터가 에러 없이 빠진다.** CI에는 EDGAR 재수집 단계가 없으므로 지우지 말 것.

---

## 자동 실행

### GitHub Actions (주 경로)

`.github/workflows/report.yml` — cron으로 국장/미국장/주간/코인 리포트를 발송하고,
누락 대비 워치독 cron이 2회 더 재시도한다. `state/last_sent.json`의 날짜 가드가
중복 발송을 막으므로 재시도가 겹쳐도 안전하다.

필요한 GitHub Secrets:

| 이름 | 용도 |
|---|---|
| `ANTHROPIC_API_KEY` | AI 검증·코멘터리 |
| `SMTP_USER` / `SMTP_PASS` | 메일 발송 계정 |
| `EMAIL_TO` | 수신 주소 |
| `KRX_ID` / `KRX_PW` | KRX 정보데이터시스템 (2025-12-27부터 로그인 필수) |

> `FMP_API_KEY`는 현재 코드 어디에서도 쓰이지 않는다 — 정리해도 된다.

### 유료 API 경로

기본적으로 **꺼져 있다.** 리포트 AI 해설은 `report.yml`의 `AI_ENABLED: "0"`으로 비활성이고
(pregen 캐시 또는 지표 기반 deterministic 리포트로 발송), `gen_profiles.py`는 로컬 claude
CLI(구독, $0)만 쓴다 — CLI가 없으면 예전처럼 유료 Batch API로 폴백하지 않고 그 자리에서
중단한다. 정말 과금 경로로 돌려야 하면 `PROFILE_API_FALLBACK=1`과 `ANTHROPIC_API_KEY`를
함께 지정한다.

### Windows 작업 스케줄러 (보조)

`scripts/`의 `.ps1`이 담당한다. PC가 꺼져 있어도 GitHub 쪽 스케줄은 그대로 도므로
보조 트리거일 뿐이다.

| 스크립트 | 역할 |
|---|---|
| `run_pregen.ps1` | 사전 검증 실행 후 결과 커밋 (구독 CLI 사용, 과금 없음) |
| `trigger_report.ps1` | `gh workflow run`으로 워크플로를 정시에 직접 깨움 |
| `register_pregen_task.ps1` / `register_report_trigger.ps1` | 위 두 개를 스케줄러에 등록 |
| `setup_crash_check_task.ps1` | 업비트 급락 체크 15분 주기 등록 |
| `test_pregen.ps1` | 수동 스모크 테스트 |

**스크립트를 옮기거나 리포 경로를 바꾸면 등록된 작업이 조용히 깨진다** (절대경로로
등록되기 때문). 그럴 땐 해당 `register_*.ps1`을 다시 실행해 재등록할 것.

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/strategy/LIVE.md](docs/strategy/LIVE.md) | **지금 뭐가 라이브고 왜** — 자산군별 요약, 가장 먼저 볼 문서 |
| [docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md) | 시스템 전체 구조 |
| [docs/USAGE.md](docs/USAGE.md) | 사용법 |
| [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md) | Actions·시크릿 설정 |
| [docs/VALIDATION_PIPELINE.md](docs/VALIDATION_PIPELINE.md) | 검증 파이프라인(PBO/DSR) |
| [docs/strategy/HISTORY.md](docs/strategy/HISTORY.md) | 시행착오 전체 기록 — 날짜별 세션 로그, "왜 이렇게 됐는지" |
| [docs/playbook/03-method/EXECUTION.md](docs/playbook/03-method/EXECUTION.md) | 체결·비용 모델 실측 감사 |
| [docs/MAINTENANCE.md](docs/MAINTENANCE.md) | 이 리포를 편집·리팩터링할 때 참고할 체크리스트 |
