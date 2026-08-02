# 코인 전용 메일 (주말 공백 채우기) — 설계

날짜: 2026-08-03
상태: 승인됨 (지호 님, 2026-08-03)

## 배경 / 문제

코인(BTC)은 24/7 거래되지만 현재 발송 스케줄(`.github/workflows/report.yml`)은:
- 국장 메일: 월~금 09:40 KST (코스피·코스닥·금·미국채10년)
- 미장 메일: 화~토 00:07대 KST (나스닥100·S&P500·**비트코인**)
- 주간 배분: 일 07:30 KST

미장 메일이 토요일 00:07(금요일 데이터)에 마지막으로 BTC를 보여준 뒤, 다음 BTC 신호는
화요일 00:07(월요일 데이터)까지 없다 — **토요일 낮~일~월요일, 약 3일간 코인 추세 신호
공백**. 이 기간에도 코인은 계속 거래되고 레짐이 바뀔 수 있어 투자 판단에 공백이 생긴다.

## 목표

토·일·월 아침에 BTC+ETH 레짐 신호를 별도 메일로 채워 이 공백을 없앤다.

## 범위

- **포함**: BTC+ETH 레짐 신호 카드(기존 미장 메일의 BTC 카드와 동일 포맷 — 시그널·이동
  평균·모멘텀 등).
- **제외**: 종목 추천, AI 검증/뉴스 검색, 보유현황, 자산배분. 화~토 미장 메일의 BTC
  섹션은 그대로 유지(중복 발송이지만 최소 변경 원칙 — 기존 흐름을 안 건드림).
- **제외(이번 설계 아님, 필요시 후속)**: 재시도(워치독) cron. 다른 메일들도 처음엔 없다가
  실제 드롭 사고 이후 추가된 이력이 있음(§ `.github/workflows/report.yml` 상단 주석) —
  이번엔 없이 시작.

## 설계

### 1. `market_signals.py`

- `CORE_ASSETS`에 이더리움 추가: `("ETH", "이더리움", "ETH-USD", "crypto", ...)`.
  `PARAMS["crypto"]`(120일선·±3%밴드·확인3일·3개월모멘텀·50일선눌림, `analyze()`가 이미
  `kind="crypto"`로 이 파라미터를 적용)를 튜닝 없이 그대로 적용 — STRATEGY.md §1에 ETH에
  BTC 파라미터를 튜닝 없이 적용해도 방향성이 일치했다는 사전 확인 이력이 있음.
- `when` 필드를 단일 문자열이 아니라 **자산 하나가 여러 메일에 동시 소속**될 수 있게
  일반화. BTC의 `when`을 `("us", "coin")`으로, ETH의 `when`을 `("coin",)`으로 설정.
- `core_for()` / `lean_for_ai()` / `signal_cards_html()`의 필터 로직을 `==` 동등비교에서
  멤버십(`in`) 체크로 변경. 하위호환을 위해 `when` 인자가 문자열로 들어오면 1개짜리
  튜플로 정규화해서 비교(`("kr",)` 등) — 기존 `core_for(sig, "kr")`/`"us"` 호출부는 코드
  변경 없이 그대로 동작.

### 2. `daily_ai_report.py`

- 신규 함수 `run_coin(no_email: bool = False, force: bool = False)`:
  - `_load_last_sent()`로 `sent_coin_kst` 확인 → 오늘 이미 발송했으면 스킵(`run_kr`/
    `run_us`와 동일한 중복발송 가드 패턴).
  - `_gather_signals()`로 신호 수집(기존 `MS.gather()` 그대로 재사용 — 이미 BTC+ETH 둘 다
    포함해서 계산됨, 필터만 나중에 함).
  - `when="coin"`으로 필터된 카드만으로 신호 섹션 HTML 조립(`MS.signal_cards_html(signals,
    sig_cids, when="coin")`, `_signal_images(signals, when="coin")`).
  - AI 호출 없음(뉴스 검색·후보선정·보유현황 전부 생략 — 순수 코드 계산, 비용 0). 최소한의
    HTML 틀만 씌워 발송(제목 예: "🪙 주말 코인 시그널 · BTC·ETH").
  - `_preview_and_send(...)`로 발송 + `{"sent_coin_kst": today_kst}` 기록.
- `_load_last_sent`/`_save_last_sent`는 이미 범용 dict라 코드 변경 불필요 — 새 키만 자연스럽게 추가됨.
- CLI 분기(`if __name__ == "__main__":` 블록)에 `--coin` 모드 추가, 기존 `--kr`/`--us`/
  `--weekly`와 동일한 자리에.

### 3. `.github/workflows/report.yml`

- 신규 cron 3개(토·일·월 KST 09:07 — 프로젝트 관행대로 정각 혼잡 회피 위해 07분 오프셋):
  ```
  - cron: "7 0 * * 6"   # KST 토 09:07 — 코인 전용
  - cron: "7 0 * * 0"   # KST 일 09:07 — 코인 전용
  - cron: "7 0 * * 1"   # KST 월 09:07 — 코인 전용
  ```
- `MODE` 분기 case문에 위 세 cron 문자열 → `MODE="coin"` 매핑 추가.
- 실행 스텝의 `case "$MODE" in ...` 에 `coin) python daily_ai_report.py --coin ;;` 추가.
- 기존 국장(월 09:40)·주간(일 07:30) cron과 시간이 겹치지 않음(각각 09:40, 07:30 vs
  09:07).

## 데이터 흐름

```
MS.gather(yf)  →  core: [KOSPI, KOSDAQ, GOLD, BOND, NDX, SPX, BTC, ETH] (신호 계산 전부 동일 로직)
                        │
                        ▼
        when="coin" 필터 (BTC, ETH만 통과 — 멤버십 체크)
                        │
                        ▼
        signal_cards_html(when="coin") + _signal_images(when="coin")
                        │
                        ▼
              최소 HTML 틀 + 제목 조립
                        │
                        ▼
        _preview_and_send() → 이메일 발송 + last_sent.json 갱신
```

## 에러 처리

- `_gather_signals()` 실패 시(야후 파이낸스 오류 등) 기존 패턴 그대로: 예외를 잡아 빈 dict
  반환, 신호 섹션이 조용히 생략되고 나머지 흐름은 계속 진행(크래시로 발송 자체가 막히지
  않음).
- 발송 실패 시 `last_sent.json`을 갱신하지 않음(기존 `_preview_and_send` 패턴) — 다음 실행
  때 재시도 가능(단, 이번 설계엔 자동 재시도 cron 없음 — 위 범위 참고).

## 테스트 방법

- 로컬: `python daily_ai_report.py --coin --no-email` → `output/coin_report.html`
  프리뷰로 카드 렌더링 확인(기존 `kr_report.html`/`us_report.html`과 동일 명명 규칙).
- 실제 발송 전: `workflow_dispatch`(mode=coin 입력)로 수동 1회 트리거해 실제 메일함에서
  확인.

## 미해결/후속 과제 (이번 설계 범위 밖)

- 재시도(워치독) cron — 필요성이 실측되면 추가.
- ETH 파라미터 자체 검증(현재는 BTC 파라미터를 튜닝 없이 전용) — STRATEGY.md §6-G 열린
  실에 이미 기록된 저우선 과제.
