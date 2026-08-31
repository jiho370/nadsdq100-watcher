# trigger_report.ps1 — Windows 작업 스케줄러가 GitHub Actions의 schedule 트리거 지연을
# 우회하기 위해 gh workflow run으로 정시에 직접 워크플로를 깨운다(2026-08-27, 지호 님
# 요청 — 국장 메일이 며칠 연속 몇 시간씩 지연되는 사고가 반복돼(.github/workflows/report.yml
# "2차 워치독" 코멘트 참고), 기존 GitHub schedule cron(본편+워치독 3회)은 그대로 두고
# PC 트리거를 보조로 추가). daily_ai_report.py의 last_sent.json 날짜 가드가 그대로라
# GitHub 쪽이 이미 보냈으면 이 트리거로 새로 뜬 실행은 조용히 스킵된다 — 중복발송 없음.
# PC가 꺼져있으면 이 보조 트리거만 안 뜨고 GitHub 쪽 스케줄은 원래대로 시도하므로
# "PC 꺼져도 발송"이라는 report.yml의 원래 설계 취지는 그대로 유지된다.
param([ValidateSet("kr","us","weekly","coin")][string]$Mode)

$log = Join-Path $PSScriptRoot "output\report_trigger.log"
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "output") | Out-Null
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] trigger attempt" | Out-File -Append -Encoding utf8 $log
gh workflow run "Daily & Weekly Market Report" --repo jiho370/nadsdq100-watcher -f mode=$Mode 2>&1 |
    Out-File -Append -Encoding utf8 $log
