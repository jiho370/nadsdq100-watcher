# trigger_report.ps1 — Windows 작업 스케줄러가 GitHub Actions의 schedule 트리거 지연을
# 우회하기 위해 gh workflow run으로 정시에 직접 워크플로를 깨운다(2026-08-27, 지호 님
# 요청 — 국장 메일이 며칠 연속 몇 시간씩 지연되는 사고가 반복돼(.github/workflows/report.yml
# "2차 워치독" 코멘트 참고), 기존 GitHub schedule cron(본편+워치독 3회)은 그대로 두고
# PC 트리거를 보조로 추가). daily_ai_report.py의 last_sent.json 날짜 가드가 그대로라
# GitHub 쪽이 이미 보냈으면 이 트리거로 새로 뜬 실행은 조용히 스킵된다 — 중복발송 없음.
# PC가 꺼져있으면 이 보조 트리거만 안 뜨고 GitHub 쪽 스케줄은 원래대로 시도하므로
# "PC 꺼져도 발송"이라는 report.yml의 원래 설계 취지는 그대로 유지된다.
param([ValidateSet("kr","us","weekly","coin")][string]$Mode)

$Root = Split-Path -Parent $PSScriptRoot   # scripts/ 의 부모 = 리포 루트
$log = Join-Path $Root "output\report_trigger.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "output") | Out-Null
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] trigger attempt" | Out-File -Append -Encoding utf8 $log
$ghOutput = gh workflow run "Daily & Weekly Market Report" --repo jiho370/nadsdq100-watcher -f mode=$Mode 2>&1
$ghExit = $LASTEXITCODE
$ghOutput | Out-File -Append -Encoding utf8 $log
# 2026-09-10: 예전엔 gh 실패(인증 만료·미설치·네트워크 오류 등)를 로그에만 남기고 스크립트
# 종료코드는 항상 0이라, 작업 스케줄러의 "마지막 실행 결과"엔 늘 성공으로만 보였다
# (register_pregen_task.ps1의 같은 유형 버그 참고). GitHub 쪽 schedule cron이 보조
# 안전망이라 발송 자체가 끊기진 않지만, 이 보조 트리거가 계속 죽어 있어도 알 방법이 없었다.
if ($ghExit -ne 0) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] [경고] gh workflow run 실패(exit=$ghExit) — GitHub 쪽 schedule cron이 대신 발송하지만, 이 보조 트리거는 죽어 있다는 뜻." |
        Out-File -Append -Encoding utf8 $log
    exit $ghExit
}
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] trigger 성공" | Out-File -Append -Encoding utf8 $log
