# 2026-08-27(지호 님 요청 — "PC 트리거를 보조로 추가"): GitHub Actions의 schedule
# 트리거가 국장 기준 며칠 연속 몇 시간씩(심하면 10시간+) 지연되는 사고가 반복 확인돼
# (.github/workflows/report.yml "2차 워치독" 코멘트 참고), PC 작업 스케줄러가 정시에
# gh workflow run으로 직접 깨우는 보조 트리거를 추가한다. 기존 GitHub schedule cron
# (본편+워치독 3회)은 그대로 두는 이중 안전망 — last_sent.json 가드 덕에 어느 쪽이
# 먼저 보내도 중복발송 없음. PC가 꺼져있으면 이 보조 트리거만 안 뜨고, 그 경우엔
# GitHub 쪽 스케줄이 원래대로(느리더라도) 시도하므로 "PC 꺼져도 발송"이라는 원래
# 취지는 깨지지 않는다.
$script = Join-Path $PSScriptRoot "trigger_report.ps1"

foreach ($t in @(
    @{Name="ReportTriggerKR";     Mode="kr";     Days=@("Monday","Tuesday","Wednesday","Thursday","Friday"); At="09:40"},
    @{Name="ReportTriggerUS";     Mode="us";     Days=@("Tuesday","Wednesday","Thursday","Friday","Saturday"); At="00:07"},
    @{Name="ReportTriggerWeekly"; Mode="weekly"; Days=@("Sunday"); At="07:30"},
    @{Name="ReportTriggerCoin";   Mode="coin";   Days=@("Saturday","Sunday","Monday"); At="09:07"}
)) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Mode $($t.Mode)"

    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $t.Days -At $t.At

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
        -Settings $settings -Description "Backup trigger to work around GitHub Actions schedule delay ($($t.Mode))" -Force

    Write-Host "Registered: $($t.Name) at $($t.At) on $($t.Days -join ',')"
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName ReportTrigger* | Get-ScheduledTaskInfo"
