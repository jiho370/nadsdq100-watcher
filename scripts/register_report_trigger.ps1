# 2026-08-27(지호 님 요청 — "PC 트리거를 보조로 추가"): GitHub Actions의 schedule
# 트리거가 국장 기준 며칠 연속 몇 시간씩(심하면 10시간+) 지연되는 사고가 반복 확인돼
# (.github/workflows/report.yml "2차 워치독" 코멘트 참고), PC 작업 스케줄러가 정시에
# gh workflow run으로 직접 깨우는 보조 트리거를 추가한다. 기존 GitHub schedule cron
# (본편+워치독 3회)은 그대로 두는 이중 안전망 — last_sent.json 가드 덕에 어느 쪽이
# 먼저 보내도 중복발송 없음. PC가 꺼져있으면 이 보조 트리거만 안 뜨고, 그 경우엔
# GitHub 쪽 스케줄이 원래대로(느리더라도) 시도하므로 "PC 꺼져도 발송"이라는 원래
# 취지는 깨지지 않는다.
#
# 2026-09-10: Register-ScheduledTask 는 실패해도 '비종료 오류'라 그냥 다음 줄로 넘어간다 —
# 예전엔 뒤의 "Registered:" 를 무조건 출력해 실패를 성공으로 오인하게 만들었다
# (register_pregen_task.ps1 의 같은 수정 참고). 종료 오류로 승격시켜 catch 한다.
$script = Join-Path $PSScriptRoot "trigger_report.ps1"
$failed = $false

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

    try {
        Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
            -Settings $settings -Description "Backup trigger to work around GitHub Actions schedule delay ($($t.Mode))" `
            -Force -ErrorAction Stop | Out-Null
        Write-Host "Registered: $($t.Name) at $($t.At) on $($t.Days -join ',')"
    } catch {
        Write-Warning "등록 실패: $($t.Name) — $($_.Exception.Message)"
        $failed = $true
    }
}

if ($failed) {
    Write-Warning "일부 작업이 등록되지 않았습니다. 'Access is denied' 라면 관리자 권한 PowerShell 에서 다시 실행하세요."
    Write-Warning "확인: Get-ScheduledTask -TaskName ReportTrigger* | Select TaskName, State"
    exit 1
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName ReportTrigger* | Get-ScheduledTaskInfo"
