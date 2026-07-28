# 2026-07-28 개편(지호 님 요청 — "처음에 생성 안되면 15분 간격으로 계속 시도"): 고정
# 2회 재시도 대신, 각 모드의 pregen.py 유효 창(run_kr=16:00~다음날10:00, run_us=06:00~21:00)
# 전체를 15분 간격으로 반복 트리거한다. 실패했을 때만 사실상 재시도되는 이유: pregen.py의
# _already_done()이 그날 몫이 이미 채워져 있으면 반복 트리거가 계속 와도 웹검색 없이
# 즉시 스킵한다(구독 사용량 낭비 방지) — pregen.py 모듈 docstring 참고.
$script = Join-Path $PSScriptRoot "run_pregen.ps1"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

function New-DailyRepeatingTrigger([string]$At, [int]$IntervalMinutes, [double]$DurationHours) {
    # Daily 트리거는 매일 반복되지만 그 자체론 하루 안에서 또 반복하지 않는다. Once 트리거로
    # 만든 Repetition(간격·지속시간) 설정을 Daily 트리거에 이식해 "매일 + 그날 안에서 15분마다"
    # 두 가지를 동시에 만족시킨다(PowerShell ScheduledTasks 모듈의 잘 알려진 조합 방법).
    $daily = New-ScheduledTaskTrigger -Daily -At $At
    $once  = New-ScheduledTaskTrigger -Once -At $At `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
        -RepetitionDuration (New-TimeSpan -Hours $DurationHours)
    $daily.Repetition = $once.Repetition
    return $daily
}

foreach ($t in @(
    # KR 유효 창(pregen.py run_kr): 16:00(저녁, 정상 경로)부터 다음날 10:00(발송 10:00 직전)
    # 까지 18시간 — 15분 간격이면 최대 72회/일 트리거(대부분 _already_done으로 즉시 스킵).
    @{Name="StockPregenKR"; Mode="kr"; At="16:00"; DurationHours=18; Desc="KR stock pregen retry (15min interval, 16:00~10:00)"},
    # US 유효 창(pregen.py run_us): 06:00(전날 세션 마감 확정 후)부터 21:00(그날 저녁 개장
    # 전)까지 15시간.
    @{Name="StockPregenUS"; Mode="us"; At="06:00"; DurationHours=15; Desc="US stock pregen retry (15min interval, 06:00~21:00)"}
)) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Mode $($t.Mode)"

    $trigger = New-DailyRepeatingTrigger -At $t.At -IntervalMinutes 15 -DurationHours $t.DurationHours

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
        -Settings $settings -Description $t.Desc -Force

    Write-Host "Registered: $($t.Name) daily at $($t.At), every 15min for $($t.DurationHours)h, start when available"
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName StockPregen* | Get-ScheduledTaskInfo"
