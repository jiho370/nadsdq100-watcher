$script = Join-Path $PSScriptRoot "run_pregen.ps1"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

foreach ($t in @(
    @{Name="StockPregenKR"; Mode="kr"; At="19:00"; Desc="KR stock pregen for next day 08:00 email"},
    @{Name="StockPregenUS"; Mode="us"; At="09:30"; Desc="US stock pregen for same day 17:00 email"}
)) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Mode $($t.Mode)"

    $trigger = New-ScheduledTaskTrigger -Daily -At $t.At

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
        -Settings $settings -Description $t.Desc -Force

    Write-Host "Registered: $($t.Name) daily at $($t.At), start when available"
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName StockPregen* | Get-ScheduledTaskInfo"
