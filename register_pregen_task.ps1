$script = Join-Path $PSScriptRoot "run_pregen.ps1"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

foreach ($t in @(
    @{Name="StockPregenKR"; Mode="kr"; At=@("19:00","22:00"); Desc="KR stock pregen retry"},
    @{Name="StockPregenUS"; Mode="us"; At=@("09:30","12:30"); Desc="US stock pregen retry"}
)) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Mode $($t.Mode)"

    $triggers = $t.At | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $triggers `
        -Settings $settings -Description $t.Desc -Force

    Write-Host "Registered: $($t.Name) daily at $($t.At -join ', '), start when available"
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName StockPregen* | Get-ScheduledTaskInfo"
