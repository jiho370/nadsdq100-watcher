# 2026-07-29 개편(지호 님 지적 — "파워쉘이 너무 자주 열린다, 한 번 생성되면 또 안 열려도
# 되는거 아냐"): 15분마다 프로세스를 새로 띄우던 반복 트리거를 걷어내고, 유효 창이 시작되는
# 시각에 딱 한 번만 실행한다 — 그 안에서 15분 간격 재시도는 run_pregen.ps1 자신의 내부
# 루프(Start-Sleep)가 담당한다(pregen.py의 done/outside_window/failed exit code로 판단).
# 그래서 ExecutionTimeLimit을 각 모드의 최대 루프 길이(KR 18시간·US 15시간)보다 넉넉히 잡아야
# Task Scheduler가 재시도 도중 프로세스를 강제 종료하지 않는다.
#
# 2026-09-10: Register-ScheduledTask 는 실패해도 '비종료 오류'라 스크립트가 그냥 다음 줄로
# 넘어간다 — 예전엔 그 뒤의 "Registered:" 를 무조건 출력해서, 관리자 권한이 없어 실제로는
# 등록에 실패했는데도 성공한 것처럼 보였다(실측: Access is denied 가 떴는데 "Registered:"
# 도 같이 출력됨). -ErrorAction Stop 으로 종료 오류로 승격시켜 catch 하고, 하나라도
# 실패하면 종료코드 1 로 끝낸다.
$script = Join-Path $PSScriptRoot "run_pregen.ps1"
$failed = $false

foreach ($t in @(
    # KR 유효 창(pregen.py run_kr): 16:00(저녁, 정상 경로)부터 다음날 10:00(발송 10:00 직전)
    # 까지 18시간.
    @{Name="StockPregenKR"; Mode="kr"; At="16:00"; MaxHours=18; Desc="KR stock pregen (single trigger, internal 15min retry loop)"},
    # US 유효 창(pregen.py run_us): 06:00(전날 세션 마감 확정 후)부터 21:00(그날 저녁 개장
    # 전)까지 15시간.
    @{Name="StockPregenUS"; Mode="us"; At="06:00"; MaxHours=15; Desc="US stock pregen (single trigger, internal 15min retry loop)"}
)) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Mode $($t.Mode)"

    $trigger = New-ScheduledTaskTrigger -Daily -At $t.At

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours ($t.MaxHours + 1))

    try {
        Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
            -Settings $settings -Description $t.Desc -Force -ErrorAction Stop | Out-Null
        Write-Host "Registered: $($t.Name) daily at $($t.At), single launch (internal 15min retry up to $($t.MaxHours)h), start when available"
    } catch {
        Write-Warning "등록 실패: $($t.Name) — $($_.Exception.Message)"
        $failed = $true
    }
}

if ($failed) {
    Write-Warning "일부 작업이 등록되지 않았습니다. 'Access is denied' 라면 관리자 권한 PowerShell 에서 다시 실행하세요."
    Write-Warning "확인: Get-ScheduledTask -TaskName StockPregen* | Select TaskName, State"
    exit 1
}

Write-Host "Check:"
Write-Host "Get-ScheduledTask -TaskName StockPregen* | Get-ScheduledTaskInfo"
