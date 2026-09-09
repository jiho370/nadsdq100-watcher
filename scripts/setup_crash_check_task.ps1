# setup_crash_check_task.ps1 — upbit_crash_check.py를 15분마다 실행하는 작업 스케줄러 등록
# (2026-09-06, realtime_circuit_breaker_paper.py의 상시 폴링 대신 경량화)
# 관리자 권한 불필요(현재 사용자 계정으로 등록). 재부팅해도 유지됨.
#
# 실행: powershell -ExecutionPolicy Bypass -File scripts\setup_crash_check_task.ps1
# 제거: Unregister-ScheduledTask -TaskName "UpbitCrashCheck" -Confirm:$false

$TaskName = "UpbitCrashCheck"
$PythonExe = "C:\Users\JH\AppData\Local\Programs\Python\Python312\python.exe"
$WorkDir = Split-Path -Parent $PSScriptRoot   # scripts/ 의 부모 = 리포 루트
$ScriptPath = Join-Path $WorkDir "upbit_crash_check.py"

$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$ScriptPath`"" -WorkingDirectory $WorkDir
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
# -AllowStartIfOnBatteries/-DontStopIfGoingOnBatteries: 노트북이 배터리로 돌아갈 때도
# 계속 체크되게(기본값은 배터리면 실행 안 함/중단함 — 노트북엔 안 맞음). WakeToRun은
# 일부러 안 씀(잠자기 상태를 억지로 깨우지 않음 — "켜져있을 때만" 돌아가면 충분).
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "업비트 BTC/ETH -5% 급락 체크(15분 주기, 페이퍼 로깅 전용, 실주문 없음)" -Force

Write-Host "등록 완료: $TaskName (15분마다 실행)"
Write-Host "확인: Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "제거: Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
