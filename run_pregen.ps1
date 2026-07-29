# run_pregen.ps1 — 작업 스케줄러가 실행하는 사전 검증(구독 CLI, 과금 없음).
# 사용: run_pregen.ps1 -Mode kr   (저녁 — 다음날 10:00 한국장 메일용)
#       run_pregen.ps1 -Mode us   (아침 — 그날 저녁 개장 30분~90분 후 미국장 메일용)
# 흐름: git pull → [pregen.py --Mode 성공할 때까지 15분 간격 반복] → output/pregen_{Mode}.json
#       (+한국장은 kospi200_cache.json) +output/pregen.log 커밋·푸시.
# 실패해도 조용히 종료 — Actions 가 API 로 자동 폴백하므로 발송엔 지장 없음.
# 로그: output\pregen.log
#
# 2026-07-29 개편(지호 님 지적 — "pregen 한번 생성되면 파워쉘이 또 안 열려도 되는거 아냐"):
#   예전엔 작업 스케줄러가 15분마다 이 스크립트 자체를 새로 실행해(하루 최대 70여회) 매번
#   PowerShell 프로세스가 새로 뜨는 게 눈에 띄게 잦았다. 이제 작업 스케줄러는 유효 창이
#   시작되는 시각에 딱 한 번만 이 스크립트를 실행하고, "성공(또는 이미 완료)할 때까지
#   15분 간격으로 재시도"는 이 스크립트 자신의 내부 루프(Start-Sleep)로 옮겼다 — 창 하나가
#   유효 창이 끝날 때까지 계속 떠 있긴 하지만(-WindowStyle Hidden), 새 프로세스가 반복
#   생성되는 일 자체가 없어져 눈에 덜 띈다. pregen.py의 exit code로 종료 여부 판단:
#   0=성공/이미완료(루프 종료) · 2=시간창 밖(루프 종료, 재시도 의미 없음) · 1=실패(재시도).
#
# 2026-07-10 수정: (1) git pull이 로컬 미커밋 변경 때문에 실패하면 이후 push까지 줄줄이
#   막힐 수 있어 pull 실패를 로그에 굵게 남김(원인 파악용 — 이 저장소를 직접 수정한 뒤
#   커밋을 안 했다면 여기서 막힌다. 한 번은 수동으로 git add/commit/push 필요).
#   (2) output/kospi200_cache.json(한국 KRX 데이터 캐시)을 이제 함께 push한다 — 이전엔
#   pregen_kr.json만 올라가서, GitHub Actions 쪽에서 KRX 접속이 안 될 때(로그인 필요 정책
#   전환 이후 클라우드 IP가 막혔을 가능성) 대체할 캐시가 없어 한국 섹션이 통째로 비었었다.
#
# 2026-07-23 수정: git pull이 "매 실행마다 100% 재현"으로 실패하던 버그 수정 — 원인은
#   (a) 이 스크립트가 로그 파일(output\pregen.log, git 추적 대상)에 "시작" 줄을 pull보다
#   먼저 써서 자기가 방금 쓴 줄 때문에 "커밋 안 된 변경"으로 rebase가 막혔고, (b) 그 로그
#   파일을 3)단계 커밋 목록에 넣지 않아 매 실행이 끝나도 로그가 영구히 dirty 상태로 남아
#   다음 실행도 똑같이 막혔다(누적 재발). 수정: pull을 이 스크립트의 첫 git 동작으로
#   옮기고(로그 파일 쓰기 전), 3)단계 커밋 목록에 pregen.log를 추가.
#   같은 세션에서 main 브랜치 강제(브랜치 미지정 pull/push로 로컬이 chore 브랜치에 남아
#   있으면 GitHub Actions(항상 main에서만 실행)에 반영 안 되던 사고 재발 방지)도 추가.

param([ValidateSet("kr","us")][string]$Mode = "kr")

Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "output\pregen.log"
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "output") | Out-Null

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] $msg" | Out-File -Append -Encoding utf8 $log }

function Push-Result {
    # pregen 파일 + pregen.log(이 실행이 방금 쓴 로그 자체 — 커밋 안 하면 다음 실행의 git
    # pull이 또 막힘, 2026-07-23 수정 참고) + (한국장만) kospi200_cache.json. 커밋 직전
    # 재-pull(2026-07-27 발견 — pregen.py 실행 중 GitHub Actions의 상태 커밋이 origin에
    # 새로 얹혀 push가 non-fast-forward로 거부되는 경우 완화).
    param([string]$CommitMsg)
    git pull --rebase 2>&1 | Out-File -Append -Encoding utf8 $log
    if ($LASTEXITCODE -ne 0) {
        Log "[경고] 커밋 직전 재-pull 실패 — 작업트리가 여전히 dirty하거나 충돌. 아래 push도 실패 가능."
    }
    $files = @("output/pregen_$Mode.json", "output/pregen.log")
    if ($Mode -eq "kr" -and (Test-Path "output/kospi200_cache.json")) {
        $files += "output/kospi200_cache.json"
    }
    $existing = $files | Where-Object { Test-Path $_ }
    if ($existing.Count -gt 0) {
        git add -f $existing 2>&1 | Out-File -Append -Encoding utf8 $log
        git commit -m $CommitMsg 2>&1 | Out-File -Append -Encoding utf8 $log
        git push 2>&1 | Out-File -Append -Encoding utf8 $log
        if ($LASTEXITCODE -ne 0) {
            Log "[경고] git push 실패 — 보통 로컬이 origin보다 뒤처져 있을 때 발생. 위 pull 경고 참고."
        }
    }
}

# 0)+1) 브랜치 확인/전환 + git pull — 로그 파일(추적 대상)에 아무것도 쓰기 전에 먼저 수행
#    해야 한다. 순서를 바꾸면 이 실행 자체가 로그를 써서 자신을 dirty하게 만들고 pull이
#    막힌다(2026-07-23 발견). 결과는 변수에 담아뒀다가 "=== pregen 시작 ===" 이후에 로그.
$branch = git rev-parse --abbrev-ref HEAD
$branchSwitchMsg = $null
if ($branch -ne "main") {
    git checkout main 2>$null
    if ($LASTEXITCODE -ne 0) {
        $branchSwitchMsg = "[경고] 브랜치가 main 아님($branch), 전환도 실패(미커밋 변경 등) — 이번 " +
                           "실행은 $branch에서 진행하지만 GitHub Actions엔 반영 안 될 수 있음."
    } else {
        $branchSwitchMsg = "브랜치가 main 아니었음($branch) → main으로 전환 완료."
    }
}

git pull --rebase 2>&1 | Tee-Object -Variable pullOutput | Out-Null
$pullFailed = ($LASTEXITCODE -ne 0)

# 유효 창 길이 — register_pregen_task.ps1의 트리거 시각(KR 16:00 / US 06:00)에서부터
# 몇 시간 동안 15분 간격으로 재시도할지(pregen.py 자체의 시간창 가드와 맞춰둠).
$maxHours = if ($Mode -eq "kr") { 18 } else { 15 }
$deadline = (Get-Date).AddHours($maxHours)

Log "=== pregen 시작(재시도 루프, 최대 ${maxHours}시간) ==="
if ($branchSwitchMsg) { Log $branchSwitchMsg }
$pullOutput | Out-File -Append -Encoding utf8 $log
if ($pullFailed) {
    Log "[경고] git pull 실패 — 로컬에 커밋 안 된 변경이 있으면 여기서 막힘. 아래 재시도에서도 계속 실패할 수 있음."
    Log "        해결: 저장소 폴더에서 'git add -A; git commit -m sync; git push' 한 번 수동 실행."
}

$attempt = 0
while ($true) {
    $attempt++
    python pregen.py --$Mode 2>&1 | Out-File -Append -Encoding utf8 $log
    $rc = $LASTEXITCODE

    if ($rc -eq 0) {
        Log "성공(또는 이미 완료) — 재시도 루프 종료(시도 ${attempt}회째)"
        Push-Result -CommitMsg "chore: pregen $Mode [skip ci]"
        break
    }
    if ($rc -eq 2) {
        Log "이번 실행 시각은 유효 시간창 밖 — 재시도 의미 없어 종료(시도 ${attempt}회째)"
        break
    }
    # rc == 1(검증 실패) 또는 그 외(예외) — 재시도 가치 있음
    if ((Get-Date) -ge $deadline) {
        Log "최대 ${maxHours}시간 초과 — 재시도 중단(시도 ${attempt}회 전부 실패). Actions가 API 폴백."
        Push-Result -CommitMsg "chore: pregen $Mode 실패 로그 [skip ci]"
        break
    }
    Log "실패(rc=$rc, 시도 ${attempt}회째) — 15분 후 재시도. 지금까지 로그만 우선 커밋."
    Push-Result -CommitMsg "chore: pregen $Mode 재시도 로그 [skip ci]"
    Start-Sleep -Seconds 900
}

Log "=== pregen 완료 ==="
