# run_pregen.ps1 — 작업 스케줄러가 실행하는 사전 검증(구독 CLI, 과금 없음).
# 사용: run_pregen.ps1 -Mode kr   (저녁 — 다음날 08:00 한국장 메일용)
#       run_pregen.ps1 -Mode us   (아침 — 당일 17:00 미국장 메일용)
# 흐름: git pull → pregen.py --Mode → output/pregen_{Mode}.json(+한국장은 kospi200_cache.json)
#       +output/pregen.log 커밋·푸시.
# 실패해도 조용히 종료 — Actions 가 API 로 자동 폴백하므로 발송엔 지장 없음.
# 로그: output\pregen.log
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

Log "=== pregen 시작 ==="
if ($branchSwitchMsg) { Log $branchSwitchMsg }
$pullOutput | Out-File -Append -Encoding utf8 $log
if ($pullFailed) {
    Log "[경고] git pull 실패 — 로컬에 커밋 안 된 변경이 있으면 여기서 막힘. 아래 push도 실패할 수 있음."
    Log "        해결: 저장소 폴더에서 'git add -A; git commit -m sync; git push' 한 번 수동 실행."
}

# 2) 사전 검증 (pregen.py 가 AI_BACKEND=cli 강제 + 시간 창 스스로 판단)
python pregen.py --$Mode 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) {
    Log "pregen.py 실패(rc=$LASTEXITCODE) — Actions 가 API 폴백. 종료."
    # 실패해도 이번 실행이 로그에 남긴 내용은 커밋해서 다음 실행의 pull이 안 막히게 한다.
    git add "output/pregen.log" 2>&1 | Out-Null
    git commit -m "chore: pregen $Mode 실패 로그 [skip ci]" 2>&1 | Out-Null
    git push 2>&1 | Out-Null
    exit 0
}

# 3) 결과 푸시 — pregen 파일 + pregen.log(이 실행이 방금 쓴 로그 자체 — 커밋 안 하면 다음
#    실행의 git pull이 또 막힘, 위 2026-07-23 수정 참고) + (한국장만) kospi200_cache.json.
#    캐시를 같이 올리는 이유: Actions 러너가 KRX에 직접 접속 못 해도(로그인 정책 전환 이후
#    빈번) 이 캐시로 코스피200 선정을 계속할 수 있게 하기 위함(kr_stocks._cached_universe).
$files = @("output/pregen_$Mode.json", "output/pregen.log")
if ($Mode -eq "kr" -and (Test-Path "output/kospi200_cache.json")) {
    $files += "output/kospi200_cache.json"
}
$existing = $files | Where-Object { Test-Path $_ }
if ($existing.Count -gt 0) {
    git add -f $existing 2>&1 | Out-File -Append -Encoding utf8 $log
    git commit -m "chore: pregen $Mode [skip ci]" 2>&1 | Out-File -Append -Encoding utf8 $log
    git push 2>&1 | Out-File -Append -Encoding utf8 $log
    if ($LASTEXITCODE -ne 0) {
        Log "[경고] git push 실패 — 보통 로컬이 origin보다 뒤처져 있을 때 발생. 위 pull 경고 참고."
    }
}
Log "=== pregen 완료 ==="
