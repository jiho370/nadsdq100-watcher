# run_pregen.ps1 — 작업 스케줄러가 실행하는 사전 검증(구독 CLI, 과금 없음).
# 사용: run_pregen.ps1 -Mode kr   (저녁 — 다음날 08:00 한국장 메일용)
#       run_pregen.ps1 -Mode us   (아침 — 당일 17:00 미국장 메일용)
# 흐름: git pull → pregen.py --Mode → output/pregen_{Mode}.json(+한국장은 kospi200_cache.json) 커밋·푸시.
# 실패해도 조용히 종료 — Actions 가 API 로 자동 폴백하므로 발송엔 지장 없음.
# 로그: output\pregen.log
#
# 2026-07-10 수정: (1) git pull이 로컬 미커밋 변경 때문에 실패하면 이후 push까지 줄줄이
#   막힐 수 있어 pull 실패를 로그에 굵게 남김(원인 파악용 — 이 저장소를 직접 수정한 뒤
#   커밋을 안 했다면 여기서 막힌다. 한 번은 수동으로 git add/commit/push 필요).
#   (2) output/kospi200_cache.json(한국 KRX 데이터 캐시)을 이제 함께 push한다 — 이전엔
#   pregen_kr.json만 올라가서, GitHub Actions 쪽에서 KRX 접속이 안 될 때(로그인 필요 정책
#   전환 이후 클라우드 IP가 막혔을 가능성) 대체할 캐시가 없어 한국 섹션이 통째로 비었었다.

param([ValidateSet("kr","us")][string]$Mode = "kr")

Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "output\pregen.log"
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "output") | Out-Null

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] $msg" | Out-File -Append -Encoding utf8 $log }

Log "=== pregen 시작 ==="

# 0) main 브랜치 강제(2026-07-23 발견: 이 스크립트가 브랜치 지정 없이 pull/push 하다 보니
#    로컬이 다른 브랜치(예: 작업용 chore/* 브랜치)에 체크아웃된 채로 몇 주째 그 브랜치로만
#    push되고 있었다 — GitHub Actions 스케줄 워크플로(.github/workflows/report.yml)는
#    schedule 트리거상 항상 default 브랜치(main)에서만 도는데 그 변경을 한 번도 못 봐서,
#    전략 코드·가중치 변경이 실제 발송 파이프라인에 반영 안 되는 사고로 이어졌다(STRATEGY.md
#    §6 저장소 상태 참고). 매 실행마다 main인지 확인하고 아니면 강제 전환.
$branch = git rev-parse --abbrev-ref HEAD
if ($branch -ne "main") {
    Log "[경고] 현재 브랜치가 main이 아님($branch) — main으로 전환 시도."
    git checkout main 2>&1 | Out-File -Append -Encoding utf8 $log
    if ($LASTEXITCODE -ne 0) {
        Log "[경고] main 전환 실패(미커밋 변경 등 — 아래 로그 참고) — 이번 실행은 $branch에서 "
        Log "        진행하지만 GitHub Actions엔 반영 안 될 수 있음. 저장소 폴더에서 수동으로"
        Log "        'git status' 확인 후 커밋/정리 필요."
    } else {
        Log "main으로 전환 완료."
    }
}

# 1) 최신 상태로 (보유목록 등 상태파일이 Actions 에서 커밋되므로 pull 필수)
git pull --rebase 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) {
    Log "[경고] git pull 실패 — 로컬에 커밋 안 된 변경이 있으면 여기서 막힘. 아래 push도 실패할 수 있음."
    Log "        해결: 저장소 폴더에서 'git add -A; git commit -m sync; git push' 한 번 수동 실행."
}

# 2) 사전 검증 (pregen.py 가 AI_BACKEND=cli 강제 + 시간 창 스스로 판단)
python pregen.py --$Mode 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) {
    Log "pregen.py 실패(rc=$LASTEXITCODE) — Actions 가 API 폴백. 종료."
    exit 0
}

# 3) 결과 푸시 — pregen 파일 + (한국장만) kospi200_cache.json.
#    캐시를 같이 올리는 이유: Actions 러너가 KRX에 직접 접속 못 해도(로그인 정책 전환 이후
#    빈번) 이 캐시로 코스피200 선정을 계속할 수 있게 하기 위함(kr_stocks._cached_universe).
$files = @("output/pregen_$Mode.json")
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
