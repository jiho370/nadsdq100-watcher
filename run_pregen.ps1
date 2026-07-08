# run_pregen.ps1 — 작업 스케줄러가 실행하는 사전 검증(구독 CLI, 과금 없음).
# 사용: run_pregen.ps1 -Mode kr   (저녁 — 다음날 08:00 한국장 메일용)
#       run_pregen.ps1 -Mode us   (아침 — 당일 17:00 미국장 메일용)
# 흐름: git pull → pregen.py --Mode → output/pregen_{Mode}.json 커밋·푸시.
# 실패해도 조용히 종료 — Actions 가 API 로 자동 폴백하므로 발송엔 지장 없음.
# 로그: output\pregen.log

param([ValidateSet("kr","us")][string]$Mode = "kr")

Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "output\pregen.log"
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "output") | Out-Null

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Mode] $msg" | Out-File -Append -Encoding utf8 $log }

Log "=== pregen 시작 ==="

# 1) 최신 상태로 (보유목록 등 상태파일이 Actions 에서 커밋되므로 pull 필수)
git pull --rebase 2>&1 | Out-File -Append -Encoding utf8 $log

# 2) 사전 검증 (pregen.py 가 AI_BACKEND=cli 강제 + 시간 창 스스로 판단)
python pregen.py --$Mode 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) {
    Log "pregen.py 실패(rc=$LASTEXITCODE) — Actions 가 API 폴백. 종료."
    exit 0
}

# 3) 결과 푸시 (해당 pregen 파일만 — 다른 상태파일은 건드리지 않음)
$file = "output/pregen_$Mode.json"
if (Test-Path $file) {
    git add -f $file 2>&1 | Out-File -Append -Encoding utf8 $log
    git commit -m "chore: pregen $Mode [skip ci]" 2>&1 | Out-File -Append -Encoding utf8 $log
    git push 2>&1 | Out-File -Append -Encoding utf8 $log
}
Log "=== pregen 완료 ==="
