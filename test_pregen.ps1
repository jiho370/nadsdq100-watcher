# test_pregen.ps1 — 수동 검토용 스모크 테스트.
# 실제 로컬 claude CLI(Pro 구독, 과금 없음)로 pregen + 리포트를 돌려 HTML 미리보기를 만든다.
# 발송(이메일)도, git 커밋/푸시도 하지 않음 — 순수 로컬 확인용.
#
# 사용: .\test_pregen.ps1              (기본: kr+us 둘 다)
#       .\test_pregen.ps1 -Mode kr     (한국장 메일만)
#       .\test_pregen.ps1 -Mode us     (미국장 메일만)
#
# 흐름(모드당): 1) pregen.py 시도 — 시간창 밖이면 자동 스킵(그래도 2번은 계속 진행)
#              2) daily_ai_report.py --no-email — pregen 있으면 그 캐시로, 없으면 그 자리에서
#                 CLI로 검증+서술까지 전부 실행(모델은 verify=sonnet, write=haiku로 분리됨)
#              3) 결과 HTML을 기본 브라우저로 열어줌

param([ValidateSet("kr", "us", "both")][string]$Mode = "both")

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -Path $PSScriptRoot
$env:AI_BACKEND = "cli"   # 로컬 Pro 구독 CLI 강제(혹시 셸에 AI_BACKEND=api 가 남아있어도 덮어씀)
$env:PYTHONIOENCODING = "utf-8"   # 파이썬 쪽 stderr/stdout 한글이 콘솔에서 안 깨지게

function Run-One([string]$m) {
    Write-Host ""
    Write-Host "=== [$m] 1) pregen 시도 (시간창 밖이면 스킵 - 정상, 2번에서 그 자리 CLI로 대체) ==="
    python pregen.py --$m

    Write-Host "=== [$m] 2) 리포트 생성 (발송 없이 HTML 미리보기만, --force 로 중복가드 무시) ==="
    python daily_ai_report.py --$m --no-email --force

    $out = if ($m -eq "kr") { "output\kr_report.html" } else { "output\us_report.html" }
    if (Test-Path $out) {
        Write-Host "완료 -> $out"
        Invoke-Item $out
    } else {
        Write-Host "[경고] $out 생성 실패 - 위 로그를 확인하세요."
    }
}

if ($Mode -eq "both") {
    Run-One "kr"
    Run-One "us"
} else {
    Run-One $Mode
}
