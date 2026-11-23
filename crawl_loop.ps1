cargo build --release
$exePath = ".\target\release\brawl-draft-core.exe"
$round = 0

Write-Host "=== 启动不间断抓取守护进程 (按 Ctrl+C 停止) ===" -ForegroundColor Green

while ($true) {
    $round++
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$now] 第 $round 轮抓取启动..." -ForegroundColor Cyan

    & $exePath crawl --limit 2000

    $code = $LASTEXITCODE
    $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$end] 第 $round 轮结束 (退出码: $code)，60 秒后开始下一轮..." -ForegroundColor Yellow
    Start-Sleep -Seconds 60
}
