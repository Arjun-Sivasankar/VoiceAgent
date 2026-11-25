$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$null = Set-Location $RepoRoot
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$WebDir = Join-Path $RepoRoot "web"
$BridgeScript = Join-Path $RepoRoot "bridge\bridge.py"
$BackendScript = Join-Path $RepoRoot "run-voice-agent.ps1"
$LogDir = Join-Path $RuntimeRoot "logs"
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) {
    throw "Python was not found on PATH"
}

$backendPort = 8765
$webPort = 7860
$bridgePort = 7870

function Test-Port([int]$port) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $port)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

if (-not (Test-Port $backendPort)) {
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $BackendScript, "serve" `
        -RedirectStandardOutput "$LogDir\serve.out.log" `
        -RedirectStandardError "$LogDir\serve.err.log" `
        -WindowStyle Hidden
    Write-Host "Backend starting (first run loads models, ~20-60s)..."
} else {
    Write-Host "Backend already running on port $backendPort."
}

$deadline = (Get-Date).AddSeconds(180)
while (-not (Test-Port $backendPort) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
}

if (Test-Port $backendPort) {
    Write-Host "Backend ready on ws://127.0.0.1:$backendPort/v1/realtime"
} else {
    Write-Host "Backend did not come up. Check $LogDir\serve.err.log"
}

if (-not (Test-Port $webPort)) {
    Start-Process -FilePath $Python `
        -ArgumentList "-m", "http.server", "$webPort", "--bind", "127.0.0.1", "--directory", "$WebDir" `
        -WindowStyle Hidden
    Write-Host "Web UI on http://127.0.0.1:$webPort"
} else {
    Write-Host "Web UI already running on port $webPort."
}

if (-not (Test-Port $bridgePort)) {
    Start-Process -FilePath $Python `
        -ArgumentList $BridgeScript `
        -RedirectStandardOutput "$LogDir\bridge.out.log" `
        -RedirectStandardError "$LogDir\bridge.err.log" `
        -WindowStyle Hidden
    Write-Host "System bridge on http://127.0.0.1:$bridgePort"
} else {
    Write-Host "System bridge already running on port $bridgePort."
}

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:$webPort"
Write-Host ""
Write-Host "Open http://127.0.0.1:$webPort and click 'Enable microphone'."
Write-Host "Stop everything with stop-voice-web.ps1"
