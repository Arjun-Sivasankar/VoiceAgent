foreach ($port in 8765, 7860, 7870) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $($conn.OwningProcess) on port $port"
    }
}

$wrappers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "powershell|speech-to-speech" -and $_.CommandLine -match "run-ultron-agent"
}
foreach ($proc in $wrappers) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "Ultron web stack stopped."
