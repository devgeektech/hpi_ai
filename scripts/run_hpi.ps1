$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# Stop leftover HPI processes — duplicate listeners on :8003 break SSE (empty body / peer closed)
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -match 'uvicorn.*apps\.hpi' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# No --reload (same Windows watcher issue as Central; also breaks SSE on Proceed)
Write-Host "Starting HPI on http://127.0.0.1:8003 ..."
& $python -m uvicorn apps.hpi.app.main:app --host 127.0.0.1 --port 8003
