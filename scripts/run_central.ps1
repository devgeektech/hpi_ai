$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:FLAGS_use_mkldnn = "0"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

# Stop any leftover Central processes so we don't share port 8000 with zombies
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -match 'uvicorn.*services\.central_ai' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# No --reload: Windows/OneDrive watchers scan venv (path too long) and kill SSE on uploads
Write-Host "Starting Central AI on http://127.0.0.1:8000 ..."
& $python -m uvicorn services.central_ai.app.main:app --host 127.0.0.1 --port 8000
