$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
uvicorn apps.hmb.app.main:app --host 0.0.0.0 --port 8001 --reload
