$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example. Add ANTHROPIC_API_KEY and OPENAI_API_KEY there for live answers."
}

$port = 8080
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port $port is already used by process $($busy.OwningProcess)."
    Write-Host "Stop that program, or start on another port:"
    Write-Host ".\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8090"
    exit 1
}

Write-Host "LLM Middleware: http://127.0.0.1:$port"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $port
