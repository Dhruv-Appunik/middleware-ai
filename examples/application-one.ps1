$ErrorActionPreference = "Stop"
$base = if ($env:MIDDLEWARE_URL) { $env:MIDDLEWARE_URL.TrimEnd("/") } else { "http://127.0.0.1:8080" }
$body = Get-Content -Raw (Join-Path $PSScriptRoot "claude.json")
$headers = @{ "Content-Type" = "application/json" }
if ($env:MIDDLEWARE_API_KEY) { $headers["X-API-Key"] = $env:MIDDLEWARE_API_KEY }
Invoke-RestMethod -Method Post -Uri "$base/api/v1/chat" -Headers $headers -Body $body
