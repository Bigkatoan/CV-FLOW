# CV-FLOW Local Startup Script
# Starts the FastAPI backend. Open http://localhost:8000 in your browser.

$ROOT = $PSScriptRoot

Write-Host ""
Write-Host "  CV-FLOW Local Server" -ForegroundColor Cyan
Write-Host "  ─────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Backend API : http://localhost:8000" -ForegroundColor Green
Write-Host "  Web UI      : http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs    : http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  Engine WS   : ws://localhost:8765 (auto-started by backend)" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

Set-Location "$ROOT\backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
