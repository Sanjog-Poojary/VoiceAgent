@echo off
title Shoppers Stop AI Voice Agent - Tester
echo =====================================================================
echo           Shoppers Stop Outbound AI Voice Agent Tester UI
echo =====================================================================
echo.
echo [1/3] Activating virtual environment (.venv)...
if not exist ".venv" (
    echo [ERROR] Virtual environment .venv not found. Please run set up first.
    pause
    exit /b
)

echo [2/3] Scheduling browser launch to http://127.0.0.1:8001 ...
start /b cmd /c "timeout /t 2 >nul && start http://127.0.0.1:8001"

echo [3/3] Launching API and ADK Workflow Server...
echo.
echo Press Ctrl+C in this terminal to shut down the server.
echo.
.venv\Scripts\python -m uvicorn mock_server:app --host 127.0.0.1 --port 8001
