@echo off
setlocal
cd /d %~dp0

if not exist venv\Scripts\python.exe (
  echo [ERROR] venv\Scripts\python.exe not found.
  pause
  exit /b 1
)

echo Starting FastAPI backend on http://localhost:8000 ...
start "Terra Sight API" cmd /k "set PYTHONIOENCODING=utf-8 && venv\Scripts\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000"

echo Opening research chat UI...
start "" research_chat_ui.html

echo Done. Keep the API terminal open.
endlocal
