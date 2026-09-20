@echo off
setlocal
cd /d %~dp0

if not exist venv\Scripts\python.exe (
  echo [ERROR] venv\Scripts\python.exe not found.
  pause
  exit /b 1
)

echo Starting Streamlit UI on http://localhost:8501 ...
venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501

endlocal
