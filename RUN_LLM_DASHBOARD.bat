@echo off
cd /d "%~dp0"

echo ==========================================
echo       LLM EVALUATION DASHBOARD
echo ==========================================
echo.
echo Starting FastAPI backend...

start "LLM Eval Backend" cmd /k ".\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001"

echo Waiting for backend...

:WAIT_BACKEND
powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/ -TimeoutSec 1; if($r.StatusCode -eq 200){exit 0}else{exit 1} } catch { exit 1 }"

if errorlevel 1 (
    timeout /t 1 >nul
    goto WAIT_BACKEND
)

echo Backend connected!
echo.
echo Starting Streamlit dashboard...

start "LLM Eval Dashboard" cmd /k ".\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501"

timeout /t 5 >nul

start http://localhost:8501

echo.
echo Dashboard started successfully.