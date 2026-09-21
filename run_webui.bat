@echo off
REM ==========================================================================
REM  maltrainer web UI launcher (Windows)
REM
REM  Starts the FastAPI backend (port 8008) and the Angular UI (port 3003), each in its
REM  own window, then opens http://localhost:3003. A server that is already listening on
REM  its port is left alone.
REM
REM  Settings come from the environment, or from webui.local.bat next to this file
REM  (git-ignored). Copy webui.local.example.bat to webui.local.bat and edit it:
REM    DATASETS_PATH        folder with one sub-folder per dataset   (default: .\datasets)
REM    COMFYUI_MODELS_PATH  your ComfyUI "models" folder             (needed for Qwen-Image 2.1)
REM  Set WEBUI_NO_BROWSER=1 to skip opening the browser.
REM ==========================================================================
setlocal
cd /d "%~dp0"

if exist "%~dp0webui.local.bat" call "%~dp0webui.local.bat"

if not defined DATASETS_PATH echo [maltrainer] DATASETS_PATH is not set, using .\datasets
if not defined COMFYUI_MODELS_PATH echo [maltrainer] COMFYUI_MODELS_PATH is not set: Qwen-Image 2.1 training will report its model files as missing

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

where npm >nul 2>&1
if errorlevel 1 (
    echo [maltrainer] npm was not found. Install Node.js from https://nodejs.org and run this again.
    exit /b 1
)

set "UI_STARTED=0"

REM ---- backend -------------------------------------------------------------
call :listening 8008
if not errorlevel 1 (
    echo [maltrainer] Backend is already running on port 8008, leaving it alone.
    goto backend_done
)
"%PYTHON%" -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [maltrainer] uvicorn / fastapi are not installed for "%PYTHON%". Run: pip install -r requirements.txt
    exit /b 1
)
echo [maltrainer] Starting backend on http://127.0.0.1:8008
start "maltrainer backend :8008" cmd /k "%PYTHON% -m uvicorn backend.app:app --host 127.0.0.1 --port 8008"
:backend_done

REM ---- frontend ------------------------------------------------------------
call :listening 3003
if not errorlevel 1 (
    echo [maltrainer] UI is already running on port 3003, leaving it alone.
    goto ui_done
)
if not exist "frontend\node_modules" (
    echo [maltrainer] First run: installing frontend dependencies, this takes a minute...
    pushd frontend
    call npm install
    popd
)
echo [maltrainer] Starting UI on http://localhost:3003
start "maltrainer UI :3003" /d frontend cmd /k "npm start"
set "UI_STARTED=1"
:ui_done

REM ---- browser -------------------------------------------------------------
if defined WEBUI_NO_BROWSER goto end
REM a freshly started dev server needs a few seconds for its first build
if "%UI_STARTED%"=="1" ping -n 13 127.0.0.1 >nul
start "" "http://localhost:3003"

:end
endlocal
exit /b 0

REM ---- helper: errorlevel 0 when something is listening on port %1 ---------
:listening
netstat -ano | findstr /R /C:":%~1 .*LISTENING" >nul
exit /b %errorlevel%
