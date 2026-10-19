@echo off
REM ==========================================================================
REM  maltrainer quality gate (Windows): every gate, as CI runs it.
REM
REM  Uses the dev environment .venv when it exists, then venv, then python on PATH. That
REM  interpreter needs the tools in pyproject.toml's dev group. Extra arguments pass through, e.g.
REM    quality.bat --junit reports
REM ==========================================================================
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -m maltrainer_quality %*
exit /b %errorlevel%
