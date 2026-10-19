@echo off
REM ==========================================================================
REM  Shrink LoRAs to a lower rank (default 8). Writes <name>_resized.safetensors
REM  next to each input; the input is never modified.
REM
REM    resize_lora.bat my_lora.safetensors [more.safetensors ...] [--rank 16] [--dtype float32]
REM
REM  Or drag .safetensors files onto this file. See docs\resize-loras.md.
REM ==========================================================================
setlocal
cd /d "%~dp0"
if exist "%~dp0webui.local.bat" call "%~dp0webui.local.bat"

if "%~1"=="" (
    echo Usage: resize_lora.bat ^<lora.safetensors^> [more files] [--rank N] [--frac F] [--dtype D]
    exit /b 1
)

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

"%PYTHON%" scripts\resize_lora.py %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
