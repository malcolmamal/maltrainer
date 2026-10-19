@echo off
REM ==========================================================================
REM  Extract a LoRA from a full finetune (default rank 64). Writes
REM  <finetuned>_lora_r<rank>.safetensors next to the finetune; neither input
REM  is modified.
REM
REM    extract_lora.bat base.safetensors finetuned.safetensors [--rank 16 32 64] [--mode fro --value 0.9]
REM
REM  See docs\extract-lora.md.
REM ==========================================================================
setlocal
cd /d "%~dp0"
if exist "%~dp0webui.local.bat" call "%~dp0webui.local.bat"

if "%~2"=="" (
    echo Usage: extract_lora.bat ^<base.safetensors^> ^<finetuned.safetensors^> [--rank N ...] [--mode M --value V]
    exit /b 1
)

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

"%PYTHON%" scripts\extract_lora.py %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
