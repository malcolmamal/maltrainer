@echo off
REM ==========================================================================
REM  Boost Krea 2 LoRAs' text-fusion blocks (default x3). Writes <name>_txt3.safetensors
REM  next to each input; the input is never modified.
REM
REM    boost_text_fusion.bat my_lora.safetensors [more.safetensors ...] [--factor 2]
REM
REM  Or drag .safetensors files onto this file. See docs\krea2-lora-training.md.
REM ==========================================================================
setlocal
cd /d "%~dp0"
if exist "%~dp0webui.local.bat" call "%~dp0webui.local.bat"

if "%~1"=="" (
    echo Usage: boost_text_fusion.bat ^<lora.safetensors^> [more files] [--factor F]
    exit /b 1
)

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

"%PYTHON%" scripts\resize_lora.py %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
