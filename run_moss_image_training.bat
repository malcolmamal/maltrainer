@echo off
REM ==========================================================================
REM  MOSS-Image LoRA training from a template, without the web UI.
REM
REM    run_moss_image_training.bat <dataset> [automagic3^|prodigy^|adamw8bit] [extra args...]
REM    run_moss_image_training.bat feliciaday
REM    run_moss_image_training.bat feliciaday prodigy
REM    run_moss_image_training.bat feliciaday automagic3 --max-steps 2000 --dry-run
REM
REM  Settings come from the environment, or from webui.local.bat next to this file:
REM    DATASETS_PATH        folder with one sub-folder per dataset   (default: .\datasets)
REM    COMFYUI_MODELS_PATH  your ComfyUI "models" folder
REM ==========================================================================
setlocal
cd /d "%~dp0"
if exist "%~dp0webui.local.bat" call "%~dp0webui.local.bat"

if "%~1"=="" (
    echo Usage: run_moss_image_training.bat ^<dataset^> [automagic3^|prodigy^|adamw8bit] [extra args]
    exit /b 1
)

set "PYTHON=python"
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"

set "OPTIMIZER=%~2"
if "%OPTIMIZER%"=="" set "OPTIMIZER=automagic3"

"%PYTHON%" scripts\train_from_template.py moss_image "%~1" --optimizer %OPTIMIZER% %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
