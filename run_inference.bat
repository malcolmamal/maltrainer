@echo off
setlocal
cd /d %~dp0

set VENV_PYTHON=venv\Scripts\python.exe
set SCRIPT=scripts\infer_zimage.py

set LORA_PATH=output\zimage_lora\zimage_lora_step_final.safetensors
set EMBEDDING_PATH=output\zimage_embedding_anya\zimage_embedding_step_final.safetensors

:menu
cls
echo ==================================================
echo Z-Image Inference Menu
echo ==================================================
echo.
echo 1. LoRA Only
echo 2. Embedding Only
echo 3. LoRA + Embedding
echo 4. Plain Model
echo 5. Exit
echo.
set /p choice=Select an option (1-5): 

if "%choice%"=="1" goto lora
if "%choice%"=="2" goto embedding
if "%choice%"=="3" goto both
if "%choice%"=="4" goto plain
if "%choice%"=="5" goto exit

echo Invalid choice, try again.
pause
goto menu

:lora
set /p PROMPT="Enter Prompt: "
echo Running Inference with LoRA...
"%VENV_PYTHON%" "%SCRIPT%" --prompt "%PROMPT%" --lora_path "%LORA_PATH%" --output_path "output\inference_lora.png"
pause
goto menu

:embedding
set /p PROMPT="Enter Prompt (include trigger word if needed): "
echo Running Inference with Embedding...
"%VENV_PYTHON%" "%SCRIPT%" --prompt "%PROMPT%" --embedding_path "%EMBEDDING_PATH%" --output_path "output\inference_embedding.png"
pause
goto menu

:both
set /p PROMPT="Enter Prompt (include trigger word if needed): "
echo Running Inference with LoRA + Embedding...
"%VENV_PYTHON%" "%SCRIPT%" --prompt "%PROMPT%" --lora_path "%LORA_PATH%" --embedding_path "%EMBEDDING_PATH%" --output_path "output\inference_combined.png"
pause
goto menu

:plain
set /p PROMPT="Enter Prompt: "
echo Running Inference with Plain Model...
"%VENV_PYTHON%" "%SCRIPT%" --prompt "%PROMPT%" --output_path "output\inference_plain.png"
pause
goto menu

:exit
endlocal
exit /b
