@echo off
title OmniVoice Standard WebUI
cd /d "%~dp0"

REM =====================================================================
REM CAU HINH HUGGINGFACE CACHE & BO NHO TAM (D:\temp_OmniVoice)
REM =====================================================================
set "HF_FOUND="
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM 1. Uu tien tim kiem model tai D:\temp_OmniVoice\hf_cache de tiet kiem o C:
if exist "D:\temp_OmniVoice\hf_cache\hub\models--k2-fsa--OmniVoice" (
    set "HF_HOME=D:\temp_OmniVoice\hf_cache"
    set "HF_FOUND=1"
)

REM 2. Neu D: chua co, kiem tra thu muc cache mac dinh cua he thong tai %USERPROFILE%
if not defined HF_FOUND (
    if exist "%USERPROFILE%\.cache\huggingface\hub\models--k2-fsa--OmniVoice" (
        set "HF_HOME=%USERPROFILE%\.cache\huggingface"
        set "HF_FOUND=1"
    )
)

if not defined HF_FOUND (
    if exist "D:\" (
        set "HF_HOME=D:\temp_OmniVoice\hf_cache"
    )
)

if exist "D:\" (
    set "TEMP=D:\temp_OmniVoice"
    set "TMP=D:\temp_OmniVoice"
    set "GRADIO_TEMP_DIR=D:\temp_OmniVoice\gradio_temp"
    set "TORCH_HOME=D:\temp_OmniVoice\torch_cache"
    set "PYTHONPYCACHEPREFIX=D:\temp_OmniVoice\pycache"
    if not exist "D:\temp_OmniVoice" mkdir "D:\temp_OmniVoice"
    if not exist "D:\temp_OmniVoice\gradio_temp" mkdir "D:\temp_OmniVoice\gradio_temp"
    if not exist "D:\temp_OmniVoice\torch_cache" mkdir "D:\temp_OmniVoice\torch_cache"
    if not exist "D:\temp_OmniVoice\pycache" mkdir "D:\temp_OmniVoice\pycache"
    if not exist "D:\temp_OmniVoice\hf_cache" mkdir "D:\temp_OmniVoice\hf_cache"
)

set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"

echo ======================================================================
echo   OMNIVOICE - ZERO-SHOT MULTILINGUAL TTS & VOICE CLONING (k2-fsa)
echo   HF Cache: %HF_HOME%
echo   Temp Dir: D:\temp_OmniVoice
echo ======================================================================
echo.
echo Dang khoi dong WebUI Gradio goc...
echo Trinh duyet se mo tai dia chi: http://localhost:7860
echo.

omnivoice-demo --device cpu --no-asr

pause
