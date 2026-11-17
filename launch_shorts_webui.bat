@echo off
title OmniVoice TikTok Studio
cd /d "%~dp0"

REM =====================================================================
REM 1. CAU HINH HUGGINGFACE CACHE & BO NHO TAM (D:\temp_OmniVoice)
REM =====================================================================
set "HF_FOUND="
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM 1.1 Uu tien tim kiem model tai D:\temp_OmniVoice\hf_cache de tiet kiem o C:
if exist "D:\temp_OmniVoice\hf_cache\hub\models--k2-fsa--OmniVoice" (
    echo [THONG TIN] Phat hien model OmniVoice co san tai D:\temp_OmniVoice\hf_cache
    set "HF_HOME=D:\temp_OmniVoice\hf_cache"
    set "HF_FOUND=1"
)

REM 1.2 Neu D: chua co, kiem tra thu muc cache mac dinh cua he thong tai %USERPROFILE%
if not defined HF_FOUND (
    if exist "%USERPROFILE%\.cache\huggingface\hub\models--k2-fsa--OmniVoice" (
        echo [THONG TIN] Phat hien model OmniVoice co san tai cache he thong:
        echo             %USERPROFILE%\.cache\huggingface
        set "HF_HOME=%USERPROFILE%\.cache\huggingface"
        set "HF_FOUND=1"
    )
)

if not defined HF_FOUND (
    if exist "D:\" (
        echo [THONG TIN] Chua co cache, se luu model vao D:\temp_OmniVoice\hf_cache
        set "HF_HOME=D:\temp_OmniVoice\hf_cache"
    ) else (
        echo [THONG TIN] Dung thu muc cache mac dinh cua he thong.
    )
)

REM 1.3 Dieu huong toan bo tac vu nang, cache temp ra D:\temp_OmniVoice de tiet kiem 4G va Google Drive
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

if not exist "outputs" mkdir "outputs"

REM =====================================================================
REM 2. TU DONG TIM TRINH PYTHON CO DU THU VIEN (GRADIO, TORCH)
REM =====================================================================
set "PY_CMD="

REM 2.1 Kiem tra virtualenv (.venv) neu hop le
if exist ".venv\Scripts\python.exe" (
    call .venv\Scripts\python.exe -c "import gradio, torch" >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=.venv\Scripts\python.exe"
    )
)

REM 2.2 Kiem tra Python 3.11 (Noi da cai day du gradio, omnivoice, torch)
if not defined PY_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        call "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -c "import gradio, torch" >nul 2>nul
        if not errorlevel 1 (
            set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        )
    )
)

REM 2.3 Kiem tra lenh python mac dinh trong PATH
if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        call python -c "import gradio, torch" >nul 2>nul
        if not errorlevel 1 (
            set "PY_CMD=python"
        )
    )
)

REM 2.4 Fallback cac ban Python khac (neu may chua cai vao 3.11)
if not defined PY_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY_CMD (
    if exist "C:\Program Files\Python310\python.exe" set "PY_CMD=C:\Program Files\Python310\python.exe"
)
if not defined PY_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)
if not defined PY_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py"
)

if not defined PY_CMD (
    echo [LOI] Khong tim thay Python co cai san gradio/torch! Vui long kiem tra lai.
    pause
    exit /b 1
)

REM =====================================================================
REM 3. TU DONG NHAN DIEN PHAN CUNG (GPU NVIDIA vs CPU INTEL / KHAC)
REM =====================================================================
set "HW_INFO=CPU Auto-detect"
call "%PY_CMD%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%g in ('call "%PY_CMD%" -c "import torch; print(torch.cuda.get_device_name(0))"') do (
        set "HW_INFO=GPU Acceleration (%%g)"
    )
) else (
    set "OMP_NUM_THREADS=4"
    set "MKL_NUM_THREADS=4"
    set "HW_INFO=CPU Multi-Core Optimized (OMP/MKL=4)"
)

echo ======================================================================
echo   OMNIVOICE - TIKTOK / SHORTS STUDIO WEBUI
echo   Python:       %PY_CMD%
echo   HF Cache:     %HF_HOME%
echo   Temp Dir:     D:\temp_OmniVoice (Bao ve Google Drive va 4G)
echo   Hardware:     %HW_INFO%
echo ======================================================================
echo.
echo Dang khoi dong WebUI danh rieng cho TikTok / Shorts...
echo Trinh duyet se san sang tai dia chi: http://localhost:7860
echo.

"%PY_CMD%" webui_shorts.py --server_port 7860

pause
