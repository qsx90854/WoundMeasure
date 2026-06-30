@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0..\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

set "INPUT_DIR=%~1"
if "%INPUT_DIR%"=="" (
    set /p INPUT_DIR=Enter video folder path: 
)

set "OUTPUT_DIR=%~2"
if "%OUTPUT_DIR%"=="" (
    "%PYTHON_EXE%" batch_process_videos.py "%INPUT_DIR%"
) else (
    "%PYTHON_EXE%" batch_process_videos.py "%INPUT_DIR%" --output-dir "%OUTPUT_DIR%"
)

pause
