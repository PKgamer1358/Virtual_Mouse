@echo off
title Proton Voice Assistant
echo ===================================================
echo   Starting Proton Voice Assistant...
echo ===================================================

cd /d "%~dp0"

if exist ".venv311\Scripts\python.exe" (
    set "PYTHON_EXE=.venv311\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python environment: %PYTHON_EXE%
cd src
"%~dp0%PYTHON_EXE%" Proton.py

echo.
echo Application closed.
pause
