@echo off
title Gesture Controlled Virtual Mouse
echo ===================================================
echo   Starting Gesture-Controlled Virtual Mouse...
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
"%~dp0%PYTHON_EXE%" Gesture_Controller.py

echo.
echo Application closed.
pause
