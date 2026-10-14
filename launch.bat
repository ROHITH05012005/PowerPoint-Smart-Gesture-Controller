@echo off
title AI-Powered PPT Gesture Controller
cd /d "%~dp0"
echo =================================================================
echo   STARTING AI GESTURE CONTROLLER PRODUCT
echo =================================================================
echo   [Q: QUIT]  [H: TOGGLE HUD]
echo.
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error. Press any key to close.
    pause >nul
)
