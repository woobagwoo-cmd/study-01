@echo off
rem Double-click this file in Windows Explorer to launch the digit recognizer.
cd /d "%~dp0"

python digit_recognition.py

if errorlevel 1 (
    echo.
    echo Something went wrong. See the error message above.
    pause
)
