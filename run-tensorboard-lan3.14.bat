@echo off
setlocal
title Tensorboard (LAN, Python 3.14)

if not exist env314\Scripts\python.exe (
    echo Please run 'run-install3.14.bat' first to set up the Python 3.14 environment.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"

env314\Scripts\python.exe core.py tensorboard --lan
echo.
pause
