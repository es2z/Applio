@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"

title %folder_name% (Python 3.14)

if not exist env314\Scripts\python.exe (
    echo Please run 'run-install3.14.bat' first to set up the Python 3.14 environment.
    pause
    exit /b 1
)

rem PyTorch Inductor templates are UTF-8, while Japanese Windows defaults to cp932.
rem Set UTF-8 mode before Python starts so compiler subprocesses inherit it too.
set "PYTHONUTF8=1"

env314\Scripts\python.exe app.py --open
echo.
pause
