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

title %folder_name%

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

rem PyTorch Inductor templates are UTF-8, while Japanese Windows defaults to cp932.
rem Set UTF-8 mode before Python starts so compiler subprocesses inherit it too.
set "PYTHONUTF8=1"

env\python.exe app.py --open
echo.
pause
