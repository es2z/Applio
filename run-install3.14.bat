@echo off
setlocal
title Applio Installer (Python 3.14)

rem Builds env314 with uv alone: a uv-managed CPython 3.14 plus requirements3.14.txt.
rem The Python 3.11 environment (env, run-install.bat, run-applio.bat) is left untouched.
rem Running this again on an existing env314 updates it in place; delete the env314
rem folder first for a clean install.

cd /d "%~dp0"
set "ENV_DIR=%~dp0env314"
set "PYTHON_VERSION=3.14"
rem PyTorch 2.15 release candidate (test channel). Once 2.15.0 is released, change this to
rem https://download.pytorch.org/whl/cu134 - the version string in requirements3.14.txt
rem stays the same.
set "TORCH_INDEX=https://download.pytorch.org/whl/test/cu134"

echo Welcome to the Applio Installer (Python %PYTHON_VERSION%)!
echo.

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Installing uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :error
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul || goto :error

echo Installing Python %PYTHON_VERSION%...
uv python install %PYTHON_VERSION% || goto :error
echo.

if exist "%ENV_DIR%\Scripts\python.exe" (
    echo Reusing the existing environment at %ENV_DIR%.
) else (
    echo Creating the virtual environment at %ENV_DIR%...
    uv venv "%ENV_DIR%" --python %PYTHON_VERSION% --managed-python || goto :error
)
echo.

echo Installing dependencies...
echo This will install PyTorch 2.15.0 with CUDA 13.4 support and all required packages.
echo.
uv pip install --python "%ENV_DIR%\Scripts\python.exe" -r requirements3.14.txt --extra-index-url %TORCH_INDEX% --index-strategy unsafe-best-match || goto :error
echo.

echo Applio (Python %PYTHON_VERSION%) has been installed successfully!
echo To start it, please run 'run-applio3.14.bat'.
echo.
pause
exit /b 0

:error
echo An error occurred during installation. Please check the output above for details.
pause
exit /b 1
