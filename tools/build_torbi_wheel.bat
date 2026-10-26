@echo off
rem Rebuilds vendor\torbi\pt215-cu134\torbi-1.4.0-cp39-abi3-win_amd64.whl for env314:
rem the unmodified maxrmorrison/torbi source at the commit the env wheel was built from,
rem compiled against env314's torch. No published torbi wheel carries a binary for
rem torch 2.15.
rem
rem Needs git, Visual Studio 2022 with the C++ tools, a CUDA 13.x toolkit and env314 with
rem torch already installed. The wheel is committed, so this is only for rebuilding it (for
rem another torch, CUDA or GPU architecture); with no wheel at all, install env314 once
rem with the torbi line of requirements3.14.txt left out.
setlocal
cd /d "%~dp0.."
set "ROOT=%cd%"
set "PYTHON=%ROOT%\env314\Scripts\python.exe"
set "OUT=%ROOT%\vendor\torbi\pt215-cu134"
set "COMMIT=ca7732cd2344ba0d7449204f8805fb594a912701"
set "WORK=%TEMP%\torbi-build-%RANDOM%"

if not exist "%PYTHON%" (
    echo env314 not found. Run run-install3.14.bat first.
    exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo Visual Studio with the C++ tools was not found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH_V13_0%"
if not defined CUDA_HOME set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
rem RTX 4090. Add other architectures here for other cards, e.g. "8.6;8.9".
set "TORCH_CUDA_ARCH_LIST=8.9"
set "DISTUTILS_USE_SDK=1"
set "PYTHONUTF8=1"
rem cl.exe prints its localized banner in the console code page, while torch's compiler
rem check decodes it as the OEM code page, so the two have to agree.
for /f "tokens=3" %%c in ('reg query HKLM\SYSTEM\CurrentControlSet\Control\Nls\CodePage /v OEMCP ^| find "OEMCP"') do chcp %%c >nul

git clone https://github.com/maxrmorrison/torbi "%WORK%" || exit /b 1
git -C "%WORK%" checkout %COMMIT% || exit /b 1

if not exist "%OUT%" mkdir "%OUT%"
uv build "%WORK%" --wheel --no-build-isolation --python "%PYTHON%" -o "%OUT%"
set "RESULT=%errorlevel%"
rmdir /s /q "%WORK%"
if not "%RESULT%"=="0" exit /b %RESULT%

echo.
echo Built into %OUT%. Record the SHA-256 below in vendor\torbi\README.md:
certutil -hashfile "%OUT%\torbi-1.4.0-cp39-abi3-win_amd64.whl" SHA256
exit /b 0
