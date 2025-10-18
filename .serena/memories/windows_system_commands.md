# Windows System Commands

This project is primarily developed and used on Windows. Here are the important Windows-specific commands and utilities.

## Shell Environments

### Command Prompt (CMD)
- Traditional Windows command line
- Batch files (`.bat`) execute in CMD
- Case-insensitive file system

### PowerShell
- Modern Windows shell
- More powerful than CMD
- Can run most CMD commands
- Additional cmdlets available

## File System Navigation

### List Directory Contents
```batch
# CMD
dir
dir /a        # Show hidden files
dir /s        # Recursive
dir /b        # Bare format (names only)

# PowerShell
ls
Get-ChildItem
Get-ChildItem -Force            # Show hidden
Get-ChildItem -Recurse          # Recursive
```

### Change Directory
```batch
cd path\to\directory
cd ..                           # Parent directory
cd \                            # Root of current drive
C:                              # Switch to C: drive
```

### Create Directory
```batch
mkdir directory_name
md directory_name               # Alias
```

### Remove Directory
```batch
rmdir directory_name
rd /s /q directory_name        # Recursive, no prompt
```

## File Operations

### Copy Files
```batch
# CMD
copy source.txt destination.txt
copy *.txt backup\              # Wildcard copy
xcopy /s /e source dest         # Directory copy

# PowerShell
cp source.txt destination.txt
Copy-Item source.txt destination.txt
Copy-Item -Recurse source dest
```

### Move/Rename Files
```batch
# CMD
move old.txt new.txt
ren old.txt new.txt            # Rename

# PowerShell
mv old.txt new.txt
Move-Item old.txt new.txt
Rename-Item old.txt new.txt
```

### Delete Files
```batch
# CMD
del file.txt
del /s *.pyc                   # Recursive delete
erase file.txt                 # Alias

# PowerShell
rm file.txt
Remove-Item file.txt
Remove-Item -Recurse folder
```

### View File Contents
```batch
# CMD
type file.txt
more file.txt                  # Paginated

# PowerShell
cat file.txt
Get-Content file.txt
```

## Search and Find

### Find Files
```batch
# CMD - Find by name
dir /s /b *.py                 # Find all .py files recursively
where python                   # Find executable in PATH

# PowerShell
Get-ChildItem -Recurse -Filter *.py
Get-ChildItem -Recurse -Include *.py,*.txt
```

### Search File Contents (grep equivalent)
```batch
# CMD
findstr "pattern" file.txt
findstr /s /i "pattern" *.py   # Recursive, case-insensitive

# PowerShell
Select-String "pattern" file.txt
Get-ChildItem -Recurse -Filter *.py | Select-String "pattern"
```

**Important**: For code searching in this project, use Serena's `search_for_pattern` tool instead of Windows commands for better results.

## Process Management

### List Processes
```batch
# CMD
tasklist
tasklist | findstr python

# PowerShell
Get-Process
Get-Process python*
```

### Kill Process
```batch
# CMD
taskkill /PID 1234
taskkill /IM python.exe /F

# PowerShell
Stop-Process -Id 1234
Stop-Process -Name python
```

## Environment Variables

### View Environment Variables
```batch
# CMD
set                            # Show all
echo %PATH%                    # Show specific

# PowerShell
Get-ChildItem Env:
$env:PATH
```

### Set Environment Variables (Session)
```batch
# CMD
set VAR_NAME=value

# PowerShell
$env:VAR_NAME = "value"
```

## Network

### Check Network Connectivity
```batch
ping google.com
ping -n 4 8.8.8.8             # 4 packets
```

### Download Files
```batch
# PowerShell
Invoke-WebRequest -Uri "url" -OutFile "file.exe"
```

## Python Environment

### Run Python (Applio specific)
```batch
# Use the virtual environment Python
env\python.exe script.py

# Check Python version
env\python.exe --version

# Install package
env\python.exe -m pip install package_name

# List installed packages
env\python.exe -m pip list
```

## Git Commands (Windows)

Git Bash or Git for Windows provides Unix-like commands:

```bash
# Standard Git operations
git status
git add .
git commit -m "message"
git push
git pull

# View history
git log
git log --oneline -10

# Diff
git diff
git diff file.py

# Branch operations
git branch
git checkout -b new-branch
git merge branch-name
```

## Batch File Basics

Common patterns in Applio batch files:

### Error Handling
```batch
@echo off                      # Don't echo commands
if errorlevel 1 goto :error    # Check error level
if not exist file.txt (        # Check file exists
    echo File not found
    exit /b 1
)
```

### Labels and Goto
```batch
:label_name
echo At label
goto :another_label

:another_label
exit /b 0
```

### Variables
```batch
set VAR=value
echo %VAR%

# Command substitution
for /f %%i in ('command') do set VAR=%%i
```

### Delayed Expansion
```batch
setlocal enabledelayedexpansion
set VAR=initial
set VAR=new
echo !VAR!                     # Shows 'new' with delayed expansion
```

## Path Separators

**Important**: Windows uses backslashes in paths

```batch
# Windows paths
C:\dev\myApplio\app.py
env\python.exe

# In Python code, use os.path.join() for cross-platform:
os.path.join("env", "python.exe")
# Results in: env\python.exe on Windows, env/python.exe on Unix
```

## File Permissions

Windows doesn't use Unix permissions (chmod). Instead:
- Right-click → Properties → Security tab
- Or use `icacls` command (complex)

For Applio, file permissions are usually not a concern.

## System Information

```batch
# System info
systeminfo
ver                            # Windows version

# Disk usage
dir                            # Shows file sizes
wmic logicaldisk get size,freespace,caption  # Disk space
```

## Conda/Miniconda (Used by Installer)

```batch
# Activate environment
call %MINICONDA_DIR%\condabin\conda.bat activate env

# Deactivate
conda deactivate

# Create environment
conda create -n env_name python=3.11

# List environments
conda env list
```

## PowerShell Aliases for Unix Commands

PowerShell provides aliases for common Unix commands:
- `ls` → Get-ChildItem
- `cat` → Get-Content
- `cp` → Copy-Item
- `mv` → Move-Item
- `rm` → Remove-Item
- `pwd` → Get-Location
- `ps` → Get-Process
- `kill` → Stop-Process
- `clear` → Clear-Host

## Tips for Working on Windows

1. **Use PowerShell** for better Unix-like experience
2. **Path separators**: Use `\` or `os.path.join()` in Python
3. **Case insensitivity**: `File.txt` == `file.txt` on Windows
4. **Line endings**: Windows uses CRLF (`\r\n`), Git may convert
5. **Admin rights**: Avoid running as Administrator (Applio checks for this)
6. **File locking**: Windows locks files when in use (unlike Unix)

## Common Issues

### "File in use" Errors
- Close programs using the file
- Restart if necessary
- Check with: `handle file.txt` (Sysinternals tool)

### Path Length Limitations
- Windows has 260 character path limit (usually)
- Enable long paths: `gpedit.msc` → Computer Configuration → Administrative Templates → System → Filesystem
- Or use `\\?\` prefix for long paths

### Execution Policy (PowerShell)
If scripts won't run:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
