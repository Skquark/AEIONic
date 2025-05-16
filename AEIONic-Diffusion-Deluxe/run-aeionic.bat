@echo off
setlocal
cd /D "%~dp0"
reg query "HKCU\Console" /v VirtualTerminalLevel >nul
IF NOT %ERRORLEVEL% EQU 0 (
  reg add "HKCU\Console" /v VirtualTerminalLevel /t REG_DWORD /d 0x1
)
reg query "hkcu\software\Python" >nul
if ERRORLEVEL 1 GOTO NOPYTHON
echo Downloading latest AEIONic Diffusion Deluxe and running in a Python Virtual Environment
if not EXIST .\venv (py -3 -m venv .\venv)
powershell -Command "if (Test-Connection -ComputerName google.com -Count 1 -Quiet) { Invoke-WebRequest https://raw.githubusercontent.com/Skquark/AEIONic/main/AEIONic-Diffusion-Deluxe/AEIONic-Diffusion-Deluxe.py -OutFile .\venv\AEIONic-Diffusion-Deluxe.py } else { if (Test-Path '.\venv\AEIONic-Diffusion-Deluxe.py') { Write-Host 'No internet connection. Using existing script version.' } else { Write-Host 'No internet connection on first run. Unable to proceed.' ; exit 1 } }"
cd .\venv
call .\Scripts\activate.bat
.\Scripts\python.exe -m pip install --upgrade --quiet pip
.\Scripts\python.exe -m pip install --upgrade --quiet flet[all]
set "ICON_PATH=..\favicon.ico"
set "FLET_EXE=.\Lib\site-packages\flet_desktop\app\flet\flet.exe"
set "RCEDIT_EXE=..\rcedit-x64.exe"
set "RCEDIT_URL=https://github.com/electron/rcedit/releases/latest/download/rcedit-x64.exe"
if not exist "%RCEDIT_EXE%" (
    echo rcedit-x64.exe not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri '%RCEDIT_URL%' -OutFile '%RCEDIT_EXE%'"
    if not exist "%RCEDIT_EXE%" (
        echo Failed to download rcedit-x64.exe.
		    pause
    )
    REM Downloaded rcedit-x64.exe successfully.
)
if not exist "%ICON_PATH%" (
    echo Favicon file not found: %ICON_PATH%
    pause
)
if not exist "%FLET_EXE%" (
    echo Flet EXE not found: %FLET_EXE%
    pause
)
"%RCEDIT_EXE%" "%FLET_EXE%" --set-icon "%ICON_PATH%"
if %ERRORLEVEL% EQU 0 (
    REM Icon updated for flet.exe.
) else (
    echo Failed to update icon.
    pause
)
cls
flet .\AEIONic-Diffusion-Deluxe.py
call .\Scripts\deactivate.bat
exit /B 1
:NOPYTHON
echo "Python for Windows is not installed. Get it from https://www.python.org/downloads/ first."
pause