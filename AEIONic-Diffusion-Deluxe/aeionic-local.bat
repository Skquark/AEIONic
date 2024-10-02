@echo off
cd /D "%~dp0"
NET SESSION >nul 2>&1
IF NOT %ERRORLEVEL% EQU 0 (
   echo Must launch app with Run as Administrator
   pause
   exit /B 1
)
pip install --upgrade --quiet flet
powershell -Command "Invoke-WebRequest https://raw.githubusercontent.com/Skquark/AEIONic/main/AEIONic-Diffusion-Deluxe/AEIONic-Diffusion-Deluxe.py -OutFile .\AEIONic-Diffusion-Deluxe.py"
flet run AEIONic-Diffusion-Deluxe.py