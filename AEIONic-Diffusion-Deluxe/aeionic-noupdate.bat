@echo off
cd /D "%~dp0"
NET SESSION >nul 2>&1
IF NOT %ERRORLEVEL% EQU 0 (
   echo Must launch app with Run as Administrator
   pause
   exit /B 1
)
reg query "hkcu\software\Python"
if ERRORLEVEL 1 GOTO NOPYTHON
echo Running AEIONic Diffusion Deluxe in a Python Virtual Environment
if not EXIST .\venv GOTO NOVENV
cd .\venv
call .\Scripts\activate.bat
py -3 -m pip install --upgrade --quiet pip
py -3 -m pip install --upgrade --quiet flet
cls
flet .\AEIONic-Diffusion-Deluxe.py
call .\Scripts\deactivate.bat
exit /B 1
:NOPYTHON
echo "Python for Windows is not installed. Get it from https://www.python.org/downloads/ first."
:NOVENV
echo "Run the regular aeionic-venv.bat to initialize before running this."