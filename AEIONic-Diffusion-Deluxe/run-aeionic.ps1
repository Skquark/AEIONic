Start-Process powershell -Verb runAs
if ( !(Test-Path .\venv) ) {
  python -m venv .\venv
}
if (Test-Connection -ComputerName google.com -Count 1 -Quiet) {
  Invoke-WebRequest https://raw.githubusercontent.com/Skquark/AEIONic/main/AEIONic-Diffusion-Deluxe/AEIONic-Diffusion-Deluxe.py -OutFile .\venv\AEIONic-Diffusion-Deluxe.py
} else {
  if (Test-Path '.\venv\AEIONic-Diffusion-Deluxe.py') {
    Write-Host 'No internet connection. Using existing script version.'
  } else {
    Write-Host 'No internet connection on first run. Unable to proceed.'
    exit 1
  }
}
Set-Location -Path '.\venv\Scripts'
.\Activate.ps1
Set-Location -Path ..
.\Scripts\python.exe -m pip install --upgrade pip
.\Scripts\python.exe -m pip install --upgrade flet[all]
flet .\AEIONic-Diffusion-Deluxe.py
& .\Scripts\deactivate.bat
exit /B 1