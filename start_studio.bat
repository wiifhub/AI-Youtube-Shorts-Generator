@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo Please run install_windows.bat first.
  pause
  exit /b 1
)
start "Shorts Studio" /b venv\Scripts\python.exe -m web.app
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:7860
echo Shorts Studio is running at http://127.0.0.1:7860
echo Close the server with Ctrl+C if running in a terminal, or stop python.exe.
pause
