@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing Python virtual environment. Follow README.md setup steps.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m flappy_auto --debug --seconds 60
pause
