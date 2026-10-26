@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing Python virtual environment. Follow README.md setup steps.
  pause
  exit /b 1
)
set "FLAPPY_PROFILE_ARGS="
if exist "profiles\demo.json" set "FLAPPY_PROFILE_ARGS=--profile profiles\demo.json"
".venv\Scripts\python.exe" -m flappy_auto --demo %FLAPPY_PROFILE_ARGS% --debug --seconds 60
pause
