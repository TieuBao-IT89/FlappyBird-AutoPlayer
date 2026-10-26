@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing Python virtual environment. Follow README.md setup steps.
  pause
  exit /b 1
)
set "FLAPPY_PROFILE_ARGS="
if exist "profiles\site.json" set "FLAPPY_PROFILE_ARGS=--profile profiles\site.json"
".venv\Scripts\python.exe" -u -m flappy_auto %FLAPPY_PROFILE_ARGS% --debug --runs 3 --seconds 0 --save-debug artifacts\latest
pause
