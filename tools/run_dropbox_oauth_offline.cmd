@echo off
setlocal

REM Runs Dropbox OAuth helper in offline mode and writes tokens into .env.
REM Usage:
REM   tools\run_dropbox_oauth_offline.cmd

set REPO_ROOT=%~dp0..
set PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe
set SCRIPT=%REPO_ROOT%\tools\dropbox_oauth_login.py

if not exist "%PYTHON%" (
  echo [ERROR] Python venv not found: %PYTHON%
  echo Create/activate the venv first.
  exit /b 2
)

pushd "%REPO_ROOT%" >nul
"%PYTHON%" "%SCRIPT%" --offline --write-env
set EXITCODE=%ERRORLEVEL%
popd >nul
exit /b %EXITCODE%
