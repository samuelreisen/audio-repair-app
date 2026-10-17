@echo off
cd /d "%~dp0"

rem Find a real Python (the Windows Store "python" stub does not count).
set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE (
  py -3 -c "import sys" >nul 2>nul && set "PYLAUNCH=py -3"
)
if not defined PYEXE if not defined PYLAUNCH (
  python -c "import sys" >nul 2>nul && set "PYLAUNCH=python"
)
if not defined PYEXE if not defined PYLAUNCH (
  echo Python is not installed. Install it with:  winget install Python.Python.3.12
  echo Then close this window and run run.bat again.
  pause
  exit /b 1
)

if not defined PYEXE (
  echo Creating virtual environment and installing packages ^(first run only^)...
  %PYLAUNCH% -m venv .venv || goto :fail
  set "PYEXE=.venv\Scripts\python.exe"
  "%PYEXE%" -m pip install --upgrade pip || goto :fail
  "%PYEXE%" -m pip install -r requirements.txt || goto :fail
)

where ffmpeg >nul 2>nul
if errorlevel 1 echo NOTE: ffmpeg not found. MP3/M4A support needs it:  winget install ffmpeg

rem Open the browser only once the server is listening (the first start takes a few seconds).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "$i=0; while(-not (Test-NetConnection 127.0.0.1 -Port 8000 -InformationLevel Quiet -WarningAction SilentlyContinue) -and $i -lt 60){Start-Sleep 1; $i++}; Start-Process 'http://127.0.0.1:8000'"
"%PYEXE%" app.py
pause
exit /b 0

:fail
echo Setup failed. See the messages above.
pause
exit /b 1
