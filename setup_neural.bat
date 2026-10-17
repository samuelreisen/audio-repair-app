@echo off
rem One-time setup of the optional neural noise remover (DeepFilterNet3). About 130 MB of downloads.
rem PyTorch has no Python 3.14 build yet, so this creates a separate Python 3.11 environment (.venv311).
cd /d "%~dp0"
py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 is needed for this step:  winget install Python.Python.3.11
  echo Then run setup_neural.bat again.
  pause
  exit /b 1
)
if not exist .venv311 py -3.11 -m venv .venv311 || goto :fail
.venv311\Scripts\python.exe -m pip install --upgrade pip || goto :fail
.venv311\Scripts\python.exe -m pip install "torch==2.1.2" "torchaudio==2.1.2" "numpy<2" deepfilternet soundfile || goto :fail
.venv311\Scripts\python.exe -c "from df.enhance import init_df; init_df(); print('DeepFilterNet3 ready')" || goto :fail
echo.
echo Done. The app uses the neural denoiser automatically from now on.
pause
exit /b 0
:fail
echo Setup failed. See the messages above.
pause
exit /b 1
