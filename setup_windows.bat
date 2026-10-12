@echo off
REM One-time setup for Windows (no PowerShell script activation needed).
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)
echo Installing packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
echo Running tests...
.venv\Scripts\python.exe -m pytest -q || goto :error
echo Building the knowledge base (uses Ollama nomic-embed-text if it is running)...
.venv\Scripts\python.exe scripts\build_index.py || goto :error
echo.
echo Setup complete. Start the app with run_app.bat
goto :eof
:error
echo.
echo Setup failed - see the message above.
exit /b 1
