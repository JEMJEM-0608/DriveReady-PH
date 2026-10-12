@echo off
REM Start the DriveReady PH web app (make sure Ollama is running).
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py
