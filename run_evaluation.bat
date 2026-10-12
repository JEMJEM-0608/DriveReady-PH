@echo off
REM Reproduce all results in reports\ (needs Ollama with llama3.1:8b and nomic-embed-text).
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\simulate_bandit.py
.venv\Scripts\python.exe scripts\evaluate.py %*
echo.
echo Done. Results are in the reports folder.
