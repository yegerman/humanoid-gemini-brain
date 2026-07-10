@echo off
REM ER-boss demo (Gemini-ER orchestrates; 3.5 authors skills). Pass-through args, e.g. --er-secs 60
"E:\huminoid\.venv\Scripts\python.exe" "%~dp0embodied\run_orchestrator_demo.py" %*
