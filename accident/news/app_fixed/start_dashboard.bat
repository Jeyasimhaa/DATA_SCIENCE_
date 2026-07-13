@echo off
REM Run this in a SECOND terminal window (after start_api.bat is running).
cd /d "%~dp0dashboard"
streamlit run app.py
