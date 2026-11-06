@echo off
REM Double-click before the presentation: opens the dashboard in your browser.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
