@echo off
REM Double-click to run every experiment and the 20 tests.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python main.py
python -m pytest -v -p no:cacheprovider
pause
