@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
python -m streamlit run app.py
pause
