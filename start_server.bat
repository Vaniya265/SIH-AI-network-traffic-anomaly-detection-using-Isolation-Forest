@echo off
echo Starting SIH1451 Backend Server...
echo.
cd src
python -m uvicorn api:app --reload --port 8000
pause
