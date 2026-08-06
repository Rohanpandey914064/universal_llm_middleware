@echo off
echo ===================================================
echo   Starting SentinelAI Demo Chatbot Application
echo ===================================================
echo.
echo Make sure the Universal Middleware is running on port 8080!
echo (In another terminal: cd universal_llm_middleware ^&^& python main.py)
echo.
echo Demo Web UI will open at: http://127.0.0.1:3000
echo.
cd /d %~dp0
python server.py
pause
