@echo off
title Elite AI Screen Assistant

echo ============================================
echo      ELITE AI SCREEN ASSISTANT
echo      GPT-5 - Claude 4.5 - Opus - Grok 4
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing elite dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements_elite.txt

echo.
echo Setup complete!
echo.
echo ============================================
echo         Starting Elite AI Assistant
echo ============================================
echo.
echo Open browser: http://localhost:5000
echo.
echo API Keys Needed (get at least one):
echo   - OpenAI GPT-5: platform.openai.com
echo   - Anthropic Claude: console.anthropic.com
echo   - OpenRouter: openrouter.ai
echo   - xAI Grok: x.ai
echo.
echo Quick Start:
echo   1. Enter your API keys
echo   2. Click 'Connect APIs'
echo   3. Choose your AI with the tabs
echo   4. Start chatting!
echo.
echo Features:
echo   - Switch between 4 elite AIs instantly
echo   - All AIs can see your screen
echo   - Natural conversation
echo   - AI can control mouse/keyboard
echo.
echo Press Ctrl+C to stop
echo ============================================
echo.

REM Run the application
python elite_ai_assistant.py

pause