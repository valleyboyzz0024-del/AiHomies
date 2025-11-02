#!/bin/bash

# Elite AI Screen Assistant - Setup and Run Script

echo "╔════════════════════════════════════════╗"
echo "║     🚀 ELITE AI SCREEN ASSISTANT 🚀    ║"
echo "║          GPT-5 | Claude 4.5            ║"
echo "║        Claude Opus | Grok 4            ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Install requirements
echo "📚 Installing elite dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements_elite.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "╔════════════════════════════════════════╗"
echo "║         Starting Elite AI Assistant     ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "🌐 Open your browser: http://localhost:5000"
echo ""
echo "🔑 API Keys Needed (get at least one):"
echo "   • OpenAI GPT-5: platform.openai.com"
echo "   • Anthropic Claude: console.anthropic.com"
echo "   • OpenRouter: openrouter.ai"
echo "   • xAI Grok: x.ai"
echo ""
echo "💡 Quick Start:"
echo "   1. Enter your API keys"
echo "   2. Click 'Connect APIs'"
echo "   3. Choose your AI with the tabs"
echo "   4. Start chatting!"
echo ""
echo "⚡ Features:"
echo "   • Switch between 4 elite AIs instantly"
echo "   • All AIs can see your screen"
echo "   • Natural conversation with each model"
echo "   • Let AI control mouse/keyboard"
echo ""
echo "Press Ctrl+C to stop"
echo "════════════════════════════════════════"
echo ""

# Run the application
python elite_ai_assistant.py