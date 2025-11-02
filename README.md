# AI Homies - Elite Multi-AI Screen Assistant

Your personal AI squad featuring the best models: GPT-5, Claude 4.5 Sonnet, Claude 4 Opus, and Grok 4!

![Elite AI Assistant](https://img.shields.io/badge/AI-Multi--Model-FF1493?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Features

### Multi-AI Model Support
- **GPT-5** - OpenAI's most advanced model
- **Claude 4.5 Sonnet** - Anthropic's latest and greatest
- **Claude 4 Opus** - Most powerful Claude via OpenRouter
- **Grok 4** - xAI's unfiltered intelligence

### Core Capabilities
- **Screen Watching** - AI monitors your screen and helps when you're stuck
- **Autonomous Error Detection** - Automatically detects errors in terminals and IDEs
- **Guardian Mode** - Proactive intervention when processes fail or get stuck
- **Command Execution** - Run safe whitelisted commands (npm, pip, git, docker, etc.)
- **Process Monitoring** - Track running processes and detect stuck operations
- **Real-time Chat** - Conversational interface with context awareness
- **Desktop App** - Native window support with pywebview

### Safety Features
- Command whitelist system - only safe commands allowed
- Localhost-only by default
- API key protection via .env file
- Process monitoring with error patterns

## Quick Start

### Prerequisites
- Python 3.8+
- API keys for at least one AI provider

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/valleyboyzz0024-del/AiHomies.git
cd AiHomies
```

2. **Install dependencies**
```bash
pip install -r requirements_elite.txt
```

3. **Set up API keys**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENROUTER_API_KEY=your-openrouter-key-here
XAI_API_KEY=your-xai-key-here
LEONARDO_API_KEY=your-leonardo-key-here  # Optional: for UI floaters
```

4. **Run the application**

**Option A: Browser-based**
```bash
python elite_ai_assistant.py
```
Open http://127.0.0.1:5000 in your browser

**Option B: Desktop App**
```bash
python elite_ai_desktop.py
```
Or use the launcher:
- Windows: `run_desktop.bat`
- Linux/Mac: `run_elite.sh`

## Usage

### Chat with AI
1. Select your preferred AI model from the tabs
2. Type your message in the chat box
3. AI responds with context from your screen (if watching is enabled)

### Screen Watching
1. Click "Start Watching"
2. AI monitors your screen every 3 seconds
3. Automatically detects errors and offers fixes
4. Click "Need Help" for instant assistance

### Command Execution
1. Type commands in the purple command input box
2. Examples: `pip list`, `npm install package`, `git status`
3. View real-time output in the chat
4. Only whitelisted safe commands are allowed

### Guardian Mode
1. Click "Guardian Mode"
2. AI autonomously monitors processes
3. Intervenes when errors are detected
4. Provides fixes and suggestions automatically

### Process Monitoring
1. Click "Start Monitoring"
2. Tracks running commands and processes
3. Detects error patterns in output
4. Alerts on stuck processes (>5 min no output)

## Architecture

```
elite_ai_assistant.py    # Main Flask app with WebSocket
elite_ai_desktop.py      # Desktop wrapper using pywebview
requirements_elite.txt   # Python dependencies
.env                     # API keys (not committed)
```

### Key Components
- **Flask + SocketIO** - Real-time web server
- **pyautogui + mss** - Screen capture and control
- **psutil** - Process monitoring
- **Multiple AI SDKs** - OpenAI, Anthropic, OpenRouter, xAI

## Security Notes

- Server runs on localhost (127.0.0.1) only by default
- Do NOT expose to the internet without proper security
- API keys stored in .env file (git-ignored)
- Command whitelist prevents dangerous operations
- Monitor your API usage and costs

## Supported Commands

### Package Managers
- `npm install`, `npm run`, `pnpm add`, `yarn install`
- `pip install`, `pip list`, `pip show`

### Version Control
- `git status`, `git add`, `git commit`, `git push`
- `git pull`, `git checkout`, `git branch`

### Containers & Databases
- `docker build`, `docker compose up`, `docker ps`
- `psql`, `mysql`, `mongo`

### Build Tools
- `make`, `cmake`, `cargo build`, `go build`

## Error Detection Patterns

The assistant detects:
- npm/pip errors (ENOENT, ERESOLVE, etc.)
- Python tracebacks and exceptions
- Build failures
- Test failures
- Port conflicts (EADDRINUSE)
- Database errors
- Docker errors
- Migration failures

## Configuration

### Environment Variables
```env
HOST=127.0.0.1          # Server host
PORT=5000               # Server port
SECRET_KEY=random-key   # Flask secret (auto-generated)
```

### Custom Command Whitelists
Edit `SAFE_COMMANDS` dictionary in `elite_ai_assistant.py`

## API Keys

Get your API keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **OpenRouter**: https://openrouter.ai/keys
- **xAI**: https://x.ai/api
- **Leonardo AI**: https://leonardo.ai/ (optional, for UI effects)

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements_elite.txt
```

### "API key not set" messages
Check your .env file has the correct API keys

### Screen capture not working
Ensure pyautogui and mss are installed correctly

### Desktop window won't open
```bash
pip install pywebview
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with Claude Code
- Powered by OpenAI, Anthropic, OpenRouter, and xAI APIs
- UI inspired by modern cyberpunk aesthetics

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [QUICK_START.md](QUICK_START.md) guide
- Review [README_ELITE.md](README_ELITE.md) for detailed docs

---

**Made with Claude Code** | **Star this repo if you find it useful!**
