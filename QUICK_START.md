# Quick Start Guide - Elite AI Assistant

Your API keys are now configured! Here's how to run it:

## Step 1: Install Dependencies

```bash
cd path/to/Downloads/files
pip install -r requirements_elite.txt
```

## Step 2: Verify Your .env File

Your `.env` file should already be set up with your API keys. Check that it exists:

```bash
# Windows
dir .env

# Mac/Linux
ls -la .env
```

## Step 3: Run the Application

```bash
python elite_ai_assistant.py
```

You should see output like:

```
============================================================
🚀 Elite AI Screen Assistant Starting...
============================================================
🔒 Server: 127.0.0.1:5000 (localhost only)
🔑 API Keys loaded from .env file

✅ Loaded API Keys:
   ✓ OPENAI: proj-AqI...Z4A
   ✓ ANTHROPIC: sk-ant-a...QAA
   ✓ OPENROUTER: sk-or-v...7b5
   ✓ XAI: xai-rwjO...WKHM

============================================================
📱 Open your browser: http://127.0.0.1:5000
============================================================

⚠️  SECURITY NOTES:
   • Server only accessible from localhost
   • Do not expose this to the internet
   • Monitor your API usage and costs
```

## Step 4: Open in Browser

Open your web browser and go to:
```
http://127.0.0.1:5000
```

or

```
http://localhost:5000
```

## Step 5: Start Chatting!

1. Select an AI model using the tabs (GPT-5, Claude 4.5, Opus, Grok 4)
2. Type your message in the chat box
3. Click "Send" or press Enter
4. The AI will respond using the API key from your .env file

## Optional: Enable Screen Watching

1. Click "Start Watching" button
2. The AI can now see your screen
3. Ask questions about what's on screen
4. Click "Stop Watching" when done

## Optional: Give AI Control

⚠️ **Be careful with this feature!**

1. Click "Give AI Control"
2. Confirm the popup
3. AI can now move mouse and type
4. Move mouse to top-left corner to emergency stop
5. Click "Take Back Control" when done

## Troubleshooting

### "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### "API key invalid"
- Check your .env file has the correct keys
- Verify keys have no extra spaces
- Make sure .env is in the same folder as elite_ai_assistant.py

### "Connection refused"
- Check that port 5000 isn't already in use
- Try changing PORT=5001 in .env file
- Make sure Windows Firewall isn't blocking Python

### Can't access from another computer
- That's correct! For security, it only works on localhost
- If you need network access, change HOST in .env (not recommended)

## Monitor Your API Usage

Keep an eye on your credits:

- **OpenAI**: https://platform.openai.com/usage
- **Anthropic**: https://console.anthropic.com/usage  
- **OpenRouter**: https://openrouter.ai/credits
- **xAI**: Check your xAI dashboard

## Stop the Server

Press `Ctrl+C` in the terminal where it's running.

## What's Next?

You're now running the Elite AI Assistant locally! Since you mentioned having ~$20 in credits per service, you should be fine for testing.

**Tips:**
- Start with Claude 3.5 Sonnet (cheapest)
- Use GPT-4 Turbo for complex tasks (GPT-5 doesn't exist yet)
- Monitor usage frequently
- Don't leave screen watching on unnecessarily

**For More Details:**
- Read `ENV_SETUP_GUIDE.md` for configuration options
- Read `EXECUTIVE_SUMMARY.md` for security recommendations
- Read `elite_ai_architecture_review.md` for full technical analysis

Enjoy your multi-AI assistant! 🚀