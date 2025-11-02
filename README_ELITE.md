# Elite AI Screen Assistant 🚀

The most powerful AI assistant featuring GPT-5, Claude 4.5 Sonnet, Claude 4 Opus, and Grok 4. Switch between cutting-edge AI models instantly while they watch your screen and help you work.

## 🌟 Elite AI Models

| Model | Provider | Strengths | API Key Required |
|-------|----------|-----------|------------------|
| **GPT-5** 🧠 | OpenAI | Most advanced reasoning, coding, vision | OpenAI API key |
| **Claude 4.5 Sonnet** ⚡ | Anthropic | Lightning fast, excellent at analysis | Anthropic API key |
| **Claude 4 Opus** 👑 | OpenRouter | Most capable Claude, deep understanding | OpenRouter API key |
| **Grok 4** 🚀 | xAI | Unfiltered, witty, real-time knowledge | xAI API key |

## 🔥 Features

- **🎯 Multi-AI System**: Switch between 4 elite AI models with one click
- **👀 Screen Vision**: All models can see and understand your screen
- **💬 Natural Chat**: Talk to AIs like colleagues, not robots  
- **🎮 AI Control**: Let any AI take control when you need help
- **🔄 Instant Switching**: Change AIs mid-conversation
- **⚡ Real-time Help**: AIs watch and jump in when you're stuck
- **🎨 Beautiful Dark UI**: Sleek interface with model-specific themes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- At least one API key (see below)

### Get Your API Keys

#### OpenAI (GPT-5)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up/login
3. Navigate to API Keys
4. Create new secret key starting with `sk-`
5. **Note**: GPT-5 access may require special approval

#### Anthropic (Claude 4.5 Sonnet)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up/login
3. Go to API Keys section
4. Generate key starting with `sk-ant-`
5. **Note**: May need Claude Pro subscription

#### OpenRouter (Claude 4 Opus)
1. Go to [openrouter.ai](https://openrouter.ai)
2. Create account
3. Add credits ($5 minimum recommended)
4. Get API key starting with `sk-or-`
5. **Pricing**: ~$15 per million tokens

#### xAI (Grok 4)
1. Go to [x.ai](https://x.ai) or platform.x.ai
2. Request API access (may have waitlist)
3. Once approved, get key starting with `xai-`
4. **Note**: Limited availability

### Installation

```bash
# Clone or download files
git clone <repository> elite-ai-assistant
cd elite-ai-assistant

# Install dependencies
pip install -r requirements_elite.txt

# Run the app
python elite_ai_assistant.py
```

### First Run
1. Open browser to `http://localhost:5000`
2. Enter your API keys (only add the ones you have)
3. Click "Connect APIs"
4. Select your preferred AI using the tabs
5. Click "Start Watching" to enable screen vision
6. Start chatting!

## 💡 Usage Tips

### Switching Between AIs
Each AI has unique strengths - switch based on your needs:

- **GPT-5**: Complex reasoning, technical documentation, creative writing
- **Claude 4.5**: Fast coding, data analysis, quick explanations
- **Claude 4 Opus**: Deep research, nuanced understanding, long documents
- **Grok 4**: Current events, humor, unfiltered responses

### Example Commands
```
"Hey, I'm stuck on this Python error"
"Can you see what's wrong with my Excel formula?"
"Take control and fix this CSS layout for me"
"What's the best way to refactor this code?"
"Help me write a response to this email"
```

### Letting AI Take Control
1. Click "Give AI Control" 
2. AI will move your mouse and type
3. Move mouse to top-left corner for emergency stop
4. Click "Take Back Control" when ready

## 🛠️ Advanced Configuration

### Custom Model Settings
Edit `elite_ai_assistant.py` to adjust:
- Temperature (creativity vs consistency)
- Max tokens (response length)
- Vision detail level (low/high)

### Adding Custom Models
Add to `AI_MODELS` dictionary:
```python
"your-model": AIModel(
    provider=AIProvider.OPENROUTER,
    model_id="provider/model-name",
    display_name="Your Model",
    description="Description",
    requires_api_key="openrouter",
    supports_vision=True,
    icon="🎯",
    color="#hexcolor"
)
```

### API Rate Limits
- OpenAI GPT-5: Varies by tier
- Anthropic: 5 requests/minute (free tier)
- OpenRouter: Based on credits
- xAI: Limited during beta

## 🔧 Troubleshooting

### "Model not available"
- GPT-5: Might not be released yet, try gpt-4-turbo-preview instead
- Claude 4.5: Ensure you have latest Anthropic API access
- Grok 4: Check if you have beta access

### "API key invalid"
- Ensure no extra spaces in keys
- Check you have credits/active subscription
- Verify key format matches provider

### Screen capture issues
- **Mac**: Grant screen recording permission
- **Windows**: Run as administrator
- **Linux**: Install `python3-tk python3-dev`

### High API costs
- OpenRouter shows costs upfront
- Use Claude 4.5 for routine tasks (cheaper)
- GPT-5/Opus for complex work only
- Monitor usage in provider dashboards

## 💰 Cost Optimization

| Model | Cost/1M tokens | Best For |
|-------|---------------|----------|
| GPT-5 | ~$30-60 | Critical complex tasks |
| Claude 4.5 | ~$3-15 | Daily coding/writing |
| Claude 4 Opus | ~$15-75 | Deep analysis |
| Grok 4 | TBD | Real-time info |

## 🔐 Security

- API keys stored only in memory
- Never committed to version control
- Use environment variables for production:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 🎮 Keyboard Shortcuts

- `Ctrl+1-4`: Quick switch between models
- `Ctrl+W`: Toggle screen watching
- `Ctrl+H`: Request help
- `Ctrl+C`: Stop AI control
- `Escape`: Clear current message

## 📊 Performance Tips

- Start screen watching only when needed
- Use lower resolution for faster processing
- Close unnecessary applications
- Use ethernet over WiFi for stability

## 🚨 Safety Features

- **Failsafe**: Mouse to top-left stops all
- **Confirmation**: Required for AI control
- **Rate limiting**: Prevents API abuse
- **Action delays**: Safe, deliberate movements
- **Status indicators**: Always know who's in control

## 📝 License

MIT License - Modify and use as needed!

## 🙏 Credits

Built with cutting-edge AI APIs:
- OpenAI GPT-5 (when available)
- Anthropic Claude 4.5 Sonnet
- Claude 4 Opus via OpenRouter
- xAI Grok 4

---

**Note**: Some models like GPT-5 and Grok 4 may not be publicly available yet. The app will work with whatever models you have access to. You can modify model IDs in the code if needed (e.g., use `gpt-4-turbo-preview` instead of `gpt-5`).

For the ultimate AI assistant experience! 🚀