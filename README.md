# Twilio Chatbot: Outbound

This project is a Pipecat-based chatbot that integrates with Twilio to make outbound calls with personalized call information. The project includes FastAPI endpoints for initiating outbound calls and handling WebSocket connections with call context.

## How It Works

When you want to make an outbound call:

1. **Send POST request**: `POST /start` with a phone number to call
2. **Server initiates call**: Uses Twilio's REST API to make the outbound call
3. **Call answered**: When answered, Twilio fetches TwiML from your server's `/twiml` endpoint
4. **Server returns TwiML**: Tells Twilio to start a WebSocket stream to your bot
5. **WebSocket connection**: Audio streams between the called person and your bot
6. **Call information**: Phone numbers are passed via TwiML Parameters to your bot

## Architecture

```
curl request → /start endpoint → Twilio REST API → Call initiated →
TwiML fetched → WebSocket connection → Bot conversation
```

## Prerequisites

### Twilio

- A Twilio account with:
  - Account SID and Auth Token
  - A purchased phone number that supports voice calls

### AI Services

- OpenAI API key for the LLM inference

### System

- Python 3.10+
- `uv` package manager
- ngrok (for local development)


## Setup

1. Set up a virtual environment and install dependencies:

```bash
cd outbound
uv sync
uv pip install -r agents/requirements.txt 
uv pip install av
uv pip install opentelemetry-api opentelemetry-sdk
uv pip install "nvidia-riva-client==2.20.0"
```




2. Get your Twilio credentials:

- **Account SID & Auth Token**: Found in your [Twilio Console Dashboard](https://console.twilio.com/)
- **Phone Number**: [Purchase a phone number](https://console.twilio.com/us1/develop/phone-numbers/manage/search) that supports voice calls

3. Set up environment variables:

```bash
cp env.example .env
# Edit .env with your API keys
```

## Environment Configuration

The bot supports two deployment modes controlled by the `ENV` variable:


## Local Development

0. Start langgraph:

  ```bash
    cd agents
    uv run langgraph dev
  ```

1. Start the outbound bot server:

   ```bash
   uv run server.py
   ```

The server will start on port 7860.

2. Using a new terminal, expose your server to the internet (for development)

   ```bash
   ngrok http 7860
   ```

   > Tip: Use the `--subdomain` flag for a reusable ngrok URL.

   Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

3. No additional Twilio configuration needed

   Unlike inbound calling, outbound calls don't require webhook configuration in the Twilio console. The server will make direct API calls to Twilio to initiate calls.

## Making an Outbound Call

With the server running and exposed via ngrok, you can initiate outbound calls:

### Basic Call

```bash
curl -X POST https://your-ngrok-url.ngrok.io/start \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+1234567890"
  }'
```

### Call with Body Data

You can include arbitrary body data that will be available to your bot:

```bash
curl -X POST https://your-ngrok-url.ngrok.io/start \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+1234567890",
    "body": {
      "user": {
        "id": "user123",
        "name": "John Doe",
        "account_type": "premium"
      }
    }
  }'
```

The body data can be any JSON structure - nested objects, arrays, etc. Your bot will receive this data automatically.

Replace:

- `your-ngrok-url.ngrok.io` with your actual ngrok URL
- `+1234567890` with the phone number you want to call

