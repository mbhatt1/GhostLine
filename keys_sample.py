import os
import openai

# Environment variables required for configuring the application
os.environ["TWILIO_ACCOUNT_SID"] = "<TWILIO_ACCOUNT_SID>"  # Your Twilio Account SID
os.environ["TWILIO_AUTH_TOKEN"] = "<TWILIO_AUTH_TOKEN>"    # Your Twilio Auth Token
os.environ["TWILIO_FROM_NUMBER"] = "<TWILIO_PHONE_NUMBER>"  # Your Twilio phone number
os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"          # Your OpenAI API Key
os.environ["DEEPGRAM_API_KEY"] = "<DEEPGRAM_API_KEY>"      # Your Deepgram API Key
os.environ["ELEVENLABS_API_KEY"] = "<ELEVENLABS_API_KEY>"  # Your ElevenLabs API Key
os.environ["NGROK_AUTHTOKEN"] = "<NGROK_AUTHTOKEN>"        # Your ngrok Auth Token

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]
