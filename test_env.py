import os
from dotenv import load_dotenv

load_dotenv()

print("Checking environment variables...")

print("LIVEKIT_URL:", os.getenv("LIVEKIT_URL"))
print("LIVEKIT_API_KEY:", os.getenv("LIVEKIT_API_KEY")[:10], "...")
print("DEEPGRAM_API_KEY:", os.getenv("DEEPGRAM_API_KEY")[:10], "...")
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")[:10], "...")
print("RIME_API_KEY:", os.getenv("RIME_API_KEY")[:10], "...")