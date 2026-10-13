import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IkCH28v1pGcqRYqIhVs16vacOzG14ZYjHSS6lHU-NOeQ")
client = genai.Client(api_key=API_KEY)

print("Auditing available models for current API key...\n")
try:
    for model in client.models.list():
        if "gemini" in model.name.lower():
            print(f"✅ {model.name}")
except Exception as e:
    print(f"❌ Error: {e}")