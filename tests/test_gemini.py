import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IkCH28v1pGcqRYqIhVs16vacOzG14ZYjHSS6lHU-NOeQ")
client = genai.Client(api_key=API_KEY)

target_model = "gemini-3.8-flash"

print(f"Testing generation on {target_model}...")
try:
    response = client.models.generate_content(
        model=target_model,
        contents="Briefly introduce yourself as an expert plant pathologist advisor in two sentences."
    )
    print("\n✅ Success! Connected using:", target_model)
    print("Response:\n", response.text)
except Exception as e:
    print(f"⚠️ Failed: {e}")