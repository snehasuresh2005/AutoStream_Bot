import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"Key found: {'Yes' if api_key else 'No'}")

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

print("Listing models with v1beta:")
try:
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
