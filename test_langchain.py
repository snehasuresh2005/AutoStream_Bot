from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

print("Testing ChatGoogleGenerativeAI with gemini-2.0-flash...")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
try:
    res = llm.invoke("Hi")
    print(res.content)
except Exception as e:
    print(f"Error: {e}")
