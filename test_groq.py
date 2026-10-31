"""Run: py test_groq.py  -> checks your Groq key and the model the app will use."""
import os
key = os.getenv("GROQ_API_KEY")
if not key: raise SystemExit("GROQ_API_KEY is not set in this window.")
import explain
from openai import OpenAI
try:
    m = explain._groq_model(key, "llama-3.1-8b-instant"); print("Model:", m)
    r = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1").chat.completions.create(
        model=m, messages=[{"role": "user", "content": "Say hi in 3 words"}], max_tokens=20)
    print("WORKS:", r.choices[0].message.content)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:400])
