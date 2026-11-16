import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model in ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]:
    t0 = time.perf_counter()
    try:
        r = client.models.generate_content(
            model=model, contents="Dịch sang tiếng Anh: khóa bảng nào chặn được SELECT?")
        print(f"OK   {model} ({time.perf_counter() - t0:.1f}s): {r.text.strip()[:150]}")
    except Exception as e:
        print(f"LỖI  {model}: {str(e)[:200]}")