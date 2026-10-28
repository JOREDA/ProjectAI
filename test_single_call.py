import time
import data_loader
from groq import Groq

client = Groq(api_key=data_loader.get_groq_api_key())
t0 = time.time()
resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": 'Respond with JSON: {"status": "ok"}'}],
    response_format={"type": "json_object"},
    temperature=0
)
t1 = time.time()
print("openai/gpt-oss-120b took:", round(t1 - t0, 2), "s. Response:", resp.choices[0].message.content)
