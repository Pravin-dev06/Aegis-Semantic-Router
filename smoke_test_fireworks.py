import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/glm-5p1",
    messages=[{
        "role": "user",
        "content": "Explain quantum computing in simple terms"
    }]
)

print(response.choices[0].message.content)