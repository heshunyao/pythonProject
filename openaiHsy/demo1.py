from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "你好"},
        {"role": "user", "content": "给我一份java代码，随便"},
        {"role": "user", "content": "再给我一份python代码，随便"},
    ]
)

print(response.choices[0].message.content)
