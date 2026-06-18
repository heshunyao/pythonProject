from openai import OpenAI

# openAi
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)

# client = OpenAI(
#     api_key="sk-35122065cc714bfd920ff1628b0055aa",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# 对话上下文
messages = [
    {"role": "system", "content": "你是一个的资深的各类编程语言面试官。"}
]

while True:
    user_input = input("\n你说：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        break

    # 添加用户消息
    messages.append({"role": "user", "content": user_input})

    print("AI：", end="", flush=True)

    # 流式响应
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    full_response = ""

    for chunk in response:
        choices = chunk.choices
        if choices and choices[0].delta.content:
            print(choices[0].delta.content, end="", flush=True)
            full_response += choices[0].delta.content

    print()  # 换行
    # 添加 AI 回答到上下文中
    messages.append({"role": "assistant", "content": full_response})
