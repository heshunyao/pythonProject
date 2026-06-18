from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)

# 创建 ChatOpenAI 实例，使用自定义 base_url（兼容 API 代理）
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    openai_api_base="https://api.chatanywhere.tech/v1"
)

# 构造消息序列
messages = [
    SystemMessage(content="你好")
]
messages.append(HumanMessage(content="给我一份java代码，随便"))
messages.append(HumanMessage(content="再给我一份python代码，随便"))

# 调用 LLM 并输出结果
response = llm.invoke(messages)
print(response.content)
