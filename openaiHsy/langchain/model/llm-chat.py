from langchain_openai import ChatOpenAI

from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

# 初始化聊天模型
chat = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# 示例1：单次对话
print("=== 示例1：单次对话 ===")
resp = chat.invoke([HumanMessage(content="把这个句子从英语翻译成法语。我喜欢编程.")])
print(resp.content)

# 示例2：带系统消息的单次对话
print("\n=== 示例2：带系统消息的单次对话 ===")
messages = [
    SystemMessage(content="你是一个优秀的翻译助理，可以将英语翻译成法语."),
    HumanMessage(content="I love programming.")
]
print(chat.invoke(messages).content)

# 示例3：多轮对话
print("\n=== 示例3：多轮对话 ===")
# 初始化对话历史
conversation_history = [
    SystemMessage(content="你是一个友好的AI助手，擅长回答各种问题。请用简洁的语言回答。")
]

def chat_with_history(user_input):
    # 添加用户消息到历史记录
    conversation_history.append(HumanMessage(content=user_input))
    # 获取AI响应
    response = chat.invoke(conversation_history)
    # 将AI响应添加到历史记录
    conversation_history.append(AIMessage(content=response.content))
    return response.content

# 进行多轮对话
print("AI: 你好！我是你的AI助手，请问有什么我可以帮你的吗？")
print("用户: 请介绍一下Python语言的特点")
print("AI:", chat_with_history("请介绍一下Python语言的特点"))
print("\n用户: 它适合用来做什么？")
print("AI:", chat_with_history("它适合用来做什么？"))
print("\n用户: 谢谢你的解释！")
print("AI:", chat_with_history("谢谢你的解释！"))

# 打印完整的对话历史
print("\n=== 完整对话历史 ===")
for message in conversation_history:
    if isinstance(message, SystemMessage):
        print(f"系统: {message.content}")
    elif isinstance(message, HumanMessage):
        print(f"用户: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
