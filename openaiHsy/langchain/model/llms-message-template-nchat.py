from langchain_community.chat_models import ChatOpenAI
from langchain_ollama.llms import OllamaLLM

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def getllm():
    try:
        ollamallm = OllamaLLM(
            model="qwen2.5:0.5b",
            temperature=0.7,
            base_url="http://127.0.0.1:11434"
        )
        return ollamallm
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查ollama qwen2.5:0.5b 是否安装并运行")


# 定义消息模板
system_template = "你是一个乐于助人的翻译助手，能精确的将 {input_language} 翻译成 {output_language}。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

print("多轮对话：-------------")

# 初始化历史消息列表
history_messages = []

ollamallm = getllm()


def chat_with_history(user_input, input_language="English", output_language="中文"):
    # 创建当前对话的消息模板
    messages = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

    # 格式化消息
    formatted_messages = messages.format_messages(
        input_language=input_language,
        output_language=output_language,
        text=user_input
    )

    # 获取响应
    resp = ollamallm.invoke(formatted_messages)

    # 更新历史记录
    history_messages.append(HumanMessage(content=user_input))
    history_messages.append(AIMessage(content=resp))

    return resp


while True:
    user_input = input("请输入：")
    if user_input == "exit":
        break
    print("收到响应：")
    resp = chat_with_history(user_input)
    print(resp)

print("----打印历史消息---------------")
for msg in history_messages:
    if isinstance(msg, HumanMessage):
        print(f"用户说：{msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"AI说：{msg.content}")
    else:
        print(f"系统消息：{msg.content}")
