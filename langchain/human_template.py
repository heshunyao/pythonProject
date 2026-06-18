from langchain_community.chat_models import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOllama
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
        ollamallm = ChatOllama(
            model="deepseek-r1:7b",
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        return ollamallm
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查ollama deepseek-r1:7b 是否安装并运行")


template = "你是一个乐于助人的翻译助手，能精确的将  {input_language}翻译{output_language}."
history_messages = [
    SystemMessage(content="你是一个乐于助人的翻译助手"),
    HumanMessage(content="user_input"),
    AIMessage(content="翻译结果"),
]

print("多轮对话：-------------")

ollamallm = getllm()
# def chat_with_history(user_input):
#
#     history_messages.append(HumanMessage(content= user_input))
#     resp = ollamallm.invoke(history_messages)
#     history_messages.append(AIMessage(content=resp))
#     return resp

system_template = "你是一个乐于助人的翻译助手，能精确的将 {input_language} 翻译为 {output_language}。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

while True:
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    messages = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 获取格式化后的消息的聊天完成结果
    resp = ollamallm.invoke(messages.format_prompt(input_language="English", output_language="中文",
                                                   text="I love programming.").to_messages())
    print(resp, "resp")
    # resp = ollamallm.invoke(messages)
# while True:
#     user_input = input("请输入：")
#     if user_input == "exit":
#         break
#     print("收到响应：")
#     resp = chat_with_history(user_input)
#     print(resp)
