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

# chatllm = ChatOpenAI(temperature=0.9 )
try:
    ollamallm = OllamaLLM(
        model ="qwen2.5:0.5b",
        temperature = 0.7,
        base_url="http://127.0.0.1:11434"
    )
    print("正在发送请求....")

    template = "你是一个乐于助人的翻译助手，能精确的将  {input_language}翻译{output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    messages = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 获取格式化后的消息的聊天完成结果
    resp = ollamallm.invoke(messages.format_prompt(input_language="English", output_language="中文",
                                   text="I love programming.").to_messages())
    # resp = ollamallm.invoke(messages)
    print("收到响应：")
    print(resp)
except Exception as e:
    print(f"发生错误: {e}")
    print("请检查ollama qwen2.5:0.5b 是否安装并运行")



# resp = chatllm([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# print(resp)
# AIMessage(content="J'aime programmer.", additional_kwargs={})

# resp = chatllm.invoke(messages)
# print(resp)