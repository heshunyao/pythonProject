from langchain_ollama import OllamaLLM

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chatllm = OllamaLLM(
    model="qwen2.5:0.5b",
    temperature=0,
    base_url="http://localhost:11434"
)

template = "你是一个乐于助人的翻译助手，能把英语翻译成中文"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = chat_prompt | chatllm
# 从格式化的消息中获取聊天完成结果
print(chain.invoke("I love programming."))
