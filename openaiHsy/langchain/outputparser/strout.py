from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化 LLM
llm = ChatOpenAI(temperature=0)

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，请回答问题。"),
    ("human", "{question}")
])

# 创建 LCEL 链
chain = prompt | llm | StrOutputParser()

# 调用链
response = chain.invoke({"question": "什么是人工智能？"})
print(response)
