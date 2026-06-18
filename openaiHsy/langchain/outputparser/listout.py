from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 初始化解析器
parser = CommaSeparatedListOutputParser()

# 初始化 LLM
llm = ChatOpenAI(temperature=0)

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "列出答案，用逗号分隔：{format_instructions}"),
    ("human", "{question}")
])

chain = prompt | llm | parser
print(chain.invoke({"question": "列出 5 个最流行的编程语言。", "format_instructions": parser.get_format_instructions()}))
