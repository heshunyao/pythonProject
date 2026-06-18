from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

# 定义结构化数据格式
class AnswerParser(BaseModel):
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="回答")
    source: str = Field(..., description="来源")

# 初始化模型
llm = Ollama(model="qwen2.5:0.5b")

# 定义 prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个问答助手，请将回答格式化为JSON，包含question、answer、source"),
    ("human", "解释: {question}"),
])

# 构造输入
_input = prompt.format_messages(question="MCP")
output = llm.invoke(_input)

# 输出原始结果
print("原始输出：", output)

# 尝试结构化解析
parser = JsonOutputParser(pydantic_object=AnswerParser)
structured = parser.parse(output)
print("结构化输出：", structured)
