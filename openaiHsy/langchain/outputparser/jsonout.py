from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

#  Pydantic 模型可以用于验证前端传来的数据是否符合预期的格式和类型
#  在FastAPI等框架中,pydantic模型直接用于请求参数解析,确保输入数据的合法性
class AnswerObject(BaseModel):
    answer: str = Field(..., description="答案")
    source: str = Field(..., description="来源")


llm = ChatOpenAI(temperature=0)
# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "将回答格式化为 JSON，包含答案和来源。"),
    ("human", "解释人工智能：{question}")
])
# 创建LCEL链
parser = JsonOutputParser(obj=AnswerObject)
chain = prompt | llm | parser
print(chain.invoke({"question": "什么是人工智能？"}))

