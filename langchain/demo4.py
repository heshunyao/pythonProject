from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field


def getllm():
    try:
        ollamallm = OllamaLLM(
            model="qwen2.5:0.5b",
            temperature=0.7,
            base_url="http://127.0.0.1:11434"
        )
        # ollamallm = Ollama(
        #     model="qwen2.5:0.5b",
        #     temperature=0.7,
        #     base_url="http://127.0.0.1:11434"
        # )
        # print(ollamallm,"ollamallm")
        print(ollamallm,"ollamallm")
        return ollamallm
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查ollama qwen2.5:0.5b 是否安装并运行")


class AnswerParser(BaseModel):
    question: str = Field(...,  description="question")
    answer: str = Field(...,  description="answer")
    source: str = Field(...,  description="source")


llm = getllm()
print(llm,"llm")

prompt = ChatPromptTemplate.from_messages([
    ("system", "将回复的内容进行JSON 格式输出, 包含question、answer和source"),
    ("human", "解释: {question}"),
])

output_parser = JsonOutputParser(pydantic_object=AnswerParser)

_input = prompt.format(question="MCP")
output = llm.invoke(_input)
print("原始输出：", output)

resp = output_parser.parse(output)
print("结构化结果：", resp)
