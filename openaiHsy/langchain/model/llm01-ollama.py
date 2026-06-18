from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
import requests
import sys

# def check_ollama_service():
#     try:
#         response = requests.get("http://localhost:11434/api/tags")
#         if response.status_code == 200:
#             print("Ollama服务正在运行")
#             return True
#         else:
#             print(f"Ollama服务返回状态码: {response.status_code}")
#             return False
#     except requests.exceptions.ConnectionError:
#         print("无法连接到Ollama服务，请确保服务已启动")
#         return False
#
# # 检查服务状态
# if not check_ollama_service():
#     print("请先启动Ollama服务：ollama serve")
#     sys.exit(1)

template = """ 
Question: {question}
Answer: 让我们一步一步地思考.
"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)


try:
    llm = OllamaLLM(
        model="qwen2.5:0.5b",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    llm_chain = prompt | llm
    question = "什么是mybatis?"
    print("正在发送请求...")
    reps = llm_chain.invoke({"question": question})
    print("收到响应：")
    print(reps)
except Exception as e:
    print(f"发生错误: {str(e)}")
    print("请确保：")
    print("1. Ollama服务正在运行")
    print("2. 已下载qwen2.5:0.5b模型 (使用命令: ollama pull qwen2.5:0.5b)")
    print("3. 模型名称正确")
