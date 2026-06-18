from langchain_ollama import OllamaLLM

# 初始化 Ollama 本地模型
llm = OllamaLLM(
    model="qwen2.5:0.5b",  # 或 qwen2.5:0.5b 具体看你本地装了哪个
    base_url="http://localhost:11434",  # 默认本地 Ollama 服务地址
)

# 发送 prompt 给模型
response = llm("用一句话解释什么是量子力学")

# 打印模型回复
print("模型回复：", response)
