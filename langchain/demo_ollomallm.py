from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate

# 初始化 Ollama 模型
def get_llm():
    try:
        ollama_llm = OllamaLLM(
            model="deepseek-coder:6.7b",  # 确保你已经用 ollama pull 下载该模型
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        return ollama_llm
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查 ollama 是否启动，模型是否安装。")

# 定义 Prompt 模板（纯文本）
prompt_template = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template="你是一个乐于助人的翻译助手，能精确地将 {input_language} 翻译为 {output_language}。以下是需要翻译的内容：\n{text}"
)

# 获取模型实例
ollama_llm = get_llm()

# 循环对话
while True:
    user_text = input("请输入需要翻译的英文句子 (输入 exit 退出)：")
    if user_text.strip().lower() == "exit":
        break

    # 格式化 prompt
    final_prompt = prompt_template.format(
        input_language="英文",
        output_language="中文",
        text=user_text
    )

    # 模型调用
    response = ollama_llm.invoke(final_prompt)
    print("翻译结果：", response)
