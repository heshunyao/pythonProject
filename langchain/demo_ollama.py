from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from typing import Optional, List, Any
import requests


class OllamaStreamLLM(LLM):
    model: str = "deepseek-r1:7b"
    base_url: str = "http://localhost:11434/api/generate"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 非流式 fallback（不推荐）
        response = requests.post(
            self.base_url,
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    def _stream(self, prompt: str, stop: Optional[List[str]] = None):
        response = requests.post(
            self.base_url,
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True,
        )
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")

    @property
    def _llm_type(self) -> str:
        return "custom-ollama-stream"


# 使用 LangChain 调用流式接口
llm = OllamaStreamLLM()

# 获取响应（非流式）
print("非流式结果：")
print(llm("给我一份简单的java代码"))

# 流式结果
print("\n流式结果：")
for chunk in llm._stream("给我一份简单的java代码"):
    print(chunk, end="", flush=True)
