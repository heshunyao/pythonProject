import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class InterviewLLM:

    def get_llm(self):
        # ---------------指定模型----gpt-3.5-turbo-----------
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
        model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        return model

    # 嵌入
    def get_embedding(self):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY").strip()
        embedding = OpenAIEmbeddings()
        return embedding
