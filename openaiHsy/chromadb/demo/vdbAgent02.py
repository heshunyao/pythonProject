from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from langchain_core.runnables import RunnablePassthrough

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_process_document(pdf_path: str):
    try:
        logger.info("开始加载PDF文档...")
        loader = PyPDFLoader(pdf_path)
        data = loader.load()

        logger.info("开始分割文档...")
        # 使用更合理的分割参数
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 更合理的块大小
            chunk_overlap=50,  # 10% 的重叠
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        splits = text_splitter.split_documents(data)
        logger.info(f"文档分割完成，共 {len(splits)} 个块")
        return splits
    except Exception as e:
        logger.error(f"文档处理过程中出错: {str(e)}")
        raise


def create_retriever(splits):
    try:
        logger.info("初始化向量数据库...")
        embedding = OpenAIEmbeddings(
            api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
            base_url="https://api.chatanywhere.tech/v1"
        )
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

        logger.info("配置检索器...")
        base_retriever = vectordb.as_retriever(
            search_kwargs={"k": 3}  # 返回最相关的3个文档
        )

        llm = ChatOpenAI(api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
                         base_url="https://api.chatanywhere.tech/v1", temperature=0.7)  # 降低温度以获得更稳定的结果

        # 配置 MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        return retriever, llm
    except Exception as e:
        logger.error(f"检索器创建过程中出错: {str(e)}")
        raise


def main():
    try:
        # 加载和处理文档
        splits = load_and_process_document("../data/HCIA-AI.pdf")

        # 创建检索器和LLM
        retriever, llm = create_retriever(splits)

        # 设置提示模板
        prompt_template = """
        您是一个专业的人工智能助手。请基于以下上下文回答问题。
        如果上下文不足以回答问题，请直接说明"根据提供的上下文无法回答这个问题"。
        
        问题: {query}
        上下文: {context}
        
        请提供清晰、准确的回答：
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context"]
        )

        # 处理查询
        query = "求职意向是什么?"
        logger.info(f"处理查询: {query}")

        docs = retriever.get_relevant_documents(query)
        logger.info(f"检索到 {len(docs)} 个相关文档")

        chain = (
                {"query": RunnablePassthrough(), "context": lambda x: docs}
                | prompt
                | llm
                | StrOutputParser()
        )
        response = chain.invoke(query)

        logger.info("生成回答完成")
        print("\n回答:", response)

    except Exception as e:
        logger.error(f"程序执行过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
