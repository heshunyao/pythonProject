from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1.pdf文档加载
# pip install pypdf
from langchain_text_splitters import CharacterTextSplitter
from langchain.agents import tool, initialize_agent, AgentType
import pymysql


# 通用数据库连接函数（根据你的配置调整）
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="sys",  # 替换为你的数据库名
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

@tool
def query_user_from_db(name: str) -> str:
    """根据名字查询用户信息"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE name = %s"
            cursor.execute(sql, (name,))
            result = cursor.fetchall()
        if result:
            return f"查询结果：{result}"
        else:
            return f"未找到名为{name}的用户。"
    except Exception as e:
        return f"查询失败: {str(e)}"
    finally:
        conn.close()

@tool
def insert_to_db(name: str, age: int) -> str:
    """向 users 表中插入一条记录"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            sql = "INSERT INTO users (name, age) VALUES (%s, %s)"
            cursor.execute(sql, (name, age))
        conn.commit()
        return f"成功插入用户：{name}, 年龄：{age}"
    except Exception as e:
        return f"插入失败: {str(e)}"
    finally:
        conn.close()

@tool
def delete_from_db(name: str) -> str:
    """删除 users 表中指定名字的用户"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            sql = "DELETE FROM users WHERE name = %s"
            cursor.execute(sql, (name,))
        conn.commit()
        return f"成功删除用户：{name}"
    except Exception as e:
        return f"删除失败: {str(e)}"
    finally:
        conn.close()
@tool
def add(x: float, y: float) -> float:
    """返回两个数字的和"""
    return x + y

@tool
def subtract(x: float, y: float) -> float:
    """返回两个数字的差（x - y）"""
    return x - y

loader = PyPDFLoader("../data/HCIA-AI.pdf")
data = loader.load()
# print(len(data))
# print(data)
# 2.分切割
text_splitters = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitters.split_documents(data)
print(docs)
# 3. 创建向量化模型
# embedding = OpenAIEmbeddings()
# 3.把分切后数据文档转成向量化，存储到chroma 向量数据库中
embeddings = OpenAIEmbeddings(
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)
vdb = Chroma.from_documents(documents=docs, embedding=embeddings)
query="人工智能的四要素?"
llm = ChatOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1",
    temperature=0
)

# # 4检索
# # 使用 LLM 多样化查询增强检索器（MultiQueryRetriever）
# # 它会调用大模型自动生成多个语义不同但等价的查询，增强检索结果的相关性和覆盖面
# retriever_llm = MultiQueryRetriever.from_llm(
#     llm=llm,                     # 传入你定义的大语言模型实例（如 ChatOpenAI 或 OpenAI）
#     retriever=vdb.as_retriever(), # 将 Chroma 向量数据库转换为一个基本的 retriever
#     prompt=MultiQueryRetriever.get_default_prompt().partial(num_queries=1)  # 限制数量
# )
# # 仅一次查询
# # retriever = vdb.as_retriever()
# docs = retriever_llm.get_relevant_documents(query=query)
# # print(docs)
#
# prompt = PromptTemplate.from_template(
#     """
# 你是一位智能AI助手，擅长从技术文档中提取关键信息并准确回答问题。
#
# 请根据以下文档内容回答用户问题：
# ----------------------
# {context}
# ----------------------
#
# 问题：{query}
#
# 请用简洁、准确的语言作答。
# """
# )
# # 构造包含 query 和 context 的 chain 输入
# chain = (
#     {"query": lambda x: x["query"], "context": lambda x: "\n".join([doc.page_content for doc in x["context"]])}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# # 调用 chain
# response = chain.invoke({"query": query, "context": docs})
# # print(response)
# print("============================")

system_prompt = """
你是一个具备数据库操作能力和基本数学运算能力的智能助手。
你可以根据用户的自然语言指令自动调用合适的工具，包括：
- 数据库插入（insert_to_db），若插入只要执行一次
- 数据库删除（delete_from_db），若插入只要执行一次
- 数学加法（add）
- 数学减法（subtract）
- 数据库插入(query_user_from_db)

请准确理解用户的意图，并调用恰当的工具执行操作。
在调用数据库操作时，请确保你理解了用户提到的名字和年龄，并以结构化方式传递参数。
如果用户的指令模糊不清，请不要猜测，而是请求澄清。
"""

# 添加工具
tools = [add, subtract, insert_to_db, delete_from_db,query_user_from_db]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# 初始化代理（Agent）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # 使用支持 function-calling 的 agent 类型
    verbose=True,
    return_intermediate_steps=True,
    agent_kwargs={"system_message": system_prompt},
    memory=memory
)


# result = agent.run("帮我计算 23.5 加上 19.2 减去 3 是多少？")
# print(result)

# response = agent.invoke("插入一个叫张三的用户，年龄25,只执行一次")
# response2 = agent.invoke("看一下这个张三用户")
# print(response2["output"])
# print("中间步骤：", response["intermediate_steps"])
# print(agent.run("请把名字叫小王的用户从数据库中删除"))

while True:
    user_input = input("请输入你的指令（输入 exit 退出）：\n> ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("已退出。")
        break
    try:
        result = agent.invoke(user_input)
        print("\n🧠 回答：", result["output"])
        print("📜 中间步骤：", result["intermediate_steps"])
    except Exception as e:
        print("❌ 出错：", e)

