from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_text_splitters import CharacterTextSplitter
from langchain.agents import tool, initialize_agent, AgentType
import pymysql


# 数据库连接配置
def get_connection():
    return pymysql.connect(
        host='localhost',      # 修改为你的数据库地址
        port=3306,             # 修改为你的端口
        user='root',           # 修改为你的用户名
        password='your_pass',  # 修改为你的密码
        database='your_db',    # 修改为你的数据库名
        charset='utf8mb4'
    )

@tool("insert_to_db", description="插入用户数据，参数为姓名和年龄")
def insert_to_db(params: dict) -> str:
    name = params["name"]
    age = params["age"]
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO users (name, age) VALUES (%s, %s)"
        cursor.execute(sql, (name, age))
        conn.commit()
        return f"成功插入用户：{name}，年龄：{age}"
    except Exception as e:
        return f"插入用户失败：{str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool(description="删除用户，传入参数为 {'name': '张三'}")
def delete_from_db(params: dict) -> str:
    name = params["name"]
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = "DELETE FROM users WHERE name = %s"
        cursor.execute(sql, (name,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"未找到名为 {name} 的用户"
        return f"成功删除用户：{name}"
    except Exception as e:
        return f"删除用户失败：{str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool(description="查询用户，传入参数为 {'name': '张三'}")
def query_user_from_db(params: dict) -> str:
    name = params["name"]
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = "SELECT id, name, age FROM users WHERE name = %s"
        cursor.execute(sql, (name,))
        results = cursor.fetchall()
        if not results:
            return f"未找到名为 {name} 的用户"
        return "\n".join([f"ID: {row[0]}, 姓名: {row[1]}, 年龄: {row[2]}" for row in results])
    except Exception as e:
        return f"查询用户失败：{str(e)}"
    finally:
        cursor.close()
        conn.close()
@tool(description="对两个数字执行加法。传入参数应为一个字典，如 {'x': 1, 'y': 2}")
def add(params: dict) -> float:
    return params["x"] + params["y"]

@tool(description="对两个数字执行减法（x - y）。传入参数应为一个字典，如 {'x': 5, 'y': 2}")
def subtract(params: dict) -> float:
    return params["x"] - params["y"]

# 向量数据库部分（可选加载 PDF 文档并嵌入）
loader = PyPDFLoader("../data/HCIA-AI.pdf")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)
vdb = Chroma.from_documents(documents=docs, embedding=embeddings)

# 初始化 LLM
llm = ChatOpenAI(
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)

tools = [add, subtract, insert_to_db, delete_from_db, query_user_from_db]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    memory=memory
)

# 与用户对话
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
