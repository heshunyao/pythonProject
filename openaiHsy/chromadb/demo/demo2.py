from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field
import json
import pymysql


# ---------------- 工具函数 ------------------

def insert_user(name: str, age: int) -> str:
    try:
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='sys', charset='utf8mb4')
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (name, age) VALUES (%s, %s)", (name, age))
            conn.commit()
            return f"成功插入用户: {name}，年龄: {age}"
    except Exception as e:
        return f"插入失败: {e}"
    finally:
        if 'conn' in locals():
            conn.close()


def delete_user(name: str) -> str:
    try:
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='sys', charset='utf8mb4')
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE name = %s", (name,))
            conn.commit()
            if cursor.rowcount > 0:
                return f"成功删除用户: {name}"
            else:
                return f"未找到用户: {name}"
    except Exception as e:
        return f"删除失败: {e}"
    finally:
        if 'conn' in locals():
            conn.close()


# ---------------- 输出解析器 ------------------

class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # 直接解析 LLM 输出的 JSON 字符串
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析 JSON 失败: {e}")


# ---------------- Prompt 模板 ------------------

prompt_template = """
你是一个数据库助手，根据用户输入判断是插入还是删除用户操作。

输入格式例子:
- 插入: 请将姓名为李雷、年龄为30的用户插入数据库
- 删除: 删除姓名为李雷的用户

请严格按照以下 JSON 格式返回：
{{
  "action": "insert_user" 或 "delete_user",
  "parameters": {{
    "name": "用户姓名",
    "age": 整数年龄(仅插入时需要)
  }}
}}

如果是删除操作，不需要 age 字段。

用户输入: {input}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
output_parser = JsonOutputParser()
llm = ChatOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1",
    temperature=0
)

# --------- 构建 chain -----------

chain = prompt | llm | output_parser

# ---------------- 主流程 ------------------

def process(user_input: str) -> str:
    # prompt_text = prompt.format(input=user_input)
    # response = llm.predict(prompt_text)
    #
    # # 解析 LLM 输出 JSON
    # parsed = output_parser.parse(response)
    #
    # action = parsed.get("action")
    # params = parsed.get("parameters", {})

    # 1. 用 chain 生成解析后的 dict
    parsed = chain.invoke({"input": user_input})

    action = parsed.get("action")
    params = parsed.get("parameters", {})
    if action == "insert_user":
        return insert_user(params.get("name"), params.get("age"))
    elif action == "delete_user":
        return delete_user(params.get("name"))
    else:
        return "无法识别操作"


# ---------------- 测试 ------------------

if __name__ == "__main__":
    # text1 = "请将姓名为李雷、年龄为30的用户插入数据库"
    # print(process(text1))
    #
    # text2 = "删除姓名为李雷的用户"
    # print(process(text2))
    while True:
        user_input = input("请输入你的指令（输入 exit 退出）：\n> ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("已退出。")
            break
        try:
            result = process(user_input)
            print(result,"===========")
            # print("\n🧠 回答：", result["output"])
            # print("📜 中间步骤：", result["intermediate_steps"])
        except Exception as e:
            print("❌ 出错：", e)
