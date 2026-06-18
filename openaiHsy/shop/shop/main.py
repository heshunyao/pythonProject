import re

from tools.calc import calculate
from tools.query_product_data import query_by_product_name
from tools.read_promotions import read_store_promotions
import os
from openai import OpenAI
from CustomerServiceAgent import CustomerServiceAgent


def main():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )
    agent = CustomerServiceAgent(client)
    query = "你们店的足球吗？如果你卖足球，现在有什么优惠政策？如果我现在买，最终会得到多少钱？"

    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }

    count = 0
    # query = input("输入您的问题或输入 '退出' 来结束: ")
    # 给智能体
    while count < 5:
        # 示例结果处理
        # query = input("输入您的问题或输入 '退出' 来结束: ")
        # if query.lower() == '退出':
        #     break
        result = agent(query)
        print(result)
        print("------1------")
        # 使用更灵活的正则表达式匹配新格式
        action_pattern = r"Action: functions\.(\w+):\s*{\s*product_name:\s*\"([^\"]+)\"\s*}"
        if match := re.search(action_pattern, result):
            tool_name, product_name = match.groups()
            print(f"工具名称: {tool_name}")
            print(f"产品名称: {product_name}")
            print("----2--------")

            if tool_name in tools:
                # 调用工具并获取响应
                try:
                    observation = tools[tool_name](product_name)
                    print(f"工具返回结果: {observation}")
                    query = f"Observation: {observation} "
                except Exception as e:
                    print(f"工具调用失败: {e}")
                    query = f"Observation: 工具调用失败: {str(e)}"

            else:
                print(f"未知的工具名称: {tool_name}")
                query = f"Observation: 未知的工具名称: '{tool_name}'"
        elif "Answer:" in result or "答复:" in result:
            # 处理英文和中文的回复格式
            if "Answer:" in result:
                response = result.split('Answer:', 1)[1].strip()
            else:
                response = result.split('答复:', 1)[1].strip()
            print(f"客服回复：{response}")
            break
        else:
            print("未找到有效的工具调用")
            # print(result)
            query = "Observation: 未找到有效的操作或答案，请提供明确的行动或答案"

        count += 1


if __name__ == '__main__':
    main()
