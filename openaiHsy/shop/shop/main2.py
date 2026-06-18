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

    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }


    # query = input("输入您的问题或输入 '退出' 来结束: ")
    # 给智能体

    while True:
        # 示例结果处理
        count = 0
        # query = "你们店有足球吗？如果你卖足球，现在有什么优惠政策？如果我现在买，最终会得到多少钱？"
        query = input("输入您的问题或输入 '退出' 来结束: ")
        while count < 5:
            result = agent(query)
            print(result)

            print("------1-调用agent end-----")
            # 使用更灵活的正则表达式匹配新格式
            action_pattern = r"Action:\s*(\S+)"
            # 读取行
            lines = result.split('\n')
            # 定义一个退出标识
            flag = False
            for line in lines:
                print(f"------1-{line}---")
                # 查找匹配
                action_match = re.search(action_pattern, line)
                # 提取并打印结果
                if action_match:
                    print(f"Action-{count+1}行: {line}")
                    action_parts = line.split("Action:", 1)[1].strip().split(": ", 1)
                    tool_name = action_parts[0]
                    tool_args = action_parts[1] if len(action_parts) > 1 else ""
                    print(f"工具名称: {tool_name}")
                    print(f"工具参数: {tool_args}")
                    print("----2. --取到 tools name and params --end ----")
                    if tool_name in tools:
                        try:
                            observation = tools[tool_name](tool_args)
                            print(f"工具返回结果: {observation}")
                            query = f"Observation: {observation} "
                        except Exception as e:
                            print(f"工具调用失败: {e}")
                            query = f"Observation: 工具调用失败: {str(e)}"
                    else:
                        print(f"未知的工具名称: {tool_name}")
                        query = f"Observation: 未知的工具名称: '{tool_name}'"
                elif "Answer:" in line or "答复:" in line:
                    print(f"客服回复：{line.split('Answer:', 1)[1].strip()}")
                    print("-----3.结束-----")
                    flag = True
                    break
                else:
                    # print("没有找到相关信息")
                    query = "Observation: 未找到有效操作或答案,请提供明确的行动或答案."
            if flag:
                break
            count += 1


if __name__ == '__main__':
    main()
# 应用
# langchain  , springai,  mcp
# 6.14 --模型训练---微调