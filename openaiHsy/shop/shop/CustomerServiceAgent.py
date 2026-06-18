import re

from tools.tool_desc import tools, get_function_details
import os
from openai import OpenAI


class CustomerServiceAgent:
    def __init__(self, client):
        self.client = client
        self.messages = []
        self.system_prompt = """
        您是电子商务平台的智能客服助理。有必要及时回答用户对产品的咨询。如果它与特定产品无关，您可以直接回答。
        将其输出为答案: [Your answer here].
       
        Example :
        Answer: 还有什么我可以帮你的吗?
        
        如果涉及到产品的具体信息，你会陷入思考、行动、观察的循环。
        用思维来描述你的分析过程。
        使用Action运行其中一个可用工具，然后等待观察。
        当你得到最终答案时, 将其输出为答案: [Your answer here].
        
        可用工具:
        1. query_by_product_name: 调用产品服务以检索与指定产品名称匹配或包含指定产品名称的产品列表，此功能可用于帮助客户通过在线平台或客户支持界面按名称查找产品。
        2. read_store_promotions: 查找与所提供产品名称相关的具体促销活动,此功能扫描文本文档，查找包含产品名称的任何促销条目.
        3. calculate: 结合产品的销售价格和优惠信息计算最终交易价格


        When using an Action, always format it as:
        Action: tool_name: argument1, argument2, ...

        Example :
        Human: 你们店里卖足球吗？如果你卖足球，现在有什么优惠政策？如果我现在买，最终会得到多少钱？
        Thought: 要回答这个问题，我需要先检查后台的数据库.
        Action: query_by_product_name: 足球

        Observation: 目前，我已经检查过这个球有库存， 知道它的价格是120元.

        Thought: 我需要进一步询问足球的优惠政策
        Action: read_store_promotions: 足球

        Observation: 足球目前的促销政策是：购买时可享受10%的折扣

        Thought: 现在需要把足球的售价和优惠政策结合起来，计算出最终的交易价格
        Action: calculate: 120 * 0.9

        Observation: 足球的最终价格是108.0元

        Thought: 我现在已经掌握了回答这个问题所需的所有信息.
        Answer:  根据您的询问，我们店确实有卖足球，目前的价格是120元。目前，我们为购买足球提供10%的折扣。因此，如果您现在购买，最终交易价格将为108元.
        """.strip()
        self.messages.append({"role": "system", "content": self.system_prompt})

        # __call__ 方法可以使得一个类的实例可以被像函数那样调用，提供了类实例的"可调用"能力。
        # 当使用类实例后面跟着括号并传递参数时，就会触发 __call__ 方法。

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        # print(f"【执行器调用大模型：{self.messages}】")
        response = self.execute()

        if not isinstance(response, str):
            raise TypeError(f"Expected string response from execute, got {type(response)}")
        self.messages.append({"role": "assistant", "content": response})
        return response

    def execute(self):

        chat_response = self.chat_completion_request(
            self.messages, tools=tools
        )
        resp = chat_response.choices[0].message.content
        if resp != None:
            return f"执行器返回结果：{resp}"
        else:
            return "执行器返回结果为空,当前没有正常的生成回复，请重新思考当前的问题，并再次进行尝试."

    def chat_completion_request(self, messages, tools=None, tool_choice=None, model="gpt-3.5-turbo"):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e


if __name__ == "__main__":
    client = OpenAI(
        api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
        base_url="https://api.chatanywhere.tech/v1"
    )
    agent = CustomerServiceAgent(client)
    query = "你们店里卖足球吗？如果你卖足球，现在有什么优惠政策？如果我现在买，最终会得到多少钱？"
    print(agent(query))
