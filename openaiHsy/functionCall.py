import io
import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)


def chat_completion_request(messages, tools=None, tool_choice=None, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        #      tool_choice={"type": "function", "function": {"name": "ccq_function"}}
        #     控制模型的行为，决定是否使用工具
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# 1. 定义函数

def ccq_function(data):
    """
    ccq算法函数，该函数定义了数据集计算过程
    :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
    :return：ccq_function函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
    """
    if isinstance(data, dict):
        data = data.get('data', '')
    data = io.StringIO(data)
    df_new = pd.read_csv(data, sep='\s+', index_col=0)
    res = df_new * 10
    return json.dumps(res.to_string())


# 2.定义function 描述  定义工具 ， 多个工具
tools = [{
    "type": "function",
    "function": {"name": "ccq_function",
                 "description": "用于执行ccq算法函数，定义了一种特殊的数据集计算过程",
                 "parameters": {"type": "object",
                                "properties": {"data": {"type": "string",
                                                        "description": "执行ccq算法的数据集"},
                                               },
                                "required": ["data"],
                                },
                 }
}]

# 4. 创建消息
df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
df_str = df.to_string()
print(df_str)

messages = [
    {"role": "system", "content": "数据集data：%s，数据集以字符串形式呈现" % df_str},
]
# 5. 创建completion，绑定工具，
chat_response = chat_completion_request(
    messages, tools=tools
)
# 6. 解析结果，获取函数调用
tool_calls = chat_response.choices[0].message.tool_calls
print(chat_response.choices[0].message.tool_calls)
# 7.执行函数
for tool_call in tool_calls:
    tool_call_id = tool_call.id
    function_call = tool_call.function
    function_args = json.loads(function_call.arguments)
    # 在全局作用域中查找这个函数
    function_to_call = globals().get(function_call.name)
    if function_to_call:
        function_result = function_to_call(**function_args)  # 使用 ** 解包参数
        print(function_result)
    else:
        print(f"Function {function_call.name} not found")
        function_result = None

    # 8. 创建completion，绑定工具，把函数调用结果加入到对话历史中
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_call.name,
                    "arguments": function_call.arguments
                }
            }]
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": function_call.name,
            "content": function_result
        }
    )
messages.append(
     {"role": "user", "content": "数据集data ，上执行ccq算法?"}
)

print(messages)

chat_response = chat_completion_request(
    messages, tools=tools
)
print("===回复===")
print(chat_response.choices[0].message.content)
