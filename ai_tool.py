import os
import io
import json
import pandas as pd
from openai import OpenAI

# 创建一个LLM 客户端
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
    base_url="https://api.chatanywhere.tech/v1"
)

# data = """
#     |   A   |   B   |   C   |
#     |:-----:|:-----:|:-----:|
#     |   1   |   2   |   3   |
#     |   4   |   5   |   6   |
#     |   7   |   8   |   9   |
# """

df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4]})
data = df.to_string()
print(data)


def a_function(data):
    """
    A算法函数，该函数定义了数据集计算过程
    :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
    :return：sunwukong_function函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
    """
    data = io.StringIO(data)
    df_new = pd.read_csv(data, sep='\s+', index_col=0)
    res = df_new * 10
    return json.dumps(res.to_string())


# 定义function
sunwukong = {
    "type": "function",
    "function": {"name": "a_function",
                 "description": "用于执行A算法函数，定义了一种特殊的数据集计算过程",
                 "parameters": {"type": "object",
                                "properties": {"data": {"type": "string",
                                                        "description": "A算法的数据集"},
                                               },
                                "required": ["data"],
                                },
                 }
}

tools = [sunwukong]

messages = [
    {"role": "system", "content": "你是一个数据处理助手，数据集data:%s" % data},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
    {"role": "user", "content": "将原来的数据集data 上执行A算法"},
]

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    temperature=0,
)
print(resp.choices[0].message.tool_calls)
print("=======第二次============")
tool_calls = resp.choices[0].message.tool_calls[0]
# 获取函数名
fn = tool_calls.function.name
tool_call_id = tool_calls.id
param = tool_calls.function.arguments
params_dict = json.loads(param)
print(params_dict, "params_dict")
tools_reps = a_function(params_dict["data"])
print(tools_reps, "tools_reps")
messages.append({
    "role": "user",
    "tool_call_id": tool_call_id,
    "name": fn,
    "content": tools_reps
})
# 二次调用LLM
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,

)
print(f"two:{response2.choices[0].message}")
messages.append({
    "role": "user",
    "tool_call_id": tool_call_id,
    "name": fn,
    "content": "根据A算法，给出例子"
})

print("=======第三次============")
# 三次调用LLM
response3 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,

)
print(f"three:{response3.choices[0].message}")
