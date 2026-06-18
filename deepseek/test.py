import requests


def ask_ai_stream(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": True
    }

    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))  # 逐行解析输出


# 调用流式输出
ask_ai_stream("给我一份简单的java代码")
