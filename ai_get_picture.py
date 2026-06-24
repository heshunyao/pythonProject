import requests
import json

url = "https://apihub.agnes-ai.com/v1/images/generations"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-XLjUamxpyqqWLpB58KlpUDgL7GmTnnHmhD09hqq61hULY0Lc"
}

payload = {
    "model": "agnes-image-2.1-flash",
    "prompt": "一只在太空漫步的柴犬，穿着宇航服，背景是绚丽的星云",
    "n":1
}

response = requests.post(url, headers=headers, json=payload)

# 检查请求是否成功
if response.status_code == 200:
    result = response.json()
    print("图片生成成功！")
    print(result)
    # 如果返回的是图片URL，可以这样获取
    if 'data' in result and len(result['data']) > 0:
        image_url = result['data'][0]['url']
        print(f"图片地址: {image_url}")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"错误信息: {response.text}")