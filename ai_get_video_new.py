import requests
import time
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 清除环境变量中的代理设置
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

API_KEY = "sk-XLjUamxpyqqWLpB58KlpUDgL7GmTnnHmhD09hqq61hULY0Lc"
BASE_URL = "https://apihub.agnes-ai.com/v1"


def generate_video(prompt, model="agnes-video-v2.0", duration=5, image_url=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration
    }

    if image_url:
        payload["image"] = image_url

    print("正在提交视频生成任务...")

    # 不使用代理
    create_response = requests.post(
        f"{BASE_URL}/videos",
        headers=headers,
        json=payload,
        verify=False,
        timeout=60
    )
    create_response.raise_for_status()

    task_data = create_response.json()
    task_id = task_data.get("id")
    if not task_id:
        print("任务创建失败：", task_data)
        return None
    print(f"任务已创建，ID: {task_id}")

    print("视频生成中，这可能需要几分钟时间...")
    while True:
        status_response = requests.get(
            f"{BASE_URL}/videos/{task_id}",
            headers=headers,
            verify=False,
            timeout=60
        )
        status_response.raise_for_status()

        status_data = status_response.json()
        current_status = status_data.get("status")
        print(f"当前状态: {current_status}")

        if current_status == "completed":
            video_url = status_data.get("remixed_from_video_id")
            print("视频生成成功！")
            return video_url
        elif current_status in ["failed", "cancelled"]:
            error_msg = status_data.get('error', {}).get('message', '未知错误')
            raise Exception(f"视频生成失败: {error_msg}")
        else:
            time.sleep(15)


if __name__ == "__main__":
    text_prompt = "一只猪"
    print(f"提示词: {text_prompt}")

    try:
        video_url = generate_video(text_prompt, duration=1)
        print(f"视频链接: {video_url}")
    except Exception as e:
        print(f"发生错误: {e}")