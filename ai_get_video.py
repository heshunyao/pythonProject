import requests
import time
import json

# 请替换成你的 Agnes API Key
API_KEY = "sk-XLjUamxpyqqWLpB58KlpUDgL7GmTnnHmhD09hqq61hULY0Lc"
BASE_URL = "https://apihub.agnes-ai.com/v1"  # 官方 API 基础 URL [citation:1][citation:2]


def generate_video(prompt, model="agnes-video-v2.0", duration=5, image_url=None):
    """
    使用 Agnes AI 生成视频
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration  # 视频时长（秒）
    }

    # 图生视频：传入首帧图片 URL
    if image_url:
        payload["image"] = image_url

    # 1. 创建视频生成任务
    print("正在提交视频生成任务...")
    create_response = requests.post(f"{BASE_URL}/videos", headers=headers, json=payload)
    create_response.raise_for_status()

    task_data = create_response.json()
    task_id = task_data.get("id")  # 获取任务 ID [citation:12]
    if not task_id:
        print("任务创建失败，请检查响应：", task_data)
        return None
    print(f"任务已创建，ID: {task_id}")

    # 2. 轮询查询任务状态，直到完成
    print("视频生成中，这可能需要几分钟时间...")
    while True:
        status_response = requests.get(f"{BASE_URL}/videos/{task_id}", headers=headers)
        status_response.raise_for_status()

        status_data = status_response.json()
        current_status = status_data.get("status")
        print(f"当前状态: {current_status}")

        if current_status == "completed":
            # 生成成功，返回视频 URL
            video_url = status_data.get("remixed_from_video_id")
            print("视频生成成功！")
            return video_url
        elif current_status in ["failed", "cancelled"]:
            error_msg = status_data.get('error', {}).get('message', '未知错误')
            raise Exception(f"视频生成失败: {error_msg}")
        else:
            # 任务还在处理中，等待 15 秒后再次查询 [citation:2]
            time.sleep(15)


if __name__ == "__main__":
    # 示例：文生视频
    text_prompt = "一只在太空漫步的柴犬，穿着宇航服，背景是绚丽的星云"
    print(f"提示词: {text_prompt}")

    try:
        video_url = generate_video(text_prompt, duration=5)
        print(f"视频链接 (有效期有限，请及时下载): {video_url}")
    except Exception as e:
        print(f"发生错误: {e}")