import os
import time
import hashlib
import random
import validators
from flask import Flask, request, jsonify
from requests import RequestException
from encrypt_util import EncryptUtil  # 导入 EncryptUtil 类
import subprocess
import numpy as np
import logging
import requests
from werkzeug.utils import secure_filename
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil
import urllib.parse
import pandas as pd
from datetime import datetime
import math
from PIL import Image
from time import sleep
import re


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_FOLDER = '/app/images'

# 确保 SAVE_FOLDER 存在且为空
if os.path.exists(SAVE_FOLDER):
    # 删除目录下所有内容
    for filename in os.listdir(SAVE_FOLDER):
        file_path = os.path.join(SAVE_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
else:
    # 创建目录
    os.makedirs(SAVE_FOLDER)

# 密钥
key = "f0faa3dac9684f13921aefd14b385914"  # 32位十六进制字符串

# 获取 token 测试接口 URL
test_get_token_url = "http://22.53.1.145:9101/sms2/getToken"
# 获取 token 正式接口 URL
formal_get_token_url = "http://25.86.160.175:29101/sms2/getToken"
# 选择使用测试接口还是正式接口，这里默认使用测试接口
get_token_url = formal_get_token_url

# 获取图片测试环境接口 URL
test_get_img_url = "http://22.53.1.145:9082/redVerificationTask/queryTaskPictures"
# 获取图片正式环境接口 URL
formal_get_img_url = "http://25.86.160.175:38080/redVerificationTask/queryTaskPictures"
# 这里选择使用测试环境
get_img_url = formal_get_img_url

# 配置参数
MAX_WORKERS = 3           # 最大推送线程数
# # 创建任务队列和线程池
# task_queue = queue.Queue(maxsize=1000)  # 限制队列最大长度，避免内存溢出
# executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# 推送接口
test_push_url = "http://22.53.1.145:9082/redVerificationTask/pushPictures"
formal_push_url = "http://25.86.160.175:38080/redVerificationTask/pushPictures"
push_url = formal_push_url

# def get_image_shooting_time(file):
#     """
#     获取图片的拍摄时间
#     :param file: 上传的文件对象
#     :return: 拍摄时间字符串，如果无法获取则返回 "未知"
#     """
#     try:
#         tags = exifread.process_file(file)
#         if 'EXIF DateTimeOriginal' in tags:
#             shooting_time = tags['EXIF DateTimeOriginal'].values
#             return shooting_time
#         else:
#             return "未知"
#     except Exception as e:
#         logger.error(f"Error getting shooting time: {e}")
#         return "未知"

def setup_and_run(image_path, output_path):
    sdk_dir = "/app"
    if not os.path.exists(sdk_dir):
        logger.error(f"SDK 目录 {sdk_dir} 不存在")
        return
    os.chdir(sdk_dir)

    dji_irp_path = os.path.join(sdk_dir, "dji_irp")
    chmod_command = ["chmod", "+x", dji_irp_path]
    try:
        subprocess.run(chmod_command, check=True, capture_output=True, text=True)
        logger.info(f"Added execute permission to {dji_irp_path} successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add execute permission to {dji_irp_path}: {e}")
        logger.error(f"Standard output: {e.stdout}")
        logger.error(f"Standard error: {e.stderr}")
        return

    # 调用设置环境变量的函数
    set_ld_library_path(["/usr/local/lib", "/app"])

    dji_irp_command = ["./dji_irp", "-s", image_path, "-a", "measure", "-o", output_path]
    # try:
    #     subprocess.run(dji_irp_command, check=True, capture_output=True, text=True)
    #     logger.info("成功调用大疆接口获取")
    # except subprocess.CalledProcessError as e:
    #     if "Argument list too long" in str(e):
    #         logger.error("可能是参数列表过长导致的错误，请检查文件路径和参数。")
    #     logger.error(f"该图片调用大疆接口获取温度失败: {e}")
    #     logger.error(f"Standard output: {e.stdout}")
    #     logger.error(f"Standard error: {e.stderr}")
    try:
        # 超时30秒，超过则抛出TimeoutExpired异常
        subprocess.run(
            dji_irp_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        logger.info("成功调用大疆接口获取")
    except subprocess.TimeoutExpired as e:
        # 捕获超时异常，单独处理
        logger.error(f"调用大疆接口超时（30秒）: {e}")
        # 可选：杀死超时的子进程（避免残留）
        if e.process:
            e.process.kill()
    except subprocess.CalledProcessError as e:
        # 保留原有的非超时异常处理
        if "Argument list too long" in str(e):
            logger.error("可能是参数列表过长导致的错误，请检查文件路径和参数。")
        logger.error(f"该图片调用大疆接口获取温度失败: {e}")
        logger.error(f"Standard output: {e.stdout}")
        logger.error(f"Standard error: {e.stderr}")

def set_ld_library_path(new_paths):
    """
    设置 LD_LIBRARY_PATH 环境变量，避免重复添加路径
    :param new_paths: 要添加的新路径列表
    """
    original_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    existing_paths = original_ld_path.split(':') if original_ld_path else []
    for path in new_paths:
        if path not in existing_paths:
            existing_paths.append(path)
    new_ld_path = ':'.join(existing_paths)
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    logger.info(f"Set LD_LIBRARY_PATH to: {new_ld_path}")

def max_point_square(cropped_region, img, x_start, y_start):
    try:
        # img 是 raw 文件解析后的原图数据
        original_height, original_width = img.shape
        # 打印图片尺寸和裁剪偏移
        logger.info(f"处理图片尺寸: {original_width}×{original_height}，裁剪偏移 x_start={x_start}, y_start={y_start}")

        # 打印裁剪区域尺寸
        logger.info(f"裁剪区域尺寸: {cropped_region.shape[1]}×{cropped_region.shape[0]}")

        # 裁剪区域内的最高温坐标（局部）
        max_index = np.unravel_index(np.argmax(cropped_region), cropped_region.shape)
        local_row, local_col = max_index
        logger.info(f"裁剪区域内最高温点: (local_row={local_row}, local_col={local_col})")
        # 转换为原图坐标
        global_row = local_row + y_start
        global_col = local_col + x_start
        logger.info(f"转换为原图坐标: (global_row={global_row}, global_col={global_col})")

        # 根据图片尺寸动态调整正方形大小
        if original_width == 640 and original_height == 512:
            square_size = 16  # 小尺寸图片保持16像素
        elif original_width == 1280 and original_height == 1024:
            square_size = 32  # 大尺寸图片增大到32像素，保持比例
        else:
            square_size = 16  # 其他尺寸默认16像素

        half_width = square_size // 2

        # 计算标记框（基于原图坐标，确保不越界）
        start_x = max(0, global_col - half_width)
        start_y = max(0, global_row - half_width)
        end_x = min(original_width - 1, start_x + square_size - 1)
        end_y = min(original_height - 1, start_y + square_size - 1)
        logger.info(f"标记框坐标: ({start_x}, {start_y}) 至 ({end_x}, {end_y})")
        return {
            "startPoint": f"{start_x},{start_y}",
            "pointWidth": float(end_x - start_x),
            "pointHeight": float(end_y - start_y),
            "row": global_row,
            "col": global_col,
            "image_height": original_height,
            "image_width": original_width,

        }
    except Exception as e:
        logger.error(f"计算最高温框失败: {e}")
        return {"startPoint": None,
                "pointWidth": None,
                "pointHeight": None,
                "row": None,
                "col": None,
                "image_height": None,
                "image_width": None,
                "isPhotoAbnormal":2,
                "isPeopleCheck": 0,
                }

# 图像裁剪处理
def img_stats(raw_img, img_name):
    try:
        # raw_img 是解析后的原图温度数据（numpy 数组）
        if raw_img is None or len(raw_img.shape) != 2:
            logger.error(f"无效的 raw 数据: {raw_img.shape if raw_img else 'None'}")
            return None
        # 原图尺寸（来自 raw 文件）
        image_height, image_width = raw_img.shape
        logger.info(f"原图尺寸: {image_width}×{image_height}")

        # 动态计算裁剪边界（向上取整，确保至少裁剪1像素）
        left_pixels = max(1, math.ceil(image_width * 0.03))
        right_pixels = max(1, math.ceil(image_width * 0.03))
        top_pixels = max(1, math.ceil(image_height * 0.02))
        bottom_pixels = max(1, math.ceil(image_height * 0.02))

        logger.info(f"裁剪参数: 左={left_pixels}, 右={right_pixels}, 上={top_pixels}, 下={bottom_pixels}")

        # 计算裁剪边界（基于原图尺寸）
        x_start = max(0, left_pixels)
        x_end = min(image_width, image_width - right_pixels)
        y_start = max(0, top_pixels)
        y_end = min(image_height, image_height - bottom_pixels)
        # 裁剪区域（基于 raw 文件的原图）
        cropped_region = raw_img[y_start:y_end, x_start:x_end]
        if cropped_region.size == 0:
            logger.error(f"裁剪区域为空: {img_name}")
            return None
        # 计算统计数据（基于裁剪后的 raw 数据）
        max_value = float(np.max(cropped_region))
        min_value = float(np.min(cropped_region))
        mean_value = float(np.mean(cropped_region))

        square_info = max_point_square(
            cropped_region=cropped_region,
            img=raw_img,  # 传入原图（raw 数据）
            x_start=x_start,
            y_start=y_start
        )
        # 返回结果
        return {
            "img_name": img_name,
            "image_width": image_width,
            "image_height": image_height,
            "裁剪后宽度": x_end - x_start,
            "裁剪后高度": y_end - y_start,
            "max_value": round(max_value, 2),
            "原图全图最高温": float(round(np.max(raw_img), 2)),
            "min_value": round(min_value, 2),
            "mean_value": round(mean_value, 2),
            "startPoint": square_info["startPoint"],
            "pointHeight": float(square_info["pointHeight"]),
            "pointWidth": float(square_info["pointWidth"]),
            "remark": f"{img_name} 红外图像裁剪区域统计数据",
            # 添加默认值
            'isPhotoAbnormal': 0,
            # isPeopleCheck = 1,不要核查，默认值赋值为1
            'isPeopleCheck': 1,
        }
    except Exception as e:
        logger.error(f"计算图像统计失败: {e}")
        return None

def merge_infos(img_infos, time_infos):
    if len(img_infos)!= len(time_infos):
        logger.error("Image information and time information length do not match.")
        return []
    merged_infos = []
    for img_info, time_info in zip(img_infos, time_infos):
        # 复制 img_info 并添加 shooting_time 信息
        merged_info = img_info.copy()
        # merged_info['shooting_time'] = time_info['shooting_time']
        merged_infos.append(merged_info)
    return merged_infos

def save_uploaded_file(file, image_dir):
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(image_dir, filename)
        file.save(file_path)
        logger.info(f"File {filename} saved successfully at {file_path}.")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {e}")
        return None

# 获取包括最高温9个点的温度值（保留两位小数），返回元组(t1-t8, tm)
def get_tem_arr(temp_matrix):

    if not isinstance(temp_matrix, np.ndarray):
        temp_matrix = np.array(temp_matrix)

    if temp_matrix.size == 0:
        raise ValueError("温度矩阵不能为空")

    # 找到最高温度点坐标
    max_value = np.max(temp_matrix)
    max_indices = np.where(temp_matrix == max_value)
    max_row, max_col = max_indices[0][0], max_indices[1][0]

    # 定义9个点的坐标偏移（固定顺序）
    offsets = [
        (-1, -1),  # t1: 左上
        (-1, 0),  # t2: 上
        (-1, 1),  # t3: 右上
        (0, -1),  # t4: 左
        (0, 0),  # tm: 中心（最高温）
        (0, 1),  # t5: 右
        (1, -1),  # t6: 左下
        (1, 0),  # t7: 下
        (1, 1)  # t8: 右下
    ]

    # 提取温度值（保留两位小数）
    rows, cols = temp_matrix.shape
    temperatures = []
    for dr, dc in offsets:
        r = max_row + dr
        c = max_col + dc
        if 0 <= r < rows and 0 <= c < cols:
            temp = round(float(temp_matrix[r, c]), 2)
            temperatures.append(temp)
        else:
            temperatures.append(None)  # 边界外的点标记为None

    return tuple(temperatures)

# 判断是否为单点发热，返回(单点发热判断, count11)
def measure(temps):

    # temps顺序：t1, t2, t3, t4, tm, t5, t6, t7, t8
    if None in temps:
        logger.warning("部分温度点超出边界，无法准确判断")
        return (-1, None)

    # 解包温度值
    t1, t2, t3, t4, tm, t5, t6, t7, t8 = temps

    # 避免除以0  不存在，只在高温发热图片中执行
    # if tm == 0:
    #     logger.error("最高温度为0，无法计算")
    #     return (-1, None)

    # 计算相对温度并获取count11
    temp11 = np.array(temps) / tm
    count11 = int(np.sum(temp11 < 0.8))

    cankao_temp1=0.6*tm
    condititon1= np.array(temps) >cankao_temp1
    condititon2= np.array(temps) >55
    print('temmm',condititon1,condititon2)
    count12=np.sum(condititon1&condititon2)
    print('num',count11,count12)

    dandianfare = 1 if count11 >= 7 and count12<=3 else 0

    return (dandianfare, count11,count12)

# 图片文件分类
def is_allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_raw_file(real_path, original_width, original_height):
    """强制使用原图尺寸处理raw文件，处理像素数不匹配的情况"""
    try:
        # 读取raw文件
        img = np.fromfile(real_path, dtype=np.int16)
        total_pixels = len(img)
        expected_pixels = original_width * original_height
        logger.info(f"raw文件总像素: {total_pixels}, 原图预期像素: {expected_pixels}")

        # 情况1：像素数完全匹配，直接reshape
        if total_pixels == expected_pixels:
            logger.info(f"使用原图尺寸解析: {original_width}×{original_height}")
            return img.reshape((original_height, original_width)) / 10.0

        # 情况2：像素数不匹配，但raw像素数是原图的1/4（可能是降采样）
        elif total_pixels * 4 == expected_pixels:
            logger.warning(f"raw像素数({total_pixels})是原图的1/4，可能是降采样版本，尝试上采样")
            # 假设raw是原图的2×2降采样，使用简单插值恢复
            img_lowres = img.reshape((original_height // 2, original_width // 2))
            img_highres = np.repeat(np.repeat(img_lowres, 2, axis=0), 2, axis=1)
            return img_highres / 10.0

        # 情况3：像素数不匹配，但raw像素数是原图的1/16（严重降采样）
        elif total_pixels * 16 == expected_pixels:
            logger.warning(f"raw像素数({total_pixels})是原图的1/16，严重降采样，尝试上采样")
            img_lowres = img.reshape((original_height // 4, original_width // 4))
            img_highres = np.repeat(np.repeat(img_lowres, 4, axis=0), 4, axis=1)
            return img_highres / 10.0

        # 情况4：像素数不匹配，但raw是原图的子区域（如中心区域）
        elif total_pixels < expected_pixels:
            logger.warning(f"raw像素数({total_pixels})小于原图，可能是子区域，尝试居中裁剪")
            # 创建全零矩阵，将raw数据填入中心区域
            img_full = np.zeros((original_height, original_width), dtype=np.int16)

            # 计算子区域尺寸
            sub_height = int(np.sqrt(total_pixels * original_height / original_width))
            sub_width = total_pixels // sub_height

            # 计算填充位置（居中）
            start_h = (original_height - sub_height) // 2
            start_w = (original_width - sub_width) // 2

            # 填充数据
            img_sub = img.reshape((sub_height, sub_width))
            img_full[start_h:start_h + sub_height, start_w:start_w + sub_width] = img_sub
            return img_full / 10.0

        # 情况5：像素数超过预期（罕见）
        else:
            logger.warning(f"raw像素数({total_pixels})超过原图预期，截断处理")
            img_truncated = img[:expected_pixels]
            return img_truncated.reshape((original_height, original_width)) / 10.0

    except Exception as e:
        logger.error(f"解析raw文件失败: {e}")
        return None

def _open_image(image_path, result_dict):
    """内部线程函数，执行图片打开逻辑"""
    try:
        with Image.open(image_path) as img:
            result_dict["size"] = img.size  # 成功则存入尺寸
    except Exception as e:
        result_dict["error"] = str(e)  # 失败则存入错误信息

def get_image_dimensions(image_path, img_name):
    result_dict = {}
    # 启动线程执行图片打开，5秒超时
    open_thread = threading.Thread(target=_open_image, args=(image_path, result_dict))
    open_thread.start()
    open_thread.join(timeout=5)  # 超过5秒未完成则视为超时

    # 处理结果
    if "size" in result_dict:
        return result_dict["size"]
    else:
        error_msg = result_dict.get("error", "超时未打开或未知错误")
        logger.error(f"图片{img_name}获取尺寸失败: {error_msg}")
        print(f"打开图片{img_name}获取尺寸失败（{error_msg}），填取默认值")
        return {"error": f"打开图片获取尺寸失败: {error_msg}"}


def download_image(image_id, url, save_path, i):
    try:
        # URL替换逻辑
        if 'oss' in url and '/pw' in url:
            pw_index = url.find('/pw')
            new_base = "http://25.86.104.33/jxzygkpt-upload-product-001"
            url = new_base + url[pw_index:]
            # 验证替换后的URL有效性
            if not validators.url(url):
                return {
                    "status": "error",  # 明确错误状态
                    "data": {
                        "minTemperature": None,
                        "avgTemperature": None,
                        "maxTemperature": None,
                        "fileId": image_id,
                        "remark": f"下载图片失败，替换后链接无效",
                        "pointHeight": None,
                        "pointWidth": None,
                        "startPoint": None,
                        'isPhotoAbnormal': 1,
                        'isPeopleCheck': 0,
                    }
                }

        # 下载逻辑
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()  # 触发HTTP错误（如404）

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        logger.info(f"第{i}张图片下载成功，保存路径: {save_path}")
        # 成功时返回status和image_id
        return {
            "status": "success",
            "image_id": image_id
        }

    except Exception as e:
        logger.error(f"下载第{i}张图片时出错: {e}")
        # 失败时也返回status和错误数据
        return {
            "status": "error",  # 明确错误状态
            "data": {  # 错误数据封装在data中
                "minTemperature": None,
                "avgTemperature": None,
                "maxTemperature": None,
                "fileId": image_id,
                "remark": f"下载图片失败:",
                "pointHeight": None,
                "pointWidth": None,
                "startPoint": None,
                'isPhotoAbnormal': 1,
                'isPeopleCheck': 0,
            }
        }

def push_image_data(token, data_array,unknown_tem_sum):

    # 定义请求头，包含 token
    headers = {
        "token": token,
        "Content-Type": "application/json"
    }

    # 构建完整的请求体
    data = data_array

    try:
        # 记录请求体内容到日志
        logging.info(f"推送请求体内容: {data}")
        logging.info(f"温度未知总数量（张）: {unknown_tem_sum}")

        # 发送 POST 请求
        response = requests.post(push_url, headers=headers, json=data, timeout=10)

        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            code = result.get("code")
            msg = result.get("msg")
            if int(code) == 200:
                print("推送成功，提示信息:", msg)
            else:
                print("推送失败，错误码:", code, "提示信息:", msg)
            return result
        else:
            print("请求失败，状态码:", response.status_code)
    except requests.RequestException as e:
        print("请求过程中出现错误:", e)
    return None

def consumer(token, unknown_tem_sum, task_queue, executor):
    while True:
        try:
            # 阻塞获取任务（设置5秒超时，避免永久卡死）
            batch_data = task_queue.get(timeout=5)
            if batch_data is None:
                task_queue.task_done()
                break  # 仅当收到退出信号时才退出
            # 处理推送任务
            future = executor.submit(push_image_data, token, batch_data, unknown_tem_sum)
            future.add_done_callback(lambda f: task_queue.task_done())
        except queue.Empty:
            # 队列空时继续循环等待，而非退出
            continue
        except Exception as e:
            print(f"任务处理异常: {e}")
            task_queue.task_done()

def process_images():
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    task_queue = queue.Queue(maxsize=1000)  # 限制队列最大长度，避免内存溢出
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    # 初始化未知温度图片计数器
    unknown_tem_sum = 0  # 普通整数计数器
    unknown_tem_lock = threading.Lock()  # 线程锁，保护计数器的读写安全
    output_dir = "/app/output"
    excel_output_dir = "/app/excel_output"  # Excel保存目录
    default_ext = ".JPG"

    # 确保 output_dir 存在且为空
    if os.path.exists(output_dir):
        # 删除目录下所有内容
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # 创建目录
        os.makedirs(output_dir)

    # 确保Excel保存目录存在
    if not os.path.exists(excel_output_dir):
        os.makedirs(excel_output_dir)
        print("excel_output 创建成功")
    else:
        print("excel_output 文件夹已存在")

    # 存储所有处理结果
    all_results = []
    # 存储所有处理结果（用于Excel保存）
    all_data_for_excel = []
    # 存储 image_name 到 image_id 的映射
    image_id_map = {}
    image_id_lock = threading.Lock()  # 保护 image_id_map 的锁
    # 当前批次的数据
    current_batch = []
    batch_count = 0
    raw_file_path = ""
    original_width = 0
    original_height = 0
    raw_img = None

    # 先获取 token
    token_response, status_code = getToken()
    if status_code != 200:
        # 返回错误信息和状态码，供调用方（定时任务或 Flask 路由）处理
        return {"error": "获取 token 失败", "detail": token_response}, status_code

    token_data = token_response.get_json()
    token = token_data.get('token')

    # 启动消费者线程（设为守护线程），这里需注意：若在纯定时任务场景，需确保线程能正确协同
    consumer_thread = threading.Thread(target=consumer, args=(token, unknown_tem_sum, task_queue, executor))
    consumer_thread.daemon = True
    consumer_thread.start()

    try:
        # 获取图片信息
        img_response, status_code = get_images()
        if status_code != 200:
            return {"error": "获取图片信息失败", "detail": img_response}, status_code

        img_data = img_response.get_json()
        image_infos = img_data.get('data')
        if not image_infos:
            return {"error": "No valid image data received"}, 400

        # 1. 先创建“下载结果列表”，存储所有图片的下载状态
        download_results = []  # 记录所有图片的下载结果
        # 下载所有图片并记录 image_id 映射
        for index, image_info in enumerate(image_infos, start=1):
            image_url = image_info.get('fileUrl')
            # 1. 解析URL并解码文件名
            parsed_url = urllib.parse.urlparse(image_url)
            # 提取路径中的最后一部分（文件名）
            img_file_path = os.path.basename(parsed_url.path)
            # 解码URL编码的字符（如%20转换为空格）
            img_name = urllib.parse.unquote(img_file_path)
            print("提取的图片名称：", img_name)

            image_id = image_info.get('fileId')
            if not image_url:
                logger.error({"error": "Missing fileUrl in image info", "image_info": image_info})
                continue
            try:
                if not validators.url(image_url):
                    return {"error": "无效的图片链接"}, 400
            except ImportError:
                pass
            # 生成唯一图片名（原始名 + URL哈希值前8位 + 时间戳）
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]

            # 生成高精度时间戳
            timestamp = int(time.time() * 1000)
            # 提取原始文件名和后缀（小写处理）
            base_name, ext = os.path.splitext(img_name)
            ext_lower = ext.lower()

            # 判断是否为图片格式，决定使用原后缀还是默认后缀
            new_ext = ext if ext_lower in IMAGE_EXTS else default_ext
            img_name = f"{os.path.splitext(img_name)[0]}_{url_hash}_{timestamp}{new_ext}"
            counter = 1
            while os.path.exists(os.path.join(SAVE_FOLDER, img_name)):
                img_name = f"{base_name}_{url_hash}_{timestamp}_{random.randint(100, 999)}{new_ext}"
                counter += 1
                if counter > 100:  # 防止无限循环
                    break

            image_path = os.path.join(SAVE_FOLDER, img_name)

            down_result = download_image(image_id, image_url, image_path, index)
            download_results.append(down_result)  # 保存结果，确保循环完成后所有下载已结束

            # 先检查status是否存在，再判断状态
            # 无论下载成败，都记录 image_id 到 image_id_map
            if "status" in down_result:
                # 关键：强制将 image_id 存入映射表（即使下载失败）
                with image_id_lock:
                    # 下载成功时用返回的 image_id，失败时用原始 image_id
                    image_id = down_result.get("image_id") or image_info.get("fileId")
                    image_id_map[img_name] = image_id  # 确保映射表一定有记录

                if down_result["status"] == "success":
                    logger.info(f"图片 {img_name} 映射 image_id: {image_id}")
                else:
                    # 此时 down_result["data"] 的 fileId 也替换为正确的 image_id
                    error_data = down_result["data"].copy()
                    error_data["fileId"] = image_id  # 覆盖原有的 None 或错误值
                    current_batch.append(error_data)
                    all_data_for_excel.append(error_data)

                    with unknown_tem_lock:
                        unknown_tem_sum += 1
            else:
                # 异常情况：返回格式不符合预期
                logger.error(f"download_image返回格式错误，缺少status键: {down_result}")
                # 手动构造错误数据
                error_data = {
                    "minTemperature": None,
                    "avgTemperature": None,
                    "maxTemperature": None,
                    "fileId": image_id,
                    "remark": "下载图片返回格式错误",
                    "pointHeight": None,
                    "pointWidth": None,
                    "startPoint": None,
                    'isPhotoAbnormal': 1,
                    'isPeopleCheck': 0,
                }
                current_batch.append(error_data)
                all_data_for_excel.append(error_data)

                with unknown_tem_lock:
                    unknown_tem_sum += 1

        # 获取所有图片文件
        image_files = [
            f for f in os.listdir(SAVE_FOLDER)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        total_images = len(image_files)
        print(f"发现 {total_images} 张图片需要处理")
        logger.info(f"下载阶段结束，共获取图片文件 {len(image_files)} 个")

        # 分批处理图片
        for index, img_name in enumerate(image_files, start=1):
            image_path = os.path.join(SAVE_FOLDER, img_name)
            print(f"处理第 {index}/{total_images} 张图片: {img_name}")

            with image_id_lock:  # 加锁读取
                file_id = image_id_map.get(img_name)
            if not file_id:
                logger.warning(f"图片 {img_name} 未找到对应的 image_id，可能是映射丢失")

            try:
                image_size = get_image_dimensions(image_path, img_name)
                if not isinstance(image_size, tuple) or len(image_size) != 2:
                    raise ValueError(f"无效的图片尺寸: {image_size}")
                original_width, original_height = image_size  # 去掉多余的逗号
            except Exception as e:
                logger.error(f"获取图片尺寸失败: {e}")
                print("打开图片获取尺寸失败，填取默认值")

                # 仅构建一次 merged_info，添加到 current_batch 和 all_data_for_excel
                file_id = ""  # 先初始化，后续补兜底逻辑
                # 1. 优先从映射表获取 file_id
                with image_id_lock:
                    file_id = image_id_map.get(img_name)
                # 2. 映射表无值则从原始 image_infos 匹配
                if not file_id:
                    for info in image_infos:
                        raw_url = info.get("fileUrl")
                        if raw_url:
                            raw_filename = os.path.basename(urllib.parse.urlparse(raw_url).path)
                            raw_base = os.path.splitext(raw_filename)[0]
                            current_base = os.path.splitext(img_name)[0].split("_")[0]
                            if raw_base in current_base:
                                file_id = info.get("fileId")
                                break
                # 3. 终极兜底：生成临时 file_id
                if not file_id:
                    file_id = f"unknown_{img_name.split('_')[0]}"
                    logger.error(f"图片 {img_name} 用兜底 fileId: {file_id}")

                merged_info = {
                    "remark": f"打开图片获取尺寸失败，填取默认值 {image_path}",
                    "img_name": img_name,
                    "image_width": None,
                    "image_height": None,
                    "max_value": None,
                    "原图全图最高温": None,
                    "min_value": None,
                    "mean_value": None,
                    "pointHeight": None,
                    "pointWidth": None,
                    "startPoint": None,
                    "count11（<0.8的点数）": None,
                    "isPhotoAbnormal": 3,
                    "isPeopleCheck": 0,
                    "image_id": file_id
                }
                # 仅添加一次数据
                current_batch.append({
                    "minTemperature": merged_info["min_value"],
                    "avgTemperature": merged_info["mean_value"],
                    "maxTemperature": merged_info["max_value"],
                    "fileId": merged_info["image_id"],
                    "remark": merged_info["remark"],
                    "pointHeight": merged_info["pointHeight"],
                    "pointWidth": merged_info["pointWidth"],
                    "startPoint": merged_info["startPoint"],
                    "isPhotoAbnormal": merged_info["isPhotoAbnormal"],
                    "isPeopleCheck": merged_info["isPeopleCheck"]
                })
                all_data_for_excel.append(merged_info)
                # 清理当前图片文件（避免残留）
                if os.path.exists(image_path):
                    os.remove(image_path)
                continue  # 跳过后续 raw 文件处理逻辑
            try:
                # 生成 .raw 文件的输出路径
                base_name = os.path.splitext(img_name)[0]
                raw_filename = f"{base_name}.raw"
                output_path = os.path.join(output_dir, raw_filename)
                raw_file_path = os.path.join(output_dir, raw_filename)

                # 运行 sdk 处理图片
                setup_and_run(image_path, output_path)

                # 处理 .raw 文件（核心：基于实际结果构建 merged_info，不强制覆盖）
                if os.path.exists(raw_file_path):
                    raw_img = process_raw_file(raw_file_path, original_width, original_height)
                    if raw_img is not None:
                        img_info = img_stats(raw_img, img_name)
                        if img_info:
                            # 情况1：raw处理成功→使用有效数据
                            merged_info = img_info.copy()
                            merged_info["image_id"] = file_id  # 补充 image_id
                            logger.info(f"第{index}张图片信息：{merged_info}")
                        else:
                            # 情况2：raw处理失败→构建失败数据
                            merged_info = {
                                "remark": f".raw文件处理失败 {raw_file_path}",
                                "img_name": img_name,
                                "image_width": original_width,  # 保留已获取的尺寸
                                "image_height": original_height,
                                "max_value": None,
                                "原图全图最高温": None,
                                "min_value": None,
                                "mean_value": None,
                                "pointHeight": None,
                                "pointWidth": None,
                                "startPoint": None,
                                "count11（<0.8的点数）": None,
                                "isPhotoAbnormal": 4,
                                "isPeopleCheck": 0,
                                "image_id": file_id
                            }
                            with unknown_tem_lock:
                                unknown_tem_sum += 1
                    else:
                        # 情况3：raw转换失败→构建失败数据
                        merged_info = {
                            "remark": f".raw文件转换成裁剪后处理失败 {raw_file_path}",
                            "img_name": img_name,
                            "image_width": original_width,
                            "image_height": original_height,
                            "max_value": None,
                            "原图全图最高温": None,
                            "min_value": None,
                            "mean_value": None,
                            "pointHeight": None,
                            "pointWidth": None,
                            "startPoint": None,
                            "count11（<0.8的点数）": None,
                            "isPhotoAbnormal": 5,
                            "isPeopleCheck": 0,
                            "image_id": file_id
                        }
                        with unknown_tem_lock:
                            unknown_tem_sum += 1
                else:
                    # 情况4：无raw文件→构建失败数据
                    merged_info = {
                        "remark": f"使用大疆接口获取温度失败，无.raw 文件",
                        "img_name": img_name,
                        "image_width": original_width,
                        "image_height": original_height,
                        "max_value": None,
                        "原图全图最高温": None,
                        "min_value": None,
                        "mean_value": None,
                        "pointHeight": None,
                        "pointWidth": None,
                        "startPoint": None,
                        "count11（<0.8的点数）": None,
                        "isPhotoAbnormal": 2,
                        "isPeopleCheck": 0,
                        "image_id": file_id
                    }
                    with unknown_tem_lock:
                        unknown_tem_sum += 1

                # 兜底处理：确保 fileId 不为空
                if not file_id:
                    file_id = f"unknown_{img_name.split('_')[0]}"
                    logger.error(f"图片 {img_name} 无法获取 fileId，使用兜底值: {file_id}")
                    merged_info["image_id"] = file_id  # 更新兜底 fileId

                # 单点发热判断（基于实际 merged_info 数据）
                if merged_info["max_value"] is not None and merged_info["max_value"] >= 90:
                    temps = get_tem_arr(raw_img)
                    if None in temps:
                        merged_info.update({
                            't1(左上)': temps[0], 't2(上)': temps[1], 't3(右上)': temps[2],
                            't4(左)': temps[3], 'tm(最高温)': temps[4], 't5(右)': temps[5],
                            't6(左下)': temps[6], 't7(下)': temps[7], 't8(右下)': temps[8],
                            '单点发热判断': '边界点获取异常', 'count11（<0.8的点数）': None,
                            'isPhotoAbnormal': 6, 'isPeopleCheck': 0
                        })
                    else:
                        dandianfare, count11, count12 = measure(temps)
                        need_check = 0 if dandianfare == 0 else 1
                        merged_info.update({
                            't1(左上)': temps[0], 't2(上)': temps[1], 't3(右上)': temps[2],
                            't4(左)': temps[3], 'tm(最高温)': temps[4], 't5(右)': temps[5],
                            't6(左下)': temps[6], 't7(下)': temps[7], 't8(右下)': temps[8],
                            'count11（<0.8的点数）': count11, 'count12': count12,
                            '单点发热判断': '是' if dandianfare == 1 else '否',
                            'isPeopleCheck': need_check,
                            'isPhotoAbnormal': 0
                        })
                else:
                    merged_info.update({
                        '单点发热判断': '低于90摄氏度，不判断单点发热'
                    })

                # 检查“原图全图最高温”是否存在且超过620度
                if merged_info.get("原图全图最高温") is not None:  # 确保最高温数据有效
                    full_max_temp = merged_info["原图全图最高温"]
                    if full_max_temp >= 620:
                        # 1. 更新异常标记为8
                        merged_info["isPhotoAbnormal"] = 8
                        # 2. 更新备注，说明异常原因（方便平台/Excel排查）
                        merged_info[
                            "remark"] = f"{merged_info['remark']} | 图片全图最高温{full_max_temp}℃，超过620℃阈值，标记为异常"
                        # 3. 记录日志（便于问题排查）
                        logger.warning(f"图片 {img_name} 异常：全图最高温{full_max_temp}℃（>620℃），isPhotoAbnormal设为8")

                # 构建推送数据（仅添加一次）
                data = {
                    "minTemperature": merged_info["min_value"],
                    "avgTemperature": merged_info["mean_value"],
                    "maxTemperature": merged_info["max_value"],
                    "fileId": merged_info["image_id"],
                    "remark": merged_info["remark"],
                    "pointHeight": merged_info["pointHeight"],
                    "pointWidth": merged_info["pointWidth"],
                    "startPoint": merged_info["startPoint"],
                    "isPhotoAbnormal": merged_info["isPhotoAbnormal"],
                    "isPeopleCheck": merged_info["isPeopleCheck"]
                }

                # 仅添加一次数据到 current_batch 和 all_data_for_excel
                current_batch.append(data)  # 唯一一次添加，解决重复问题
                all_data_for_excel.append(merged_info)

                # 清理文件（逻辑不变）
                for f in [image_path, raw_file_path]:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                            logger.info(f"已成功删除文件: {f}")
                        except Exception as e:
                            logger.error(f"删除文件 {f} 时出错: {e}")

                logger.info(f"已处理完第{index}张图片")

            except Exception as e:
                # 异常分支：仅添加一次错误数据（逻辑不变）
                logger.error(f"处理图片 {image_path} 过程出错: {e}")
                merged_info = {
                    "remark": f"最外层try未知错误 {e}",
                    "img_name": img_name,
                    "image_width": None,
                    "image_height": None,
                    "max_value": None,
                    "原图全图最高温": None,
                    "min_value": None,
                    "mean_value": None,
                    "pointHeight": None,
                    "pointWidth": None,
                    "startPoint": None,
                    'count11（<0.8的点数）': None,
                    'count12': None,
                    'isPhotoAbnormal': 7,
                    'isPeopleCheck': 0,
                    "image_id": image_id_map.get(img_name, f"unknown_{img_name.split('_')[0]}")
                }
                # 仅添加一次错误数据
                current_batch.append({
                    "minTemperature": merged_info["max_value"],
                    "avgTemperature": merged_info["mean_value"],
                    "maxTemperature": merged_info["max_value"],
                    "fileId": merged_info["image_id"],
                    "remark": merged_info["remark"],
                    "pointHeight": merged_info["pointHeight"],
                    "pointWidth": merged_info["pointWidth"],
                    "startPoint": merged_info["startPoint"],
                    'isPhotoAbnormal': 7,
                    'isPeopleCheck': 0,
                })
                all_data_for_excel.append(merged_info)

            finally:
                # 尝试清理临时文件，这里可根据实际情况调整要清理的文件
                for f in [image_path, raw_file_path]:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                            logger.info(f"已删除文件: {f}")
                        except Exception as e:
                            logger.error(f"删除文件 {f} 时出错: {e}")

            # 每处理200张图片或最后一批，进行一次数据推送（这里逻辑需确认，原代码是用于 HTTP 场景的异步推送，定时任务场景可能需调整）
            # 注意：定时任务场景下，若不需要异步队列推送，可直接处理结果，或简化这部分逻辑
            if len(current_batch) >= 200 or index == total_images:
                batch_count += 1
                print(f"准备推送第 {batch_count} 批，共 {len(current_batch)} 张图片")

                # 异步推送当前批次（定时任务场景下，若不需要异步，可直接处理，比如打印或存储）
                task_queue.put(current_batch.copy())

                # 记录结果并清空当前批次
                all_results.append({"batch": batch_count, "count": len(current_batch)})
                current_batch = []

        # 循环结束后，检查是否有剩余的图片需要推送
        if current_batch:
            batch_count += 1
            print(f"推送最后一批，共 {len(current_batch)} 张图片")
            task_queue.put(current_batch.copy())
            all_results.append({"batch": batch_count, "count": len(current_batch)})
            current_batch = []

        start_time = time.time()
        while not task_queue.empty() and (time.time() - start_time) < 30:
            time.sleep(0.1)  # 轮询检查队列，最多等30秒
        task_queue.join()  # 等待任务完成
        task_queue.put(None)
        # 等待消费者线程处理完所有任务
        consumer_thread.join()

        # 保存所有数据到Excel
        # if all_data_for_excel:
        #     df = pd.DataFrame(all_data_for_excel)
        #     # 获取当前时间并格式化为指定字符串，比如年-月-日_时-分-秒
        #     now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     # 拼接文件名
        #     excel_filename = f"max_tem_message_{now}.xlsx"
        #     excel_path = os.path.join(excel_output_dir, excel_filename)
        #     df.to_excel(excel_path, index=False)
        #     logger.info(f"Excel文件保存路径: {excel_path}")
        # else:
        #     logger.warning("没有数据可保存到Excel")

        # 关闭线程池
        print("关闭线程池")
        executor.shutdown(wait=True, cancel_futures=False)
        print("所有图片处理和推送完成")

        # 最后返回处理结果（状态码固定 200 可根据实际情况调整，或在出错时返回对应状态码）
        # 表示进程处理完毕，返回200，而不是全部图片推送完成
        return {
            "status": "success",
            "total_images": total_images,
            "batches": batch_count,
            "unknown_tem_count": unknown_tem_sum,
            "results": all_results
        }, 200

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        # 推送剩余数据（若有）
        if current_batch:
            batch_count += 1
            print(f"推送最后一批，共 {len(current_batch)} 张图片")
            task_queue.put(current_batch.copy())
            all_results.append({"batch": batch_count, "count": len(current_batch)})
            current_batch = []
        # 确保异常发生时也能清理资源
        try:
            task_queue.put(None)
            task_queue.join()
            # 线程池关闭时强制取消未完成任务
            executor.shutdown(wait=True, cancel_futures=False)
        except:
            print("异常发生时清理资源出错")
    finally:
        # 【关键】放在 finally 块，确保无论成功/失败都执行
        try:
            # 线程池关闭时强制取消未完成任务
            if 'executor' in locals():  # 确保 executor 已定义
                executor.shutdown(wait=True, cancel_futures=False)
                logger.info("线程池已强制关闭")
        except Exception as e:
            logger.error(f"线程池关闭失败: {e}")

        try:
            # 队列清理时强制标记完成
            if 'task_queue' in locals():  # 确保 task_queue 已定义
                while not task_queue.empty():
                    try:
                        task_queue.get_nowait()
                        task_queue.task_done()  # 强制标记完成
                    except queue.Empty:
                        break
                # 设置超时，避免永久阻塞
                task_queue.join()
                logger.info("任务队列已清理完毕")
        except Exception as e:
            logger.error(f"队列清理失败: {e}")

@app.route('/getToken', methods=['POST'])
def getToken():
    try:
        # 获取登录名和密码
        login_name = "17891863362"
        password = "E@1dkbzb10"
        # 测试环境
        # login_name = "fj_gly"
        # password = "Pwfj@2022"

        if not login_name or not password:
            return jsonify({"error": "loginName and password are required"}), 400

        # 获取当前时间戳（毫秒）
        timestamp = str(int(time.time() * 1000))

        # 生成要加密的数据
        data_to_encrypt = f'{{"loginName":"{login_name}","password":"{password}","time":"{timestamp}"}}'

        # 加密
        encrypted = EncryptUtil.encrypt(key, data_to_encrypt)

        # 解密操作
        decrypted = EncryptUtil.decrypt(key, encrypted)
        print("Decrypted:", decrypted)

        # 构建请求体
        request_body = encrypted

        # 发送 POST 请求获取 token
        response = requests.post(get_token_url, data=request_body, timeout=10)
        # response,status_code = requests.post(get_token_url, data=request_body)
        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            code = result.get("code")
            msg = result.get("msg")
            if code == 200:
                token = result.get("token")
                print(f"成功获取 token: {token}")
                return jsonify({"encrypted": encrypted, "Decrypted": decrypted, "token": token}), 200
            else:
                print(f"获取 token 失败，错误码: {code}，提示信息: {msg}")
                return jsonify({"encrypted": encrypted, "Decrypted": decrypted,
                                "error": f"获取 token 失败，错误码: {code}，提示信息: {msg}"}), 500
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return jsonify({"encrypted": encrypted, "Decrypted": decrypted,
                            "error": f"请求失败，状态码: {response.status_code}"}), 500


    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getImg', methods=['POST'])
def get_images():
    try:
        # 先获取 token
        token_response, status_code = getToken()
        if status_code != 200:
            return token_response, status_code

        token_data = token_response.get_json()
        token = token_data.get('token')

        # 设置请求头，包含 token
        headers = {
            "token": token
        }

        # 发送 POST 请求获取图片
        response = requests.post(get_img_url, headers=headers, json={}, timeout=10)

        # 检查响应状态码
        if response.status_code == 200:
            # 获取响应的 JSON 数据
            result = response.json()
            code = result.get("code")
            msg = result.get("msg")
            data = result.get("data")

            if int(code) == 200:
                print(f"成功获取 msg 图片: {msg}")
                return jsonify({
                    "code": code,
                    "msg": msg,
                    "data": data
                }),200

            else:
                return jsonify({"error": f"请求失败，错误码: {code}，提示信息: {msg}"}), 500
        else:
            return jsonify({"error": f"请求失败，状态码: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Flask 路由函数（仅负责 HTTP 适配）
@app.route('/receptionImgUrl', methods=['POST'])
def push_images_info():
    try:
        # 调用业务函数
        data, status_code = process_images()
        return jsonify(data), status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500