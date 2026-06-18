
from flask import Flask, request, jsonify
import paddleocr
import requests
from PIL import Image
import io
# image_url = data['image_url']
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")
# 下载图片
response = requests.get("https://pss.bdstatic.com/static/superman/img/qrcode_download-02b84e1f66.png")
image = Image.open(io.BytesIO(response.content))

# 解析 OCR
result = ocr.ocr(image, cls=True)
text = "\n".join([line[1][0] for line in result[0]])  # 提取 OCR 结果文本
print(jsonify({"ocr_result": text}))
# return jsonify({"ocr_result": text})