import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 创建图像画布（大小为28x28像素，黑底）
fig = plt.figure(figsize=(1, 1), dpi=28)  # 1英寸 * 28dpi = 28像素
plt.text(0.5, 0.5, '7', fontsize=24, color='white',
         ha='center', va='center')  # 白色数字，居中显示
plt.axis('off')  # 不显示坐标轴
fig.patch.set_facecolor('black')  # 设置背景为黑色

# 2. 保存为图像文件（注意设置边距）
plt.savefig('my_digit.png', bbox_inches='tight', pad_inches=0, facecolor='black')
plt.close()

# 3. 再用 PIL 确保为灰度图，强制大小为 28x28
img = Image.open("my_digit.png").convert("L").resize((28, 28))
img.save("my_digit.png")

# 可选：显示图片进行验证
img.show()
