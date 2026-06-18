# 初始化模型
import os
from torchvision.transforms import ToTensor, Compose
import torch
from torchvision.transforms import ToTensor, Compose
from MnistNetDemo import MnistNet  # 从你定义的模块中导入模型类
from torchvision import datasets, transforms
from PIL import Image
def predict_image(image_path, model_path='./results/mnist_net.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ✅ 重新创建模型并加载权重
    net = MnistNet()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net.to(device)

    # ✅ 加载图片并处理
    image = Image.open(image_path).convert('L')
    transform = Compose([
        ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    image = transform(image).unsqueeze(0).to(device)

    # ✅ 推理
    with torch.no_grad():
        output = net(image)
        pred = output.argmax(dim=1).item()
        print(f"图片 {image_path} 的预测数字为：{pred}")


net = MnistNet()

# 加载模型参数
model_path = './results/mnist_net.pth'
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("✅ 已加载训练好的模型参数")
else:
    print("❌ 模型文件不存在，请先训练模型！")
    exit()
predict_image("my_digit.png")
# 运行测试