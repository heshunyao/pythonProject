import torch.nn as nn
from torch.nn import functional as F


# 定义一个用于手写数字识别的神经网络模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        # 定义第一层全连接层：输入 28x28 的图像（784维）映射到 28*3 = 84 维
        self.fc1 = nn.Linear(28 * 28, 28 * 3)

        # 定义第二层全连接层：将 84 维映射到 10 类（数字 0~9）
        self.fc2 = nn.Linear(28 * 3, 10)

    def forward(self, x):
        # 将输入的图像张量展开成一维向量，形状从 [batch_size, 1, 28, 28] 变成 [batch_size, 784]
        x = x.view(-1, 28 * 28)

        # 第一层：全连接后使用 ReLU 激活函数
        x = F.relu(self.fc1(x))

        # 第二层：输出后使用 log_softmax，得到每类的对数概率
        out = F.log_softmax(self.fc2(x), dim=-1)

        return out
