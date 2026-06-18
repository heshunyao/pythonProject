import os

import datasets  # ⚠️ 这行可以删除，没用到
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose

from MnistNetDemo import MnistNet  # 从你定义的模块中导入模型类


# 定义获取 DataLoader 的函数，参数 train=True 表示获取训练集
def get_dataloader(train=True):
    # 定义预处理操作：将图片转换为张量，并进行标准化（适用于 MNIST）
    fn_compose = Compose([
        ToTensor(),  # 把图片转为 Tensor，值变为 [0,1]
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # 标准化：减去均值再除以标准差
    ])

    # 加载 MNIST 数据集，train 决定是训练集还是测试集
    mnist = datasets.MNIST(
        root='./data',  # 数据保存位置
        train=train,  # 是否为训练集
        download=False,  # 是否下载（你已设置为 False）
        transform=fn_compose  # 应用预处理
    )

    # 使用 DataLoader 封装数据集，设置每个 batch 大小为 100，打乱顺序
    train_dataloader = DataLoader(
        mnist,
        batch_size=100,
        shuffle=True
    )
    return train_dataloader  # 返回 DataLoader 对象

import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import torch
def train(epoch):
    # 获取训练数据的 DataLoader
    train_dataloader = get_dataloader(train=True)

    # 自动选择设备：如果有GPU就用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型移动到指定设备上（GPU或CPU）
    net.to(device)

    # 遍历训练数据
    for idx, (data, target) in enumerate(train_dataloader):
        # 将输入数据和标签也移动到同一设备上
        data, target = data.to(device), target.to(device)

        # 清除上一次迭代的梯度
        optimizer.zero_grad()

        # 前向传播：计算模型输出
        output = net(data)

        # 计算损失值（负对数似然损失函数，适用于分类任务）
        loss = F.nll_loss(output, target).to(device)

        # 反向传播：根据损失计算梯度
        loss.backward()

        # 优化器更新模型参数
        optimizer.step()

        # 每100个batch打印一次当前训练状态
        if idx % 100 == 0:
            print("train epoch: {} [{}/{} ({:.0f}%)]\tloss:{:.6f}".format(
                epoch,
                idx * len(data),                      # 当前处理了多少样本
                len(train_dataloader.dataset),        # 总样本数
                100. * idx / len(train_dataloader),   # 当前进度（百分比）
                loss.item()                           # 当前的损失值
            ))

    # 创建目录用于保存模型参数和优化器状态
    os.makedirs('./results', exist_ok=True)

    # ✅ 保存模型的权重和偏置参数，只保存参数而不保存模型结构
    torch.save(net.state_dict(), './results/mnist_net.pth')

    # ✅ 保存优化器的状态（包括学习率、动量、已积累的梯度等）
    torch.save(optimizer.state_dict(), './results/mnist_optimizer.pth')

def test():
    list_loss = []  # 用于保存每个 batch 的损失值
    list_acc = []   # 用于保存每个 batch 的准确率

    # 获取测试集的 DataLoader
    test_dataloader = get_dataloader(False)

    # 遍历测试数据
    for idx, (input, target) in enumerate(test_dataloader):
        # 计算模型的输出（未使用 .to(device)，假设训练和测试都在 CPU 上，或模型和数据默认在同一设备）
        output = net(input)

        # ✅ 在评估阶段关闭梯度计算，提高推理效率，节省显存
        with torch.no_grad():
            # 计算当前 batch 的负对数似然损失
            cur_loss = nll_loss(output, target)

            # 获取每个样本预测结果中概率最大的类别索引（即分类结果）
            out_pred = output.max(dim=-1)[1]  # 取最后一维（分类维度）中最大值的索引

            # 计算当前 batch 的准确率：预测等于真实标签的数量 / 总数量
            acc = (out_pred == target).float().mean()

            # 打印当前 batch 的损失和准确率
            print(f"Test loss: {cur_loss.item():.4f}, Test acc:{acc.item():.4f}")

            # 把当前 batch 的损失和准确率加入列表
            list_loss.append(cur_loss.item())
            list_acc.append(acc.item())

    # ✅ 计算所有 batch 的平均损失和平均准确率，并打印
    print(f"total test loss: {sum(list_loss)/len(list_loss):.4f}, total Test acc: {sum(list_acc)/len(list_acc):.4f}")
from PIL import Image

def predict_image(image_path):
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()  # 设置为评估模式
    net.to(device)

    # 加载并预处理图片
    image = Image.open(image_path).convert('L')  # 转为灰度图
    transform = Compose([
        ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度 [1, 1, 28, 28]

    # 预测
    with torch.no_grad():
        output = net(image)
        pred = output.argmax(dim=1).item()  # 取最大值索引作为预测结果
        print(f"图片 {image_path} 的预测数字为：{pred}")


# 测试 get_dataloader 函数：打印一个 batch 的尺寸
if __name__ == '__main__':
    net = MnistNet()
    net.train()
    optimizer = Adam(net.parameters(), lr=0.001)
    if os.path.exists('./results/mnist_net.pth') and os.path.exists('./results/mnist_optimizer.pth'):
        net.load_state_dict(torch.load('./results/mnist_net.pth'))
        optimizer.load_state_dict(torch.load('./results/mnist_optimizer.pth'))
        print("已加载模型和优化器参数")
    train(1000)
    test()
    predict_image("my_digit.png")

