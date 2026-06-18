import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子，确保实验可复现
torch.manual_seed(42)

# 每类颜色生成 100 个样本
num_samples = 100

# ========== 生成高斯分布的 RGB 数据 ==========

# 红色样本以 [1, 0, 0] 为中心，标准差为 0.1
mean_red = torch.tensor([1.0, 0.0, 0.0]).expand(num_samples, 3)  # 形状：(100, 3)
std_red = torch.full((num_samples, 3), 0.1)  # 所有维度标准差为0.1
red = torch.normal(mean_red, std_red)       # 生成红色样本

# 绿色样本以 [0, 1, 0] 为中心
mean_green = torch.tensor([0.0, 1.0, 0.0]).expand(num_samples, 3)
green = torch.normal(mean_green, std_red)   # 使用同样的标准差生成绿色样本

# 蓝色样本以 [0, 0, 1] 为中心
mean_blue = torch.tensor([0.0, 0.0, 1.0]).expand(num_samples, 3)
blue = torch.normal(mean_blue, std_red)     # 使用同样的标准差生成蓝色样本

# ========== 构造训练集 ==========

# 将红、绿、蓝样本合并为一个大张量
X = torch.cat([red, green, blue], dim=0)    # 形状：(300, 3)

# 构造对应标签：红=0，绿=1，蓝=2（各100个）
y = torch.tensor([0] * num_samples + [1] * num_samples + [2] * num_samples)

# ========== 定义神经网络模型 ==========

class MyRGBModel(nn.Module):
    def __init__(self):
        super(MyRGBModel, self).__init__()
        self.hidden = nn.Linear(3, 8)    # 输入层：RGB 3维 -> 隐藏层8维
        self.output = nn.Linear(8, 3)    # 隐藏层 -> 输出层（3类）

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # ReLU 激活函数
        return self.output(x)           # 输出未归一化的 logits（供 CrossEntropyLoss 使用）

# 实例化模型
model = MyRGBModel()

# 交叉熵损失函数，用于多分类任务（自动包含 Softmax）
criterion = nn.CrossEntropyLoss()

# Adam 优化器（学习率设为 0.01）
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ========== 模型训练 ==========

# 训练1000轮
for epoch in range(1000):
    output = model(X)          # 前向传播，得到预测 logits
    loss = criterion(output, y)  # 计算损失（内部做 softmax）

    optimizer.zero_grad()      # 清空梯度
    loss.backward()            # 反向传播，计算梯度
    optimizer.step()           # 更新模型参数

    # 每10轮打印一次损失和准确率
    if (epoch+1) % 10 == 0:
        pred = torch.argmax(output, dim=1)  # 取最大概率对应的类别作为预测值
        acc = (pred == y).float().mean()    # 计算分类准确率
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")
