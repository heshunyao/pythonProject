import torch
import numpy as np

# 打印 PyTorch 版本 和 当前是否可用 CUDA（GPU）
print(torch.__version__)
print(torch.cuda.is_available())

# 生成一个 100x1 的随机张量，每个值 ∈ [0,1)
x = torch.rand([100, 1])

# 根据 y = 4x + 6 的线性关系生成目标值 y_ture
y_ture = 4 * x + 6

# 打印输入数据 x（100 行）
print(x)

# 初始化模型参数 w 和 b
# w 是一个需要梯度的 1x1 张量，表示权重
w = torch.rand([1, 1], requires_grad=True, dtype=torch.float32)

# b 是一个需要梯度的偏置项，初始化为 1
b = torch.tensor([1], requires_grad=True, dtype=torch.float32)

# 训练循环，执行 10000 次迭代
for i in range(10000):
    # 计算预测值 y_pred = x*w + b
    y_pred = x * w + b

    # 使用均方误差（MSE）作为损失函数
    loss = (y_pred - y_ture).pow(2).mean()

    # 反向传播，计算损失对 w 和 b 的梯度
    loss.backward()

    # 在 no_grad 模式下更新参数，避免这些操作被记录在计算图中
    with torch.no_grad():
        # 使用学习率 0.01 更新参数
        w -= w.grad * 0.01
        b -= b.grad * 0.01

        # 清空梯度，否则 PyTorch 会累积梯度
        w.grad.zero_()
        b.grad.zero_()

        # 每 100 次迭代，生成新的训练数据并打印当前损失和参数
        if i % 100 == 0:
            # 重新生成训练数据，模拟在线学习
            x = torch.rand([100, 1])
            y_ture = 4 * x + 6

            # 打印当前 epoch、损失值、w 和 b 的值
            print(f"epoch:{i}, loss:{loss.item()}, w:{w}, b:{b}")
