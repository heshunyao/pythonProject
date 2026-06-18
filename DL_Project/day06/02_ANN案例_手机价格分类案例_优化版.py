import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import StandardScaler


# 构建数据集
def create_dataset():
	# 使用pandas读取数据
	data = pd.read_csv('./data/手机价格预测.csv')
	# 特征值和目标值
	x, y = data.iloc[:, :-1], data.iloc[:, -1]
	# 类型转换：特征值，目标值
	x = x.astype(np.float32)
	y = y.astype(np.int64)
	# 数据集划分
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)
	# 优化①:数据标准化
	transfer = StandardScaler()
	x_train = transfer.fit_transform(x_train)
	x_valid = transfer.transform(x_valid)
	# 构建数据集,转换为pytorch的形式
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
	valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.tensor(y_valid.values))
	# 返回结果
	return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


# 构建网络模型
class PhonePriceModel(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(PhonePriceModel, self).__init__()
		# 优化②:增加网络深度
		# 1. 第一层: 输入为维度为 20, 输出维度为: 128
		self.linear1 = nn.Linear(input_dim, 128)
		# 2. 第二层: 输入为维度为 128, 输出维度为: 256
		self.linear2 = nn.Linear(128, 256)
		# 3. 第三层: 输入为维度为 256, 输出维度为: 512
		self.linear3 = nn.Linear(256, 512)
		# 4. 第四层: 输入为维度为 512, 输出维度为: 128
		self.linear4 = nn.Linear(512, 128)
		# 5. 输出层: 输入为维度为 128, 输出维度为: 4
		self.linear5 = nn.Linear(128, output_dim)

	def forward(self, x):
		# 前向传播过程
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		x = torch.relu(self.linear3(x))
		x = torch.relu(self.linear4(x))
		# 后续CrossEntropyLoss损失函数中包含softmax过程, 所以当前步骤不进行softmax操作
		output = self.linear5(x)
		# 获取数据结果
		return output

def train(train_dataset, input_dim, class_num):
	"""
    训练手机价格分类模型的主函数

    参数:
    train_dataset: 训练数据集
    input_dim: 输入特征的维度
    class_num: 分类类别数量
    """

	# 固定随机数种子
	# 作用: 确保每次训练结果可复现，相同的参数设置会得到相同的结果
	# 注释: 设置PyTorch的随机种子为0，使随机初始化和数据shuffle具有确定性
	torch.manual_seed(0)

	# 初始化数据加载器
	# 作用: 将数据集分批加载，便于模型训练
	# 注释: batch_size=8表示每次训练使用8个样本，shuffle=True表示每个epoch打乱数据顺序
	#       这有助于模型学习更通用的特征，防止过拟合到特定的数据顺序
	dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

	# 初始化模型
	# 作用: 创建神经网络模型实例
	# 注释: PhonePriceModel是根据input_dim和class_num构建的神经网络
	#       input_dim决定输入层大小，class_num决定输出层大小（分类数）
	model = PhonePriceModel(input_dim, class_num)

	# 损失函数: CrossEntropyLoss = softmax + 损失计算
	# 作用: 计算模型预测结果与真实标签之间的差异
	# 注释: nn.CrossEntropyLoss()是多分类任务的常用损失函数
	#       它内部包含了softmax激活函数和负对数似然损失计算
	#       适用于分类问题，特别是当类别互斥时
	criterion = nn.CrossEntropyLoss()

	# 优化器: 使用Adam优化方法, 学习率变为1e-4
	# 作用: 根据损失函数的梯度更新模型参数
	# 注释: optim.Adam是自适应矩估计优化器，结合了RMSprop和动量的优点
	#       lr=1e-4是学习率，控制参数更新的步长
	#       较小的学习率可以使训练更稳定，但可能需要更多epoch
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	# 遍历每个轮次的数据
	# 作用: 多次遍历整个数据集，让模型逐步学习数据特征
	# 注释: num_epoch=50表示训练50个完整的epoch
	#       每个epoch表示模型看过一次完整的训练集
	num_epoch = 50
	for epoch_idx in range(num_epoch):

		# 记录训练开始时间
		# 作用: 计算每个epoch的训练耗时
		# 注释: time.time()返回当前时间戳，用于性能监控
		start = time.time()

		# 初始化损失和样本计数
		# 作用: 累积本epoch的总损失，用于计算平均损失
		# 注释: total_loss记录所有batch的损失总和
		#       total_num记录本epoch处理的样本总数
		total_loss = 0.0
		total_num = 0

		# 遍历每个batch数据进行处理
		# 作用: 分批处理训练数据，减少内存使用并支持梯度下降
		# 注释: dataloader每次返回一个batch的数据(x为特征，y为标签)
		for x, y in dataloader:
			# 设置模型为训练模式
			# 作用: 启用dropout和batch normalization的训练行为
			# 注释: model.train()会改变某些层（如Dropout、BatchNorm）的行为
			#       训练模式下，这些层会进行随机失活或使用batch统计量
			model.train()

			# 前向传播
			# 作用: 将输入数据通过神经网络计算输出
			# 注释: model(x)调用模型的forward方法，计算预测结果
			#       output是模型的预测概率分布（未经过softmax，由CrossEntropyLoss处理）
			output = model(x)

			# 计算损失
			# 作用: 量化模型预测与真实标签的差异
			# 注释: criterion(output, y)计算预测输出和真实标签之间的交叉熵损失
			#       损失值越小表示模型预测越准确
			loss = criterion(output, y)

			# 梯度清零
			# 作用: 清除上一次反向传播计算的梯度
			# 注释: 在PyTorch中，梯度会累积，每个batch训练前需要清零
			#       防止梯度信息从上一个batch泄漏到当前batch
			optimizer.zero_grad()

			# 反向传播
			# 作用: 计算损失函数关于模型参数的梯度
			# 注释: loss.backward()通过链式法则计算所有可训练参数的梯度
			#       梯度指示了如何调整参数可以减少损失
			loss.backward()

			# 参数更新
			# 作用: 根据梯度更新模型参数
			# 注释: optimizer.step()使用优化器算法（这里是Adam）更新参数
			# 参数沿着梯度下降方向移动，学习率控制移动步长
			# Adam优化器的更新（简化理解）
			# w = w - lr * adjusted_grad   调整后的新权重
			# 可能变成 w = 1.5 - 0.001 * 0.3 = 1.4997
			optimizer.step()

			# 累积损失和样本数
			# 作用: 统计本epoch的总损失，用于后续计算平均损失
			# 注释: loss.item()获取损失的标量值
			#       loss.item() * len(y)将平均损失转换为批次总损失
			#       这是为了正确计算整个epoch的平均损失
			total_num += len(y)
			total_loss += loss.item() * len(y)

		# 打印每个epoch的训练结果
		# 作用: 监控训练过程，观察损失变化和训练速度
		# 注释: 输出格式: epoch编号、平均损失、训练时间
		#       total_loss / total_num: 本epoch的平均损失
		#       time.time() - start: 本epoch的训练耗时
		print('epoch: %4s loss: %.2f, time: %.2fs' %
			  (epoch_idx + 1, total_loss / total_num, time.time() - start))

	# 模型保存
	# 作用: 将训练好的模型参数保存到文件
	# 注释: torch.save保存模型的状态字典（state_dict）
	#       './model/phone-price-model2.pth'是保存路径
	#       只保存参数不保存整个模型，便于后续加载和推理
	#       state_dict包含所有可学习参数（权重和偏置）
	torch.save(model.state_dict(), './model/phone-price-model2.pth')

	# 返回训练好的模型
	# 作用: 使函数外部可以继续使用训练好的模型
	return model

def evaluate(valid_dataset, input_dim, class_num):
	# 加载模型和训练好的网络参数
	model = PhonePriceModel(input_dim, class_num)
	# load_state_dict:将加载的参数字典应用到模型上
	# load:加载用来保存模型参数的文件
	model.load_state_dict(torch.load('./model/phone-price-model2.pth'))
	# 构建加载器
	dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
	# 评估测试集
	correct = 0
	# 遍历测试集中的数据
	for x, y in dataloader:
		# 将其送入网络中
		# model.eval()
		output = model(x)
		# 获取预测类别结果
		y_pred = torch.argmax(output, dim=1)
		# 获取预测正确的个数
		correct += (y_pred == y).sum()
	# 求预测精度
	print('Acc: %.5f' % (correct / len(valid_dataset)))


if __name__ == '__main__':
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	train(train_dataset, input_dim, class_num)
	evaluate(valid_dataset, input_dim, class_num)