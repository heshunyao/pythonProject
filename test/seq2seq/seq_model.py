import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层维度

        # 嵌入层：将输入的 token id 转换为 embedding 向量（维度为 hidden_size）
        self.embedding = nn.Embedding(input_size, hidden_size)

        # GRU 层：输入为 embedding 向量，输出为隐藏状态
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # 将输入 token id 转换为 embedding 向量
        embedded = self.embedding(input)  # shape: [1, hidden_size]

        # 调整为 GRU 所需的三维输入：shape [seq_len=1, batch=1, hidden_size]
        # 参数	值	含义
        # 1 (第1维)	seq_len = 1	序列长度（一次处理一个时间步）
        # 1 (第2维)	batch_size = 1	批次大小（一次只处理一个样本）
        # -1 (第3维)	自动推断为 embedding_dim	嵌入维度大小（由 embedding 层输出决定）
        out = embedded.view(1, 1, -1)

        # 将 embedding 向量送入 GRU，得到输出和新的隐藏状态
        output, hidden = self.gru(out, hidden)
        return output, hidden  # 返回当前时间步的输出和隐藏状态

    def forward_sequence(self, input_seq, hidden=None):
        """
        一次处理多个 token（一个完整序列）
        input_seq: shape [seq_len] or [seq_len, 1]
        """
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(1)  # shape: [seq_len, 1]

        embedded = self.embedding(input_seq)  # shape: [seq_len, 1, hidden_size]

        if hidden is None:
            hidden = self.initHidden()

        output, hidden = self.gru(embedded, hidden)  # output: [seq_len, 1, hidden_size]
        return output, hidden

    def initHidden(self):
        # 初始化隐藏状态为 0，shape: [1层, 1批次, hidden_size]
        return torch.zeros(1, 1, self.hidden_size)


# 定义解码器
# 定义解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层维度

        # 嵌入层：将输出的 token id 转换为 embedding 向量
        self.embedding = nn.Embedding(output_size, hidden_size)

        # GRU 层：接收 embedding 输入，更新隐藏状态
        self.gru = nn.GRU(hidden_size, hidden_size)

        # 全连接层：将 GRU 输出映射到词表大小的输出空间（用于分类）
        self.out = nn.Linear(hidden_size, output_size)

        # softmax 层：将输出转换为 log 概率，用于计算损失
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 嵌入输入 token，shape: [1, hidden_size]
        output = self.embedding(input).view(1, 1, -1)

        # 应用 ReLU 激活函数（可以加一些非线性）
        output = F.relu(output)

        # 将嵌入向量传入 GRU，得到输出和新的隐藏状态
        output, hidden = self.gru(output, hidden)

        # GRU 的输出是 [seq_len, batch, hidden_size]，取第一个时间步 output[0]
        # 然后通过全连接和 softmax 得到词的 log 概率分布
        output = self.softmax(self.out(output[0]))  # shape: [1, output_size]

        return output, hidden  # 返回 log 概率向量和新的隐藏状态

    def initHidden(self):
        # 初始化隐藏状态为 0
        return torch.zeros(1, 1, self.hidden_size)
