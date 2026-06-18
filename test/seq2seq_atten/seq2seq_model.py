import torch
import torch.nn as nn
import random

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        # 编码器使用 1 层 RNN，因此设置 num_layers 为 1
        self.num_layers = 1

        # 隐藏层维度大小
        self.hid_dim = hid_dim

        # 嵌入层：将输入的 token ID 映射为向量，大小为 [input_dim, emb_dim]
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # 单层的单向 GRU 网络，输入维度是 emb_dim，输出 hidden_dim
        # batch_first=True 意味着输入/输出的第一个维度是 batch
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=False,
            batch_first=True,
            num_layers=self.num_layers
        )

        # Dropout 用于防止过拟合，作用于 embedding 层输出
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 获取当前批次的大小，src shape: [batch_size, src_len]
        batch_size = src.shape[0]

        # 初始化隐藏状态为 0，形状为 [num_layers, batch_size, hidden_dim]
        hidden = self.init_hidden(batch_size)

        # 对输入进行词嵌入，得到 shape: [batch_size, src_len, emb_dim]
        # 输入: src.shape = [batch_size, src_len]    # 比如 [2, 5]
        # 输出: embedded.shape = [batch_size, src_len, emb_dim]  # 比如 [2, 5, 256]
        embedded = self.embedding(src)

        # 对嵌入进行 Dropout
        # Dropout 是一种正则化手段，在训练过程中会随机将输入的一部分设置为 0，防止模型过拟合某些特征。
        # 对于嵌入矩阵来说，Dropout 会随机“屏蔽”部分特征维度
        # 原始嵌入: [0.3, 0.2, 0.6, 0.4]
        # Dropout后: [0.3, 0.0, 0.6, 0.0]
        embedded = self.dropout(embedded)

        # 将嵌入后的输入送入 GRU 网络，outputs 包含所有时间步的隐藏状态
        # hidden 是最后一个时间步的隐藏状态
        outputs, hidden = self.rnn(embedded, hidden)

        # 打印调试信息
        print("encoder outputs=", outputs.shape)  # [batch_size, src_len, hidden_dim]
        print("encoder hidden=", hidden.shape)  # [num_layers, batch_size, hidden_dim]

        # 返回所有时间步的输出和最后的隐藏状态
        return outputs, hidden

    def init_hidden(self, batch_size):
        # 返回初始隐藏状态，值为全 0，形状为 [num_layers, batch_size, hidden_dim]
        return torch.zeros(self.num_layers, batch_size, self.hid_dim)


# 定义注意力机制类，使用加性注意力（Bahdanau Attention）
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # 将 decoder 的 hidden 和 encoder 的 output 拼接后通过线性层映射到 hid_dim（energy 计算）
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        # 将 energy 映射到 1 维的分数（注意力权重的“能量”值）
        self.v = nn.Linear(hid_dim, 1, bias=False)

    # forward 函数执行注意力权重的计算
    # 参数：
    # hidden: decoder 当前时间步的隐藏状态，shape = [1, batch_size, hid_dim]
    # encoder_outputs: encoder 所有时间步的隐藏状态，shape = [batch_size, src_len, hid_dim]
    # batch_size: 一次处理的句子数（样本数）
    #
    # src_len: 输入句子的长度（分词后的 token 数）
    #
    # hidden_size: 每个 token 编码后的向量维度
    def forward(self, hidden, encoder_outputs):
        print("Attention hidden:", hidden.shape)
        print("Attention encoder_outputs:", encoder_outputs.shape)

        batch_size = encoder_outputs.shape[0]  # 批大小
        src_len = encoder_outputs.shape[1]     # 源序列的长度（时间步）

        # 将 hidden 从 [1, batch_size, hid_dim] 转为 [batch_size, 1, hid_dim]
        # 再在时间维度上（dim=1）复制 src_len 次，扩展为 [batch_size, src_len, hid_dim]
        hidden_expanded = hidden.permute(1, 0, 2).repeat(1, src_len, 1)

        # 将 decoder 的 hidden 状态和 encoder 的每个时间步拼接
        # 得到 shape 为 [batch_size, src_len, hid_dim * 2]
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)

        # 将拼接结果映射为 energy，维度变为 [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(combined))  # 加性注意力核心：用全连接层学习 energy

        # 再通过一层线性映射，将 energy 转成 attention 得分，shape 为 [batch_size, src_len, 1]
        # 然后 squeeze(2) 去掉最后一维，变成 [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        print("attention =", attention.shape)

        # 使用 softmax 沿时间步维度（dim=1）归一化，得到最终注意力权重
        return torch.softmax(attention, dim=1)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim,batch_first=True)
        self.fc_out = nn.Linear(2*hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # 扩展 input 的维度：从 (batch_size,) → (batch_size, 1)，用于后续嵌入
        input = input.unsqueeze(1)
        print("decoder input=", input.shape)  # torch.Size([batch_size, seq_len=1])
        print("decoder encoder_outputs=", encoder_outputs.shape)  # (batch_size, src_seq_len, hidden_size)
        print("decoder hidden=", hidden.shape)  # torch.Size([batch_size, hid_dim])00

        # 将 input 映射为词向量表示： (batch_size, 1, emb_dim)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)  # 防止过拟合
        print("decoder embedded=", embedded.shape)  # torch.Size([batch_size, 1, emb_dim])

        # 记录当前解码器的输入长度，通常为1（每次只输入一个token）
        input_len = embedded.shape[1]

        # 计算注意力权重： (batch_size, src_seq_len)
        a = self.attention(hidden, encoder_outputs)

        # 扩展维度以进行 batch 矩阵乘法：→ (batch_size, 1, src_seq_len)
        a = a.unsqueeze(1)

        # 上下文向量：注意力加权求和 → (batch_size, 1, hidden_size)
        weighted = torch.bmm(a, encoder_outputs)
        print("decoder weighted1=", weighted.shape)  # torch.Size([batch_size, 1, hidden_size])

        # 扩展上下文向量维度，使其与 embedded 在时间步维度一致：→ (batch_size, 1, hidden_size)
        weighted = weighted.expand(-1, input_len, -1)
        print("decoder weighted2=", weighted.shape)  # torch.Size([batch_size, 1, hidden_size])

        # 将词向量和上下文向量拼接，作为 RNN 的输入：(batch_size, 1, emb_dim + hidden_size)
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # 传入RNN进行解码，得到当前时间步的输出与新的隐藏状态
        output, hidden = self.rnn(rnn_input, hidden)
        print("decoder rnn output=", output.shape)  # torch.Size([batch_size, 1, hidden_dim])
        print("decoder rnn hidden=", hidden.shape)  # 通常是 torch.Size([1, batch_size, hidden_dim])

        # 拼接RNN输出、上下文向量和嵌入向量，然后送入全连接层预测词分布
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        print("decoder fc_out=", prediction.shape)  # torch.Size([batch_size, 1, vocab_size])

        # 返回预测值和新的隐藏状态
        return prediction, hidden


# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 获取目标序列的批大小和长度
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        print("seq2seq trg_len=", trg_len)
        print("seq2seq batch_size=", batch_size)
        print("seq2seq trg_vocab_size=", trg_vocab_size)

        # 将输入源序列传入编码器，获取编码器输出和隐藏状态
        encoder_outputs, hidden = self.encoder(src)
        print("seq2seq encoder_outputs,hidden=", encoder_outputs.shape, hidden.shape)

        # 初始化一个张量来保存 decoder 的每一步输出，形状为 [batch_size, trg_len, vocab_size]
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 取每个目标序列的第一个 token（通常是 <BOS> 或 <SOS>），作为 decoder 的初始输入
        input = trg[:, 0]
        print("seq2seq input=", input.shape)

        # 对每一个时间步进行解码，跳过第一个（因为第一个是输入）
        for t in range(1, trg_len):
            # 将当前输入 token，隐藏状态和编码器输出传入 decoder
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # 移除多余的维度 output: [batch_size, 1, vocab_size] → [batch_size, vocab_size]
            output = output.squeeze(1)

            # 保存当前时间步的输出到 outputs 中
            outputs[:, t] = output

            # 选择输出中概率最高的 token 索引，作为下一个时间步的输入（如果不用 teacher forcing）
            top1 = output.argmax(1)
            print("decoder top1: ", top1.shape)

            # 决定是否使用 teacher forcing（用真实标签当下一步输入）
            teacher_force = random.random() < teacher_forcing_ratio

            # 如果使用 teacher forcing，则用目标序列的当前时间步；否则用模型预测值
            input = trg[:, t] if teacher_force else top1
            print("decoder next_token: ", input.shape)

        # 返回 decoder 所有时间步的输出结果，形状为 [batch_size, trg_len, vocab_size]
        return outputs
