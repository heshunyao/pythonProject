import torch
import torch.nn as nn
from unicodedata import bidirectional

import RNNConfig as CONF


class RNNModel(nn.Module):
    """
    初始化 RNN 模型。

    参数:
        input_size (int): 输入特征的维度（如 one-hot 则是 vocab_size，embedding 则是 embedding_size）。
        hidden_size (int): 隐藏层维度。
        output_size (int): 输出维度（通常是 vocab_size）。
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size

        # 定义 RNN 层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=CONF.num_layers,
            batch_first=True,  # 输入为 [batch, seq, feature]
        )

        # 输出层：将隐藏状态映射到输出空间
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        前向传播。
        参数:
            x: 输入张量，shape=[batch_size, seq_len, input_size]
            hidden: 初始隐藏状态，shape=[num_layers, batch_size, hidden_size]
        返回:
            output: 每个时间步的输出，shape=[batch_size, seq_len, output_size]
            hidden: 最后一层的隐藏状态
        """
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.fc(rnn_out)
        return output, hidden

    def init_hidden(self, batch_size=CONF.batch_size):
        """
        初始化隐藏状态，全 0。
        返回:
            hidden: shape=[num_layers, batch_size, hidden_size]
        """
        return torch.zeros(CONF.num_layers, batch_size, self.hidden_size)
