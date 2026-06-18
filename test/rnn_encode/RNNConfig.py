# RNNConfig.py

batch_size = 16             # 每个 batch 的样本数
hidden_size = 64           # RNN 隐藏层维度 ✅ 就是你缺少的这个
num_layers = 1            # RNN 层数
learning_rate = 0.001       # 学习率
spec_tokens = ["<PAD>", "<UNK>", "<EOS>"]  # 特殊词：填充、未知、句子结束
num_epoches = 1000  # 你需要的训练轮数，可以自定义
end_token = "<EOS>"  # 结束标记
