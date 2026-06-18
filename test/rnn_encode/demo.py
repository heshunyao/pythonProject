import jieba



def make_vocab(sentences, spec_tokens=None):
    """
    对给定的句子列表进行分词，生成一个唯一的词汇表。

    参数:
        sentences (list): 包含多个句子的列表。
        spec_tokens (list): 特殊标记列表，例如 ["<PAD>", "<UNK>"]。

    返回:
        dict: 包含词汇和对应索引的字典。
        int: 词汇表的大小。
    """
    vocab = {}

    if not sentences or not isinstance(sentences, list):
        return {}, 0

    ind = 0  # 起始索引为0，也可以从10开始，预留空间

    # 分词并构建基础词表
    for sentence in sentences:
        for word in jieba.cut(sentence):
            if word not in vocab:
                vocab[word] = ind
                ind += 1

    # 添加特殊标记
    if spec_tokens:
        for i, token in enumerate(spec_tokens):
            if token not in vocab:  # 避免重复添加
                vocab[token] = len(vocab)

    vocab_size = len(vocab)
    return vocab, vocab_size

import numpy as np
def encode_onehot_by_sentence(sentence, vocab_dict):
    """
    将句子编码为 one-hot 向量序列。

    参数:
        sentence (str): 输入的中文句子。
        vocab_dict (dict): 词汇表，键是词，值是索引。

    返回:
        np.ndarray: shape 为 (seq_len, vocab_size) 的 one-hot 编码。
    """
    if not sentence or not isinstance(sentence, str):
        return np.array([])

    tokens = list(jieba.cut(sentence))
    vocab_size = len(vocab_dict)
    seq_len = len(tokens)

    # 初始化 one-hot 矩阵
    seq_vector = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, word in enumerate(tokens):
        index = vocab_dict.get(word, vocab_dict.get("<UNK>", -1))
        if index != -1 and index < vocab_size:
            seq_vector[i][index] = 1.0

    return seq_vector  # shape: (seq_len, vocab_size)


def encode_onehot_by_token_id(index, vocab_dict):
    """
    将单个词索引编码为 one-hot 向量。

    参数:
        index (int): 词在词表中的索引。
        vocab_dict (dict): 词汇表。

    返回:
        np.ndarray: 长度为 vocab_size 的 one-hot 向量。
    """
    vocab_size = len(vocab_dict)
    one_hot = np.zeros(vocab_size, dtype=np.float32)
    if 0 <= index < vocab_size:
        one_hot[index] = 1.0
    return one_hot
def convert_sentence_to_token_ids(sentence, vocab_dict):
    """
    对给定的句子进行索引编码，将句子中的单词转换为对应的索引token_id。
    参数:
        sentence (str): 需要编码的句子。
        vocab_dict (dict): 词汇字典，key是词，value是token_id。
    返回:
        list: 句子分词编码后token_ids，每个单词对应一个token_id。
    """
    if not sentence:
        return []
    # jieba.cut对句子进行分词，返回词的生成器
    token_ids = [vocab_dict.get(word, vocab_dict.get("<UNK>")) for word in jieba.cut(sentence)]
    print(token_ids,"token_ids")
    return token_ids

def convert_word_to_token_id(word, vocab_dict):
    """
    将单词转换为对应的 token_id。
    如果单词不在词表中，返回 <UNK> 的 token_id。
    """
    if word in vocab_dict:
        return vocab_dict[word]
    else:
        return vocab_dict.get("<UNK>")


def get_token(word, vocab_dict):
    return vocab_dict.get(word, vocab_dict.get("<UNK>"))



# 1.构建词汇表

import RNNConfig as CONF
import torch

# 示例句子
sentences = [
    "中国的首都是北京",
    "福建的省会是福州"
]

# 生成词汇表并赋值给配置模块中的变量
CONF.vocab_dict, CONF.vocab_size = make_vocab(sentences, spec_tokens=CONF.spec_tokens)

# 构造反向词典：索引 → 词
CONF.vocab_dict_rev = {v: k for k, v in CONF.vocab_dict.items()}

# 打印结果
print("词汇表：", CONF.vocab_dict)
print("词汇表大小：", CONF.vocab_size)


# 2.初始化训练模型=======================
# 初始化模型
from RNNModel import RNNModel
model = RNNModel(
    input_size=CONF.vocab_size,
    hidden_size=CONF.hidden_size,
    output_size=CONF.vocab_size  # 输出是词表大小（做分类）
)

# 初始化优化器
optimizer = torch.optim.Adam(model.parameters(), lr=CONF.learning_rate)

# 初始化损失函数：用于多分类问题，要求 target 是 class index（如 0, 1, 2, ...）
critition = torch.nn.CrossEntropyLoss()

# 3、语料处理/分词
# 将每个句子编码为 one-hot 形式的序列，生成 input_tensor
# input = [encode_onehot_by_sentence(sentence, CONF.vocab_dict) for sentence in sentences]
# input_tensor = torch.tensor(input, dtype=torch.float32)
eos_id = CONF.vocab_dict["<EOS>"]
vocab_size = len(CONF.vocab_dict)
input = []
for sentence in sentences:
    # 原句的 one-hot 编码
    onehot_matrix = encode_onehot_by_sentence(sentence, CONF.vocab_dict)

    # 构造 <EOS> 的 one-hot
    eos_vector = np.zeros(vocab_size, dtype=np.float32)
    eos_vector[eos_id] = 1.0

    # 拼接上去
    onehot_with_eos = np.vstack([onehot_matrix, eos_vector])
    input.append(onehot_with_eos)

input_tensor = torch.tensor(np.array(input), dtype=torch.float32)
print("input_tensor:", input_tensor.shape)
# 初始化隐藏状态时传入对应的 batch_size
batch_size = input_tensor.size(0)  # 取输入的 batch_size
# print(input_tensor)
# print(input_tensor.shape)


# 目标输出：将每个句子转为 token id 序列
eos_token_id = CONF.vocab_dict.get("<EOS>", CONF.vocab_dict.get("<UNK>"))
target = [convert_sentence_to_token_ids(sentence, CONF.vocab_dict) for sentence in sentences]
for t in target:
    t.append(eos_token_id)
# 构建目标张量
target_tensor = torch.tensor(target)
# 打印目标的形状确认：[batch_size, seq_len]
print("target_tensor:", target_tensor.shape)


# 4、训练train和预测Predict函数
def train(model, input, target):
    print("-------train start--------")
    print("input:", input.shape)  # 输出输入张量的形状：[batch_size, seq_len, vocab_size]

    # 训练多个 epoch（轮）
    for epoch in range(CONF.num_epoches):
        # 从模型中初始化隐藏状态（根据当前 batch_size）
        batch_size = input.size(0)
        hidden = model.init_hidden(batch_size=batch_size)

        model.zero_grad()  # 清空上一次的梯度

        # 前向传播：将输入传入模型，得到输出和新的隐藏状态
        out, hidden = model(input, hidden)

        # 将输出和目标变形为二维，计算交叉熵损失
        # out: [batch_size * seq_len, vocab_size]
        # target: [batch_size * seq_len]
        loss = critition(out.view(-1, CONF.vocab_size), target.view(-1))

        # 反向传播计算梯度
        loss.backward()

        # 使用优化器更新模型参数
        optimizer.step()

        # 每 10 个 epoch 打印一次损失值
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{CONF.num_epoches}], Loss: {loss.item():.4f}')

    print("-------train over--------")

def predict(model, token_tensor, hidden=None, device="cpu"):
    model.eval()  # 设置模型为推理模式（禁用 Dropout 等）

    # 初始化隐藏状态（默认 batch_size=2，但建议与 token_tensor 匹配）
    if hidden is None:
        hidden = model.init_hidden(batch_size=1).to(device)

    # token_tensor 是 shape 为 [vocab_size] 的 one-hot 向量
    # 需要调整维度为 [1, 1, vocab_size]：1个样本、1个时间步、one-hot 向量
    token_tensor = token_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # 前向传播预测
    output, hidden = model(token_tensor, hidden)  # output shape: [1, 1, vocab_size]

    # 从输出中找到预测概率最大的词索引
    _, predicted_idx = torch.max(output, 2)  # 返回最大值所在索引，shape: [1, 1]

    # 转换为整数索引
    predicted_token_id = predicted_idx.item()

    # 通过反向词典查找对应的词
    predicted_word = CONF.vocab_dict_rev[predicted_token_id]

    print("Predicted:", predicted_word)
    return predicted_word, hidden


# 5、训练train
#train
train(model,input_tensor,target_tensor)
#训练后保存模型
torch.save(model.state_dict(),"rnnmodel.pth")


# 6.测试
# ✅ 1. 实例化模型（必须先实例后才能加载参数）

# ✅ 2. 加载已保存的模型参数（确保 rnnmodel.pth 是对应结构的）
model.load_state_dict(torch.load("rnnmodel.pth"))

# ✅ 3. 初始化输入词
input_word = "中国_是北京"  # 初始输入词
seq_predict = input_word  # 最终预测出的句子
hidden = None  # 初始化隐藏状态
next_word = ""  # 预测的下一个词

# ✅ 4. 进入预测循环，直到遇到结束标记 <EOS>
while next_word != CONF.end_token:
    # 将词转换为 token id（索引）
    token_id = get_token(input_word, CONF.vocab_dict)
    print("token_id:", token_id)

    # 将 token id 编码为 one-hot 向量
    input_tensor = encode_onehot_by_token_id(token_id, CONF.vocab_dict)
    print("input_tensor (one-hot):", input_tensor)

    # 转换为 Torch Tensor，类型为 float32
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

    # ✅ 使用模型预测下一个词
    next_word, hidden = predict(model, input_tensor, hidden=hidden)

    # 拼接预测结果到完整句子中
    seq_predict += next_word
    input_word = next_word  # 下一轮输入就是刚刚预测出的词

# ✅ 5. 打印最终生成的完整句子
print("生成的句子：", seq_predict)
