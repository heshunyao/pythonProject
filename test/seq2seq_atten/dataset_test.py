import pandas as pd
import spacy
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import torch.nn as nn
from vocab_spacy import vocab_spacy as SpacyTokenizer

class Translate_Dataset(Dataset):
    def __init__(self, data_pd):
        self.data_pd = data_pd

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        src_sentence = self.data_pd.iloc[idx,0]
        tgt_sentence = self.data_pd.iloc[idx,1]
        return {"src": src_sentence, "tgt": tgt_sentence}

    
data = pd.read_csv("../../datasets/damo_mt_testsets_zh2en_news_wmt18.csv")
data  = data.dropna()
data = data[1:10]
#建立词汇表
tokenizer_zh = SpacyTokenizer(language="zh",min_freq=1)
tokenizer_zh.build(data["0"])
tokenizer_en = SpacyTokenizer(language="en",min_freq=1)
tokenizer_en.build(data["1"])
# print(tokenizer_zh.decode(tokenizer_zh.encode("你好")))


class Translate_Collate():
    def __init__(self,tokenizer_zh,tokenizer_en):
        self.tokenizer_zh = tokenizer_zh
        self.tokenizer_en = tokenizer_en
    def __call__(self, batch):
        src_seqs = [item['src'] for item in batch]
        tgt_seqs = [item['tgt'] for item in batch]
        src_ids = [torch.tensor(tokenizer_zh.encode(src)) for src in src_seqs]
        tgt_ids = [torch.tensor(tokenizer_en.encode(tgt)) for tgt in tgt_seqs]
        # 填充序列到最大长度
        src_padded = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=0)
        return {
            "src_seqs": src_seqs,
            "tgt_seqs": tgt_seqs,
            "src_ids": src_padded,
            "tgt_ids": tgt_padded
            }

collate_func = Translate_Collate(tokenizer_zh,tokenizer_en)


my_dataset  = Translate_Dataset(data)
data_loader = DataLoader(my_dataset,shuffle=False,batch_size=2,collate_fn=collate_func)


# 手动计算损失，验证原理
def manual_cross_entropy(logits, targets):
    # 对预测值应用 softmax 和 log
    log_probs = torch.log_softmax(logits, dim=1)  # [6, 5]
    
    # 根据 targets 索引选择对应的对数概率
    selected_log_probs = log_probs[range(len(targets)), targets]  # [6]
    
    # 取负平均
    return -torch.mean(selected_log_probs)

if __name__ == "__main__":

    # 假设参数
    batch_size = 2
    seq_len = 3
    vocab_size = 5

    # 创建模拟预测值（未归一化的分数）
    # 形状: [batch_size*seq_len, vocab_size] = [6, 5]
    logits = torch.randn(batch_size * seq_len, vocab_size)
    print("预测值 logits 形状:", logits.shape)
    print("预测值示例:\n", logits)

    # 创建模拟实际值（词索引）
    # 形状: [batch_size*seq_len] = [6]
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    print("\n实际值 targets 形状:", targets.shape)
    print("实际值示例:", targets)

    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 计算损失
    loss = criterion(logits, targets)
    print("\n损失值:", loss.item())
    manual_loss = manual_cross_entropy(logits, targets)
    print("手动计算的损失值:", manual_loss.item())
    print("两者是否相等:", torch.allclose(loss, manual_loss))  # 应输出 True