import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from torch.nn.utils.rnn import pad_sequence
from vocab_spacy import vocab_spacy as SpacyTokenizer
import pandas as pd
import os

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
data = data[1:500]
#建立词汇表
tokenizer_zh = SpacyTokenizer(language="zh",min_freq=1)
tokenizer_en = SpacyTokenizer(language="en",min_freq=1)
tokenizer_zh.build(data["0"])
tokenizer_en.build(data["1"])

# print(tokenizer_zh.decode(tokenizer_zh.encode("你好")))


class Translate_Collate():
    def __init__(self,tokenizer_zh,tokenizer_en):
        self.tokenizer_zh = tokenizer_zh
        self.tokenizer_en = tokenizer_en
        self.sos_token = tokenizer_en.token2idx["<sos>"]
        self.eos_token = tokenizer_en.token2idx["<eos>"]
        self.pad_token = tokenizer_en.token2idx["<pad>"]
    def __call__(self, batch):
        src_seqs = [item['src'] for item in batch]
        trg_seqs = [item['tgt'] for item in batch]
        src_ids = [tokenizer_zh.encode(src) for src in src_seqs]
        trg_ids = [[self.sos_token]+ tokenizer_en.encode(trg) + [self.eos_token] for trg in trg_seqs]
        
        max_len_src = max([len(src) for src in src_ids])
        src_ids = [src + [self.pad_token] * (max_len_src - len(src)) for src in src_ids]
        max_len_trg = max([len(trg) for trg in trg_ids])
        trg_ids = [trg + [self.pad_token] * (max_len_trg - len(trg)) for trg in trg_ids]
        
        return {
            "src_seqs": src_seqs,
            "trg_seqs": trg_seqs,
            "src_ids": torch.tensor(src_ids),
            "trg_ids": torch.tensor(trg_ids)
            }

collate_func = Translate_Collate(tokenizer_zh,tokenizer_en)
my_dataset  = Translate_Dataset(data)
train_dataset,test_dataset = random_split(my_dataset, [0.8, 0.2])

train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=16,collate_fn=collate_func)
test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=16,collate_fn=collate_func)


if  __name__ == "__main__":
    print("<sos>=",tokenizer_en.token2idx["<sos>"])
    for item in train_data_loader:
        print(item)
        print(item["trg_ids"])
        break