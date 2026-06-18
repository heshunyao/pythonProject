from collections import Counter
import spacy
import torch

class vocab_spacy(object):
    def __init__(self, language="en",min_freq=2, specials=["<pad>", "<unk>", "<sos>", "<eos>"]):
        self.min_freq = min_freq
        self.specials = specials
        self.token2idx = {}
        self.idx2token = []
        self.counter = Counter()
        if language == "zh":
            self.nlp = spacy.load("zh_core_web_trf")
        elif language == "en":
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise ValueError("Invalid language specified.")
        
    def build(self, texts):
        # 1. 使用spaCy分词并统计词频
        for text in texts:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            self.counter.update(tokens)
        
        # 2. 添加特殊符号
        for token in self.specials:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        
        # 3. 添加满足频率要求的词
        for token, freq in self.counter.items():
            if freq >= self.min_freq and token not in self.specials:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1
        
        # 4. 设置未知词默认索引
        self.unk_idx = self.token2idx["<unk>"]
    
    def save(self, path):
        torch.save(self.token2idx, path)
    
    def load(self, path):
        self.token2idx = torch.load(path)
        self.idx2token = [None] * len(self.token2idx)
        for token, idx in self.token2idx.items():
            self.idx2token[idx] = token
        
    def __len__(self):
        return len(self.idx2token)
    
    def encode(self, text):
        """将文本转换为索引序列"""
        doc = self.nlp(text)
        return [self.token2idx.get(token.text, self.unk_idx) for token in doc]
    
    def decode(self, indices):
        """将索引序列转换回文本"""
        return [self.idx2token[idx] for idx in indices]
    
if __name__ == "__main__":
    # 使用示例
    corpus = [
        "Hello, world!",
        "This is a test.",
        "Natural Language Processing is fun."
    ]

    vocab = vocab_spacy(language="en",min_freq=1)
    vocab.build(corpus)

    # 测试
    text = "Hello, this is a spaCy test."
    indices = vocab.encode(text)
    decoded = vocab.decode(indices)

    print(f"原始文本: {text}")
    print(f"索引序列: {indices}")
    print(f"解码回文本: {decoded}")