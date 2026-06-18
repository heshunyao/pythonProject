import jieba
import torch
import numpy as np
from seq_config import *

def make_vocab(sentences,spec_tokens=None):
    """
    对给定的句子列表进行分词，生成一个唯一的词汇表。
    
    参数:
    sentences (list): 包含多个句子的列表，用于生成词汇表。
    
    返回:
    dict: 包含词汇和对应索引的字典。
    int: 词汇表的大小。
    """
    vocab = {}
    
    if not sentences:
        return
    
    if not isinstance(sentences,list):
        return
    
    ind = 0    #从10开始，避免0被用来表示填充或未知的索引
    for sentence in sentences:
        for word in jieba.cut(sentence=sentence):
            if word in vocab:
                continue
            else:
                vocab[word] = ind		# vocab:  {"我":0,“爱":1,"中国":2....}
                ind += 1
    #合并特殊字符
    vocab_size = len(vocab)
    for i,t in enumerate(spec_tokens):
        vocab[t] = vocab_size+i
    
    vocab_size = len(vocab)
    return vocab,vocab_size

def encode_onehot_by_sentence(sentence,vocab_dict):
    """
    对给定的句子进行编码，将句子中的单词转换为对应的索引。

    参数:
    sentence (str): 需要编码的句子。

    返回:
    list: 编码后的句子，每个单词对应一个索引。

    """
    if not sentence:
        return
    dim = len(vocab_dict)
    tokens = list(jieba.cut(sentence))
    seq_len = len(tokens)
    seq_vector = np.zeros((seq_len,dim))

    for i,word in enumerate(tokens):
        ind = vocab_dict.get(word,vocab_dict.get("<UNK>"))
        # print(i,"----",ind)
        seq_vector[i][ind] = 1.0
    return seq_vector			#[[独日编码],[],[]]    (seq_len,one_hot_dim)

def encode_onehot_by_token_id(index,vocab_dict):
    """
    对给定的索引列表进行编码，将索引转换为对应的单词。

    参数:
    index_list (list): 需要编码的索引列表。

    返回:
    list: 索引编码后的句子，每个单词对应一个索引。

    """
    dim = len(vocab_dict)
    seq_vector = np.zeros(dim)
    seq_vector[index] = 1.0
    return seq_vector


def convert_sentence_to_token_ids(sentence,vocab_dict):
    """
    对给定的句子进行索引编码，将句子中的单词转换为对应的索引token_id。

    参数:
    sentence (str): 需要编码的句子。

    返回:
    list: 句子分词编码后token_ids，每个单词对应一个token_id。

    """
    if not sentence:
        return

    token_ids = [vocab_dict.get(word,None) for word in jieba.cut(sentence)]
    return token_ids

def convert_token_id_to_word(token_ids,vocab_dict_rev):
    """
    对给定的索引列表进行解码，将索引转换为对应的单词。

    参数:
    token_ids (list): 需要解码的索引列表。

    返回:
    str: 解码后的句子，每个单词对应一个索引。

    """
    if not token_ids:
        return

    sentence = [vocab_dict_rev[ind] for ind in token_ids]
    return sentence

def convert_word_to_token_id(word,vocab_dict):
    if word in vocab_dict:
        return vocab_dict.get(word,None)
    else:
        return vocab_dict.get("<UNK>")