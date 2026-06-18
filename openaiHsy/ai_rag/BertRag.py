# 用于读取 JSON 格式的知识库文件
import json

# 类型注解：List、Dict 用于增强代码可读性
from typing import List, Dict

# 用于向量和数值计算
import numpy as np

# 这里其实没用到 torch.cosine_similarity（你自己实现了一个）
from torch import cosine_similarity

# HuggingFace 的 BERT 分词器和模型
from transformers import BertTokenizer, BertModel


class BertRag:
    """
    基于 BERT 的简单 RAG Retriever（检索器）
    功能：将知识库文本向量化，并通过余弦相似度检索 TopK 文档
    """

    def __init__(self, knowledge_base_path: str):
        """
        初始化 RAG 检索器
        :param knowledge_base_path: 知识库 JSON 文件路径
        """

        # 指定本地已下载的中文 BERT 模型路径
        # 原本可以用 "bert-base-chinese"，这里是本地缓存
        self.model_name = "D:/modelscope_cache/models/bert-base-chinese"

        # 加载 BERT 分词器（Tokenizer）
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # 加载 BERT 模型（仅用于向量编码）
        self.model = BertModel.from_pretrained(self.model_name)

        # 从 JSON 文件中加载知识库（原始文本）
        self.knowledge_txt = self.load_knowledge_base(knowledge_base_path)

        # 对整个知识库进行向量化（构建向量索引）
        self.knowledge_embeddings = self._encode_knowledge_base()


    def load_knowledge_base(self, knowledge_base_path: str) -> List[Dict]:
        """
        从 JSON 文件加载知识库
        :param knowledge_base_path: 知识库路径
        :return: 知识条目列表，每一项是一个 Dict
        """
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            # 将 JSON 内容解析成 Python 对象
            return json.load(f)


    def _encode_knowledge_base(self) -> np.ndarray:
        """
        将知识库中的每一条文本编码成向量
        :return: 知识库向量矩阵 (N, 768)
        """

        # 对 knowledge_txt 中的每一条 content 调用 encode_text
        # 最终得到一个 numpy 数组
        return np.array([
            self.encode_text(item['content'])  # 提取文本并编码
            for item in self.knowledge_txt
        ])


    def encode_text(self, question: str) -> np.ndarray:
        """
        使用 BERT 将文本编码为向量
        :param question: 输入文本（问题或知识文本）
        :return: 768 维向量
        """

        # 使用 tokenizer 将文本转换为 BERT 可接受的输入
        input_ids = self.tokenizer(
            question,                 # 输入文本
            return_tensors="pt",       # 返回 PyTorch Tensor
            padding=True,              # 自动补齐长度
            truncation=True            # 超过 512 token 自动截断
        )

        # 将分词后的输入送入 BERT 模型
        output = self.model(**input_ids)

        # 取 pooler_output（即 [CLS] 向量），并转换为 numpy
        # shape: (1, 768) -> (768,)
        return output.pooler_output.detach().numpy()[0]


    def my_cosine_similarity(self, vec1, vec2) -> float:
        """
        手动实现余弦相似度计算
        :param vec1: 向量 1
        :param vec2: 向量 2
        :return: 余弦相似度值（-1 ~ 1）
        """

        # 转换为 numpy 数组
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # 计算向量点积
        dot_product = np.dot(vec1, vec2)

        # 计算向量的 L2 范数
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # 余弦相似度公式
        return dot_product / (norm_vec1 * norm_vec2)


    def retrieve(self, query: str, topk: int = 1):
        """
        根据查询语句检索最相似的 TopK 知识条目
        :param query: 用户查询问题
        :param topk: 返回最相似的条数
        :return: TopK 知识条目
        """

        # 将查询语句编码成向量
        query_embedding = self.encode_text(query)

        # 存储每一条知识与查询的相似度
        similarities = []

        # 遍历知识库向量
        for knowledge_emb in self.knowledge_embeddings:
            # 计算查询向量与知识向量的余弦相似度
            sim = self.my_cosine_similarity(query_embedding, knowledge_emb)
            similarities.append(sim)

        # 根据相似度排序，取 TopK 的索引（从大到小）
        topk_indices = np.argsort(similarities)[-topk:][::-1]

        # 根据索引取出对应的知识文本
        results = [self.knowledge_txt[i] for i in topk_indices]

        return results


def rag_main():
    """
    RAG 检索流程入口函数
    """

    # 初始化 RAG 检索器，并加载知识库
    rag = BertRag('knowledge_base/java.json')

    # 用户查询问题
    query = "Java中的内存模型是什么？"

    # 从知识库中检索最相似的 3 条
    results = rag.retrieve(query, topk=3)

    # 打印检索结果
    print(results)

    # 注意：results 中是 Dict，而不是 (content, score)
    for item in results:
        print(item['content'])


# 程序入口
if __name__ == '__main__':
    rag_main()
