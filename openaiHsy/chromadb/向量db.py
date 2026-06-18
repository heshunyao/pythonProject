import requests
from sympy import false

import chromadb
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

model_name = "D:/modelscope_cache/models/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def _encode_text(text: str) -> np.ndarray:
    """将文本通用BERT 进行向量化 -encode"""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()

def ask_ai_stream(prompt,isStream=False):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": isStream
    }
    headers = {"Content-Type": "application/json"}
    if isStream:
        with requests.post(url, json=data, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    print(line.decode("utf-8"))  # 逐行解析输出
    else:
        resp = requests.post(url, json=data, headers=headers,stream=false)
        if resp.status_code == 200:
            return resp.json()['response']
        else:
            raise Exception(f"请求ollama失败，状态码：{resp.status_code}")



chroma_client = chromadb.PersistentClient(path="./chromadb")

# chroma_client = chromadb.HttpClient()
# chroma_client = chromadb.Client()  # 使用内存模式
# 创建一个名为 "my_collection" 的向量集合
# embedding_function=None 表示我们将手动传入向量（即使用自定义的 BERT 编码）
# metadata={"hnsw:space": "cosine"} 表示使用 HNSW 算法进行相似度搜索，且采用余弦相似度作为度量方式
# my_collection = chroma_client.create_collection(
#     name="my_collection1",
#     embedding_function=None,  # 不使用默认嵌入函数，使用自定义向量
#     metadata={"hnsw:space": "cosine"}  # 使用 cosine 相似度进行搜索
# )
my_collection = chroma_client.get_collection("my_collection")

text_files = ["data/a.txt", "data/b.txt", "data/c.txt", "data/d.txt"]

documents = []
metadatas = []
ids = []

for i, fname in enumerate(text_files, start=1):
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            context = f.read()
            documents.append(context)
            metadatas.append({"source": f"{fname}{i}"})
            ids.append(f"id{i}")

    except  FileNotFoundError:
        print(f"文件{fname}不存在")
    except  Exception as e:
        print(f"读取文件{fname}时出错误：{str(e)}")

# 数据添加
if len(documents) > 0:
    print("开始添加数据")
    print(documents)
    embeddings = [_encode_text(doc)[0].tolist() for doc in documents]
    # documents = ["英雄联盟是一款热门游戏"]
    # metadatas = [{"source": "a.txt"}]
    # embeddings = [[0.123, 0.456, ..., 0.768维]]
    # ids = ["id1"]
    my_collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,  # 👈 这就是“设置”维度的地方
        ids=ids
    )
    print("数据添加成功")
else:
    print("没有数据添加")

question_text = "有哪些游戏"
qu_embeddings = _encode_text(question_text)

# 查询数据
results = my_collection.query(
    query_embeddings=qu_embeddings.tolist(),  # 不要加中括号
    n_results=2
)
# 取出那句话
sentence = results['documents'][0]
for i in sentence:
    print(ask_ai_stream(i))

