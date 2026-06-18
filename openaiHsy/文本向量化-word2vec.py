import jieba
from gensim.models import Word2Vec

# 示例英文文本数据
# texts = [
#     "Apple and pear are very common fruits.",
#     "I like to eat mango and passion fruit.",
#     "Apple has launched a new phone this year.",
#     "HuaWei's new phone has excellent camera performance."
# ]


# 示例中文文本数据
texts = [
    '苹果和梨子是很常见的水果。',
    '我喜欢吃芒果和百香果。',
    '苹果今年推出了新的手机。',
    '华为的新手机拍照性能非常好。'
]

# 使用jieba进行分词
texts_cut = [list(jieba.cut(text)) for text in texts]

# 训练Word2Vec模型
# 指定生成的词嵌入维度为30，学习的上下文窗口指定为5，用于构建词汇表的单词出现频次不低于min_count
# **sg**：取值为0表示使用CBOW模式，为1表示使用skip-gram模式，默认为0。
model = Word2Vec(sentences=texts_cut, vector_size=30, window=5, min_count=1, sg=1)

# 使用模型

# pear
vector = model.wv['梨子']  # 获取单词的向量表示
similar_words = model.wv.most_similar('梨子')  # 找到最相似的词汇

# 打印结果
print("向量表示:", vector)
print("与'梨子'最相似的词汇:", similar_words[:3])

