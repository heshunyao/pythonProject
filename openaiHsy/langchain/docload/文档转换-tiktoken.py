 # 读取demo2.txt文件
with open("data04/demo2.txt", "r", encoding="utf-8") as f:
    state_of_the_union = f.read()
# pip install tiktoken
# 导入文本分割器，用于将长文本切分成小块
from langchain.text_splitter import CharacterTextSplitter

# 创建一个基于tiktoken的文本分割器
# tiktoken是OpenAI开发的分词器，可以准确计算token数量
# chunk_size=128: 每个文本块的目标token数量为128
# chunk_overlap=10: 相邻文本块之间重叠10个token，保持上下文连贯性
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=10   #注意chunk_size >138
)

# 使用分割器将长文本state_of_the_union切分成多个小块
# 返回一个文本块列表
texts = text_splitter.split_text(state_of_the_union)

# 打印所有分割后的文本块
print(texts)

# 遍历每个文本块，打印其长度（字符数）
# 这可以帮助我们验证分割是否符合预期
for text in texts:
    print(len(text))

print("=======直接TokenText  注意 chunk_size >138 =============")
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=178, chunk_overlap=0)

textss = text_splitter.split_text(state_of_the_union)
print(textss[0])