# 导入文本分割器
from langchain.text_splitter import CharacterTextSplitter

# 读取demo2.txt文件，使用utf-8编码确保中文正确显示
with open("data04/demo2.txt", "r", encoding="utf-8") as f:
    demo2_text = f.read()

# 创建文本分割器实例
# CharacterTextSplitter用于将长文本分割成较小的块
text_splitter = CharacterTextSplitter(
    separator="\n\n",  # 使用双换行符作为分隔符，将文本按段落分割
    chunk_size=200,    # 每个文本块的最大字符数
    chunk_overlap=100, # 相邻文本块之间的重叠字符数，用于保持上下文连贯性
    length_function=len,  # 用于计算文本长度的函数，这里使用Python内置的len函数
)

# 使用分割器处理文本，将长文本分割成多个小块
# create_documents方法接收文本列表作为输入，返回文档对象列表
demo2_texts = text_splitter.create_documents([demo2_text])

# 打印分割结果
print("=== Demo2文档分切结果 ===")
print(f"分切后的文档数量: {len(demo2_texts)}")
# 遍历并打印每个文本块的内容
for text in demo2_texts:
    print("分块内容:")
    print(text)
    print("\n")

