from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 加载PDF文档
loader = PyPDFLoader("data04/斗破苍穹.pdf")
documents = loader.load()  # 使用load()而不是load_and_split()

# 创建文本分割器
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,  # 分块长度
    chunk_overlap=100,  # 重合的文本长度
    length_function=len,
)

# 处理文档
split_docs = text_splitter.split_documents(documents)  # 使用split_documents而不是create_documents
print("=== PDF文档分切结果 ===")
print(f"分切后的文档数量: {len(split_docs)}")
for doc in split_docs:
    print("分块内容:")
    print(doc.page_content)  # 使用page_content来获取文档内容
    print("\n")