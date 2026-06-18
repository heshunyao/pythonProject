from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document


# 定义一个函数来加载 JSON 文件
def load_json_file(file_path: str) -> list[Document]:
    # 创建 JSONLoader 实例，指定 JSON 文件的路径和内容键
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",  # 使用 jq 查询语言来提取 JSON 数组中的每个对象
        content_key="name",  # 指定要作为文档内容的键
        metadata_func=lambda x, idx: {"id": x["id"], "age": x["age"], "name": x["name"]}  # 定义一个函数来提取元数据
    )
    # 加载 JSON 文件并返回文档列表
    return loader.load()


# 指定 JSON 文件的路径
file_path = "data04/loadtest.json"

# 加载 JSON 文件并打印结果
docs = load_json_file(file_path)
for doc in docs:
    print(doc)
