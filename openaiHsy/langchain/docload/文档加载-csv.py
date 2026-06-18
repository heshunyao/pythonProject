import os
from io import StringIO
from pathlib import Path
import pandas as pd
import chardet
# from langchain.document_loaders import UnstructuredCSVLoader
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_community.document_loaders import CSVLoader

# 可能文件有文件


# EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "./data04"
# print(EXAMPLE_DIRECTORY)
EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "./docload/data04"
print(EXAMPLE_DIRECTORY)
#本文本中有bug
# ln.decode(self._encoding or "utf-8") for ln in file.readlines(num_bytes)
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position 0: invalid continuation byte
# 没有解决

# 测试下文件 的编码
def detect_encoding() -> str:
    """检测文件的编码

    Returns:
        str: 检测到的文件编码
    """
    file_path = os.path.join(EXAMPLE_DIRECTORY, "test.csv")
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            # 自动识别文件的编码，而不需要手动指定 chardet 库来检测二进制数据的编码格式
            result = chardet.detect(raw_data)
            if result['encoding']:
                return result['encoding']
            return 'utf-8'  # 如果检测失败，默认返回utf-8
    except Exception as e:
        print(f"检测编码时出错: {str(e)}")
        return 'utf-8'  # 发生错误时默认返回utf-8


encoding= detect_encoding()
print(f"Detected encoding: {encoding}")


def test_unstructured_csv_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "demo.csv")
    df = pd.read_csv(file_path, encoding='utf-8')
    # 将 DataFrame 转换为 CSV 字符串
    csv_string = df.to_csv(index=False)

    string_io = StringIO(csv_string)
    loader = UnstructuredCSVLoader(string_io)
    # loader = UnstructuredCSVLoader(str(file_path))
    docs = loader.load()
    print(docs)
    assert len(docs) == 1


def test_csv_loader():
    file_path = os.path.join(EXAMPLE_DIRECTORY, "test.csv")
    loader = CSVLoader(file_path)
    docs = loader.load()
    print(docs)


# test_unstructured_csv_loader2()
test_csv_loader()
