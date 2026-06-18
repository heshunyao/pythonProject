from langchain_community.document_loaders import PyPDFLoader
# from pdfminer.utils import open_filename
#  第一种用法
loader = PyPDFLoader("data04/ict.pdf")
pages = loader.load_and_split()

print(pages[0])

# # 第二种用法
# please add an environment variable `MATHPIX_API_KEY`

# from langchain_community.document_loaders import MathpixPDFLoader
# loader = MathpixPDFLoader("data04/ict.pdf")
#
# data = loader.load()
# print(data[0])

print("========第3种用法=====")
# '''
# 第三种用法
# '''
# from langchain_community.document_loaders import UnstructuredPDFLoader
#
# loader = UnstructuredPDFLoader("data04/ict.pdf")
#
# data = loader.load()
# print(data[0])


# 报错
# from pdfminer.utils import open_filename
# ImportError: cannot import name 'open_filename' from 'pdfminer.utils'
# 解决1
# pip uninstall pdfminer.six  # 先卸载当前版本
# pip install pdfminer.six==20220319  # 安装旧版本
# 解决2
# pip install --upgrade unstructured
# 还不行，用方法2