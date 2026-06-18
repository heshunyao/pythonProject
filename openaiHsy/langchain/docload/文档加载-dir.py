# from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('../../data04/',
                         glob="**/*.txt",  # 遍历txt文件
                         show_progress=True,  # 显示进度
                         use_multithreading=True,  # 使用多线程
                         loader_cls=TextLoader,  # 使用加载数据的方式
                         silent_errors=True,  # 遇到错误继续
                         loader_kwargs=text_loader_kwargs)  # 可以使用字典传入参数

docs = loader.load()
print("\n")
print(docs[0])