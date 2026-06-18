from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
import chardet
import os

def detect_encoding(file_path: str) -> str:
    """检测文件的编码
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 检测到的文件编码
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            if result['encoding']:
                return result['encoding']
            return 'utf-8'  # 如果检测失败，默认返回utf-8
    except Exception as e:
        print(f"检测编码时出错: {str(e)}")
        return 'utf-8'

def load_html_file(file_path: str):
    """加载HTML文件，包含编码检测和错误处理"""
    # 检测文件编码
    encoding = detect_encoding(file_path)
    print(f"检测到的文件编码: {encoding}")
    
    try:
        # 尝试使用BSHTMLLoader
        loader = BSHTMLLoader(file_path, encoding=encoding)
        docs = loader.load()
        print("使用BSHTMLLoader成功加载文档:")
        print(docs[0])
        return docs
    except Exception as e:
        print(f"使用BSHTMLLoader加载失败: {str(e)}")
        try:
            # 如果BSHTMLLoader失败，尝试使用UnstructuredHTMLLoader
            print("尝试使用UnstructuredHTMLLoader...")
            loader = UnstructuredHTMLLoader(file_path)
            docs = loader.load()
            print("使用UnstructuredHTMLLoader成功加载文档:")
            print(docs[0])
            return docs
        except Exception as e:
            print(f"使用UnstructuredHTMLLoader加载失败: {str(e)}")
            # 尝试其他常见编码
            common_encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
            for enc in common_encodings:
                if enc == encoding:
                    continue
                try:
                    print(f"尝试使用 {enc} 编码...")
                    loader = BSHTMLLoader(file_path, encoding=enc)
                    docs = loader.load()
                    print(f"使用 {enc} 编码成功!")
                    print(docs[0])
                    return docs
                except Exception as e:
                    print(f"使用 {enc} 编码失败: {str(e)}")
                    continue
    return None

# 使用函数加载HTML文件
file_path = "data04/login.html"
docs = load_html_file(file_path)