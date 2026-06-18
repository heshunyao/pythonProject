# -*- coding: utf-8 -*-

def calculate(what):
    """
    计算给定的数学表达式
    
    参数:
        what (str): 要计算的数学表达式字符串，例如 "120 * 0.9"
        
    返回:
        float/int: 计算结果
        
    警告:
        此函数使用eval()执行计算，在生产环境中应该使用更安全的方式
        比如使用ast.literal_eval()或专门的数学表达式解析库
        eval() 函数是Python内置函数之一，它的主要功能是评估一个字符串表达式，并返回表达式的值
    """
    return eval(what)


if __name__ == '__main__':
    # 测试代码：计算120的9折价格
    print(calculate("120 * 0.8"))
