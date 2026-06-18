from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 初始化语言模型，temperature设置为0.9以增加创造性
llm = OpenAI(temperature=0.9)

# 创建中文提示模板
prompt = PromptTemplate(
    input_variables=["product"],
    template="请为一家生产{product}的公司起一个好听的中文名字，并简单解释这个名字的含义。",
)

# 方法1：使用传统的LLMChain方式， 注意版本，0.1.0以上版本以上 废弃
chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("智能手表"))

# 方法2：使用管道操作符方式（更现代的写法）
# chain = llm | prompt
# print(chain.invoke("智能手表"))

# chain.apply 传入多个问题
input_list = [
    {"product": "小鸟"},
    {"product": "大象"},
    {"product": "老虎"}
]
# print(chain.apply(input_list))

# 有没有一种办法，传入多个问题，可不可以不要一次行返回结果，调用一次返回一个当然有
print(chain.generate(input_list))




