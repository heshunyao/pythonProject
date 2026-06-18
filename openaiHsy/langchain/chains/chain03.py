from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 初始化语言模型，temperature设置为0.9以增加创造性
llm = OpenAI(temperature=0.9)

# 创建中文提示模板
prompt = PromptTemplate(
    input_variables=["product","company"],
    template="请为一家生产{product}{company}的公司起一个好听的中文名字，并简单解释这个名字的含义。",
)
llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))
# resp = llm_chain.predict(product="sad", company="ducks")
# print(resp)

resp2 = llm_chain.predict_and_parse(product="sad", company="ducks")
print(resp2)