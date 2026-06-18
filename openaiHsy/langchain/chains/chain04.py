from langchain_openai import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI(temperature=0.9)

chain = ConversationChain(llm=llm)

# 使用 input 作为输入键
print(chain.invoke({"input": "请为一家生产苹果的公司起一个好听的中文名字，并简单解释这个名字的含义。"}))
print(chain.invoke({"input": "请为一家生产梨子的公司起一个好听的中文名字，并简单解释这个名字的含义。"}))
print(chain.invoke({"input": "请为一家生产葡萄的公司起一个好听的中文名字，并简单解释这个名字的含义。"}))

