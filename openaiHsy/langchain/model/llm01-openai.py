from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """
Question: {question}
Answer: 让我们一步一步地思考.
"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)
llm = OpenAI(temperature=0)

llm_chain = prompt | llm
question = "什么是mybatis?"
reps = llm_chain.invoke({"question": question})
print(reps)