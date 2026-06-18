# from langchain.chains.llm import LLMChain
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.prompts import PromptTemplate

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is a good name for a company that makes {product}?",
        input_variables=["product"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)

chain = chat | chat_prompt_template
print(chain.invoke("colorful socks"))
