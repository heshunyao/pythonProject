from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
#流式输出

# 创建流式输出的回调处理器
streaming_handler = StreamingStdOutCallbackHandler()

# 初始化聊天模型，启用流式输出
chatllm = ChatOpenAI(
    temperature=0,
    streaming=True,  # 启用流式输出
    callbacks=[streaming_handler]  # 添加流式输出处理器
)

# 定义系统提示模板
template = "你是一个乐于助人的翻译助手，能把中文翻译成英文"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 定义示例对话
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("你好")  # 修改为更合适的中文回复

# 定义用户输入模板
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合所有提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    example_human,
    example_ai,
    human_message_prompt
])

# 创建处理链
chain = chat_prompt | chatllm

# 测试流式输出
print("\n开始翻译（流式输出）：")
str = """

5月14日，国家主席习近平致电让-吕西安·萨维·德托夫，祝贺他就任多哥共和国总统。同日，习近平主席还致电福雷·埃索齐姆纳·纳辛贝，祝贺他就任多哥共和国部长会议主席。

习近平指出，中多友好关系由两国历代领导人共同缔造和精心培育。半个多世纪以来，双方始终坚持真诚友好、平等互信、合作共赢，在涉及彼此核心利益和重大关切问题上坚定相互支持，成为大小国家平等相待和全球南方团结合作的典范。2024年中非合作论坛北京峰会期间，中多关系提升为全面战略伙伴关系，开启了两国关系新篇章。我高度重视中多关系发展，愿同多哥领导人一道努力，以落实中非合作论坛北京峰会成果为契机，赓续传统友好，拓展各领域合作，不断丰富两国全面战略伙伴关系内涵，更好造福两国人民。

同日，国务院总理李强也致电祝贺福雷就任多哥共和国部长会议主席
"""
chain.invoke({"text": str})
