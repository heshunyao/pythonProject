from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode

checkpointer = InMemorySaver()

llm = ChatOpenAI(temperature=0.9)

# 文本摘要实例
summarize = SummarizationNode(
    token_counter= count_tokens_approximately,
    model=llm,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key= "llm_input_messages"
)
store = InMemorySaver()
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    weather_data={
        "福州":"今天天气阴, 气温26度，空气很差",
        "厦门":"今天天气阴, 气温26度，空气很差",
        "莆田":"今天天气阴, 气温27度，空气很差",
        "泉州":"今天天气阴, 气温26度，空气很差",

    }
    return weather_data.get(city, f"没有找到该城市的天气信息")

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]
# context  用于存储上下文信息，如历史对话，上一次摘要等

agent = create_react_agent(
    model= llm,
    tools=[get_weather],
    checkpointer=checkpointer,
    pre_model_hook=summarize   # 模型执行之前做的预处理，文本摘要
)


# Run the agent
config = {
    "configurable": {
        "thread_id": "1"
    }
}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in 莆田"}]},
    config
)
print(sf_response)
# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about 福州?"}]},
    config
)
print("=====================")
print(ny_response)
print(ny_response.get("messages")[-1].content)