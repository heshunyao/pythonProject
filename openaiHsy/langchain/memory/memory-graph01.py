from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

llm = ChatOpenAI(temperature=0.9 )

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    weather_data={
        "福州":"今天天气阴, 气温26度，空气很差",
        "厦门":"今天天气阴, 气温26度，空气很差",
        "莆田":"今天天气阴, 气温27度，空气很差",
        "泉州":"今天天气阴, 气温26度，空气很差",

    }
    return weather_data.get(city, f"没有找到该城市的天气信息")


agent = create_react_agent(
    model= llm,
    tools=[get_weather],
    checkpointer=checkpointer
)


# get_tuple   tuple 组元 （）    ---  list 列表 【】



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

# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about 福州?"}]},
    config
)
print("=====================")

print(ny_response.get("messages")[-1].content)