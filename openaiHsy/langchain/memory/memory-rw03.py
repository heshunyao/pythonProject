from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


llm = ChatOpenAI(temperature=0.9 )

store = InMemoryStore()

class UserInfo(TypedDict):
    name: str

# class CustomState(AgentState):
#     user_id: str

# def get_user_info(
#     state: Annotated[CustomState, InjectedState]
# ) -> str:
#     """Look up user info."""
#     user_id = state["user_id"]
#     return "我是智能体" if user_id == "user_123" else "Unknown user"

def save_user_info(user_info : UserInfo,config: RunnableConfig)->str:
    # 得到存储对象
    store = get_store()
    user_id = config["configurable"].get("user_id")
    print(user_id)
    store.put(("users",),user_id,user_info)
    return "保存用户信息成功"


agent = create_react_agent(
    model=llm,
    tools=[save_user_info],
    store= store
)

#  保存用户信息



resp =agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    config={"configurable": {"user_id": "user_123"}}

)

print(resp.get("messages")[-1].content)
# resp = store.get(("users",), "user_123").value