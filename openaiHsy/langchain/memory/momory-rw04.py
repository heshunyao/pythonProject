from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama

store = InMemoryStore()
llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.7,
    base_url="http://localhost:11434"
)


class UserInfo(TypedDict):
    name: str


def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str:
    """Save user info."""
    # Same as that provided to `create_react_agent`
    store = get_store()
    user_id = config["configurable"].get("user_id")
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."


def get_user_info(config: RunnableConfig
                  ) -> str:
    """Look up user info."""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


agent = create_react_agent(
    model=llm,
    tools=[save_user_info, get_user_info],
    store=store
)

# Run the agent
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is 大模型"}]},
    config={"configurable": {"user_id": "user_123"}}
)
# print(resp)
# You can access the store directly to get the value
# store.get(("users",), "user_123").value
resp2 = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    config={"configurable": {"user_id": "user_123"}}
)
print(resp2.get("messages")[-1].content)
