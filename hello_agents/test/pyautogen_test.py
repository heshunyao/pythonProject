from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager
)

llm_config = {
    "config_list": [
        {
            "model": "deepseek-v4-pro",
            "api_key": "sk-2b14a33b892c402080d9be58743b124c",
            "base_url": "https://api.deepseek.com"
        }
    ]
}

pm = AssistantAgent(
    name="PM",
    system_message="你是项目经理，负责拆解任务",
    llm_config=llm_config
)

coder = AssistantAgent(
    name="Coder",
    system_message="你是Java性能专家",
    llm_config=llm_config
)

tester = AssistantAgent(
    name="Tester",
    system_message="你负责验证方案",
    llm_config=llm_config
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False
)

groupchat = GroupChat(
    agents=[pm, coder, tester],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

user.initiate_chat(
    manager,
    message="分析Java项目性能问题"
)