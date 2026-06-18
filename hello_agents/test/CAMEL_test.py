from camel.agents import ChatAgent
from camel.messages import BaseMessage

researcher = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Java Architect",
        content="负责分析性能问题"
    )
)

engineer = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="Java Developer",
        content="负责给出优化方案"
    )
)

task = """
分析SpringBoot系统TPS低的问题
"""

msg = researcher.step(task)

while True:
    response = engineer.step(msg.msg.content)

    print(response.msg.content)

    msg = researcher.step(response.msg.content)