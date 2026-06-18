# test_mysql_tool.py
from dotenv import load_dotenv
from hello_agents.hello_agents.agents.my_simple_agent import MySimpleAgent
from hello_agents.hello_agents.core.llm import HelloAgentsLLM
from hello_agents.hello_agents.tools.builtin.mysql_tool import create_mysql_tool
from hello_agents.hello_agents.tools.registry import ToolRegistry

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM("deepseek-v4-pro", "sk-2b14a33b892c402080d9be58743b124c", "https://api.deepseek.com")

# 创建MySQL工具
# 方式1: 使用环境变量（推荐）
# mysql_tool = create_mysql_tool()

# 方式2: 直接传入参数
mysql_tool = create_mysql_tool(
    host="localhost",
    port=3306,
    database="ry-cloud",
    user="root",
    password="123456"
)

# 创建工具注册表并注册MySQL工具
tool_registry = ToolRegistry()
tool_registry.register_tool(mysql_tool)

# 创建带MySQL工具的Agent
mysql_agent = MySimpleAgent(
    name="数据库助手",
    llm=llm,
    system_prompt="你是一个数据库查询助手，可以帮助用户查询MySQL数据库中的数据。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)

# 测试查询
print("=== 测试MySQL查询 ===")
response = mysql_agent.run("请查询所有用户在sys_user的姓名user_name和邮箱email")
print(f"查询结果: {response}\n")

# 测试条件查询
# print("=== 测试条件查询 ===")
# response = mysql_agent.run("请查询年龄大于25岁的用户")
# print(f"查询结果: {response}\n")
#
# # 测试统计查询
# print("=== 测试统计查询 ===")
# response = mysql_agent.run("请统计用户总数")
# print(f"查询结果: {response}\n")

# 关闭数据库连接
mysql_tool.close()