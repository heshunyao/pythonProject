from typing import TypedDict


class AgentState(TypedDict):
    question: str
    architect_result: str
    dba_result: str
    tester_result: str
    final_report: str

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="你的key",
    base_url="https://api.deepseek.com",
    temperature=0.2
)
# 架构师节点
def architect_node(state: AgentState):

    prompt = f"""
    你是一名Java架构师。

    问题：
    {state['question']}

    请分析系统性能瓶颈。
    """

    result = llm.invoke(prompt)

    return {
        "architect_result": result.content
    }

# DBA节点
def dba_node(state: AgentState):

    prompt = f"""
    你是一名DBA。

    架构师分析：

    {state['architect_result']}

    请从数据库角度给出优化建议。
    """

    result = llm.invoke(prompt)

    return {
        "dba_result": result.content
    }

# 测试节点
def tester_node(state: AgentState):

    prompt = f"""
    你是一名性能测试工程师。

    架构建议：

    {state['architect_result']}

    DBA建议：

    {state['dba_result']}

    请验证这些方案是否合理。
    """

    result = llm.invoke(prompt)

    return {
        "tester_result": result.content
    }

# 报告节点
def report_node(state: AgentState):

    report = f"""
        # Java性能优化报告
        
        ## 架构师分析
        
        {state['architect_result']}
        
        ## DBA建议
        
        {state['dba_result']}
        
        ## 测试验证
        
        {state['tester_result']}
        """

    return {
        "final_report": report
    }


# 构建Graph
from langgraph.graph import StateGraph
from langgraph.graph import END

builder = StateGraph(AgentState)

builder.add_node(
    "architect",
    architect_node
)

builder.add_node(
    "dba",
    dba_node
)

builder.add_node(
    "tester",
    tester_node
)

builder.add_node(
    "report",
    report_node
)

# 配置流程
builder.set_entry_point("architect")

builder.add_edge(
    "architect",
    "dba"
)

builder.add_edge(
    "dba",
    "tester"
)

builder.add_edge(
    "tester",
    "report"
)

builder.add_edge(
    "report",
    END
)

# 编译
graph = builder.compile()

#执行
result = graph.invoke({
    "question": "SpringBoot系统TPS只有200，CPU利用率很低"
})

print(result["final_report"])
