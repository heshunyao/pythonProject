import agentscope
from agentscope.agents import DialogAgent

agentscope.init(
    model_configs=[
        {
            "config_name": "deepseek",
            "model_type": "openai_chat",
            "model_name": "deepseek-chat",
            "api_key": "你的key",
            "client_args": {
                "base_url": "https://api.deepseek.com"
            }
        }
    ]
)


clue_agent = DialogAgent(
    name="ClueAgent",
    model_config_name="deepseek",
    sys_prompt="""
你是一名案件分析专家。

职责：
1. 分析线索内容
2. 提取关键信息
3. 判断涉及哪些违法行为
"""
)

risk_agent = DialogAgent(
    name="RiskAgent",
    model_config_name="deepseek",
    sys_prompt="""
你是一名风险评估专家。

职责：
1. 根据案件内容判断风险等级
2. 输出高、中、低
3. 给出原因
"""
)

law_agent = DialogAgent(
    name="LawAgent",
    model_config_name="deepseek",
    sys_prompt="""
你是一名法规专家。

职责：
1. 根据案件内容匹配法规
2. 输出相关法律依据
"""
)

report_agent = DialogAgent(
    name="ReportAgent",
    model_config_name="deepseek",
    sys_prompt="""
你是一名案件审核人员。

职责：
整合所有分析结果，
生成最终处理意见。
"""
)

clue = """
某商户涉嫌销售来源不明卷烟，
库存约300条，
无法提供合法进货凭证。
"""

# 6 执行流程
# 第一步
clue_result = clue_agent(clue)

print("=====线索分析=====")
print(clue_result.content)

# 第二步
risk_result = risk_agent(clue_result.content)

print("=====风险评估=====")
print(risk_result.content)

# 第三步
law_result = law_agent(clue_result.content)

print("=====法规匹配=====")
print(law_result.content)

# 第四步
final_prompt = f"""
线索分析：

{clue_result.content}

风险评估：

{risk_result.content}

法规依据：

{law_result.content}

请生成处理意见。
"""

report_result = report_agent(final_prompt)

print("=====最终报告=====")
print(report_result.content)