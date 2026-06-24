"""压缩策略对比演示"""
import sys, os
sys.path.insert(0, '/Users/heshunyao/PycharmProjects/pythonProject/hello_agents')

from gssc.context_builder import ContextBuilder
from gssc.models import ContextConfig, ContextPacket
from datetime import datetime
import importlib.util

# 动态加载 4.compress
cpath = '/Users/heshunyao/PycharmProjects/pythonProject/hello_agents/gssc/4.compress.py'
spec = importlib.util.spec_from_file_location('gssc_compress', cpath)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

builder = ContextBuilder(
    system_instructions='你是一个知识图谱工程师，擅长 Neo4j。',
    config=ContextConfig(max_total_tokens=500),
)

# 伪造超长信息
packets = []
packets.append(ContextPacket('你是一个知识图谱工程师，擅长 Neo4j。', datetime.now(), 0, 1.0, {'type': 'system_instruction'}))
for i in range(15):
    score = 0.95 - i * 0.04
    content = f'证据{i+1}：关于 Cypher 的 MATCH 语法详解，解释了模式匹配的各种用法...'
    packets.append(ContextPacket(content, datetime.now(), 0, score, {'type': 'rag_result'}))
for i in range(10):
    packets.append(ContextPacket(f'用户：问题{i+1}。助手：回答{i+1}', datetime.now(), 0, 0.5, {'type': 'conversation_history'}))

structured = builder.structure(packets, '如何存储知识图谱？')
print(f'原始上下文：~{c._default_count_tokens(structured)} tokens\n')

# ====== 旧策略 ======
old = c.compress_context(structured, max_tokens=80)
print('=' * 40)
print('【旧策略】字符串按位置截断')
print('=' * 40)
old_has_evidence = "[Evidence]" in old

print(old[:250])
print(f'\n是否保留 Evidence：{old_has_evidence}')
print(f'长度：{len(old)} 字符')
print()

# ====== 新策略 ======
new = c.smart_compress(packets, '如何存储知识图谱？', max_tokens=80)
print('=' * 40)
print('【新策略】按区块优先级 + 相关性排序')
print('=' * 40)
new_has_evidence = "[Evidence]" in new

print(new)
print(f'\n是否保留 Evidence：{new_has_evidence}')
print(f'长度：{len(new)} 字符')
print()

# ====== 极端压缩测试（30 tokens，只能存系统指令+问题） ======
tiny_new = c.smart_compress(packets, '如何存储？', max_tokens=30)
print('=' * 40)
print('【极端测试】仅 30 tokens 预算')
print('=' * 40)
print(tiny_new)
print()

# ====== LLM 二次摘要（激进压缩） ======
from hello_agents.core.llm import HelloAgentsLLM
llm = HelloAgentsLLM(
    model='deepseek-v4-pro',
    apiKey='sk-2b14a33b892c402080d9be58743b124c',
    baseUrl='https://api.deepseek.com',
)
llm_ver = c.smart_compress(packets, '如何在 Neo4j 中存储知识图谱？', max_tokens=200, llm=llm, llm_summarize=True)
print('=' * 40)
print('【激进压缩】LLM 摘要模式（200 tokens）')
print('=' * 40)
print(llm_ver[:500])
print(f'\n长度：{len(llm_ver)} 字符')
print(f'tokens 估算：{c._default_count_tokens(llm_ver)}')
