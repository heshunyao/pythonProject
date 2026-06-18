"""GSSC 四阶段上下文构建器测试

运行方式:
    python3 gssc/test_gssc.py        # 从项目根目录
    python3 test_gssc.py              # 从 gssc/ 目录内
"""

import os
import sys

# 把父目录加入路径，这样 `from gssc.models import ...` 和 `from hello_agents.core.llm import ...` 都能 work
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from datetime import datetime, timedelta

from gssc.models import ContextPacket, ContextConfig
from gssc.context_builder import ContextBuilder


# ============================================================
# Mock 工具：模拟记忆系统和 RAG 工具
# ============================================================
class MockMemoryTool:
    """模拟记忆工具 — 可被 ContextBuilder 调用"""

    def __init__(self):
        self.items = [
            {"content": "用户上周问过图数据库 Neo4j 的基础使用方法", "score": 0.8},
            {"content": "用户对 Cypher 查询语言很感兴趣", "score": 0.75},
            {"content": "用户正在开发一个知识图谱项目", "score": 0.9},
            {"content": "用户之前使用过 Python neo4j 驱动", "score": 0.6},
        ]

    def run(self, params):
        action = params.get("action")
        if action == "search":
            return self.items
        return []


class MockRagTool:
    """模拟 RAG 检索工具"""

    def __init__(self):
        self.docs = [
            {"content": "Neo4j 是一个开源的图数据库，使用 Cypher 作为查询语言。", "score": 0.92},
            {"content": "Cypher 的基本语法：MATCH (n:Label) RETURN n", "score": 0.88},
            {"content": "图数据库非常适合表示人际关系、知识图谱等数据。", "score": 0.85},
        ]

    def run(self, params):
        action = params.get("action")
        if action == "search":
            return self.docs
        return []


# ============================================================
# 测试 1：验证 import 和类初始化
# ============================================================
def test_1_import_and_init():
    print("=" * 50)
    print("🔬 Test 1: 模块导入与 ContextBuilder 初始化")
    print("=" * 50)

    builder = ContextBuilder(
        system_instructions="你是一个有帮助的 AI 助手，请始终基于事实回答。",
        config=ContextConfig(
            max_total_tokens=2000,
            min_relevance=0.1,
            relevance_weight=0.7,
            recency_weight=0.3,
            history_window=5,
        ),
        memory_tool=MockMemoryTool(),
        rag_tool=MockRagTool(),
    )
    print(f"✅ ContextBuilder 实例化成功")
    print(f"   system_instructions 长度: {len(builder.system_instructions)}")
    print(f"   max_total_tokens: {builder.config.max_total_tokens}")
    print(f"   memory_tool: {builder.memory_tool}")
    print(f"   rag_tool: {builder.rag_tool}")
    return builder


# ============================================================
# 测试 2：Step 1 — Gather 汇集
# ============================================================
def test_2_gather(builder):
    print("\n" + "=" * 50)
    print("🔬 Test 2: Gather 汇集阶段")
    print("=" * 50)

    user_query = "如何用 Neo4j 存储知识图谱？"
    conversation_history = [
        {"role": "user", "content": "你好，我想了解 Neo4j", "timestamp": datetime.now() - timedelta(hours=2)},
        {"role": "assistant", "content": "当然可以，Neo4j 是一款图数据库。请问你具体想了解哪方面？", "timestamp": datetime.now() - timedelta(hours=2)},
        {"role": "user", "content": "我想知道怎么存数据", "timestamp": datetime.now() - timedelta(minutes=30)},
    ]

    packets = builder.gather(user_query, conversation_history=conversation_history)

    assert len(packets) > 0, "packets 不应为空"

    types = set(p.metadata.get("type") for p in packets)
    print(f"✅ 共汇集 {len(packets)} 个信息包")
    print(f"   类型分布: {types}")
    for p in packets:
        print(f"   - [{p.metadata.get('type', '?')}] score={p.relevance_score:.3f} tokens={p.token_count}")

    return packets, user_query


# ============================================================
# 测试 3：Step 2 — Select 筛选
# ============================================================
def test_3_select(builder, packets, user_query):
    print("\n" + "=" * 50)
    print("🔬 Test 3: Select 筛选阶段")
    print("=" * 50)

    selected = builder.select(packets, user_query)

    assert len(selected) > 0, "selected 不应为空"
    assert len(selected) <= len(packets), "选中的数量不应超过汇集的数量"

    total_tokens = sum(p.token_count for p in selected)
    print(f"✅ 从 {len(packets)} 个中选择了 {len(selected)} 个")
    print(f"   总 tokens: {total_tokens} / {builder.config.max_total_tokens}")
    for p in selected:
        print(f"   - [{p.metadata.get('type', '?')}] score={p.relevance_score:.3f}")

    return selected


# ============================================================
# 测试 4：Step 3 — Structure 组织
# ============================================================
def test_4_structure(builder, selected, user_query):
    print("\n" + "=" * 50)
    print("🔬 Test 4: Structure 组织阶段")
    print("=" * 50)

    structured = builder.structure(selected, user_query)

    assert "[Task]" in structured, "应包含 [Task] 区块"
    assert "[Output]" in structured, "应包含 [Output] 区块"
    assert user_query in structured, "应包含用户查询"

    print(f"✅ 结构化上下文生成成功")
    print(f"   总长度: {len(structured)} 字符")
    print("-" * 30)
    print(structured)
    print("-" * 30)
    return structured


# ============================================================
# 测试 5：Step 4 — Compress 压缩（超限场景）
# ============================================================
def test_5_compress(builder, structured):
    print("\n" + "=" * 50)
    print("🔬 Test 5: Compress 压缩阶段（超限场景）")
    print("=" * 50)

    # 故意设置很低的 token 上限，触发压缩
    small_tokens = 50
    compressed = builder.compress(structured, max_tokens=small_tokens)

    assert len(compressed) <= len(structured), "压缩后不应更长"

    before_tokens = builder.token_counter(structured)
    after_tokens = builder.token_counter(compressed)
    print(f"   压缩前: {before_tokens} tokens")
    print(f"   压缩后: {after_tokens} tokens (限制: {small_tokens})")
    print(f"   压缩前长度: {len(structured)} 字符")
    print(f"   压缩后长度: {len(compressed)} 字符")
    print("-" * 30)
    print(compressed)
    print("-" * 30)
    print("✅ 压缩完成")

    # 测试非超限场景：不应被修改
    big_tokens = 10000
    unmodified = builder.compress(structured, max_tokens=big_tokens)
    assert unmodified == structured, "非超限场景内容应保持不变"
    print("✅ 非超限场景正常跳过压缩")


# ============================================================
# 测试 6：完整端到端流程
# ============================================================
def test_6_full_pipeline():
    print("\n" + "=" * 50)
    print("🔬 Test 6: 完整 GSSC 端到端流程")
    print("=" * 50)

    builder = ContextBuilder(
        system_instructions="你是一个资深的知识图谱工程师，擅长使用 Neo4j。",
        config=ContextConfig(max_total_tokens=1500),
        memory_tool=MockMemoryTool(),
        rag_tool=MockRagTool(),
    )

    user_query = "如何在 Neo4j 中存储一个知识图谱？Cypher 怎么写？"
    conversation_history = [
        {"role": "user", "content": "你好，我想了解 Neo4j", "timestamp": datetime.now() - timedelta(hours=1)},
        {"role": "assistant", "content": "Neo4j 是图数据库，你想了解什么？", "timestamp": datetime.now() - timedelta(hours=1)},
    ]

    final_context = builder.build(
        user_query=user_query,
        conversation_history=conversation_history,
    )

    assert final_context, "输出不应为空"
    assert "[Task]" in final_context
    assert "[Output]" in final_context

    print(f"✅ 端到端流程成功")
    print(f"   输出长度: {len(final_context)} 字符")
    print(f"   Token 估算: {builder.token_counter(final_context)}")
    print("-" * 30)
    print(final_context)
    print("-" * 30)


# ============================================================
# 测试 7：仅使用基本信息（无记忆/RAG）
# ============================================================
def test_7_basic_mode():
    print("\n" + "=" * 50)
    print("🔬 Test 7: 基础模式（无记忆/RAG）")
    print("=" * 50)

    builder = ContextBuilder(
        system_instructions="你是一个简洁的助手。",
        config=ContextConfig(max_total_tokens=500),
    )

    result = builder.build("今天天气怎么样？")
    assert result
    print(f"✅ 基础模式正常工作")
    print(f"   输出: {result[:100]}...")


# ============================================================
# 测试 8：自定义 ContextPacket 注入
# ============================================================
def test_8_custom_packets():
    print("\n" + "=" * 50)
    print("🔬 Test 8: 自定义 ContextPacket 注入")
    print("=" * 50)

    builder = ContextBuilder()
    custom = [
        ContextPacket(
            content="用户是一个 Python 开发者，正在学习图数据库。",
            timestamp=datetime.now(),
            token_count=50,
            relevance_score=0.9,
            metadata={"type": "user_profile"},
        )
    ]

    packets = builder.gather("Neo4j", custom_packets=custom)
    assert any(p.metadata.get("type") == "user_profile" for p in packets)
    print(f"✅ 自定义包已成功注入，共 {len(packets)} 个")


# ============================================================
# 测试 9：相关性计算函数验证
# ============================================================
def test_9_relevance_calculation():
    print("\n" + "=" * 50)
    print("🔬 Test 9: 相关性计算验证")
    print("=" * 50)

    gssc_dir = os.path.dirname(os.path.abspath(__file__))
    import importlib.util as _util
    spec = _util.spec_from_file_location("gssc_select", os.path.join(gssc_dir, "", "2.select.py"))
    select_mod = _util.module_from_spec(spec)
    spec.loader.exec_module(select_mod)

    test_cases = [
        ("Neo4j 是一个图数据库", "Neo4j 图数据库"),
        ("这是一段完全无关的文本", "天气怎么样"),
        ("苹果公司发布了新款 iPhone", "苹果 iPhone 发布"),
    ]

    for content, query in test_cases:
        score = select_mod._calculate_relevance(content, query)
        print(f"   相关性({query[:20]} | {content[:30]}...) = {score}")
        assert 0.0 <= score <= 1.0, "分数应在 0~1 之间"

    print("✅ 相关性计算正常")


# ============================================================
# 测试 10：时间近因性衰减验证
# ============================================================
def test_10_recency():
    print("\n" + "=" * 50)
    print("🔬 Test 10: 时间近因性衰减验证")
    print("=" * 50)

    gssc_dir = os.path.dirname(os.path.abspath(__file__))
    import importlib.util as _util2
    spec = _util2.spec_from_file_location("gssc_select2", os.path.join(gssc_dir, "", "2.select.py"))
    select_mod = _util2.module_from_spec(spec)
    spec.loader.exec_module(select_mod)

    now = datetime.now()
    times = [
        ("刚刚", now),
        ("1小时前", now - timedelta(hours=1)),
        ("1天前", now - timedelta(days=1)),
        ("30天前", now - timedelta(days=30)),
        ("1年前", now - timedelta(days=365)),
    ]

    for label, ts in times:
        score = select_mod._calculate_recency(ts)
        bar = "█" * int(score * 40)
        print(f"   {label:<10} {score:.3f} {bar}")

    print("✅ 时间衰减正常")


# ============================================================
# 测试 11：GSSC 四阶段 + DeepSeek 端到端问答（真实 API 调用）
# ============================================================
def test_11_deepseek_pipeline():
    print("\n" + "=" * 50)
    print("🔬 Test 11: GSSC + DeepSeek 端到端问答")
    print("=" * 50)

    from hello_agents.core.llm import HelloAgentsLLM

    # 初始化 DeepSeek 大模型
    llm = HelloAgentsLLM(
        model="deepseek-v4-pro",
        apiKey="sk-2b14a33b892c402080d9be58743b124c",
        baseUrl="https://api.deepseek.com",
    )
    print(f"✅ LLM 实例化: {llm.model}")

    # 初始化 GSSC ContextBuilder
    builder = ContextBuilder(
        system_instructions="你是一个知识图谱工程师，擅长 Neo4j。回答要基于所给证据，简洁明了。",
        config=ContextConfig(max_total_tokens=2000),
        memory_tool=MockMemoryTool(),
        rag_tool=MockRagTool(),
        llm=llm,
    )
    print(f"✅ ContextBuilder 实例化")

    # 用户问题
    user_query = "如何在 Neo4j 中存储一个知识图谱？Cypher 的基础语法是什么？"
    conversation_history = [
        {"role": "user", "content": "你好，我想了解 Neo4j", "timestamp": datetime.now()},
    ]

    # 调用 GSSC + LLM
    answer, context = builder.build_and_answer(
        user_query=user_query,
        conversation_history=conversation_history,
        temperature=0.1,
    )

    assert answer, "LLM 未返回有效答案"
    print(f"\n✅ 端到端流程成功，上下文 tokens ≈ {builder.token_counter(context)}")
    print(f"✅ 最终答案长度: {len(answer)} 字符")
    return answer, context


# ============================================================
# 主程序
# ============================================================
def main():
    print("🚀 开始 GSSC 测试套件")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    passed = 0
    failed = 0

    tests = [
        ("Test 1", lambda: test_1_import_and_init()),
    ]

    # Test 1: init
    try:
        builder = test_1_import_and_init()
        passed += 1
    except Exception as e:
        print(f"❌ Test 1 失败: {e}")
        failed += 1
        return

    # Test 2-4 依赖前序结果
    try:
        packets, user_query = test_2_gather(builder)
        passed += 1
    except Exception as e:
        print(f"❌ Test 2 失败: {e}")
        failed += 1
        packets = user_query = None

    try:
        selected = test_3_select(builder, packets, user_query)
        passed += 1
    except Exception as e:
        print(f"❌ Test 3 失败: {e}")
        failed += 1
        selected = None

    try:
        structured = test_4_structure(builder, selected, user_query)
        passed += 1
    except Exception as e:
        print(f"❌ Test 4 失败: {e}")
        failed += 1
        structured = None

    try:
        test_5_compress(builder, structured)
        passed += 1
    except Exception as e:
        print(f"❌ Test 5 失败: {e}")
        failed += 1

    # 独立测试
    for name, fn in [
        ("Test 6 - 端到端", test_6_full_pipeline),
        ("Test 7 - 基础模式", test_7_basic_mode),
        ("Test 8 - 自定义包", test_8_custom_packets),
        ("Test 9 - 相关性", test_9_relevance_calculation),
        ("Test 10 - 时间衰减", test_10_recency),
        ("Test 11 - GSSC + DeepSeek 端到端问答", test_11_deepseek_pipeline),
    ]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 总结
    print("\n" + "=" * 50)
    print(f"📊 测试总结: {passed}/{passed + failed} 通过, {failed} 失败")
    if failed == 0:
        print("🎉 全部测试通过！GSSC 运行正常。")
    else:
        print(f"⚠️  有 {failed} 个测试失败，请检查日志。")
    print("=" * 50)


if __name__ == "__main__":
    main()
