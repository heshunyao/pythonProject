from datetime import datetime
from typing import List, Optional


def gather_packets(
    user_query: str,
    conversation_history: Optional[List[dict]] = None,
    system_instructions: Optional[str] = None,
    custom_packets: Optional[List] = None,
    memory_tool=None,
    rag_tool=None,
    config=None,
    token_counter=None,
) -> List:
    """汇集所有候选信息

    Args:
        user_query: 用户查询
        conversation_history: 对话历史（dict 列表，含 role/content/timestamp）
        system_instructions: 系统指令
        custom_packets: 自定义信息包
        memory_tool: 记忆工具对象（需有 run(action, query, limit) 方法）
        rag_tool: RAG 工具对象
        config: ContextConfig 对象
        token_counter: token 计数函数（接收字符串返回 int）

    Returns:
        List[ContextPacket]: 候选信息列表
    """
    from gssc.models import ContextPacket

    packets = []
    tc = token_counter or (lambda t: _default_count_tokens(t))

    # 1. 系统指令（最高优先级，不参与评分）
    if system_instructions:
        packets.append(ContextPacket(
            content=system_instructions,
            timestamp=datetime.now(),
            token_count=tc(system_instructions),
            relevance_score=1.0,
            metadata={"type": "system_instruction", "priority": "high"}
        ))

    # 2. 记忆系统检索
    if memory_tool is not None:
        try:
            mem_limit = config.memory_limit if config else 10
            memory_results = memory_tool.run({
                "action": "search",
                "query": user_query,
                "limit": mem_limit,
                "min_importance": 0.3
            })
            memory_packets = _parse_memory_results(memory_results, tc)
            packets.extend(memory_packets)
            print(f"[Gather] 从记忆系统获得 {len(memory_packets)} 条")
        except Exception as e:
            print(f"[Gather][WARNING] 记忆检索失败: {e}")

    # 3. RAG 知识库检索
    if rag_tool is not None:
        try:
            rag_limit = config.rag_limit if config else 5
            rag_results = rag_tool.run({
                "action": "search",
                "query": user_query,
                "limit": rag_limit,
                "min_score": 0.3
            })
            rag_packets = _parse_rag_results(rag_results, tc)
            packets.extend(rag_packets)
            print(f"[Gather] 从 RAG 获得 {len(rag_packets)} 条")
        except Exception as e:
            print(f"[Gather][WARNING] RAG 检索失败: {e}")

    # 4. 对话历史（最近 N 条）
    if conversation_history:
        window = config.history_window if config else 5
        recent = conversation_history[-window:]
        for msg in recent:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
            ts = msg.get("timestamp") if isinstance(msg, dict) and "timestamp" in msg else datetime.now()
            packets.append(ContextPacket(
                content=f"{role}: {content}",
                timestamp=ts,
                token_count=tc(content),
                relevance_score=0.6,
                metadata={"type": "conversation_history", "role": role}
            ))

    # 5. 自定义信息包
    if custom_packets:
        packets.extend(custom_packets)

    print(f"[Gather] 共汇集 {len(packets)} 个候选信息包")
    return packets


def _parse_memory_results(memory_results, tc) -> List:
    """解析记忆检索结果为 ContextPacket 列表"""
    from gssc.models import ContextPacket

    if not memory_results:
        return []

    packets = []
    results = []
    # 兼容多种返回格式
    if isinstance(memory_results, list):
        results = memory_results
    elif isinstance(memory_results, dict) and "results" in memory_results:
        results = memory_results["results"]
    elif isinstance(memory_results, str):
        return [ContextPacket(
            content=memory_results,
            timestamp=datetime.now(),
            token_count=tc(memory_results),
            relevance_score=0.7,
            metadata={"type": "memory", "source": "memory_tool"}
        )]

    for item in results:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or item.get("memory", str(item))
            score = float(item.get("relevance") or item.get("score") or item.get("importance", 0.5))
            ts = item.get("timestamp") or datetime.now()
            packets.append(ContextPacket(
                content=str(content),
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                token_count=tc(str(content)),
                relevance_score=score,
                metadata={"type": "memory", "source": "memory_tool", **item}
            ))
        else:
            packets.append(ContextPacket(
                content=str(item),
                timestamp=datetime.now(),
                token_count=tc(str(item)),
                relevance_score=0.5,
                metadata={"type": "memory", "source": "memory_tool"}
            ))

    return packets


def _parse_rag_results(rag_results, tc) -> List:
    """解析 RAG 检索结果为 ContextPacket 列表"""
    from gssc.models import ContextPacket

    if not rag_results:
        return []

    packets = []
    results = []
    if isinstance(rag_results, list):
        results = rag_results
    elif isinstance(rag_results, dict) and "results" in rag_results:
        results = rag_results["results"]
    elif isinstance(rag_results, str):
        return [ContextPacket(
            content=rag_results,
            timestamp=datetime.now(),
            token_count=tc(rag_results),
            relevance_score=0.8,
            metadata={"type": "rag_result", "source": "rag_tool"}
        )]

    for item in results:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or item.get("document", str(item))
            score = float(item.get("relevance") or item.get("score") or item.get("similarity", 0.7))
            packets.append(ContextPacket(
                content=str(content),
                timestamp=datetime.now(),
                token_count=tc(str(content)),
                relevance_score=score,
                metadata={"type": "rag_result", "source": "rag_tool"}
            ))
        else:
            packets.append(ContextPacket(
                content=str(item),
                timestamp=datetime.now(),
                token_count=tc(str(item)),
                relevance_score=0.7,
                metadata={"type": "rag_result", "source": "rag_tool"}
            ))

    return packets


def _default_count_tokens(text: str) -> int:
    """默认 token 估算 — 中文 1 字 ≈ 1 token，英文 1 词 ≈ 1.3 tokens"""
    if not text:
        return 0
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    english_words = len([w for w in text.split() if w and not ("\u4e00" <= w[0] <= "\u9fff")])
    return int(chinese_chars + english_words * 1.3) + 1
