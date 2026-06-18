import math
from datetime import datetime
from typing import List


def select_packets(
    packets: List,
    user_query: str,
    available_tokens: int,
    config=None,
    relevance_fn=None,
) -> List:
    """选择最相关的信息包

    Args:
        packets: 候选信息包列表
        user_query: 用户查询（用于计算相关性）
        available_tokens: 可用的 token 数量
        config: ContextConfig 对象
        relevance_fn: 自定义相关性计算函数（可选）

    Returns:
        List[ContextPacket]: 选中的信息包列表
    """
    relevance_weight = getattr(config, "relevance_weight", 0.7)
    recency_weight = getattr(config, "recency_weight", 0.3)
    min_relevance = getattr(config, "min_relevance", 0.1)
    calc_relevance = relevance_fn or _calculate_relevance

    # 1. 分离系统指令和其他信息
    system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
    other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

    # 2. 计算系统指令占用的 token
    system_tokens = sum(p.token_count for p in system_packets)
    remaining_tokens = available_tokens - system_tokens

    if remaining_tokens <= 0:
        print("[Select][WARNING] 系统指令已占满所有 token 预算")
        return system_packets

    # 3. 为其他信息计算综合分数
    scored_packets = []
    for packet in other_packets:
        if packet.relevance_score == 0.5:
            packet.relevance_score = calc_relevance(packet.content, user_query)

        recency = _calculate_recency(packet.timestamp)
        combined_score = relevance_weight * packet.relevance_score + recency_weight * recency

        if packet.relevance_score >= min_relevance:
            scored_packets.append((combined_score, packet))

    # 4. 按分数降序排序
    scored_packets.sort(key=lambda x: x[0], reverse=True)

    # 5. 贪心选择
    selected = list(system_packets)
    current_tokens = system_tokens

    for score, packet in scored_packets:
        if current_tokens + packet.token_count <= available_tokens:
            selected.append(packet)
            current_tokens += packet.token_count
        else:
            break

    print(f"[Select] 选择了 {len(selected)} 个信息包，共 {current_tokens} tokens")
    return selected


def _calculate_relevance(content: str, query: str) -> float:
    """计算内容与查询的相关性（基于关键词重叠的简易算法）

    支持中英文混排。
    """
    if not content or not query:
        return 0.0

    # 中文提取：单字符 + 英文单词
    content_terms = set()
    for ch in content:
        if "\u4e00" <= ch <= "\u9fff":
            content_terms.add(ch)
    content_terms.update(w.lower() for w in query.split() if w)

    query_terms = set()
    for ch in query:
        if "\u4e00" <= ch <= "\u9fff":
            query_terms.add(ch)
    query_terms.update(w.lower() for w in query.split() if w)

    if not query_terms:
        return 0.0

    # Jaccard 相似度
    intersection = content_terms & query_terms
    union = content_terms | query_terms

    return round(len(intersection) / len(union) if union else 0.0, 3)


def _calculate_recency(timestamp) -> float:
    """计算时间近因性分数（指数衰减模型）

    24 小时内保持高分，之后逐渐衰减。
    """
    try:
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        age_hours = max(0, (datetime.now() - timestamp).total_seconds() / 3600)
        decay_factor = 0.1
        recency_score = math.exp(-decay_factor * age_hours / 24)
        return max(0.1, min(1.0, recency_score))
    except Exception:
        return 0.5
