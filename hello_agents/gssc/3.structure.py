from typing import List


def structure_context(selected_packets: List, user_query: str) -> str:
    """将选中的信息包组织成结构化的上下文模板

    Args:
        selected_packets: 选中的信息包列表
        user_query: 用户查询

    Returns:
        str: 结构化的上下文字符串
    """
    # 按类型分组
    system_instructions = []
    evidence = []
    context = []

    for packet in selected_packets:
        packet_type = packet.metadata.get("type", "general")

        if packet_type == "system_instruction":
            system_instructions.append(packet.content)
        elif packet_type in ("rag_result", "knowledge", "memory"):
            evidence.append(packet.content)
        else:
            context.append(packet.content)

    # 构建结构化模板
    sections = []

    # [Role & Policies]
    if system_instructions:
        sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

    # [Task]
    sections.append(f"[Task]\n{user_query}")

    # [Evidence]
    if evidence:
        numbered_evidence = []
        for idx, item in enumerate(evidence, 1):
            numbered_evidence.append(f"[{idx}] {item}")
        sections.append("[Evidence]\n" + "\n---\n".join(numbered_evidence))

    # [Context]
    if context:
        sections.append("[Context]\n" + "\n".join(context))

    # [Output]
    sections.append("[Output]\n请基于以上信息，提供准确、有据的回答。")

    final_context = "\n\n".join(sections)
    print(f"[Structure] 上下文已组织完成，共 {len(sections)} 个区块")
    return final_context
