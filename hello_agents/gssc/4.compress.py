"""Compress —— 压缩阶段（增强版）

策略：按区块优先级分配 token 配额：
    1. System Instructions  —— 全量保留
    2. Task              —— 全量保留
    3. Evidence          —— 按相关性分数逐条填充，每条完整保留
    4. Context           —— 保留最近对话
    5. Output 指令     —— 可选保留
"""

import re
from typing import List, Optional


SECTION_PRIORITY = {
    "system": 1,
    "task": 2,
    "evidence": 3,
    "evidence_item": 3,  # evidence 内部条目
    "context": 4,
    "output": 5,
    "other": 99,
}


# ============================================================
# 主入口：兼容旧 API（纯字符串压缩 + 简单截断）
# ============================================================
def compress_context(context: str, max_tokens: int, token_counter=None) -> str:
    """基础版（兼容旧签名）——按区块优先级简单截断

    Args:
        context: 原始上下文
        max_tokens: 最大 token 限制
        token_counter: token 计数函数

    Returns:
        str: 压缩后的上下文
    """
    tc = token_counter or _default_count_tokens
    current_tokens = tc(context)

    if current_tokens <= max_tokens:
        print(f"[Compress] 上下文 {current_tokens} tokens，无需压缩")
        return context

    print(f"[Compress] 上下文超限 ({current_tokens} > {max_tokens})，执行压缩")

    # 按区块分段压缩，按优先级保留
    sections = _parse_sections(context)

    compressed_sections = []
    current_total = 0

    # 1) 先保留高优先级区块
    for section in sections:
        section_tokens = tc(section["text"])
        section_priority = SECTION_PRIORITY.get(section["type"], 99)

        if section_priority <= 2:  # system / task
            if current_total + section_tokens <= max_tokens * 0.8:  # 前两个区块占比不超过 80%
                compressed_sections.append(section["text"])
                current_total += section_tokens
                continue
            else:
                print(f"[Compress] 前两个区块已占 {current_total} tokens，不再新增 {section['type']}")
                break

    # 2) Evidence 按相关性逐条保留
    evidence_section = _get_section(sections, "evidence")
    if evidence_section and current_total < max_tokens:
        remaining = max_tokens - current_total
        compressed_evidence = _compress_evidence_section(
            evidence_section, remaining, tc
        )
        if compressed_evidence:
            compressed_sections.append(compressed_evidence)
            current_total += tc(compressed_evidence)

    # 3) Context 仅保留最近一条
    context_section = _get_section(sections, "context")
    if context_section and current_total < max_tokens:
        remaining = max_tokens - current_total
        context_text = context_section["text"]
        if tc(context_text) <= remaining:
            # 只保留最近一条对话（最后一句）
            lines = context_text.split("\n")
            short_context = "\n".join(lines[:3])  # 最近 3 行
            if tc(short_context) <= remaining:
                compressed_sections.append(short_context)
                current_total += tc(short_context)

    # 4) Output 指令
    output_section = _get_section(sections, "output")
    if output_section and current_total < max_tokens:
        remaining = max_tokens - current_total
        output_text = output_section["text"]
        if tc(output_text) <= remaining:
            compressed_sections.append(output_text)
            current_total += tc(output_text)

    compressed_result = "\n\n".join(compressed_sections)
    final_tokens = tc(compressed_result)
    print(f"[Compress] 压缩完成：{current_tokens} → {final_tokens} tokens")
    return compressed_result


# ============================================================
# 增强版：基于选中的 ContextPacket 列表的智能压缩（推荐）
# ============================================================
def smart_compress(
    selected_packets: list,
    user_query: str,
    max_tokens: int,
    llm=None,
    token_counter=None,
    llm_summarize: bool = False,
    system_instructions: str = "",
) -> str:
    """智能压缩 —— 基于 ContextPacket 做按相关性 + LLM 摘要

    Args:
        selected_packets: List[ContextPacket] 列表
        user_query: 用户查询（仅 llm 摘要时使用
        max_tokens: 总 token 上限
        llm: HelloAgentsLLM 实例；若传了且 llm_summarize=True 时启用 LLM 摘要
        token_counter: token 计数函数
        llm_summarize: 是否在最后使用 LLM 二次摘要（更激进压缩
        system_instructions: 系统指令文本
    """
    tc = token_counter or _default_count_tokens

    # 分类型分组
    system_packets = [p for p in selected_packets if p.metadata.get("type") == "system_instruction"]
    evidence_packets = [p for p in selected_packets if p.metadata.get("type") in ("rag_result", "memory")]
    context_packets = [p for p in selected_packets if p.metadata.get("type") == "conversation_history"]
    other_packets = [p for p in selected_packets if p.metadata.get("type") not in ("system_instruction", "rag_result", "memory", "conversation_history")]

    # 估算各部分
    system_text = "\n\n".join(p.content for p in system_packets) if system_packets else system_instructions
    task_text = f"[Task]\n{user_query}"

    sections_used = tc(system_text) + tc(task_text) + 20  # 留一点冗余
    if sections_used >= max_tokens:
        print(f"[Compress] 系统指令和问题已占 {sections_used} tokens，超过上限")
        return system_text + "\n\n" + task_text

    remaining = max_tokens - sections_used
    print(f"[Compress] 系统指令+问题占 {sections_used} tokens，剩余 {remaining} tokens")

    # Evidence —— 按相关性分数降序逐条填充（只保留完整条目）
    evidence_lines = []
    for idx, packet in enumerate(sorted(evidence_packets, key=lambda p: p.relevance_score, reverse=True), 1):
        entry_text = f"[{idx}] {packet.content}"
        if tc(entry_text) <= remaining * 0.9:
            evidence_lines.append(entry_text)
            remaining -= tc(entry_text)
        else:
            break

    evidence_text = "[Evidence]\n" + "\n---\n".join(evidence_lines) if evidence_lines else ""

    # Context —— 保留最近 3 条对话
    context_text = ""
    if context_packets and remaining > 50:
        recent = context_packets[-3:]
        context_text = "[Context]\n" + "\n".join(p.content for p in recent)
        if tc(context_text) > remaining:
            context_text = "[Context]\n" + recent[-1].content

    # Output
    output_text = "[Output]\n请基于以上信息，提供准确、有据的回答。"

    # 组装
    final_sections = [system_text, task_text]
    if evidence_text:
        final_sections.append(evidence_text)
    if context_text and tc(context_text) < remaining:
        final_sections.append(context_text)
    final_sections.append(output_text)

    result = "\n\n".join(final_sections)
    final_tokens = tc(result)
    print(f"[Compress] 智能压缩完成：共 {final_tokens} tokens")

    # 可选：LLM 二次摘要（激进压缩）
    if llm_summarize and llm and final_tokens > max_tokens:
        return _llm_summarize_compress(result, max_tokens, llm, tc)

    return result


# ============================================================
# 工具函数：LLM 二次摘要（最激进压缩，丢信息少）
# ============================================================
def _llm_summarize_compress(text: str, max_tokens: int, llm, tc) -> str:
    """用 LLM 把超长上下文重新写成紧凑摘要；失败时退化为字符截断。"""
    print(f"[Compress] LLM 摘要模式：{tc(text)} → {max_tokens}")
    try:
        summary = llm.invoke([
            {"role": "system", "content": "你是一个文档摘要者。保留所有结论与事实依据，其余删除。"},
            {"role": "user", "content": f"请将以下文本压缩到 {max_tokens} tokens 以内，保留关键信息：\n\n{text}"},
        ], temperature=0.0)
        if summary and tc(summary) <= max_tokens:
            print(f"[Compress] LLM 摘要成功：{tc(summary)} tokens")
            return summary
    except Exception as e:
        print(f"[Compress] LLM 摘要失败: {e}")
    # 退化：字符截断
    chars = int(max_tokens * 1.5)
    return text[:chars] + "\n[... 内容已压缩 ...]"


# ============================================================
# 内部辅助：区块解析
# ============================================================
def _parse_sections(context: str) -> list:
    """把结构化上下文解析成类型字典的列表"""
    sections = []
    blocks = re.split(r"\n\n(?=\[)", context)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # 识别类型
        lower = block.lower()
        if "role & policies" in lower or "[role" in lower:
            s_type = "system"
        elif "task" in lower and "[task]" in lower:
            s_type = "task"
        elif "evidence" in lower and "[evidence]" in lower:
            s_type = "evidence"
        elif "context" in lower and "[context]" in lower:
            s_type = "context"
        elif "output" in lower and "[output]" in lower:
            s_type = "output"
        else:
            s_type = "other"
        sections.append({"type": s_type, "text": block})
    return sections


def _get_section(sections: list, s_type: str) -> Optional[dict]:
    """从 sections 中取指定类型区块"""
    for s in sections:
        if s["type"] == s_type:
            return s
    return None


def _compress_evidence_section(evidence_section: dict, remaining_tokens: int, tc) -> str:
    """Evidence 区块内部按条目截断，保留高相关的条目"""
    text = evidence_section["text"]
    lines = text.split("\n---\n")
    kept = []
    used = 0
    for line in lines:
        if not line.strip():
            continue
        line_tokens = tc(line)
        if used + line_tokens <= remaining_tokens:
            kept.append(line)
            used += line_tokens
        else:
            break
    if len(kept) < len(lines):
        kept.append("\n[... 其余证据已省略 ...]")
    return "\n---\n".join(kept)


# ============================================================
# Token 计数（默认值，已兼容中文/英文混合估算
# ============================================================
def _default_count_tokens(text: str) -> int:
    """默认 token 估算 —— 中文 1 字 ≈ 1 token，英文 1 词 ≈ 1.3 tokens"""
    if not text:
        return 0
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    english_words = len([w for w in text.split() if w and not ("\u4e00" <= w[0] <= "\u9fff")])
    return int(chinese_chars + english_words * 1.3) + 1


def _truncate_text(text: str, max_tokens: int, token_counter) -> str:
    """截断文本到指定 token 数量（备用，保留中文字符/句末边界）"""
    tcount = token_counter(text) or 1
    char_per_token = len(text) / tcount
    max_chars = int(max_tokens * char_per_token)
    if max_chars >= len(text):
        return text
    # 在最近的句末标点处截断
    cut = text[:max_chars]
    last_punct = max(
        cut.rfind("。"), cut.rfind("."), cut.rfind("\n"),
        cut.rfind("；"), cut.rfind(";"), cut.rfind("！"), cut.rfind("!"),
        cut.rfind("?"), cut.rfind("？")
    )
    if last_punct > max_chars * 0.7:
        return text[:last_punct + 1]
    return cut
