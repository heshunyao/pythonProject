"""GSSC — Gather / Select / Structure / Compress 四阶段上下文构建器"""

import importlib.util
import os
import sys

from datetime import datetime
from typing import List, Optional

from gssc.models import ContextPacket, ContextConfig


def _load_module(name, filename):
    """通过文件路径加载模块（解决数字开头文件名问题）"""
    gssc_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(gssc_dir, filename)
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gather_mod = _load_module("gssc_gather", "1.gather.py")
select_mod = _load_module("gssc_select", "2.select.py")
structure_mod = _load_module("gssc_structure", "3.structure.py")
compress_mod = _load_module("gssc_compress", "4.compress.py")


class ContextBuilder:
    """GSSC 上下文构建器

    两档使用方式:
        1) 纯上下文构建 —— build() 只跑 4 个阶段，返回结构化上下文字符串
        2) 带 LLM —— build_and_answer() 在上一步之后调用 LLM 返回最终答案

    示例:
        from hello_agents.core.llm import HelloAgentsLLM
        llm = HelloAgentsLLM(
            model='deepseek-v4-pro',
            apiKey='sk-xxxx',
            baseUrl='https://api.deepseek.com',
        )
        builder = ContextBuilder(
            system_instructions="你是一个有帮助的助手",
            llm=llm,
        )
        answer, context = builder.build_and_answer("如何在 Neo4j 中存储知识图谱？")
    """

    def __init__(
        self,
        system_instructions: Optional[str] = None,
        config: Optional[ContextConfig] = None,
        memory_tool=None,
        rag_tool=None,
        token_counter=None,
        relevance_fn=None,
        llm=None,
    ):
        self.system_instructions = system_instructions
        self.config = config or ContextConfig()
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.token_counter = token_counter or compress_mod._default_count_tokens
        self.relevance_fn = relevance_fn
        # HelloAgentsLLM 实例（可选）；没传则 build_and_answer 不可用
        self.llm = llm

    # ---- 主入口：只构建上下文 ----
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[dict]] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
    ) -> str:
        """执行四阶段流程并返回结构化上下文字符串"""
        print("\n========== GSSC 开始 ==========")
        print(f"[Input] 用户查询: {user_query}")

        # Step 1 — Gather 汇集
        packets = gather_mod.gather_packets(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=self.system_instructions,
            custom_packets=custom_packets,
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=self.config,
            token_counter=self.token_counter,
        )

        # Step 2 — Select 筛选
        selected = select_mod.select_packets(
            packets=packets,
            user_query=user_query,
            available_tokens=self.config.max_total_tokens,
            config=self.config,
            relevance_fn=self.relevance_fn,
        )

        # Step 3 — Structure 组织
        structured = structure_mod.structure_context(
            selected_packets=selected,
            user_query=user_query,
        )

        # Step 4 — Compress 压缩
        final_context = compress_mod.compress_context(
            context=structured,
            max_tokens=self.config.max_total_tokens,
            token_counter=self.token_counter,
        )

        print("========== GSSC 完成 ==========\n")
        return final_context

    # ---- 主入口：上下文构建 + LLM 回答 ----
    def build_and_answer(
        self,
        user_query: str,
        conversation_history: Optional[List[dict]] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
        temperature: float = 0,
    ):
        """先跑 GSSC 四阶段，再调用 LLM 返回最终答案

        返回 (answer: str, context: str)
        """
        if self.llm is None:
            raise ValueError("ContextBuilder 没有配置 llm，请在初始化时传入 llm 参数")

        context = self.build(
            user_query=user_query,
            conversation_history=conversation_history,
            custom_packets=custom_packets,
        )

        # 把结构化文本直接送给 LLM：整个 context 作为 system 唯一一条 system message；
        # 用户问题作为 user message
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": f"请基于上述上下文，简洁回答：{user_query}"},
        ]

        print("\n[LLM] 正在生成最终答案...\n")
        # try both (answer\
        try:
            answer = self.llm.invoke(messages, temperature=temperature)
        except Exception as e:
            print(f"[LLM] ❌ 调用失败: {e}")
            answer = ""

        if answer:
            print("--- LLM 回答:")
            print(answer)
        return answer, context

    # ---- 单独调用各阶段（调试用）----
    def gather(self, user_query, conversation_history=None, custom_packets=None):
        return gather_mod.gather_packets(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=self.system_instructions,
            custom_packets=custom_packets,
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=self.config,
            token_counter=self.token_counter,
        )

    def select(self, packets, user_query):
        return select_mod.select_packets(
            packets=packets,
            user_query=user_query,
            available_tokens=self.config.max_total_tokens,
            config=self.config,
            relevance_fn=self.relevance_fn,
        )

    def structure(self, selected_packets, user_query):
        return structure_mod.structure_context(selected_packets, user_query)

    def compress(self, context, max_tokens=None):
        return compress_mod.compress_context(
            context=context,
            max_tokens=max_tokens or self.config.max_total_tokens,
            token_counter=self.token_counter,
        )
