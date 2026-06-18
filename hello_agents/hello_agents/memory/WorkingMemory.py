import math
from datetime import datetime
from typing import List

from .base import MemoryConfig, MemoryItem


class WorkingMemory:
    """工作记忆实现
    特点：
    - 容量有限（默认50条）+ TTL自动清理
    - 纯内存存储，访问速度极快
    - 混合检索：关键词匹配 + 时间衰减
    """

    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl or 60
        self.memories = []

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆"""
        self._expire_old_memories()  # 过期清理

        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory()  # 容量管理

        self.memories.append(memory_item)
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：关键词匹配 + 时间衰减"""
        self._expire_old_memories()

        # 计算综合分数
        scored_memories = []
        for memory in self.memories:
            keyword_score = self._calculate_keyword_score(query, memory.content)

            # 混合评分
            base_relevance = keyword_score
            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)

            final_score = base_relevance * time_decay * importance_weight
            if final_score > 0:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def delete(self, memory_id: str) -> bool:
        """删除工作记忆"""
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                del self.memories[i]
                return True
        return False

    def clear(self):
        """清空所有工作记忆"""
        self.memories = []

    def _expire_old_memories(self):
        """清理过期的记忆"""
        current_time = datetime.now()
        self.memories = [
            memory for memory in self.memories
            if self._is_memory_valid(memory.timestamp, current_time)
        ]

    def _is_memory_valid(self, timestamp: str, current_time: datetime) -> bool:
        """检查记忆是否有效（未过期）"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_minutes = (current_time - memory_time).total_seconds() / 60
            return age_minutes <= self.max_age_minutes
        except Exception:
            return False

    def _remove_lowest_priority_memory(self):
        """移除最低优先级的记忆"""
        if not self.memories:
            return

        # 找到优先级最低的记忆（时间最久且重要性最低）
        lowest_priority = None
        lowest_index = -1

        for i, memory in enumerate(self.memories):
            try:
                memory_time = datetime.fromisoformat(memory.timestamp)
                age_minutes = (datetime.now() - memory_time).total_seconds() / 60
                # 优先级 = 时间衰减 * 重要性
                priority = self._calculate_time_decay(memory.timestamp) * memory.importance

                if lowest_priority is None or priority < lowest_priority:
                    lowest_priority = priority
                    lowest_index = i
            except Exception:
                # 如果时间格式无效，优先移除
                lowest_index = i
                break

        if lowest_index >= 0:
            del self.memories[lowest_index]

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """计算关键词匹配分数"""
        if not query or not content:
            return 0.0

        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Jaccard相似度
        intersection = query_words & content_words
        return len(intersection) / len(query_words)

    def _calculate_time_decay(self, timestamp: str) -> float:
        """计算时间衰减因子"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_minutes = (datetime.now() - memory_time).total_seconds() / 60

            # 指数衰减
            decay_rate = 0.01  # 每分钟衰减1%
            decay_factor = math.exp(-decay_rate * age_minutes)

            return max(0.1, decay_factor)
        except Exception:
            return 0.5

    def _try_tfidf_search(self, query: str):
        """尝试TF-IDF搜索（简单实现）"""
        # 简单实现：返回空字典，使用关键词匹配代替
        return {}