import math
from datetime import datetime
from typing import List, Dict

from .base import BaseMemory, MemoryConfig, MemoryItem, StorageManager


class PerceptualMemory(BaseMemory):
    """感知记忆实现

    特点：
    - 支持多模态数据（文本、图像、音频等）
    - 跨模态相似性搜索
    - 感知数据的语义理解
    - 支持内容生成和检索
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.text_embedder = self._get_text_embedder()
        self._image_dim = 512
        self._audio_dim = 512

        # 使用StorageManager获取向量存储
        self.vector_stores = {
            "text": StorageManager.get_qdrant_store(config, "perceptual_text"),
            "image": StorageManager.get_qdrant_store(config, "perceptual_image"),
            "audio": StorageManager.get_qdrant_store(config, "perceptual_audio")
        }

    def add(self, memory_item: MemoryItem) -> str:
        """添加感知记忆"""
        modality = memory_item.metadata.get("modality", "text")
        store = self._get_vector_store_for_modality(modality)

        embedding = self._encode_data(memory_item.content, modality)

        metadata = {
            "memory_id": memory_item.id,
            "content": memory_item.content,
            "modality": modality,
            "memory_type": "perceptual",
            "timestamp": memory_item.timestamp,
            "importance": memory_item.importance,
            **memory_item.metadata
        }

        store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[memory_item.id]
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆"""
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality = kwargs.get("query_modality", target_modality or "text")

        store = self._get_vector_store_for_modality(target_modality or query_modality)
        query_vector = self._encode_data(query, query_modality)

        where = {"memory_type": "perceptual"}
        if user_id:
            where["user_id"] = user_id
        if target_modality:
            where["modality"] = target_modality

        hits = store.search_similar(
            query_vector=query_vector,
            limit=max(limit * 5, 20),
            where=where
        )

        results = []
        for hit in hits:
            vector_score = float(hit.get("score", 0.0))
            recency_score = self._calculate_recency_score(hit["metadata"]["timestamp"])
            importance = hit["metadata"].get("importance", 0.5)

            base_relevance = vector_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight

            results.append((combined_score, self._create_memory_item(hit)))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

    def delete(self, memory_id: str) -> bool:
        """删除感知记忆"""
        for modality, store in self.vector_stores.items():
            if store.delete_vector(memory_id):
                return True
        return False

    def _calculate_recency_score(self, timestamp: str) -> float:
        """计算时间近因性得分"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    def _get_text_embedder(self):
        """获取文本编码器"""
        class SimpleEmbedder:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:100]]
        return SimpleEmbedder()

    def _encode_data(self, data, modality):
        """编码数据"""
        if modality == "text":
            return self.text_embedder.encode(data)
        return [0] * self.vector_dim

    def _get_vector_store_for_modality(self, modality):
        """获取指定模态的向量存储"""
        return self.vector_stores.get(modality, self.vector_stores["text"])