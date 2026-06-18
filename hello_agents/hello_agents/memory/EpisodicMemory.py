import math
from datetime import datetime
from typing import List, Dict

from .base import MemoryConfig, MemoryItem, Episode, StorageManager


class EpisodicMemory:
    """情景记忆实现
    特点：
    - SQLite+Qdrant混合存储架构
    - 支持时间序列和会话级检索
    - 结构化过滤 + 语义向量检索
    """

    def __init__(self, config: MemoryConfig):
        # 使用StorageManager获取存储
        self.doc_store = StorageManager.get_document_store(config)
        self.vector_store = StorageManager.get_qdrant_store(config, "episodic_memory")
        self.embedder = self._create_embedding_model()
        self.sessions = {}

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        episode = Episode(
            episode_id=memory_item.id,
            session_id=memory_item.metadata.get("session_id", "default"),
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=memory_item.metadata
        )

        session_id = episode.session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        self._persist_episode(episode)
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：结构化过滤 + 语义向量检索"""
        candidate_ids = self._structured_filter(**kwargs)
        hits = self._vector_search(query, limit * 5, kwargs.get("user_id"))

        results = []
        for hit in hits:
            if self._should_include(hit, candidate_ids, kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item(hit)
                results.append((score, memory_item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

    def delete(self, memory_id: str) -> bool:
        """删除情景记忆"""
        self.doc_store.delete_document(memory_id)
        self.vector_store.delete_vector(memory_id)

        for session_id, episodes in self.sessions.items():
            if memory_id in episodes:
                episodes.remove(memory_id)

        return True

    def _persist_episode(self, episode: Episode):
        """持久化情景"""
        self.doc_store.add_episode({
            "episode_id": episode.episode_id,
            "session_id": episode.session_id,
            "timestamp": episode.timestamp,
            "content": episode.content,
            "context": episode.context
        })

        self.doc_store.add_document({
            "id": episode.episode_id,
            "content": episode.content,
            "timestamp": episode.timestamp,
            "memory_type": "episodic",
            "importance": episode.context.get("importance", 0.5),
            "metadata": episode.context
        })

        embedding = self.embedder.encode(episode.content)
        metadata = {
            "memory_id": episode.episode_id,
            "content": episode.content,
            "session_id": episode.session_id,
            "memory_type": "episodic",
            "timestamp": episode.timestamp,
            "importance": episode.context.get("importance", 0.5),
            **episode.context
        }

        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[episode.episode_id]
        )

    def _structured_filter(self, **kwargs) -> List[str]:
        """结构化过滤"""
        query = {"memory_type": "episodic"}

        if "session_id" in kwargs:
            query["session_id"] = kwargs["session_id"]
        if "start_time" in kwargs:
            query["start_time"] = kwargs["start_time"]
        if "end_time" in kwargs:
            query["end_time"] = kwargs["end_time"]

        documents = self.doc_store.query_documents(query, limit=1000)
        return [doc["id"] for doc in documents]

    def _vector_search(self, query: str, limit: int, user_id=None):
        """向量搜索"""
        query_vector = self.embedder.encode(query)

        where = {"memory_type": "episodic"}
        if user_id:
            where["user_id"] = user_id

        hits = self.vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
            where=where
        )

        results = []
        for hit in hits:
            results.append({
                "id": hit["id"],
                "content": hit.get("content", ""),
                "score": hit.get("score", 0.0),
                "metadata": hit.get("metadata", {})
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _should_include(self, hit, candidate_ids, kwargs) -> bool:
        """判断是否应该包含该结果"""
        memory_id = hit.get("id", hit.get("memory_id"))
        return memory_id in candidate_ids if candidate_ids else True

    def _calculate_episode_score(self, hit) -> float:
        """情景记忆评分算法"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"]["timestamp"])
        importance = hit["metadata"].get("importance", 0.5)

        base_relevance = vec_score * 0.8 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)

        return base_relevance * importance_weight

    def _calculate_recency(self, timestamp: str) -> float:
        """计算时间近因性"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    def _create_memory_item(self, hit) -> MemoryItem:
        """创建MemoryItem"""
        metadata = hit.get("metadata", {})
        return MemoryItem(
            id=hit.get("id", hit.get("memory_id", str(id(hit)))),
            content=hit.get("content", ""),
            timestamp=metadata.get("timestamp", datetime.now().isoformat()),
            memory_type="episodic",
            importance=metadata.get("importance", 0.5),
            metadata=metadata
        )

    def _create_embedding_model(self):
        """创建嵌入模型"""
        class SimpleEmbedder:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:100]]
        return SimpleEmbedder()