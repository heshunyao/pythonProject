from typing import List, Dict, Any, Optional
import uuid


class QdrantVectorStore:
    """Qdrant向量存储实现

    提供高性能的向量检索能力，支持：
    - 相似度搜索
    - 条件过滤
    - 批量插入
    - 自动集合管理

    如果Qdrant不可用，会自动降级到内存存储。
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None, collection_name: str = "memory", vector_size: int = 384):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._qdrant_client = None
        self._use_memory_fallback = False
        self._memory_store = []

        self._init_client()

    def _init_client(self):
        """初始化Qdrant客户端"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse

            self._qdrant_client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )

            # 检查集合是否存在，不存在则创建
            try:
                self._qdrant_client.get_collection(self.collection_name)
            except UnexpectedResponse:
                self._qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": self.vector_size,
                        "distance": "Cosine"
                    }
                )

            print(f"✅ 成功连接到Qdrant: {self.url}, 集合: {self.collection_name}")
        except ImportError:
            self._use_memory_fallback = True
            print(f"⚠️ Qdrant客户端未安装，使用内存存储降级模式")
        except Exception as e:
            self._use_memory_fallback = True
            print(f"⚠️ 无法连接Qdrant ({str(e)}), 使用内存存储降级模式")

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """批量添加向量"""
        if self._use_memory_fallback:
            return self._add_vectors_memory(vectors, metadata, ids)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        from qdrant_client.models import PointStruct

        points = []
        for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=meta
            ))

        self._qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def _add_vectors_memory(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """内存模式添加向量"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
            self._memory_store.append({
                "id": point_id,
                "vector": vector,
                "payload": meta
            })

        return ids

    def search_similar(self, query_vector: List[float], limit: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if self._use_memory_fallback:
            return self._search_similar_memory(query_vector, limit, where)

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if where:
            must_conditions = []
            for key, value in where.items():
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            qdrant_filter = Filter(must=must_conditions)

        results = self._qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            filter=qdrant_filter,
            with_payload=True
        )

        hits = []
        for result in results:
            hits.append({
                "id": str(result.id),
                "content": result.payload.get("content", ""),
                "score": result.score,
                "metadata": result.payload
            })

        return hits

    def _search_similar_memory(self, query_vector: List[float], limit: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """内存模式搜索相似向量"""
        results = []

        for item in self._memory_store:
            # 条件过滤
            if where:
                payload = item["payload"]
                match = True
                for key, value in where.items():
                    if payload.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, item["vector"])
            results.append({
                "id": item["id"],
                "content": item["payload"].get("content", ""),
                "score": similarity,
                "metadata": item["payload"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """获取向量"""
        if self._use_memory_fallback:
            for item in self._memory_store:
                if item["id"] == vector_id:
                    return item
            return None

        try:
            result = self._qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True
            )

            if result:
                return {
                    "id": str(result[0].id),
                    "vector": result[0].vector,
                    "payload": result[0].payload
                }
            return None
        except Exception:
            return None

    def delete_vector(self, vector_id: str) -> bool:
        """删除向量"""
        if self._use_memory_fallback:
            for i, item in enumerate(self._memory_store):
                if item["id"] == vector_id:
                    del self._memory_store[i]
                    return True
            return False

        try:
            self._qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            return True
        except Exception:
            return False

    def count_vectors(self) -> int:
        """统计向量数量"""
        if self._use_memory_fallback:
            return len(self._memory_store)

        try:
            info = self._qdrant_client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """创建集合"""
        if self._use_memory_fallback:
            return True

        try:
            from qdrant_client.http.exceptions import UnexpectedResponse
            try:
                self._qdrant_client.get_collection(collection_name)
                return True
            except UnexpectedResponse:
                self._qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": vector_size,
                        "distance": "Cosine"
                    }
                )
                return True
        except Exception:
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        if self._use_memory_fallback:
            return True

        try:
            self._qdrant_client.delete_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def clear_collection(self) -> bool:
        """清空集合"""
        if self._use_memory_fallback:
            self._memory_store = []
            return True

        try:
            self._qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector="all"
            )
            return True
        except Exception:
            return False