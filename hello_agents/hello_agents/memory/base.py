from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MemoryConfig:
    """内存配置类"""
    database_path: str = "./data/memory.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "memory"

    working_memory_capacity: int = 50
    working_memory_ttl: int = 60

    vector_dim: int = 384

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


@dataclass
class MemoryItem:
    """记忆项类"""
    id: str
    content: str
    timestamp: str
    memory_type: str
    importance: float = 0.5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BaseMemory(ABC):
    """基础内存抽象类"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage_backend = storage_backend
        self.vector_dim = config.vector_dim

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆"""
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索记忆"""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    def _create_memory_item(self, hit) -> MemoryItem:
        """从检索结果创建MemoryItem"""
        metadata = hit.get("metadata", {})
        return MemoryItem(
            id=hit.get("id", hit.get("memory_id", str(id(hit)))),
            content=hit.get("content", hit.get("payload", {}).get("content", "")),
            timestamp=metadata.get("timestamp", datetime.now().isoformat()),
            memory_type=metadata.get("memory_type", "unknown"),
            importance=metadata.get("importance", 0.5),
            metadata=metadata
        )


@dataclass
class Episode:
    """情景类"""
    episode_id: str
    session_id: str
    timestamp: str
    content: str
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class Entity:
    """实体类"""
    entity_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relation:
    """关系类"""
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class StorageManager:
    """存储管理器"""

    _document_store = None
    _qdrant_store = None
    _neo4j_store = None

    @classmethod
    def get_document_store(cls, config: MemoryConfig) -> 'SQLiteDocumentStore':
        """获取SQLite文档存储实例"""
        if cls._document_store is None:
            from .storage.document_store import SQLiteDocumentStore
            cls._document_store = SQLiteDocumentStore(config.database_path)
        return cls._document_store

    @classmethod
    def get_qdrant_store(cls, config: MemoryConfig, collection_name: str = None) -> 'QdrantVectorStore':
        """获取Qdrant向量存储实例"""
        if cls._qdrant_store is None:
            from .storage.qdrant_store import QdrantVectorStore
            cls._qdrant_store = QdrantVectorStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name=collection_name or config.qdrant_collection,
                vector_size=config.vector_dim
            )
        return cls._qdrant_store

    @classmethod
    def get_neo4j_store(cls, config: MemoryConfig) -> 'Neo4jGraphStore':
        """获取Neo4j图存储实例"""
        if cls._neo4j_store is None:
            from .storage.neo4j_store import Neo4jGraphStore
            cls._neo4j_store = Neo4jGraphStore(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password
            )
        return cls._neo4j_store

    @classmethod
    def init_all_stores(cls, config: MemoryConfig):
        """初始化所有存储"""
        cls.get_document_store(config)
        cls.get_qdrant_store(config)
        cls.get_neo4j_store(config)

    @classmethod
    def clear_all(cls):
        """清空所有存储"""
        if cls._document_store:
            cls._document_store.clear_all()
        if cls._qdrant_store:
            cls._qdrant_store.clear_collection()
        if cls._neo4j_store:
            cls._neo4j_store.clear_all()

    @classmethod
    def reset(cls):
        """重置存储管理器"""
        cls._document_store = None
        cls._qdrant_store = None
        cls._neo4j_store = None
