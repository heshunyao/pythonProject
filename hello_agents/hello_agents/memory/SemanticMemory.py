from typing import List, Dict

from .base import BaseMemory, MemoryConfig, MemoryItem, Entity, Relation, StorageManager


class SemanticMemory(BaseMemory):
    """语义记忆实现

    特点：
    - 使用文本嵌入进行向量检索
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+关键词匹配
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.embedding_model = self._get_text_embedder()

        # 使用StorageManager获取存储
        self.vector_store = StorageManager.get_qdrant_store(config, "semantic_memory")
        self.graph_store = StorageManager.get_neo4j_store(config)

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        embedding = self.embedding_model.encode(memory_item.content)

        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)

        for entity in entities:
            self.entities[entity.entity_id] = entity
            self.graph_store.add_entity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=entity.properties
            )

        for relation in relations:
            self.relations.append(relation)
            self.graph_store.add_relation(
                source_entity_id=relation.source_entity_id,
                target_entity_id=relation.target_entity_id,
                relation_type=relation.relation_type,
                properties=relation.properties
            )

        metadata = {
            "memory_id": memory_item.id,
            "content": memory_item.content,
            "entities": [e.entity_id for e in entities],
            "entity_count": len(entities),
            "relation_count": len(relations),
            "memory_type": "semantic",
            "timestamp": memory_item.timestamp,
            "importance": memory_item.importance
        }

        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[memory_item.id]
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆"""
        user_id = kwargs.get("user_id")

        vector_results = self._vector_search(query, limit * 2, user_id)
        graph_results = self._graph_search(query, limit * 2, user_id)

        combined_results = self._combine_and_rank_results(
            vector_results, graph_results, query, limit
        )

        return combined_results[:limit]

    def delete(self, memory_id: str) -> bool:
        """删除语义记忆"""
        return self.vector_store.delete_vector(memory_id)

    def _vector_search(self, query: str, limit: int, user_id=None):
        """向量搜索"""
        query_vector = self.embedding_model.encode(query)

        where = {"memory_type": "semantic"}
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
                "memory_id": hit["id"],
                "content": hit.get("content", ""),
                "score": hit.get("score", 0.0),
                **hit.get("metadata", {})
            })

        return results

    def _graph_search(self, query: str, limit: int, user_id=None):
        """图搜索"""
        entities = self.graph_store.get_entities_by_type("person", limit)
        entities.extend(self.graph_store.get_entities_by_type("organization", limit))
        entities.extend(self.graph_store.get_entities_by_type("location", limit))

        results = []
        for entity in entities:
            if query.lower() in entity["name"].lower():
                results.append({
                    "memory_id": entity["entity_id"],
                    "content": f"实体: {entity['name']} ({entity['entity_type']})",
                    "similarity": 0.8
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        """混合排序结果"""
        combined = {}

        for result in vector_results:
            combined[result["memory_id"]] = {
                **result,
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0
            }

        for result in graph_results:
            memory_id = result["memory_id"]
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            else:
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0)
                }

        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)

            base_relevance = vector_score * 0.7 + graph_score * 0.3
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight
            result["combined_score"] = combined_score

        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        memory_items = []
        for result in sorted_results[:limit]:
            memory_items.append(MemoryItem(
                id=result["memory_id"],
                content=result.get("content", ""),
                timestamp=result.get("timestamp", ""),
                memory_type="semantic",
                importance=result.get("importance", 0.5),
                metadata={k: v for k, v in result.items() if k not in ["memory_id", "content", "timestamp", "importance"]}
            ))

        return memory_items

    def _extract_entities(self, text: str) -> List[Entity]:
        """简单实体提取"""
        entities = []
        entity_types = ["person", "organization", "location"]
        keywords = {
            "person": ["张三", "李四", "王五", "用户"],
            "organization": ["公司", "团队", "部门"],
            "location": ["北京", "上海", "广州"]
        }

        for entity_type, words in keywords.items():
            for word in words:
                if word in text:
                    entities.append(Entity(
                        entity_id=f"{entity_type}_{word}",
                        name=word,
                        entity_type=entity_type
                    ))
        return entities

    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """简单关系提取"""
        relations = []
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    relations.append(Relation(
                        source_entity_id=entities[i].entity_id,
                        target_entity_id=entities[j].entity_id,
                        relation_type="related_to"
                    ))
        return relations

    def _get_text_embedder(self):
        """获取文本编码器"""
        class SimpleEmbedder:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:100]]
        return SimpleEmbedder()

    def _init_nlp(self):
        """初始化NLP处理器"""
        return None