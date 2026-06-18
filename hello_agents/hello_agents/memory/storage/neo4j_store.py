from typing import List, Dict, Any, Optional
import uuid


class Neo4jGraphStore:
    """Neo4j图存储实现

    提供知识图谱管理能力，支持：
    - 实体创建和查询
    - 关系创建和查询
    - 图遍历
    - 路径搜索

    如果Neo4j不可用，会自动降级到内存存储。
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self._neo4j_driver = None
        self._use_memory_fallback = False

        # 内存降级存储
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: List[Dict[str, Any]] = []

        self._init_driver()

    def _init_driver(self):
        """初始化Neo4j驱动"""
        try:
            from neo4j import GraphDatabase

            self._neo4j_driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # 测试连接
            with self._neo4j_driver.session() as session:
                result = session.run("RETURN 1")
                result.single()

            print(f"✅ 成功连接到Neo4j: {self.uri}")
        except ImportError:
            self._use_memory_fallback = True
            print(f"⚠️ Neo4j驱动未安装，使用内存存储降级模式")
        except Exception as e:
            self._use_memory_fallback = True
            print(f"⚠️ 无法连接Neo4j ({str(e)}), 使用内存存储降级模式")

    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """添加实体"""
        if properties is None:
            properties = {}

        if self._use_memory_fallback:
            self._entities[entity_id] = {
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "properties": properties
            }
            return entity_id

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MERGE (e:{entity_type} {{entity_id: $entity_id}})
                    SET e.name = $name, e.properties = $properties
                    RETURN e.entity_id
                """.format(entity_type=self._sanitize_label(entity_type))

                result = session.run(
                    query,
                    entity_id=entity_id,
                    name=name,
                    properties=properties
                )

                return result.single()[0]
        except Exception as e:
            # 降级到内存存储
            self._use_memory_fallback = True
            self._entities[entity_id] = {
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "properties": properties
            }
            return entity_id

    def _sanitize_label(self, label: str) -> str:
        """清理标签名称，移除非法字符"""
        return ''.join(c for c in label.title() if c.isalnum())

    def add_relation(self, source_entity_id: str, target_entity_id: str, relation_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """添加关系"""
        if properties is None:
            properties = {}

        if self._use_memory_fallback:
            relation_id = str(uuid.uuid4())
            self._relations.append({
                "relation_id": relation_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relation_type": relation_type,
                "properties": properties
            })
            return relation_id

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH (s {{entity_id: $source_id}})
                    MATCH (t {{entity_id: $target_id}})
                    MERGE (s)-[r:{relation_type}]->(t)
                    SET r.properties = $properties
                    RETURN id(r)
                """.format(relation_type=self._sanitize_label(relation_type))

                result = session.run(
                    query,
                    source_id=source_entity_id,
                    target_id=target_entity_id,
                    properties=properties
                )

                return str(result.single()[0])
        except Exception as e:
            self._use_memory_fallback = True
            relation_id = str(uuid.uuid4())
            self._relations.append({
                "relation_id": relation_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relation_type": relation_type,
                "properties": properties
            })
            return relation_id

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体"""
        if self._use_memory_fallback:
            return self._entities.get(entity_id)

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH (e {{entity_id: $entity_id}})
                    RETURN e.entity_id, e.name, labels(e)[0] as entity_type, e.properties
                """

                result = session.run(query, entity_id=entity_id)
                record = result.single()

                if record:
                    return {
                        "entity_id": record["e.entity_id"],
                        "name": record["e.name"],
                        "entity_type": record["entity_type"],
                        "properties": record["e.properties"] or {}
                    }
                return None
        except Exception:
            return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """按类型获取实体"""
        if self._use_memory_fallback:
            return [
                entity for entity in self._entities.values()
                if entity["entity_type"] == entity_type
            ][:limit]

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH (e:{entity_type})
                    RETURN e.entity_id, e.name, labels(e)[0] as entity_type, e.properties
                    LIMIT $limit
                """.format(entity_type=self._sanitize_label(entity_type))

                results = session.run(query, limit=limit)

                entities = []
                for record in results:
                    entities.append({
                        "entity_id": record["e.entity_id"],
                        "name": record["e.name"],
                        "entity_type": record["entity_type"],
                        "properties": record["e.properties"] or {}
                    })

                return entities
        except Exception:
            return [
                entity for entity in self._entities.values()
                if entity["entity_type"] == entity_type
            ][:limit]

    def get_relations(self, source_entity_id: Optional[str] = None, relation_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取关系"""
        if self._use_memory_fallback:
            results = self._relations[:]

            if source_entity_id:
                results = [r for r in results if r["source_entity_id"] == source_entity_id]
            if relation_type:
                results = [r for r in results if r["relation_type"] == relation_type]

            return results[:limit]

        try:
            with self._neo4j_driver.session() as session:
                query_parts = ["MATCH (s)-[r]->(t)"]
                params = {}

                if source_entity_id:
                    query_parts.append("WHERE s.entity_id = $source_id")
                    params["source_id"] = source_entity_id

                if relation_type:
                    if "WHERE" in query_parts[0]:
                        query_parts.append("AND type(r) = $relation_type")
                    else:
                        query_parts.append("WHERE type(r) = $relation_type")
                    params["relation_type"] = relation_type

                query_parts.append("""
                    RETURN s.entity_id as source_id, t.entity_id as target_id, 
                           type(r) as relation_type, r.properties as properties, id(r) as relation_id
                    LIMIT $limit
                """)

                params["limit"] = limit
                query = " ".join(query_parts)

                results = session.run(query, **params)

                relations = []
                for record in results:
                    relations.append({
                        "relation_id": str(record["relation_id"]),
                        "source_entity_id": record["source_id"],
                        "target_entity_id": record["target_id"],
                        "relation_type": record["relation_type"],
                        "properties": record["properties"] or {}
                    })

                return relations
        except Exception:
            results = self._relations[:]

            if source_entity_id:
                results = [r for r in results if r["source_entity_id"] == source_entity_id]
            if relation_type:
                results = [r for r in results if r["relation_type"] == relation_type]

            return results[:limit]

    def get_neighbors(self, entity_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取实体的邻居节点"""
        if self._use_memory_fallback:
            neighbors = []

            for relation in self._relations:
                if relation["source_entity_id"] == entity_id:
                    target = self._entities.get(relation["target_entity_id"])
                    if target:
                        neighbors.append({
                            "entity_id": target["entity_id"],
                            "name": target["name"],
                            "entity_type": target["entity_type"],
                            "relation_type": relation["relation_type"]
                        })
                elif relation["target_entity_id"] == entity_id:
                    source = self._entities.get(relation["source_entity_id"])
                    if source:
                        neighbors.append({
                            "entity_id": source["entity_id"],
                            "name": source["name"],
                            "entity_type": source["entity_type"],
                            "relation_type": relation["relation_type"]
                        })

            return neighbors[:limit]

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH (e {{entity_id: $entity_id}})-[r]->(n)
                    RETURN n.entity_id, n.name, labels(n)[0] as entity_type, type(r) as relation_type
                    UNION
                    MATCH (n)-[r]->(e {{entity_id: $entity_id}})
                    RETURN n.entity_id, n.name, labels(n)[0] as entity_type, type(r) as relation_type
                    LIMIT $limit
                """

                results = session.run(query, entity_id=entity_id, limit=limit)

                neighbors = []
                for record in results:
                    neighbors.append({
                        "entity_id": record["n.entity_id"],
                        "name": record["n.name"],
                        "entity_type": record["entity_type"],
                        "relation_type": record["relation_type"]
                    })

                return neighbors
        except Exception:
            return self.get_neighbors(entity_id, limit)

    def find_path(self, source_entity_id: str, target_entity_id: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """查找两个实体之间的路径"""
        if self._use_memory_fallback:
            return self._find_path_memory(source_entity_id, target_entity_id, max_depth)

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH path = shortestPath((s {{entity_id: $source_id}})-[*..{max_depth}]->(t {{entity_id: $target_id}}))
                    RETURN [node in nodes(path) | {{entity_id: node.entity_id, name: node.name}}] as nodes,
                           [rel in relationships(path) | {{relation_type: type(rel)}}] as relations
                """.format(max_depth=max_depth)

                result = session.run(query, source_id=source_entity_id, target_id=target_entity_id)
                record = result.single()

                if record:
                    path = []
                    nodes = record["nodes"]
                    relations = record["relations"]

                    for i, node in enumerate(nodes):
                        path.append({"entity": node})
                        if i < len(relations):
                            path.append({"relation": relations[i]})

                    return path
                return None
        except Exception:
            return self._find_path_memory(source_entity_id, target_entity_id, max_depth)

    def _find_path_memory(self, source_entity_id: str, target_entity_id: str, max_depth: int) -> Optional[List[Dict[str, Any]]]:
        """内存模式查找路径（简化实现）"""
        from collections import deque

        visited = set()
        queue = deque([(source_entity_id, [])])

        while queue:
            current_id, path = queue.popleft()

            if current_id == target_entity_id:
                full_path = []
                for i, entity_id in enumerate(path + [target_entity_id]):
                    entity = self._entities.get(entity_id)
                    if entity:
                        full_path.append({"entity": {"entity_id": entity_id, "name": entity.get("name", "")}})
                        if i < len(path):
                            full_path.append({"relation": {"relation_type": "related_to"}})
                return full_path

            if current_id in visited or len(path) >= max_depth:
                continue

            visited.add(current_id)

            for relation in self._relations:
                if relation["source_entity_id"] == current_id:
                    next_id = relation["target_entity_id"]
                    if next_id not in visited:
                        queue.append((next_id, path + [current_id]))
                elif relation["target_entity_id"] == current_id:
                    next_id = relation["source_entity_id"]
                    if next_id not in visited:
                        queue.append((next_id, path + [current_id]))

        return None

    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if self._use_memory_fallback:
            if entity_id in self._entities:
                del self._entities[entity_id]
                self._relations = [
                    r for r in self._relations
                    if r["source_entity_id"] != entity_id and r["target_entity_id"] != entity_id
                ]
                return True
            return False

        try:
            with self._neo4j_driver.session() as session:
                query = """
                    MATCH (e {{entity_id: $entity_id}})
                    DETACH DELETE e
                """

                result = session.run(query, entity_id=entity_id)
                return result.consume().counters.nodes_deleted > 0
        except Exception:
            return self.delete_entity(entity_id)

    def clear_all(self):
        """清空所有数据"""
        if self._use_memory_fallback:
            self._entities = {}
            self._relations = []
            return

        try:
            with self._neo4j_driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception:
            self._entities = {}
            self._relations = []

    def count_entities(self) -> int:
        """统计实体数量"""
        if self._use_memory_fallback:
            return len(self._entities)

        try:
            with self._neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n)")
                return result.single()[0]
        except Exception:
            return len(self._entities)

    def count_relations(self) -> int:
        """统计关系数量"""
        if self._use_memory_fallback:
            return len(self._relations)

        try:
            with self._neo4j_driver.session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r)")
                return result.single()[0]
        except Exception:
            return len(self._relations)