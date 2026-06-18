import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime


class SQLiteDocumentStore:
    """SQLite文档存储实现

    提供结构化数据的持久化存储，支持：
    - CRUD操作
    - 条件查询
    - 批量操作
    """

    def __init__(self, database_path: str = "./data/memory.db"):
        self.database_path = database_path
        self._ensure_database()

    def _ensure_database(self):
        """确保数据库和表存在"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT DEFAULT '{}',
                FOREIGN KEY (episode_id) REFERENCES memories(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}'
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                FOREIGN KEY (source_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(entity_id)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)
        ''')

        conn.commit()
        conn.close()

    def add_document(self, document: Dict[str, Any]) -> str:
        """添加文档"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        memory_id = document["id"]
        metadata = json.dumps(document.get("metadata", {}))

        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, content, timestamp, memory_type, importance, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_id,
            document["content"],
            document.get("timestamp", datetime.now().isoformat()),
            document.get("memory_type", "unknown"),
            document.get("importance", 0.5),
            metadata,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()
        return memory_id

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM memories WHERE id = ?
        ''', (document_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "content": row[1],
                "timestamp": row[2],
                "memory_type": row[3],
                "importance": row[4],
                "metadata": json.loads(row[5]),
                "created_at": row[6],
                "updated_at": row[7]
            }
        return None

    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """更新文档"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        set_clauses = []
        params = []

        if "content" in updates:
            set_clauses.append("content = ?")
            params.append(updates["content"])
        if "importance" in updates:
            set_clauses.append("importance = ?")
            params.append(updates["importance"])
        if "metadata" in updates:
            set_clauses.append("metadata = ?")
            params.append(json.dumps(updates["metadata"]))

        set_clauses.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(document_id)

        if set_clauses:
            query = f'''UPDATE memories SET {", ".join(set_clauses)} WHERE id = ?'''
            cursor.execute(query, params)
            conn.commit()

        conn.close()
        return cursor.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''DELETE FROM memories WHERE id = ?''', (document_id,))
        conn.commit()
        conn.close()

        return cursor.rowcount > 0

    def query_documents(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """条件查询文档"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if "memory_type" in query:
            where_clauses.append("memory_type = ?")
            params.append(query["memory_type"])
        if "user_id" in query:
            where_clauses.append("metadata LIKE ?")
            params.append(f'%user_id": "{query["user_id"]}%')
        if "start_time" in query:
            where_clauses.append("timestamp >= ?")
            params.append(query["start_time"])
        if "end_time" in query:
            where_clauses.append("timestamp <= ?")
            params.append(query["end_time"])

        query_str = "SELECT * FROM memories"
        if where_clauses:
            query_str += " WHERE " + " AND ".join(where_clauses)
        query_str += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query_str, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "timestamp": row[2],
                "memory_type": row[3],
                "importance": row[4],
                "metadata": json.loads(row[5]),
                "created_at": row[6],
                "updated_at": row[7]
            })

        return results

    def add_episode(self, episode: Dict[str, Any]) -> str:
        """添加情景"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        episode_id = episode["episode_id"]
        context = json.dumps(episode.get("context", {}))

        cursor.execute('''
            INSERT OR REPLACE INTO episodes 
            (episode_id, session_id, timestamp, content, context)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            episode_id,
            episode.get("session_id", "default"),
            episode.get("timestamp", datetime.now().isoformat()),
            episode["content"],
            context
        ))

        conn.commit()
        conn.close()
        return episode_id

    def get_episodes_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话的所有情景"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM episodes WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (session_id, limit))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "episode_id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "content": row[3],
                "context": json.loads(row[4])
            })

        return results

    def add_entity(self, entity: Dict[str, Any]) -> str:
        """添加实体"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        entity_id = entity["entity_id"]
        properties = json.dumps(entity.get("properties", {}))

        cursor.execute('''
            INSERT OR REPLACE INTO entities 
            (entity_id, name, entity_type, properties)
            VALUES (?, ?, ?, ?)
        ''', (
            entity_id,
            entity["name"],
            entity.get("entity_type", "unknown"),
            properties
        ))

        conn.commit()
        conn.close()
        return entity_id

    def add_relation(self, relation: Dict[str, Any]) -> int:
        """添加关系"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        properties = json.dumps(relation.get("properties", {}))

        cursor.execute('''
            INSERT INTO relations 
            (source_entity_id, target_entity_id, relation_type, properties)
            VALUES (?, ?, ?, ?)
        ''', (
            relation["source_entity_id"],
            relation["target_entity_id"],
            relation["relation_type"],
            properties
        ))

        conn.commit()
        relation_id = cursor.lastrowid
        conn.close()

        return relation_id

    def get_entities(self, entity_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取实体"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        query = "SELECT * FROM entities"
        params = []

        if entity_type:
            query += " WHERE entity_type = ?"
            params.append(entity_type)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "entity_id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "properties": json.loads(row[3])
            })

        return results

    def get_relations(self, source_entity_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取关系"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        query = "SELECT * FROM relations"
        params = []

        if source_entity_id:
            query += " WHERE source_entity_id = ?"
            params.append(source_entity_id)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "relation_id": row[0],
                "source_entity_id": row[1],
                "target_entity_id": row[2],
                "relation_type": row[3],
                "properties": json.loads(row[4])
            })

        return results

    def clear_all(self):
        """清空所有数据"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''DELETE FROM relations''')
        cursor.execute('''DELETE FROM entities''')
        cursor.execute('''DELETE FROM episodes''')
        cursor.execute('''DELETE FROM memories''')

        conn.commit()
        conn.close()

    def count_documents(self, memory_type: Optional[str] = None) -> int:
        """统计文档数量"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM memories"
        params = []

        if memory_type:
            query += " WHERE memory_type = ?"
            params.append(memory_type)

        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()

        return count