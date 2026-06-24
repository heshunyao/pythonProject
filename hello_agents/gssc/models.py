from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class ContextPacket:
    """上下文信息包 — GSSC 处理的最小信息单元"""

    content: str
    timestamp: datetime
    token_count: int = 0
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """GSSC 上下文构建配置"""

    max_total_tokens: int = 3000
    min_relevance: float = 0.1
    relevance_weight: float = 0.7
    recency_weight: float = 0.3
    history_window: int = 5
    memory_limit: int = 10
    rag_limit: int = 5
