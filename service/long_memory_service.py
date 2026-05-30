from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from agent import llm_service
from agent.llm_service import LlmService
from db.db_config import DbSessionLocal, LongTermMemory, MemoryType
from db.db_service import DbService
from db.hybrid_retrieval import rank_hybrid_results
from model.memory_model import LongTermMemoryCreate, LongTermMemoryRead
from service.embedding_service import EmbeddingService


class LongMemoryService:
    """长期记忆服务，负责摘要生成、长期写入和检索。"""

    def __init__(
        self,
        db: DbService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db = db or DbService()
        self._owns_db = db is None
        self.embedding_service = embedding_service or EmbeddingService()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_long_term_memory_from_messages(
        self,
        player_id: str,
        npc_id: str,
        messages: list[dict[str, Any]],
        llm_service: Any | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 生成摘要，并写入一条长期记忆。"""

        if not player_id or not npc_id:
            return {"ok": False, "error": "缺少 player_id 或 npc_id"}
        if not messages:
            return {"ok": False, "error": "缺少待归档消息"}

        summary = self.summarize_messages(
            messages=messages,
            player_id=player_id,
            npc_id=npc_id,
            llm_service=llm_service,
        )
        result = self.create_long_term_memory(
            player_id=player_id,
            npc_id=npc_id,
            title=summary["title"],
            content=summary["content"],
            memory_type=summary["memory_type"],
            importance=summary["importance"],
            source_message_ids=summary["source_message_ids"],
            metadata={
                "source": "short_term_archive",
                "archived_message_count": len(messages),
                "created_by": "LongMemoryService",
            },
        )
        # result["summary"] = summary
        return result

    def create_long_term_memory(
        self,
        player_id: str,
        npc_id: str,
        title: str,
        content: str,
        memory_type: str = "event",
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写入长期记忆，并尽量补充 embedding。"""

        if not player_id or not npc_id or not content:
            return {"ok": False, "error": "player_id、npc_id、content 不能为空"}

        memory_type = self._safe_memory_type(memory_type)
        importance = self._safe_importance(importance)
        embedding = self._embed_memory_safely(
            title=title,
            content=content,
            memory_type=memory_type,
        )

        with DbSessionLocal() as session:
            row = LongTermMemory(
                memory_id=f"mem_{uuid4().hex[:16]}",
                player_id=player_id,
                npc_id=npc_id,
                title=title or "一条长期记忆",
                content=content,
                memory_type=MemoryType(memory_type),
                importance=importance,
                embedding=embedding,
                embedding_model=(
                    self.embedding_service.model_name
                    if embedding is not None
                    else None
                ),
                source_message_ids=source_message_ids or [],
                metadata_json=metadata or {},
            )
            session.add(row)
            session.commit()
            session.refresh(row)

            return {"ok": True, "memory": self._serialize_long_term_memory(row)}

    def create_long_memory(self, data: LongTermMemoryCreate) -> LongTermMemoryRead:
        """兼容旧工具层的 Pydantic 创建入口。"""

        memory = self.db.create_long_memory(**data.model_dump())
        return LongTermMemoryRead.model_validate(memory)

    def list_long_memories(
        self,
        player_id: str,
        npc_id: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemoryRead]:
        """按重要性和时间返回长期记忆。"""

        rows = self.db.list_long_memories(
            player_id,
            npc_id=npc_id,
            limit=max(1, min(int(limit), 100)),
        )
        return [LongTermMemoryRead.model_validate(row) for row in rows]

    def search_long_memories(
        self,
        player_id: str,
        query: str,
        npc_id: str | None = None,
        limit: int = 20,
        memory_type: str | None = None,
    ) -> list[dict[str, Any] | LongTermMemoryRead]:
        """优先向量检索，失败时回退到轻量文本排序。"""

        if not query:
            print("没有长期记忆检索query\n")
            return self.list_long_memories(player_id=player_id, npc_id=npc_id, limit=limit)

        query_embedding = self._embed_query_safely(query)
        return self._search_by_hybrid(
            player_id=player_id,
            npc_id=npc_id,
            query=query,
            query_embedding=query_embedding,
            memory_type=memory_type,
            limit=limit,
        )

    def summarize_messages(
        self,
        messages: list[dict[str, Any]],
        player_id: str,
        npc_id: str,
        llm_service: Any | None = None,
    ) -> dict[str, Any]:
        """将短期消息总结为长期记忆结构。"""

        source_message_ids = [
            str(msg.get("message_id"))
            for msg in messages
            if msg.get("message_id")
        ]
        dialogue_text = "\n".join(
            f"{msg.get('role', 'unknown')}：{msg.get('content', '')}"
            for msg in messages
            if msg.get("content")
        )

        if llm_service is not None:
            try:
                prompt = json.dumps(
                    {
                        "task": "将NPC 对话压缩为一条长期记忆，保留事实和关键信息，只输出 JSON。",
                        "player_id": player_id,
                        "npc_id": npc_id,
                        "dialogue": dialogue_text,
                        "required_schema": {
                            "title": "不超过 30 字",
                            "content": "长期记忆摘要",
                            "memory_type": "preference|relationship|event|habit|warning",
                            "importance": "0 到 1 的浮点数",
                        },
                    },
                    ensure_ascii=False,
                )
                response, _ = llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt="你是游戏 NPC 记忆摘要模块，只输出合法 JSON。",
                )
                data = json.loads(response)
                return {
                    "title": str(data.get("title", "一段重要互动")),
                    "content": str(data.get("content", dialogue_text[:500])),
                    "memory_type": self._safe_memory_type(
                        str(data.get("memory_type", "event"))
                    ),
                    "importance": self._safe_importance(data.get("importance", 0.5)),
                    "source_message_ids": source_message_ids,
                }
            except Exception:
                pass

        return self._rule_based_summary(
            dialogue_text=dialogue_text,
            source_message_ids=source_message_ids,
        )

    def _search_by_hybrid(
        self,
        player_id: str,
        npc_id: str | None,
        query: str,
        query_embedding: list[float] | None,
        memory_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """使用候选记忆的 BM25 分和向量分进行混合排序。"""

        candidate_limit = max(20, min(max(1, int(limit)) * 10, 200))
        rows = self._load_memory_candidates(
            player_id=player_id,
            npc_id=npc_id,
            query_embedding=query_embedding,
            memory_type=memory_type,
            limit=candidate_limit,
        )
        if not rows:
            return []

        ranked = rank_hybrid_results(
            rows,
            query=query,
            content_getter=lambda item: (
                f"{item['memory'].title}\n{item['memory'].content}"
            ),
            distance_getter=lambda item: item.get("distance"),
            limit=limit,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        memories = []
        for row in ranked:
            item = self._serialize_long_term_memory(row["item"]["memory"])
            item["distance"] = row["distance"]
            item["similarity"] = row["vector_score"]
            item["vector_score"] = row["vector_score"]
            item["bm25_score"] = row["bm25_score"]
            item["hybrid_score"] = row["hybrid_score"]
            memories.append(item)
        return memories

    def _load_memory_candidates(
        self,
        player_id: str,
        npc_id: str | None,
        query_embedding: list[float] | None,
        memory_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """加载长期记忆候选集，存在向量时一并计算 cosine distance。"""

        with DbSessionLocal() as session:
            if query_embedding is not None:
                stmt = (
                    select(
                        LongTermMemory,
                        LongTermMemory.embedding.cosine_distance(query_embedding).label(
                            "distance"
                        ),
                    )
                    .where(LongTermMemory.player_id == player_id)
                    .where(LongTermMemory.embedding.is_not(None))
                )
                if npc_id:
                    stmt = stmt.where(LongTermMemory.npc_id == npc_id)
                if memory_type:
                    stmt = stmt.where(LongTermMemory.memory_type == MemoryType(memory_type))
                rows = session.execute(stmt.order_by("distance").limit(limit)).all()
                candidates = [
                    {"memory": memory, "distance": float(distance)}
                    for memory, distance in rows
                ]
                seen_ids = {item["memory"].memory_id for item in candidates}
                remaining_limit = max(0, limit - len(candidates))
                if remaining_limit:
                    null_stmt = (
                        select(LongTermMemory)
                        .where(LongTermMemory.player_id == player_id)
                        .where(LongTermMemory.embedding.is_(None))
                    )
                    if npc_id:
                        null_stmt = null_stmt.where(LongTermMemory.npc_id == npc_id)
                    if memory_type:
                        null_stmt = null_stmt.where(
                            LongTermMemory.memory_type == MemoryType(memory_type)
                        )
                    null_rows = session.scalars(
                        null_stmt.order_by(LongTermMemory.importance.desc()).limit(
                            remaining_limit
                        )
                    ).all()
                    candidates.extend(
                        {"memory": memory, "distance": None}
                        for memory in null_rows
                        if memory.memory_id not in seen_ids
                    )
                return candidates

            stmt = select(LongTermMemory).where(LongTermMemory.player_id == player_id)
            if npc_id:
                stmt = stmt.where(LongTermMemory.npc_id == npc_id)
            if memory_type:
                stmt = stmt.where(LongTermMemory.memory_type == MemoryType(memory_type))
            rows = session.scalars(
                stmt.order_by(LongTermMemory.importance.desc()).limit(limit)
            ).all()
            return [{"memory": memory, "distance": None} for memory in rows]

    def _search_by_vector(
        self,
        player_id: str,
        npc_id: str | None,
        query_embedding: list[float],
        memory_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """使用 pgvector 做相似度检索。"""

        with DbSessionLocal() as session:
            stmt = (
                select(
                    LongTermMemory,
                    LongTermMemory.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .where(LongTermMemory.player_id == player_id)
                .where(LongTermMemory.embedding.is_not(None))
            )
            if npc_id:
                stmt = stmt.where(LongTermMemory.npc_id == npc_id)
            if memory_type:
                stmt = stmt.where(LongTermMemory.memory_type == MemoryType(memory_type))

            rows = session.execute(stmt.order_by("distance").limit(limit)).all()

        memories = []
        for memory, distance in rows:
            item = self._serialize_long_term_memory(memory)
            item["distance"] = float(distance) if distance is not None else None
            item["similarity"] = 1.0 - float(distance) if distance is not None else None
            memories.append(item)
        return memories

    def _search_by_text(
        self,
        player_id: str,
        npc_id: str | None,
        query: str,
        limit: int,
    ) -> list[LongTermMemoryRead]:
        """没有可用向量时，用标题和内容命中做轻量排序。"""

        candidates = self.list_long_memories(
            player_id=player_id,
            npc_id=npc_id,
            limit=max(limit * 3, limit),
        )
        query_text = str(query or "").strip().lower()
        if not query_text:
            return candidates[:limit]

        def memory_score(memory: LongTermMemoryRead) -> tuple[int, float]:
            # 标题命中权重更高，内容命中作为兜底。
            title = str(memory.title or "").lower()
            content = str(memory.content or "").lower()
            score = 0
            if query_text in title:
                score += 2
            if query_text in content:
                score += 1
            return score, float(memory.importance or 0)

        return sorted(candidates, key=memory_score, reverse=True)[:limit]

    def _rule_based_summary(
        self,
        dialogue_text: str,
        source_message_ids: list[str],
    ) -> dict[str, Any]:
        """LLM 不可用时的规则摘要兜底。"""

        merged = dialogue_text.strip()
        memory_type = "event"
        importance = 0.5
        title = "玩家与 NPC 的一段互动"

        if any(word in merged for word in ["喜欢", "偏好", "想喝", "拿铁", "美式", "手冲"]):
            memory_type = "preference"
            importance = 0.7
            title = "玩家表达了偏好或喜好"
        elif any(word in merged for word in ["不要", "讨厌", "生气", "冒犯", "压力"]):
            memory_type = "warning"
            importance = 0.8
            title = "玩家互动中出现需要注意的信息"
        elif any(word in merged for word in ["谢谢", "信任", "熟悉", "抱歉", "开心"]):
            memory_type = "relationship"
            importance = 0.6
            title = "玩家与 NPC 的关系发生变化"
        elif any(word in merged for word in ["经常", "每天", "上午", "下午", "常来", "习惯"]):
            memory_type = "habit"
            importance = 0.65
            title = "玩家表现出了某些习惯或规律"

        return {
            "title": title,
            "content": merged[:500] or "玩家与 NPC 进行了一段普通互动。",
            "memory_type": memory_type,
            "importance": importance,
            "source_message_ids": source_message_ids,
        }

    def _embed_memory_safely(
        self,
        title: str,
        content: str,
        memory_type: str,
    ) -> list[float] | None:
        """embedding 失败不阻断长期记忆写入。"""

        try:
            return self.embedding_service.embed_memory(
                title=title,
                content=content,
                memory_type=memory_type,
            )
        except Exception:
            return None

    def _embed_query_safely(self, query: str) -> list[float] | None:
        """查询向量生成失败时回退文本检索。"""

        try:
            return self.embedding_service.embed_query(query)
        except Exception:
            return None

    def _serialize_long_term_memory(self, row: LongTermMemory) -> dict[str, Any]:
        """将 ORM 长期记忆转换为工具层可返回结构。"""

        return {
            "memory_id": row.memory_id,
            "player_id": row.player_id,
            "npc_id": row.npc_id,
            "title": row.title,
            "content": row.content,
            "memory_type": (
                row.memory_type.value
                if hasattr(row.memory_type, "value")
                else row.memory_type
            ),
            "importance": row.importance,
            "embedding_model": row.embedding_model,
            "source_message_ids": row.source_message_ids or [],
            "metadata": row.metadata_json or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def _safe_memory_type(self, value: str) -> str:
        allowed = {item.value for item in MemoryType}
        return value if value in allowed else MemoryType.EVENT.value

    def _safe_importance(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.5
        return max(0.0, min(score, 1.0))

    def backfill_long_term_memory_embeddings(
        self,
        player_id: str | None = None,
        npc_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """为已经存在但 embedding 为空的长期记忆补充向量。"""

        limit = max(1, min(int(limit), 1000))
        updated_count = 0
        failed_items: list[dict[str, Any]] = []

        with DbSessionLocal() as session:
            query = session.query(LongTermMemory).filter(LongTermMemory.embedding.is_(None))
            if player_id:
                query = query.filter(LongTermMemory.player_id == player_id)
            if npc_id:
                query = query.filter(LongTermMemory.npc_id == npc_id)

            rows = query.order_by(LongTermMemory.created_at.asc()).limit(limit).all()
            for row in rows:
                try:
                    memory_type = (
                        row.memory_type.value
                        if hasattr(row.memory_type, "value")
                        else str(row.memory_type)
                    )
                    embedding = self.embedding_service.embed_memory(
                        title=row.title,
                        content=row.content,
                        memory_type=memory_type,
                    )
                    row.embedding = embedding
                    row.embedding_model = self.embedding_service.model_name
                    row.updated_at = datetime.now(timezone.utc)
                    updated_count += 1
                except Exception as exc:
                    failed_items.append({"memory_id": row.memory_id, "error": str(exc)})

            session.commit()

        return {
            "ok": True,
            "tool_name": "backfill_long_term_memory_embeddings",
            "player_id": player_id,
            "npc_id": npc_id,
            "updated_count": updated_count,
            "failed_count": len(failed_items),
            "failed_items": failed_items,
        }

if __name__ == "__main__":
    service = LongMemoryService()
    result = service.search_long_memories(
        player_id="p001",
        npc_id="passerby_001",
        query="NPC喜爱拍照和甜品，期待下次一起喝焦糖拿铁拍照",
    )
    print(result)
