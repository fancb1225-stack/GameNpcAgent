from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.db_config import DbSessionLocal, ShortTermMemory, LongTermMemory
from db.db_service import DbService
from service.embedding_service import EmbeddingService

memory_executor = ThreadPoolExecutor(max_workers=2)


class MemoryService:
    """
    Short-term + pgvector long-term memory service.

    - Short-term memory remains a JSON sliding window.
    - Long-term memory is stored in PostgreSQL with pgvector embedding.
    - Archiving can run asynchronously and will not block NPC reply return.
    """

    def __init__(self,  db: DbService | None = None, embedding_service: Optional[EmbeddingService] = None) -> None:
        self.db = db or DbService()
        self._owns_db = db is None
        self.embedding_service = embedding_service or EmbeddingService()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    """
        记忆服务：
        - 维护短期记忆滑动窗口
        - 当短期记忆超过阈值时，自动归档为长期摘要记忆
    """
    SHORT_TERM_LIMIT = 20
    ARCHIVE_BATCH_SIZE = 10

    def append_short_term_message(
        self,
        player_id: str,
        npc_id: str,
        message: Dict[str, Any],
        llm_service: Optional[Any] = None,
        async_archive: bool = True,
    ) -> Dict[str, Any]:
        result = self._append_short_term_message_only(
            player_id=player_id,
            npc_id=npc_id,
            message=message,
        )

        if not result.get("ok"):
            return result

        if result.get("short_term_count", 0) > self.SHORT_TERM_LIMIT:
            if async_archive:
                memory_executor.submit(
                    self.archive_short_term_memory,
                    player_id,
                    npc_id,
                    llm_service,
                )
                result["archive_submitted"] = True
                result["archived"] = False
            else:
                archive_result = self.archive_short_term_memory(
                    player_id=player_id,
                    npc_id=npc_id,
                    llm_service=llm_service,
                )
                result.update(archive_result)

        return result

    def _append_short_term_message_only(
        self,
        player_id: str,
        npc_id: str,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not player_id:
            return {"ok": False, "error": "缺少 player_id"}
        if not npc_id:
            return {"ok": False, "error": "缺少 npc_id"}
        if not message or not message.get("content"):
            return {"ok": False, "error": "缺少 message.content"}

        normalized_message = self._normalize_message(message)

        with DbSessionLocal() as session:
            memory = (
                session.query(ShortTermMemory)
                .filter(
                    ShortTermMemory.player_id == player_id,
                    ShortTermMemory.npc_id == npc_id,
                )
                .one_or_none()
            )

            if memory is None:
                memory = ShortTermMemory(
                    player_id=player_id,
                    npc_id=npc_id,
                    messages=[],
                    message_count=0,
                    last_message_at=datetime.now(timezone.utc),
                )
                session.add(memory)
                session.flush()

            messages = list(memory.messages or [])
            messages.append(normalized_message)

            memory.messages = messages
            memory.message_count = len(messages)
            memory.last_message_at = datetime.now(timezone.utc)

            session.commit()

            return {
                "ok": True,
                "archived": False,
                "archive_submitted": False,
                "short_term_count": len(messages),
            }

    def archive_short_term_memory(
        self,
        player_id: str,
        npc_id: str,
        llm_service: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with DbSessionLocal() as session:
            memory = (
                session.query(ShortTermMemory)
                .filter(
                    ShortTermMemory.player_id == player_id,
                    ShortTermMemory.npc_id == npc_id,
                )
                .one_or_none()
            )

            if memory is None:
                return {"ok": True, "archived": False, "reason": "没有短期记忆"}

            messages = list(memory.messages or [])
            if len(messages) <= self.SHORT_TERM_LIMIT:
                return {
                    "ok": True,
                    "archived": False,
                    "reason": "短期记忆未超过上限",
                    "short_term_count": len(messages),
                }

            archive_messages = messages[: self.ARCHIVE_BATCH_SIZE]
            remaining_messages = messages[self.ARCHIVE_BATCH_SIZE :]

            long_term_memory = self._create_long_term_memory_from_messages(
                session=session,
                player_id=player_id,
                npc_id=npc_id,
                messages=archive_messages,
                llm_service=llm_service,
            )

            memory.messages = remaining_messages
            memory.message_count = len(remaining_messages)
            memory.last_message_at = datetime.now(timezone.utc)

            session.commit()

            return {
                "ok": True,
                "archived": True,
                "long_term_memory_id": long_term_memory.memory_id,
                "archived_message_count": len(archive_messages),
                "short_term_count": len(remaining_messages),
            }

    def create_long_term_memory(
        self,
        player_id: str,
        npc_id: str,
        title: str,
        content: str,
        memory_type: str = "event",
        importance: float = 0.5,
        source_message_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not player_id or not npc_id or not content:
            return {"ok": False, "error": "player_id、npc_id、content 不能为空"}

        memory_type = self._safe_memory_type(memory_type)
        importance = self._safe_importance(importance)
        embedding = self.embedding_service.embed_memory(
            title=title,
            content=content,
            memory_type=memory_type,
        )

        with DbSessionLocal() as session:
            row = LongTermMemory(
                memory_id=f"mem_{uuid.uuid4().hex[:16]}",
                player_id=player_id,
                npc_id=npc_id,
                title=title or "一条长期记忆",
                content=content,
                memory_type=memory_type,
                importance=importance,
                embedding=embedding,
                embedding_model=self.embedding_service.model_name,
                source_message_ids=source_message_ids or [],
                meta_data=metadata or {},
            )
            session.add(row)
            session.commit()
            session.refresh(row)

            return {"ok": True, "memory": self._serialize_long_term_memory(row)}

    def search_long_term_memories(
        self,
        player_id: str,
        query: str,
        npc_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        if not player_id:
            return {"ok": False, "error": "缺少 player_id", "memories": []}
        if not query:
            return {"ok": False, "error": "缺少 query", "memories": []}

        limit = max(1, min(int(limit), 50))
        query_embedding = self.embedding_service.embed_query(query)

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
                stmt = stmt.where(LongTermMemory.memory_type == memory_type)

            stmt = stmt.order_by("distance").limit(limit)
            rows = session.execute(stmt).all()

            memories = []
            for memory, distance in rows:
                item = self._serialize_long_term_memory(memory)
                item["distance"] = float(distance) if distance is not None else None
                item["similarity"] = 1.0 - float(distance) if distance is not None else None
                memories.append(item)

            return {
                "ok": True,
                "query": query,
                "player_id": player_id,
                "npc_id": npc_id,
                "count": len(memories),
                "memories": memories,
            }

    def get_long_term_memories(
        self,
        player_id: str,
        npc_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 50))

        with DbSessionLocal() as session:
            query = session.query(LongTermMemory).filter(LongTermMemory.player_id == player_id)
            if npc_id:
                query = query.filter(LongTermMemory.npc_id == npc_id)
            if memory_type:
                query = query.filter(LongTermMemory.memory_type == memory_type)

            rows = (
                query.order_by(LongTermMemory.importance.desc(), LongTermMemory.created_at.desc())
                .limit(limit)
                .all()
            )

            return {
                "ok": True,
                "count": len(rows),
                "memories": [self._serialize_long_term_memory(row) for row in rows],
            }

    def _create_long_term_memory_from_messages(
        self,
        session,
        player_id: str,
        npc_id: str,
        messages: List[Dict[str, Any]],
        llm_service: Optional[Any] = None,
    ) -> LongTermMemory:
        summary = self._summarize_messages(
            messages=messages,
            player_id=player_id,
            npc_id=npc_id,
            llm_service=llm_service,
        )

        embedding = self.embedding_service.embed_memory(
            title=summary["title"],
            content=summary["content"],
            memory_type=summary["memory_type"],
        )

        row = LongTermMemory(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            player_id=player_id,
            npc_id=npc_id,
            title=summary["title"],
            content=summary["content"],
            memory_type=summary["memory_type"],
            importance=summary["importance"],
            embedding=embedding,
            embedding_model=self.embedding_service.model_name,
            source_message_ids=summary.get("source_message_ids", []),
            meta_data={
                "source": "short_term_archive",
                "archived_message_count": len(messages),
                "created_by": "MemoryService",
            },
        )
        session.add(row)
        session.flush()
        return row

    def _summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        player_id: str,
        npc_id: str,
        llm_service: Optional[Any] = None,
    ) -> Dict[str, Any]:
        source_message_ids = [msg.get("message_id") for msg in messages if msg.get("message_id")]
        dialogue_text = "\n".join(
            f"{msg.get('role', 'unknown')}：{msg.get('content', '')}"
            for msg in messages
            if msg.get("content")
        )

        if llm_service is not None:
            try:
                prompt = f"""
请将以下咖啡厅 NPC 对话压缩为一条长期记忆。
只返回 JSON，不要解释。

字段：title, content, memory_type, importance。
memory_type 只能是 preference、relationship、event、habit、warning。
importance 是 0 到 1 的浮点数。

player_id: {player_id}
npc_id: {npc_id}

对话：
{dialogue_text}
"""
                response, _ = llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt="你是游戏 NPC 记忆摘要模块，只输出合法 JSON。",
                )
                data = json.loads(response)
                return {
                    "title": str(data.get("title", "一段重要互动")),
                    "content": str(data.get("content", dialogue_text[:500])),
                    "memory_type": self._safe_memory_type(data.get("memory_type", "event")),
                    "importance": self._safe_importance(data.get("importance", 0.5)),
                    "source_message_ids": source_message_ids,
                }
            except Exception:
                pass

        return self._rule_based_summary(messages, source_message_ids)

    def _rule_based_summary(
        self,
        messages: List[Dict[str, Any]],
        source_message_ids: List[str],
    ) -> Dict[str, Any]:
        contents = [msg.get("content", "") for msg in messages if msg.get("content")]
        merged = "；".join(contents)

        preference_keywords = ["喜欢", "偏好", "想喝", "拿铁", "美式", "手冲", "甜", "酸", "苦", "奶"]
        relationship_keywords = ["谢谢", "信任", "熟悉", "抱歉", "开心"]
        habit_keywords = ["经常", "每天", "上午", "下午", "常来", "习惯"]
        warning_keywords = ["不要", "别", "讨厌", "生气", "冒犯", "压力"]

        if any(word in merged for word in preference_keywords):
            return {
                "title": "玩家表达了咖啡偏好",
                "content": merged[:500] or "玩家表达了咖啡偏好。",
                "memory_type": "preference",
                "importance": 0.7,
                "source_message_ids": source_message_ids,
            }
        if any(word in merged for word in warning_keywords):
            return {
                "title": "玩家互动中出现需要注意的信息",
                "content": merged[:500] or "玩家互动中出现需要注意的信息。",
                "memory_type": "warning",
                "importance": 0.8,
                "source_message_ids": source_message_ids,
            }
        if any(word in merged for word in relationship_keywords):
            return {
                "title": "玩家与 NPC 的关系发生变化",
                "content": merged[:500] or "玩家与 NPC 的关系发生变化。",
                "memory_type": "relationship",
                "importance": 0.6,
                "source_message_ids": source_message_ids,
            }
        if any(word in merged for word in habit_keywords):
            return {
                "title": "玩家表现出到店或互动习惯",
                "content": merged[:500] or "玩家表现出到店或互动习惯。",
                "memory_type": "habit",
                "importance": 0.65,
                "source_message_ids": source_message_ids,
            }

        return {
            "title": "玩家与 NPC 的一段互动",
            "content": merged[:500] or "玩家与 NPC 进行了一段普通互动。",
            "memory_type": "event",
            "importance": 0.5,
            "source_message_ids": source_message_ids,
        }

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(message)
        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if data.get("role") not in {"player", "npc", "system"}:
            data["role"] = "system"
        data["content"] = str(data.get("content", "")).strip()
        return data

    def _serialize_long_term_memory(self, row: LongTermMemory) -> Dict[str, Any]:
        return {
            "memory_id": row.memory_id,
            "player_id": row.player_id,
            "npc_id": row.npc_id,
            "title": row.title,
            "content": row.content,
            "memory_type": row.memory_type,
            "importance": row.importance,
            "embedding_model": getattr(row, "embedding_model", None),
            "source_message_ids": row.source_message_ids or [],
            "metadata": getattr(row, "meta_data", None) or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def _safe_memory_type(self, value: str) -> str:
        allowed = {"preference", "relationship", "event", "habit", "warning"}
        return value if value in allowed else "event"

    def _safe_importance(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.5
        return max(0.0, min(score, 1.0))

    def backfill_long_term_memory_embeddings(
            self,
            player_id: Optional[str] = None,
            npc_id: Optional[str] = None,
            limit: int = 100,
    ) -> Dict[str, Any]:
        """
        为已经存在但 embedding 为空的长期记忆补充向量。

        适用场景：
        - 原来 long_term_memories 表已有数据
        - 后来才新增 pgvector embedding 字段
        - 旧数据 embedding 为 NULL，无法被向量检索命中
        """

        limit = max(1, min(int(limit), 1000))

        updated_count = 0
        failed_items: List[Dict[str, Any]] = []

        with DbSessionLocal() as session:
            query = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.embedding.is_(None))
            )

            if player_id:
                query = query.filter(LongTermMemory.player_id == player_id)

            if npc_id:
                query = query.filter(LongTermMemory.npc_id == npc_id)

            rows = (
                query.order_by(LongTermMemory.created_at.asc())
                .limit(limit)
                .all()
            )

            for row in rows:
                try:
                    embedding = self.embedding_service.embed_memory(
                        title=row.title,
                        content=row.content,
                        memory_type=(
                            row.memory_type.value
                            if hasattr(row.memory_type, "value")
                            else str(row.memory_type)
                        ),
                    )

                    row.embedding = embedding
                    row.embedding_model = self.embedding_service.model_name
                    row.updated_at = datetime.now(timezone.utc)

                    updated_count += 1

                except Exception as exc:
                    failed_items.append(
                        {
                            "memory_id": row.memory_id,
                            "error": str(exc),
                        }
                    )

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
    memory_service = MemoryService()


    print("\nStep 1: 使用 pgvector 查询长期记忆")

    search_result = memory_service.search_long_term_memories(
        player_id="p001",
        npc_id="barista_001",
        query="玩家喜欢什么咖啡？",
        limit=5,
    )

    print(search_result)

    print("\nStep 3: 打印简化结果")

    for idx, memory in enumerate(search_result.get("memories", []), 1):
        print(f"\n[{idx}]")
        print("memory_id:", memory.get("memory_id"))
        print("title:", memory.get("title"))
        print("memory_type:", memory.get("memory_type"))
        print("importance:", memory.get("importance"))
        print("distance:", memory.get("distance"))
        print("similarity:", memory.get("similarity"))
        print("content:", memory.get("content"))