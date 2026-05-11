from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from uuid import uuid4

from db.db_service import DbService
from model.memory_model import LongTermMemoryCreate, LongTermMemoryRead, ShortMemoryMessage, ShortTermMemoryRead
from db.db_config import (
    MemoryType,
    DbSessionLocal,
    ShortTermMemory,
    LongTermMemory,
)

class MemoryService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

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
    ) -> Dict[str, Any]:
        """
        追加一条短期记忆。
        如果短期记忆超过 20 条，则自动将最早 10 条摘要为长期记忆。

        Args:
            player_id: 玩家 ID
            npc_id: NPC ID
            message: 单条消息，例如：
                {
                    "role": "player",
                    "content": "今天推荐什么咖啡？",
                    "timestamp": "...",
                    "emotion": "calm",
                    "action": "recommend_coffee"
                }
            llm_service: 可选 LLM 服务。若传入，则用 LLM 生成摘要；否则使用规则摘要。

        Returns:
            {
                "ok": bool,
                "archived": bool,
                "short_term_count": int,
                "long_term_memory_id": str | None
            }
        """

        if not player_id:
            return {
                "ok": False,
                "error": "缺少 player_id",
                "archived": False,
            }

        if not npc_id:
            return {
                "ok": False,
                "error": "缺少 npc_id",
                "archived": False,
            }

        if not message or not message.get("content"):
            return {
                "ok": False,
                "error": "缺少 message.content",
                "archived": False,
            }

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

            archived = False
            long_term_memory_id = None

            if len(messages) > self.SHORT_TERM_LIMIT:
                archive_messages = messages[: self.ARCHIVE_BATCH_SIZE]
                remaining_messages = messages[self.ARCHIVE_BATCH_SIZE:]

                long_term_memory = self._create_long_term_memory_from_messages(
                    session=session,
                    player_id=player_id,
                    npc_id=npc_id,
                    messages=archive_messages,
                    llm_service=llm_service,
                )

                long_term_memory_id = long_term_memory.memory_id
                messages = remaining_messages
                archived = True

            memory.messages = messages
            memory.message_count = len(messages)
            memory.last_message_at = datetime.now(timezone.utc)

            session.commit()

            return {
                "ok": True,
                "archived": archived,
                "short_term_count": len(messages),
                "long_term_memory_id": long_term_memory_id,
            }

    def get_short_term_memory(
            self,
            player_id: str,
            npc_id: str,
    ) -> Dict[str, Any]:
        """查询玩家与 NPC 的短期记忆。"""

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
                return {
                    "ok": True,
                    "player_id": player_id,
                    "npc_id": npc_id,
                    "messages": [],
                    "message_count": 0,
                }

            return {
                "ok": True,
                "player_id": player_id,
                "npc_id": npc_id,
                "messages": memory.messages or [],
                "message_count": memory.message_count or 0,
                "last_message_at": (
                    memory.last_message_at.isoformat()
                    if memory.last_message_at
                    else None
                ),
            }

    def get_long_term_memories(
            self,
            player_id: str,
            npc_id: Optional[str] = None,
            memory_type: Optional[str] = None,
            limit: int = 10,
    ) -> Dict[str, Any]:
        """查询长期记忆。"""

        limit = max(1, min(int(limit), 50))

        with DbSessionLocal() as session:
            query = session.query(LongTermMemory).filter(
                LongTermMemory.player_id == player_id
            )

            if npc_id:
                query = query.filter(LongTermMemory.npc_id == npc_id)

            if memory_type:
                query = query.filter(LongTermMemory.memory_type == memory_type)

            rows = (
                query.order_by(
                    LongTermMemory.importance.desc(),
                    LongTermMemory.created_at.desc(),
                )
                .limit(limit)
                .all()
            )

            memories = []
            for row in rows:
                memories.append(
                    {
                        "memory_id": row.memory_id,
                        "player_id": row.player_id,
                        "npc_id": row.npc_id,
                        "title": row.title,
                        "content": row.content,
                        "memory_type": row.memory_type,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "metadata": row.metadata or {},
                        "created_at": (
                            row.created_at.isoformat()
                            if row.created_at
                            else None
                        ),
                    }
                )

            return {
                "ok": True,
                "count": len(memories),
                "memories": memories,
            }

    def _create_long_term_memory_from_messages(
            self,
            session,
            player_id: str,
            npc_id: str,
            messages: List[Dict[str, Any]],
            llm_service: Optional[Any] = None,
    ) -> LongTermMemory:
        """
        将一批短期消息压缩为长期记忆。
        """

        summary = self._summarize_messages(
            messages=messages,
            player_id=player_id,
            npc_id=npc_id,
            llm_service=llm_service,
        )

        memory = LongTermMemory(
            memory_id=f"mem_{uuid.uuid4().hex[:16]}",
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
                "created_by": "MemoryService",
            },
        )

        session.add(memory)
        session.flush()

        return memory

    def _summarize_messages(
            self,
            messages: List[Dict[str, Any]],
            player_id: str,
            npc_id: str,
            llm_service: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        生成长期记忆摘要。
        优先使用 LLM；没有 LLM 时使用规则摘要。
        """

        source_message_ids = [
            msg.get("message_id")
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
                prompt = f"""
    你是游戏 NPC 记忆摘要模块，负责将玩家与咖啡厅 NPC 的对话压缩为一条长期记忆，只输出合法 JSON。

    要求：
    1. 标题不超过 30 字。
    2. content 是一段摘要，根据prompt输入的对话总结；需要保留玩家偏好、关系变化、重要事件或需要注意的信息。
    3. memory_type 只能是 preference、relationship、event、habit、warning 之一。
    4. importance 是 0 到 1 的浮点数。
    5. 只返回 JSON，不要输出额外解释。

    player_id: {player_id}
    npc_id: {npc_id}

    返回格式：
    {{
      "title": "...",
      "content": "...",
      "memory_type": "preference",
      "importance": 0.7
    }}
    """
                print("==========================\n")
                print(prompt)
                print("==========================\n")
                response, _ = llm_service.chat(
                    prompt=dialogue_text,
                    history=[],
                    system_prompt="你是游戏 NPC 记忆摘要模块，只输出合法 JSON。",
                )

                import json

                data = json.loads(response)

                return {
                    "title": str(data.get("title", "一段重要互动")),
                    "content": str(data.get("content", dialogue_text[:300])),
                    "memory_type": self._safe_memory_type(
                        data.get("memory_type", "event")
                    ),
                    "importance": self._safe_importance(
                        data.get("importance", 0.5)
                    ),
                    "source_message_ids": source_message_ids,
                }

            except Exception:
                pass

        return self._rule_based_summary(
            messages=messages,
            source_message_ids=source_message_ids,
        )

    def _rule_based_summary(
            self,
            messages: List[Dict[str, Any]],
            source_message_ids: List[str],
    ) -> Dict[str, Any]:
        """
        无 LLM 时的规则摘要。
        """

        contents = [
            msg.get("content", "")
            for msg in messages
            if msg.get("content")
        ]

        merged = "；".join(contents)

        memory_type = "event"
        importance = 0.5

        preference_keywords = ["喜欢", "偏好", "想喝", "拿铁", "美式", "手冲", "甜", "酸", "苦", "奶"]
        relationship_keywords = ["谢谢", "讨厌", "信任", "熟悉", "抱歉", "开心", "生气"]
        habit_keywords = ["经常", "每天", "上午", "下午", "常来", "习惯"]
        warning_keywords = ["不要", "别", "讨厌", "生气", "冒犯", "压力"]

        if any(word in merged for word in preference_keywords):
            memory_type = "preference"
            importance = 0.7
            title = "玩家表达了咖啡偏好"
        elif any(word in merged for word in warning_keywords):
            memory_type = "warning"
            importance = 0.8
            title = "玩家互动中出现需要注意的信息"
        elif any(word in merged for word in relationship_keywords):
            memory_type = "relationship"
            importance = 0.6
            title = "玩家与 NPC 的关系发生变化"
        elif any(word in merged for word in habit_keywords):
            memory_type = "habit"
            importance = 0.65
            title = "玩家表现出到店或互动习惯"
        else:
            title = "玩家与 NPC 的一段互动"

        content = merged[:500] if merged else "玩家与 NPC 进行了一段普通互动。"

        return {
            "title": title,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "source_message_ids": source_message_ids,
        }

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """标准化短期记忆消息结构。"""

        data = dict(message)

        print(data)
        # data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        allowed_roles = {"player", "npc", "system"}
        if data.get("role") not in allowed_roles:
            data["role"] = "system"

        # data["content"] = str(data.get("content", "")).strip()
        data = dict({
            "role": data["role"],
            "content": data["content"],
            "emotion": data["emotion"],
            "timestamp": message["timestamp"]
        })
        return data

    def _safe_memory_type(self, value: str) -> str:
        allowed = {"preference", "relationship", "event", "habit", "warning"}
        return value if value in allowed else "event"

    def _safe_importance(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.5

        return max(0.0, min(score, 1.0))

    def get_short_memory(self, player_id: str, npc_id: str) -> ShortTermMemoryRead | None:
        memory = self.db.get_short_memory(player_id, npc_id)
        if memory is None:
            print("原始数据库记忆：未找到\n")
            return None

        short_memory = ShortTermMemoryRead.model_validate(memory)
        print(f"原始数据库记忆：{short_memory}\n")
        return short_memory

    def append_short_memory(self, player_id: str, npc_id: str, message: ShortMemoryMessage | dict) -> ShortTermMemoryRead:
        payload = message.model_dump() if isinstance(message, ShortMemoryMessage) else dict(message)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        memory = self.db.append_short_memory(player_id, npc_id, payload, max_messages=20)
        return ShortTermMemoryRead.model_validate(memory)

    def create_long_memory(self, data: LongTermMemoryCreate) -> LongTermMemoryRead:
        memory = self.db.create_long_memory(**data.model_dump())
        return LongTermMemoryRead.model_validate(memory)

    def list_long_memories(self, player_id: str, npc_id: str | None = None, limit: int = 20) -> list[LongTermMemoryRead]:
        rows = self.db.list_long_memories(player_id, npc_id=npc_id, limit=limit)
        return [LongTermMemoryRead.model_validate(row) for row in rows]

    def summarize_oldest_short_messages(self, player_id: str, npc_id: str, take: int = 10) -> LongTermMemoryRead | None:
        """Minimal non-LLM summary trigger. Real summary text can later be replaced by LLM output."""
        short_memory = self.db.get_short_memory(player_id, npc_id)
        if short_memory is None or len(short_memory.messages or []) < take:
            return None
        messages = list(short_memory.messages or [])[:take]
        joined = "；".join([str(item.get("content", "")) for item in messages if isinstance(item, dict)])[:500]
        data = LongTermMemoryCreate(
            memory_id=f"mem_{uuid4().hex[:16]}",
            player_id=player_id,
            npc_id=npc_id,
            title="短期对话摘要",
            content=joined or "玩家与 NPC 发生了一段对话。",
            memory_type=MemoryType.EVENT,
            importance=0.5,
            source_message_ids=[],
            metadata_json={"source": "short_term_memory", "message_count": len(messages)},
        )
        return self.create_long_memory(data)


if __name__ == "__main__":
    memory_service = MemoryService()

    short_memory = memory_service.get_short_memory(npc_id="barista_001", player_id="test_001")

    if short_memory is not None:
        for message in short_memory.messages:
            if len(message['content']) < 100:
                print({
                    "role": message["role"],
                    "content": message["content"],
                    "emotion": message["emotion"],
                    "timestamp": message["timestamp"]
                })
