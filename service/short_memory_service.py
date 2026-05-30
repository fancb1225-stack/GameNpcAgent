from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from db.db_config import DbSessionLocal, ShortTermMemory
from db.db_service import DbService
from model.memory_model import ShortMemoryMessage, ShortTermMemoryRead


class ShortMemoryService:
    """短期记忆服务，只维护玩家与 NPC 的滑动窗口。"""

    SHORT_TERM_LIMIT = 10
    ARCHIVE_BATCH_SIZE = 5

    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def append_short_term_message(
        self,
        player_id: str,
        npc_id: str,
        message: dict[str, Any],
        async_archive: bool = False,
    ) -> dict[str, Any]:
        """追加短期记忆，并在超出阈值时返回待归档消息。"""

        if not player_id:
            return {"ok": False, "error": "缺少 player_id", "needs_archive": False}
        if not npc_id:
            return {"ok": False, "error": "缺少 npc_id", "needs_archive": False}
        if not message or not message.get("content"):
            return {"ok": False, "error": "缺少 message.content", "needs_archive": False}

        normalized_message = self._normalize_message(message)

        with DbSessionLocal() as session:
            # 短期服务只写短期窗口，不创建长期记忆。
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

        needs_archive = len(messages) > self.SHORT_TERM_LIMIT
        archive_messages = messages[: self.ARCHIVE_BATCH_SIZE] if needs_archive else []

        return {
            "ok": True,
            "needs_archive": needs_archive,
            "archive_messages": archive_messages,
            "archive_message_count": len(archive_messages),
            "remaining_message_count": max(0, len(messages) - len(archive_messages)),
            "short_term_count": len(messages),
            "async_archive": async_archive,
        }

    def trim_short_term_memory(
        self,
        player_id: str,
        npc_id: str,
        remove_count: int,
    ) -> dict[str, Any]:
        """长期记忆写入成功后，移除已经归档的短期消息。"""

        remove_count = max(0, int(remove_count))
        if remove_count <= 0:
            return {"ok": True, "trimmed": False, "short_term_count": None}

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
                return {"ok": True, "trimmed": False, "short_term_count": 0}

            messages = list(memory.messages or [])
            remaining_messages = messages[remove_count:]
            memory.messages = remaining_messages
            memory.message_count = len(remaining_messages)
            memory.last_message_at = datetime.now(timezone.utc)
            session.commit()

        return {
            "ok": True,
            "trimmed": True,
            "removed_count": min(remove_count, len(messages)),
            "short_term_count": len(remaining_messages),
        }

    def get_short_term_memory(self, player_id: str, npc_id: str) -> dict[str, Any]:
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

    def get_short_memory(self, player_id: str, npc_id: str) -> ShortTermMemoryRead | None:
        """兼容工具层读取短期记忆的 Pydantic 返回。"""

        memory = self.db.get_short_memory(player_id, npc_id)
        return ShortTermMemoryRead.model_validate(memory) if memory else None

    def append_short_memory(
        self,
        player_id: str,
        npc_id: str,
        message: ShortMemoryMessage | dict[str, Any],
    ) -> ShortTermMemoryRead:
        """兼容旧调用：只追加短期窗口，不触发长期归档。"""

        payload = message.model_dump() if isinstance(message, ShortMemoryMessage) else dict(message)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        memory = self.db.append_short_memory(
            player_id,
            npc_id,
            payload,
            max_messages=self.SHORT_TERM_LIMIT,
        )
        return ShortTermMemoryRead.model_validate(memory)

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """标准化短期记忆消息结构。"""

        data = dict(message)
        data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        data.setdefault("emotion", None)

        if data.get("role") not in {"player", "npc", "system"}:
            data["role"] = "system"

        return {
            "role": data["role"],
            "content": str(data.get("content", "")).strip(),
            "emotion": data.get("emotion"),
            "timestamp": data["timestamp"],
            "metadata": data.get("metadata", {}),
        }
