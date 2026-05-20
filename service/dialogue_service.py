from __future__ import annotations

import time
from typing import Any

from db.db_config import MessageRole, NpcAction, SessionStatus, SourceType
from db.db_service import DbService
from model.dialogue_model import (
    DialogueMessageCreate,
    DialogueMessageRead,
    DialogueSessionCreate,
    DialogueSessionRead,
    NpcReplyCreate,
    PlayerMessageCreate,
)


class DialogueService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_session(self, data: DialogueSessionCreate) -> DialogueSessionRead:
        session = self.db.create_dialogue_session(**data.model_dump())
        return DialogueSessionRead.model_validate(session)

    def get_session(self, session_id: str) -> DialogueSessionRead | None:
        session = self.db.get_dialogue_session(session_id)
        return DialogueSessionRead.model_validate(session) if session else None

    def select_npc(self, session_id: str, npc_id: str) -> DialogueSessionRead | None:
        session = self.db.update_by_field(
            model=self.db._model("dialogue_sessions"),
            field_name="session_id",
            value=session_id,
            data={"selected_npc_id": npc_id},
        )
        return DialogueSessionRead.model_validate(session) if session else None

    def end_session(self, session_id: str) -> DialogueSessionRead | None:
        session = self.db.update_by_field(
            model=self.db._model("dialogue_sessions"),
            field_name="session_id",
            value=session_id,
            data={"status": SessionStatus.ENDED},
        )
        return DialogueSessionRead.model_validate(session) if session else None

    def add_message(self, data: DialogueMessageCreate) -> DialogueMessageRead:
        message = self.db.add_dialogue_message(**data.model_dump())
        return DialogueMessageRead.model_validate(message)

    def add_player_message(self, data: PlayerMessageCreate) -> DialogueMessageRead:
        return self.add_message(
            DialogueMessageCreate(
                session_id=data.session_id,
                player_id=data.player_id,
                npc_id=data.npc_id,
                role=MessageRole.PLAYER,
                content=data.content,
            )
        )

    def add_npc_reply(self, data: NpcReplyCreate) -> DialogueMessageRead:
        message = self.add_message(
            DialogueMessageCreate(
                session_id=data.session_id,
                player_id=data.player_id,
                npc_id=data.npc_id,
                role=MessageRole.NPC,
                content=data.content,
                intent=data.intent,
                action=data.action,
                emotion=data.emotion,
                relationship_delta=data.relationship_delta.as_dict(),
                state_delta=data.state_delta,
                metadata_json=data.metadata_json,
            )
        )
        if data.relationship_delta:
            self.db.apply_relationship_delta(
                SourceType.PLAYER,
                data.player_id,
                SourceType.NPC,
                data.npc_id,
                data.relationship_delta.as_dict(),
            )
        if data.action == NpcAction.END_DIALOGUE:
            self.end_session(data.session_id)
        if data.reply_started_at is not None:
            latency_ms = max(0, int(round((time.perf_counter() - data.reply_started_at) * 1000)))
            message_orm = self.db.get_by_id(self.db._model("dialogue_messages"), message.id)
            if message_orm is not None:
                message_orm = self.db.update(
                    message_orm,
                    {"reply_latency_ms": latency_ms},
                )
                return DialogueMessageRead.model_validate(message_orm)
        return message

    def list_messages(
        self,
        session_id: str | None = None,
        player_id: str | None = None,
        npc_id: str | None = None,
        limit: int = 50,
    ) -> list[DialogueMessageRead]:
        from db.db_config import DialogueMessage
        filters: dict[str, Any] = {}
        if session_id:
            filters["session_id"] = session_id
        if player_id:
            filters["player_id"] = player_id
        if npc_id:
            filters["npc_id"] = npc_id
        rows = self.db.list(
            DialogueMessage,
            filters=filters or None,
            limit=limit,
            order_by="created_at",
            desc=False,
        )
        return [DialogueMessageRead.model_validate(row) for row in rows]
