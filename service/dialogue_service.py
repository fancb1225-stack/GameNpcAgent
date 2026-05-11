from __future__ import annotations

from datetime import datetime, timezone

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
from model.memory_model import ShortMemoryMessage
from service.memory_service import MemoryService


class DialogueService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None
        self.memory_service = MemoryService(self.db)

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
        if data.npc_id and data.role in (MessageRole.PLAYER, MessageRole.NPC):
            self.memory_service.append_short_memory(
                data.player_id,
                data.npc_id,
                ShortMemoryMessage(
                    role=str(data.role.value if hasattr(data.role, "value") else data.role),
                    content=data.content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    emotion=str(data.emotion.value) if data.emotion else None,
                    metadata={"message_id": str(message.id), "intent": data.intent, "action": str(data.action) if data.action else None},
                ),
            )
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
        return message

    def list_messages(self, session_id: str, limit: int = 50) -> list[DialogueMessageRead]:
        return [DialogueMessageRead.model_validate(row) for row in self.db.list_dialogue_messages(session_id, limit=limit)]
