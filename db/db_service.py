"""
CRUD service for CoffeeNpcAgent database.

Example:
    from db_service import DbService

    with DbService() as db:
        player = db.create_player("p001", nickname="玩家A")
        npc = db.get_npc("barista_001")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence, TypeVar

from sqlalchemy import Select, and_, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db_config import (
    Base,
    CafeEvent,
    CafeWorldState,
    CoffeeKnowledge,
    DialogueMessage,
    DialogueSession,
    EncounterRule,
    LongTermMemory,
    MODEL_REGISTRY,
    MessageRole,
    Npc,
    Player,
    Relationship,
    DbSessionLocal,
    ShortTermMemory,
    SourceType,
)

ModelT = TypeVar("ModelT", bound=Base)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def model_to_dict(obj: Base) -> dict[str, Any]:
    return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}


class DbService:
    def __init__(self, session: Session | None = None):
        self.db = session or DbSessionLocal()
        self._owns_session = session is None

    def __enter__(self) -> "DbService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            self.db.rollback()
        if self._owns_session:
            self.db.close()

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def close(self) -> None:
        self.db.close()

    # ---------- generic CRUD ----------

    def create(self, model: type[ModelT], data: dict[str, Any], commit: bool = True) -> ModelT:
        obj = model(**data)
        self.db.add(obj)
        try:
            if commit:
                self.db.commit()
                self.db.refresh(obj)
            return obj
        except IntegrityError:
            self.db.rollback()
            raise

    def get_by_id(self, model: type[ModelT], object_id: str | uuid.UUID) -> ModelT | None:
        if isinstance(object_id, str):
            object_id = uuid.UUID(object_id)
        return self.db.get(model, object_id)

    def get_one_by_field(self, model: type[ModelT], field_name: str, value: Any) -> ModelT | None:
        return self.db.scalar(select(model).where(getattr(model, field_name) == value))

    def list(
        self,
        model: type[ModelT],
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str | None = None,
        desc: bool = False,
    ) -> list[ModelT]:
        stmt: Select = select(model)
        if filters:
            stmt = stmt.where(*[getattr(model, key) == value for key, value in filters.items()])
        if order_by:
            column = getattr(model, order_by)
            stmt = stmt.order_by(column.desc() if desc else column.asc())
        stmt = stmt.limit(limit).offset(offset)
        return list(self.db.scalars(stmt).all())

    def update(self, obj: ModelT, data: dict[str, Any], commit: bool = True) -> ModelT:
        for key, value in data.items():
            if not hasattr(obj, key):
                raise ValueError(f"Unknown field: {key}")
            setattr(obj, key, value)
        if commit:
            self.db.commit()
            self.db.refresh(obj)
        return obj

    def update_by_field(
        self,
        model: type[ModelT],
        field_name: str,
        value: Any,
        data: dict[str, Any],
        commit: bool = True,
    ) -> ModelT | None:
        obj = self.get_one_by_field(model, field_name, value)
        if obj is None:
            return None
        return self.update(obj, data, commit=commit)

    def delete_obj(self, obj: ModelT, commit: bool = True) -> None:
        self.db.delete(obj)
        if commit:
            self.db.commit()

    def delete_by_field(self, model: type[ModelT], field_name: str, value: Any, commit: bool = True) -> int:
        result = self.db.execute(delete(model).where(getattr(model, field_name) == value))
        if commit:
            self.db.commit()
        return int(result.rowcount or 0)

    def upsert_by_field(
        self,
        model: type[ModelT],
        field_name: str,
        value: Any,
        data: dict[str, Any],
        commit: bool = True,
    ) -> ModelT:
        obj = self.get_one_by_field(model, field_name, value)
        if obj is None:
            obj = model(**{field_name: value}, **data)
            self.db.add(obj)
        else:
            for key, item in data.items():
                setattr(obj, key, item)
        if commit:
            self.db.commit()
            self.db.refresh(obj)
        return obj

    def create_by_table_name(self, table_name: str, data: dict[str, Any], commit: bool = True) -> Base:
        return self.create(self._model(table_name), data, commit=commit)

    def list_by_table_name(self, table_name: str, filters: dict[str, Any] | None = None, limit: int = 50) -> list[Base]:
        return self.list(self._model(table_name), filters=filters, limit=limit)

    def _model(self, table_name: str) -> type[Base]:
        try:
            return MODEL_REGISTRY[table_name]
        except KeyError as exc:
            raise ValueError(f"Unknown table name: {table_name}") from exc

    # ---------- Player ----------

    def create_player(self, player_id: str, nickname: str | None = None, **kwargs) -> Player:
        return self.create(Player, {"player_id": player_id, "nickname": nickname, **kwargs})

    def get_player(self, player_id: str) -> Player | None:
        return self.get_one_by_field(Player, "player_id", player_id)

    def update_player(self, player_id: str, **kwargs) -> Player | None:
        return self.update_by_field(Player, "player_id", player_id, kwargs)

    def delete_player(self, player_id: str) -> int:
        return self.delete_by_field(Player, "player_id", player_id)

    # ---------- NPC ----------

    def create_npc(self, npc_id: str, name: str, npc_type, role: str, **kwargs) -> Npc:
        return self.create(Npc, {"npc_id": npc_id, "name": name, "npc_type": npc_type, "role": role, **kwargs})

    def get_npc(self, npc_id: str) -> Npc | None:
        return self.get_one_by_field(Npc, "npc_id", npc_id)

    def list_active_npcs(self) -> list[Npc]:
        return self.list(Npc, filters={"is_active": True}, limit=100, order_by="npc_id")

    def update_npc(self, npc_id: str, **kwargs) -> Npc | None:
        return self.update_by_field(Npc, "npc_id", npc_id, kwargs)

    def delete_npc(self, npc_id: str) -> int:
        return self.delete_by_field(Npc, "npc_id", npc_id)

    # ---------- Relationship ----------

    def get_relationship(
        self,
        source_type: SourceType | str,
        source_id: str,
        target_type: SourceType | str,
        target_id: str,
    ) -> Relationship | None:
        return self.db.scalar(
            select(Relationship).where(
                Relationship.source_type == source_type,
                Relationship.source_id == source_id,
                Relationship.target_type == target_type,
                Relationship.target_id == target_id,
            )
        )

    def get_or_create_relationship(
        self,
        source_type: SourceType | str,
        source_id: str,
        target_type: SourceType | str,
        target_id: str,
    ) -> Relationship:
        rel = self.get_relationship(source_type, source_id, target_type, target_id)
        if rel is not None:
            return rel
        rel = Relationship(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            last_interacted_at=utc_now(),
        )
        self.db.add(rel)
        self.db.commit()
        self.db.refresh(rel)
        return rel

    def apply_relationship_delta(
        self,
        source_type: SourceType | str,
        source_id: str,
        target_type: SourceType | str,
        target_id: str,
        delta: dict[str, int],
    ) -> Relationship:
        rel = self.get_or_create_relationship(source_type, source_id, target_type, target_id)
        for field in ("trust", "familiarity", "fondness", "dislike", "stress"):
            setattr(rel, field, getattr(rel, field) + int(delta.get(field, 0)))
        rel.last_interacted_at = utc_now()
        self.db.commit()
        self.db.refresh(rel)
        return rel

    # ---------- Dialogue session/message ----------

    def create_dialogue_session(self, session_id: str, player_id: str, **kwargs) -> DialogueSession:
        return self.create(DialogueSession, {"session_id": session_id, "player_id": player_id, **kwargs})

    def get_dialogue_session(self, session_id: str) -> DialogueSession | None:
        return self.get_one_by_field(DialogueSession, "session_id", session_id)

    def add_dialogue_message(
        self,
        session_id: str,
        player_id: str,
        role: MessageRole | str,
        content: str,
        npc_id: str | None = None,
        **kwargs,
    ) -> DialogueMessage:
        return self.create(
            DialogueMessage,
            {
                "session_id": session_id,
                "player_id": player_id,
                "npc_id": npc_id,
                "role": role,
                "content": content,
                **kwargs,
            },
        )

    def list_dialogue_messages(self, session_id: str, limit: int = 50) -> list[DialogueMessage]:
        return self.list(
            DialogueMessage,
            filters={"session_id": session_id},
            limit=limit,
            order_by="created_at",
        )

    # ---------- Memory ----------

    def get_short_memory(self, player_id: str, npc_id: str) -> ShortTermMemory | None:
        return self.db.scalar(
            select(ShortTermMemory).where(
                ShortTermMemory.player_id == player_id,
                ShortTermMemory.npc_id == npc_id,
            )
        )

    def append_short_memory(self, player_id: str, npc_id: str, message: dict[str, Any], max_messages: int = 20) -> ShortTermMemory:
        memory = self.get_short_memory(player_id, npc_id)
        if memory is None:
            memory = ShortTermMemory(player_id=player_id, npc_id=npc_id, messages=[])
            self.db.add(memory)

        messages = list(memory.messages or [])
        messages.append(message)
        memory.messages = messages[-max_messages:]
        memory.message_count = len(memory.messages)
        memory.last_message_at = utc_now()
        self.db.commit()
        self.db.refresh(memory)
        return memory

    def create_long_memory(self, memory_id: str, player_id: str, npc_id: str, title: str, content: str, memory_type, **kwargs) -> LongTermMemory:
        return self.create(
            LongTermMemory,
            {
                "memory_id": memory_id,
                "player_id": player_id,
                "npc_id": npc_id,
                "title": title,
                "content": content,
                "memory_type": memory_type,
                **kwargs,
            },
        )

    def list_long_memories(self, player_id: str, npc_id: str | None = None, limit: int = 20) -> list[LongTermMemory]:
        filters = {"player_id": player_id}
        if npc_id:
            filters["npc_id"] = npc_id
        return self.list(LongTermMemory, filters=filters, limit=limit, order_by="importance", desc=True)

    # ---------- World, event, knowledge, encounter ----------

    def get_world_state(self, session_id: str = "default_morning_session") -> CafeWorldState | None:
        return self.get_one_by_field(CafeWorldState, "session_id", session_id)

    def update_world_state(self, session_id: str, **kwargs) -> CafeWorldState | None:
        return self.update_by_field(CafeWorldState, "session_id", session_id, kwargs)

    def list_active_events(self, time_period=None) -> list[CafeEvent]:
        stmt = select(CafeEvent).where(CafeEvent.is_active.is_(True))
        if time_period is not None:
            stmt = stmt.where(CafeEvent.time_period == time_period)
        return list(self.db.scalars(stmt).all())

    def list_coffee_knowledge(self, category=None, active_only: bool = True, limit: int = 50) -> list[CoffeeKnowledge]:
        stmt = select(CoffeeKnowledge)
        conditions = []
        if category is not None:
            conditions.append(CoffeeKnowledge.category == category)
        if active_only:
            conditions.append(CoffeeKnowledge.is_active.is_(True))
        if conditions:
            stmt = stmt.where(and_(*conditions))
        return list(self.db.scalars(stmt.limit(limit)).all())

    def get_encounter_rule(self, time_period) -> EncounterRule | None:
        return self.db.scalar(
            select(EncounterRule).where(
                EncounterRule.time_period == time_period,
                EncounterRule.is_active.is_(True),
            )
        )


def create_service() -> DbService:
    return DbService()
