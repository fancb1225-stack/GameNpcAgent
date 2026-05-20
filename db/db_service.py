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
from typing import Any, TypeVar

from sqlalchemy import Select, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .hybrid_retrieval import rank_hybrid_results
from .db_config import (
    Base,
    Event,
    WorldState,
    DialogueMessage,
    DialogueSession,
    EncounterRule,
    GameDocument,
    GameDocumentType,
    GameSettingChunk,
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

    # ---------- Game documents ----------

    def create_game_document(
        self,
        document_id: str,
        document_type: str,
        filename: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> GameDocument:
        document = GameDocument(
            document_id=document_id,
            document_type=GameDocumentType(document_type),
            filename=filename,
            metadata_json=metadata_json or {},
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def replace_game_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> int:
        self.db.execute(delete(GameSettingChunk).where(GameSettingChunk.document_id == document_id))
        for chunk in chunks:
            self.db.add(GameSettingChunk(**chunk))
        self.db.commit()
        return len(chunks)

    def search_game_chunks(
        self,
        query_embedding: list[float],
        query: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        candidate_limit = max(20, min(max(1, int(limit)) * 10, 200))
        stmt = (
            select(
                GameSettingChunk,
                GameSettingChunk.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .where(GameSettingChunk.embedding.is_not(None))
            .order_by("distance")
            .limit(candidate_limit)
        )
        rows = self.db.execute(stmt).all()
        candidates = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": chunk.metadata_json or {},
                "distance": float(distance) if distance is not None else None,
            }
            for chunk, distance in rows
        ]
        ranked = rank_hybrid_results(
            candidates,
            query=query,
            content_getter=lambda item: str(item.get("content") or ""),
            distance_getter=lambda item: item.get("distance"),
            limit=limit,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        results = []
        for row in ranked:
            item = dict(row["item"])
            item["hybrid_score"] = row["hybrid_score"]
            item["bm25_score"] = row["bm25_score"]
            item["vector_score"] = row["vector_score"]
            results.append(item)
        return results

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

    def upsert_npc_from_setting(self, data: dict[str, Any]) -> Npc:
        npc_id = data["npc_id"]
        values = {key: value for key, value in data.items() if key != "npc_id"}
        return self.upsert_by_field(Npc, "npc_id", npc_id, values)

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

    def get_world_state(self, session_id: str = "default_morning_session") -> WorldState | None:
        return self.get_one_by_field(WorldState, "session_id", session_id)

    def update_world_state(self, session_id: str, **kwargs) -> WorldState | None:
        return self.update_by_field(WorldState, "session_id", session_id, kwargs)

    def list_active_events(self, time_period=None) -> list[Event]:
        stmt = select(Event).where(Event.is_active.is_(True))
        if time_period is not None:
            stmt = stmt.where(Event.time_period == time_period)
        return list(self.db.scalars(stmt).all())

    def get_encounter_rule(self, time_period) -> EncounterRule | None:
        return self.db.scalar(
            select(EncounterRule).where(
                EncounterRule.time_period == time_period,
                EncounterRule.is_active.is_(True),
            )
        )


def create_service() -> DbService:
    return DbService()
