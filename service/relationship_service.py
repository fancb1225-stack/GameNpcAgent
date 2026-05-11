from __future__ import annotations

from db.db_config import Relationship, SourceType
from db.db_service import DbService
from model.relationship_model import RelationshipDelta, RelationshipRead


class RelationshipService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def get_relationship(
        self,
        source_type: SourceType | str,
        source_id: str,
        target_type: SourceType | str,
        target_id: str,
    ) -> RelationshipRead | None:
        rel = self.db.get_relationship(source_type, source_id, target_type, target_id)
        return RelationshipRead.model_validate(rel) if rel else None

    def get_or_create_player_npc_relationship(self, player_id: str, npc_id: str) -> RelationshipRead:
        rel = self.db.get_or_create_relationship(SourceType.PLAYER, player_id, SourceType.NPC, npc_id)
        return RelationshipRead.model_validate(rel)

    def apply_player_npc_delta(self, player_id: str, npc_id: str, delta: RelationshipDelta | dict[str, int]) -> RelationshipRead:
        delta_dict = delta.as_dict() if isinstance(delta, RelationshipDelta) else delta
        rel = self.db.apply_relationship_delta(SourceType.PLAYER, player_id, SourceType.NPC, npc_id, delta_dict)
        return RelationshipRead.model_validate(rel)

    def list_player_relationships(self, player_id: str) -> list[RelationshipRead]:
        rows = self.db.list(Relationship, filters={"source_type": SourceType.PLAYER, "source_id": player_id}, limit=100)
        return [RelationshipRead.model_validate(row) for row in rows]
