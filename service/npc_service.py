from __future__ import annotations

from db.db_config import Emotion, Npc
from db.db_service import DbService
from model.npc_model import NpcCreate, NpcRead, NpcUpdate


class NpcService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_npc(self, data: NpcCreate) -> NpcRead:
        npc = self.db.create_npc(**data.model_dump())
        return NpcRead.model_validate(npc)

    def get_npc(self, npc_id: str) -> NpcRead | None:
        npc = self.db.get_npc(npc_id)
        return NpcRead.model_validate(npc) if npc else None

    def list_active_npcs(self) -> list[NpcRead]:
        return [NpcRead.model_validate(npc) for npc in self.db.list_active_npcs()]

    def list_npcs_by_ids(self, npc_ids: list[str]) -> list[NpcRead]:
        result: list[NpcRead] = []
        for npc_id in npc_ids:
            npc = self.db.get_npc(npc_id)
            if npc:
                result.append(NpcRead.model_validate(npc))
        return result

    def update_npc(self, npc_id: str, data: NpcUpdate) -> NpcRead | None:
        npc = self.db.update_npc(npc_id, **data.to_update_dict())
        return NpcRead.model_validate(npc) if npc else None

    def update_emotion(self, npc_id: str, emotion: Emotion | str) -> NpcRead | None:
        npc = self.db.update_npc(npc_id, current_emotion=emotion)
        return NpcRead.model_validate(npc) if npc else None

    def move_location(self, npc_id: str, location: str) -> NpcRead | None:
        npc = self.db.update_npc(npc_id, current_location=location)
        return NpcRead.model_validate(npc) if npc else None

    def delete_npc(self, npc_id: str) -> int:
        return self.db.delete_npc(npc_id)
