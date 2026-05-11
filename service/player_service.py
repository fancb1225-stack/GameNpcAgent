from __future__ import annotations

from db.db_service import DbService
from model.player_model import PlayerCreate, PlayerRead, PlayerUpdate


class PlayerService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_player(self, data: PlayerCreate) -> PlayerRead:
        player = self.db.create_player(**data.model_dump())
        return PlayerRead.model_validate(player)

    def get_player(self, player_id: str) -> PlayerRead | None:
        player = self.db.get_player(player_id)
        return PlayerRead.model_validate(player) if player else None

    def get_or_create_player(self, player_id: str, nickname: str | None = None) -> PlayerRead:
        player = self.db.get_player(player_id)
        if player is None:
            player = self.db.create_player(player_id=player_id, nickname=nickname)
        return PlayerRead.model_validate(player)

    def update_player(self, player_id: str, data: PlayerUpdate) -> PlayerRead | None:
        player = self.db.update_player(player_id, **data.to_update_dict())
        return PlayerRead.model_validate(player) if player else None

    def delete_player(self, player_id: str) -> int:
        return self.db.delete_player(player_id)

    def add_known_event(self, player_id: str, event_id: str) -> PlayerRead | None:
        player = self.db.get_player(player_id)
        if player is None:
            return None
        known_events = list(player.known_events or [])
        if event_id not in known_events:
            known_events.append(event_id)
        updated = self.db.update(player, {"known_events": known_events})
        return PlayerRead.model_validate(updated)

    def add_frequent_npc(self, player_id: str, npc_id: str) -> PlayerRead | None:
        player = self.db.get_player(player_id)
        if player is None:
            return None
        npc_ids = list(player.frequent_npc_ids or [])
        if npc_id not in npc_ids:
            npc_ids.append(npc_id)
        updated = self.db.update(player, {"frequent_npc_ids": npc_ids})
        return PlayerRead.model_validate(updated)
