from __future__ import annotations

import random
from uuid import uuid4

from db.db_config import SessionStatus, TimePeriod
from db.db_service import DbService
from model.dialogue_model import DialogueSessionCreate, DialogueSessionRead
from model.npc_model import NpcRead
from model.player_model import PlayerCreate, PlayerRead
from model.world_model import WorldStateRead, EncounterRuleRead
from service.dialogue_service import DialogueService
from service.npc_service import NpcService
from service.player_service import PlayerService
from service.world_state_service import WorldStateService


class GameSceneResult(dict):
    """Plain dict-compatible result for simple API serialization."""


class SceneService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None
        self.players = PlayerService(self.db)
        self.npcs = NpcService(self.db)
        self.dialogues = DialogueService(self.db)
        self.world = WorldStateService(self.db)

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def enter_morning_cafe(self, player_id: str, nickname: str | None = None, session_id: str | None = None) -> dict:
        """Create/reuse a player and create an active morning cafe session."""
        player = self.players.get_player(player_id)
        if player is None:
            player = self.players.create_player(PlayerCreate(player_id=player_id, nickname=nickname))

        session_id = session_id or f"session_{uuid4().hex[:12]}"
        session = self.dialogues.create_session(
            DialogueSessionCreate(
                session_id=session_id,
                player_id=player_id,
                scene="morning_cafe",
                time_period=TimePeriod.MORNING,
                status=SessionStatus.ACTIVE,
            )
        )
        rule = self.world.get_encounter_rule(TimePeriod.MORNING)
        world_state = self.world.get_world_state("default_morning_session")
        encountered_npcs = self._resolve_encountered_npcs(rule, session_id)
        return {
            "player": player.model_dump(mode="json"),
            "session": session.model_dump(mode="json"),
            "world_state": world_state.model_dump(mode="json") if world_state else None,
            "encounter_rule": rule.model_dump(mode="json") if rule else None,
            "npcs": [npc.model_dump(mode="json") for npc in encountered_npcs],
        }

    def _resolve_encountered_npcs(self, rule: EncounterRuleRead | None, session_id: str) -> list[NpcRead]:
        if rule is None:
            return self.npcs.list_active_npcs()

        npc_ids = list(rule.fixed_npc_ids or [])
        pool = list(rule.random_npc_pool or [])
        if pool and rule.random_count > 0:
            rng = random.Random(session_id)
            random_ids = rng.sample(pool, k=min(rule.random_count, len(pool)))
            npc_ids.extend(random_ids)
        return self.npcs.list_npcs_by_ids(npc_ids)

    def select_dialogue_npc(self, session_id: str, npc_id: str) -> DialogueSessionRead | None:
        return self.dialogues.select_npc(session_id, npc_id)
