from __future__ import annotations

from db.db_config import CafeEvent, CafeWorldState, TimePeriod
from db.db_service import DbService
from model.world_model import CafeEventRead, CafeWorldStateCreate, CafeWorldStateRead, CafeWorldStateUpdate, EncounterRuleRead


class WorldStateService:
    def __init__(self, db: DbService | None = None):
        self.db = db or DbService()
        self._owns_db = db is None

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def create_world_state(self, data: CafeWorldStateCreate) -> CafeWorldStateRead:
        state = self.db.create(CafeWorldState, data.model_dump())
        return CafeWorldStateRead.model_validate(state)

    def get_world_state(self, session_id: str = "default_morning_session") -> CafeWorldStateRead | None:
        state = self.db.get_world_state(session_id)
        return CafeWorldStateRead.model_validate(state) if state else None

    def update_world_state(self, session_id: str, data: CafeWorldStateUpdate) -> CafeWorldStateRead | None:
        state = self.db.update_world_state(session_id, **data.to_update_dict())
        return CafeWorldStateRead.model_validate(state) if state else None

    def list_active_events(self, time_period: TimePeriod | str | None = None) -> list[CafeEventRead]:
        rows = self.db.list_active_events(time_period=time_period)
        return [CafeEventRead.model_validate(row) for row in rows]

    def get_encounter_rule(self, time_period: TimePeriod | str = TimePeriod.MORNING) -> EncounterRuleRead | None:
        rule = self.db.get_encounter_rule(time_period)
        return EncounterRuleRead.model_validate(rule) if rule else None

    def list_available_menu(self, session_id: str = "default_morning_session") -> list[dict]:
        state = self.db.get_world_state(session_id)
        if state is None:
            return []
        return [item for item in (state.today_menu or []) if isinstance(item, dict) and item.get("available", True)]

    def decrease_inventory(self, session_id: str, item_key: str, amount: int = 1) -> CafeWorldStateRead | None:
        state = self.db.get_world_state(session_id)
        if state is None:
            return None
        inventory = dict(state.inventory or {})
        inventory[item_key] = max(0, int(inventory.get(item_key, 0)) - amount)
        updated = self.db.update(state, {"inventory": inventory})
        return CafeWorldStateRead.model_validate(updated)
