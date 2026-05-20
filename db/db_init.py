"""
Create all CoffeeNpcAgent tables and seed the default morning cafe scene.

Run:
    python db_init.py
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db_config import (
    Base,
    EncounterRule,
    DbSessionLocal,
    TimePeriod,
    check_database_connection,
    engine,
)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    Base.metadata.drop_all(bind=engine)

def seed_encounter_rules(db: Session) -> None:
    rule = db.scalar(
        select(EncounterRule).where(
            EncounterRule.time_period == TimePeriod.MORNING,
            EncounterRule.is_active.is_(True),
        )
    )
    values = {
        "time_period": TimePeriod.MORNING,
        "fixed_npc_ids": ["owner_001", "barista_001", "customer_001"],
        "random_npc_pool": ["passerby_001"],
        "random_count": 1,
        "is_active": True,
    }
    if rule is None:
        db.add(EncounterRule(**values))
    else:
        for field, value in values.items():
            setattr(rule, field, value)


def seed_default_data() -> None:
    with DbSessionLocal() as db:
        seed_encounter_rules(db)
        db.commit()


def init_database(drop_existing: bool = False, seed: bool = True) -> None:
    check_database_connection()
    if drop_existing:
        drop_tables()
    create_tables()
    if seed:
        seed_default_data()


if __name__ == "__main__":
    init_database(drop_existing=False, seed=True)
    print("database initialized")
