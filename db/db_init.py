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
    CafeEvent,
    CafeWorldState,
    CoffeeKnowledge,
    CoffeeKnowledgeCategory,
    CustomerFlow,
    Emotion,
    EncounterRule,
    EventType,
    Npc,
    NpcType,
    DbSessionLocal,
    TimePeriod,
    check_database_connection,
    engine,
)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    Base.metadata.drop_all(bind=engine)


def get_or_create_by_key(db: Session, model, key_name: str, key_value: str, values: dict):
    instance = db.scalar(select(model).where(getattr(model, key_name) == key_value))
    if instance is None:
        instance = model(**{key_name: key_value}, **values)
        db.add(instance)
    else:
        for field, value in values.items():
            setattr(instance, field, value)
    return instance


def seed_npcs(db: Session) -> None:
    npc_rows = [
        {
            "npc_id": "owner_001",
            "name": "沈岚",
            "npc_type": NpcType.STAFF,
            "role": "咖啡厅老板",
            "age": 36,
            "gender": "female",
            "mbti": "ISTJ",
            "job": "店主",
            "hobbies": ["手账", "旧唱片", "店铺经营"],
            "coffee_preferences": {"favorite": "深烘美式", "avoid": ["过甜饮品"]},
            "personality": "沉稳、克制，重视秩序和店内气氛。",
            "speaking_style": "简短直接，偶尔带一点关照。",
            "background": "经营这家街角咖啡厅多年，对常客的习惯记得很清楚。",
            "current_emotion": Emotion.CALM,
            "current_location": "cashier_counter",
            "is_active": True,
        },
        {
            "npc_id": "barista_001",
            "name": "林澈",
            "npc_type": NpcType.STAFF,
            "role": "咖啡师",
            "age": 27,
            "gender": "male",
            "mbti": "ENFP",
            "job": "咖啡师",
            "hobbies": ["拉花", "风味实验", "城市散步"],
            "coffee_preferences": {"favorite": "埃塞俄比亚手冲", "style": "明亮果酸"},
            "personality": "外向、细腻，喜欢根据客人状态推荐咖啡。",
            "speaking_style": "轻松自然，喜欢解释风味。",
            "background": "曾参加手冲比赛，正在研究新的季节限定菜单。",
            "current_emotion": Emotion.PLEASED,
            "current_location": "brew_bar",
            "is_active": True,
        },
        {
            "npc_id": "customer_001",
            "name": "周祈",
            "npc_type": NpcType.FIXED_CUSTOMER,
            "role": "固定顾客",
            "age": 31,
            "gender": "male",
            "mbti": "INTP",
            "job": "自由撰稿人",
            "hobbies": ["阅读", "观察行人", "写作"],
            "coffee_preferences": {"favorite": "冰拿铁", "sweetness": "low"},
            "personality": "安静、观察力强，说话慢但信息量高。",
            "speaking_style": "克制、略带旁观者语气。",
            "background": "几乎每天上午坐在靠窗位置写稿。",
            "current_emotion": Emotion.CALM,
            "current_location": "window_seat",
            "is_active": True,
        },
        {
            "npc_id": "passerby_001",
            "name": "唐棠",
            "npc_type": NpcType.RANDOM_CUSTOMER,
            "role": "路人顾客",
            "age": 23,
            "gender": "female",
            "mbti": "ESFJ",
            "job": "设计实习生",
            "hobbies": ["甜品", "拍照", "探店"],
            "coffee_preferences": {"favorite": "焦糖拿铁", "sweetness": "high"},
            "personality": "活泼，容易被新品吸引。",
            "speaking_style": "轻快，常带感叹。",
            "background": "第一次来到这家咖啡厅。",
            "current_emotion": Emotion.CURIOUS,
            "current_location": "entrance",
            "is_active": True,
        },
    ]
    for row in npc_rows:
        npc_id = row.pop("npc_id")
        get_or_create_by_key(db, Npc, "npc_id", npc_id, row)


def seed_events(db: Session) -> None:
    rows = [
        {
            "event_id": "event_morning_rush_001",
            "title": "上午熟客小高峰",
            "content": "上午九点后熟客陆续到店，吧台需要保持出杯效率。",
            "event_type": EventType.DAILY,
            "time_period": TimePeriod.MORNING,
            "is_active": True,
            "metadata_json": {"risk": "queue", "mood": "busy"},
        },
        {
            "event_id": "event_weather_001",
            "title": "小雨天气",
            "content": "窗外下着小雨，热饮和奶咖更容易被推荐。",
            "event_type": EventType.WEATHER,
            "time_period": TimePeriod.MORNING,
            "is_active": True,
            "metadata_json": {"weather": "rain", "recommendation_bias": ["hot", "milk"]},
        },
    ]
    for row in rows:
        event_id = row.pop("event_id")
        get_or_create_by_key(db, CafeEvent, "event_id", event_id, row)


def seed_knowledge(db: Session) -> None:
    rows = [
        {
            "knowledge_id": "coffee_menu_001",
            "title": "埃塞俄比亚手冲",
            "content": "具有柑橘、花香和轻盈茶感，适合喜欢明亮酸质的客人。",
            "category": CoffeeKnowledgeCategory.MENU,
            "tags": ["handbrew", "citrus", "floral", "light"],
            "is_active": True,
        },
        {
            "knowledge_id": "coffee_menu_002",
            "title": "热拿铁",
            "content": "奶香柔和，酸苦平衡，适合雨天或偏好顺滑口感的客人。",
            "category": CoffeeKnowledgeCategory.MENU,
            "tags": ["milk", "hot", "smooth"],
            "is_active": True,
        },
        {
            "knowledge_id": "coffee_brew_001",
            "title": "手冲推荐原则",
            "content": "推荐手冲时应说明豆子产区、主要风味、酸甜苦平衡和适合人群。",
            "category": CoffeeKnowledgeCategory.BREW,
            "tags": ["recommendation", "handbrew"],
            "is_active": True,
        },
    ]
    for row in rows:
        knowledge_id = row.pop("knowledge_id")
        get_or_create_by_key(db, CoffeeKnowledge, "knowledge_id", knowledge_id, row)


def seed_world_state(db: Session) -> None:
    get_or_create_by_key(
        db,
        CafeWorldState,
        "session_id",
        "default_morning_session",
        {
            "time_period": TimePeriod.MORNING,
            "customer_flow": CustomerFlow.NORMAL,
            "inventory": {
                "ethiopia_beans": 12,
                "espresso_beans": 20,
                "milk": 15,
                "caramel_syrup": 4,
            },
            "today_menu": [
                {"name": "埃塞俄比亚手冲", "available": True, "tags": ["handbrew", "citrus", "floral"]},
                {"name": "热拿铁", "available": True, "tags": ["milk", "hot", "smooth"]},
                {"name": "焦糖拿铁", "available": True, "tags": ["milk", "sweet"]},
            ],
            "weather": "小雨",
            "background_music": "低音量爵士乐",
            "seat_occupancy": {"window_seat": "customer_001", "bar_seat": None, "corner_table": None},
            "current_event_ids": ["event_morning_rush_001", "event_weather_001"],
        },
    )


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
        seed_npcs(db)
        seed_events(db)
        seed_knowledge(db)
        seed_world_state(db)
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
