from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260507_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def uuid_column() -> sa.Column:
    """创建 UUID 主键列。"""
    return sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True)


def timestamps() -> list[sa.Column]:
    """创建通用时间戳列。"""
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    """创建 CoffeeNpcAgent 首版数据库表。"""
    op.create_table(
        "players",
        uuid_column(),
        sa.Column("player_id", sa.String(128), nullable=False),
        sa.Column("nickname", sa.String(128), nullable=True),
        sa.Column("taste_preferences", sa.JSON(), nullable=False),
        sa.Column("common_visit_periods", sa.JSON(), nullable=False),
        sa.Column("frequent_npc_ids", sa.JSON(), nullable=False),
        sa.Column("consumption_habits", sa.JSON(), nullable=False),
        sa.Column("personality_tendency", sa.JSON(), nullable=False),
        sa.Column("known_events", sa.JSON(), nullable=False),
        sa.Column("npc_subjective_notes", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("player_id"),
    )
    op.create_index("ix_players_player_id", "players", ["player_id"])

    op.create_table(
        "npcs",
        uuid_column(),
        sa.Column("npc_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("npc_type", sa.String(32), nullable=False),
        sa.Column("role", sa.String(128), nullable=False),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("gender", sa.String(32), nullable=False),
        sa.Column("mbti", sa.String(16), nullable=False),
        sa.Column("job", sa.String(128), nullable=False),
        sa.Column("hobbies", sa.JSON(), nullable=False),
        sa.Column("coffee_preferences", sa.JSON(), nullable=False),
        sa.Column("personality", sa.Text(), nullable=False),
        sa.Column("speaking_style", sa.Text(), nullable=False),
        sa.Column("background", sa.Text(), nullable=False),
        sa.Column("current_emotion", sa.String(32), nullable=False),
        sa.Column("current_location", sa.String(128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("npc_id"),
    )
    op.create_index("ix_npcs_npc_id", "npcs", ["npc_id"])
    op.create_index("idx_npcs_npc_type", "npcs", ["npc_type"])

    op.create_table(
        "relationships",
        uuid_column(),
        sa.Column("source_type", sa.String(16), nullable=False),
        sa.Column("source_id", sa.String(128), nullable=False),
        sa.Column("target_type", sa.String(16), nullable=False),
        sa.Column("target_id", sa.String(128), nullable=False),
        sa.Column("trust", sa.Integer(), nullable=False),
        sa.Column("familiarity", sa.Integer(), nullable=False),
        sa.Column("fondness", sa.Integer(), nullable=False),
        sa.Column("dislike", sa.Integer(), nullable=False),
        sa.Column("stress", sa.Integer(), nullable=False),
        sa.Column("last_interacted_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.UniqueConstraint(
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            name="uq_relationship_endpoints",
        ),
    )
    op.create_index("idx_relationship_source_id", "relationships", ["source_id"])
    op.create_index("idx_relationship_target_id", "relationships", ["target_id"])
    op.create_index(
        "idx_relationship_last_interacted_at",
        "relationships",
        ["last_interacted_at"],
    )

    op.create_table(
        "cafe_world_states",
        uuid_column(),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("time_period", sa.String(32), nullable=False),
        sa.Column("customer_flow", sa.String(32), nullable=False),
        sa.Column("inventory", sa.JSON(), nullable=False),
        sa.Column("today_menu", sa.JSON(), nullable=False),
        sa.Column("weather", sa.String(64), nullable=False),
        sa.Column("background_music", sa.String(128), nullable=False),
        sa.Column("seat_occupancy", sa.JSON(), nullable=False),
        sa.Column("current_event_ids", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("session_id"),
    )
    op.create_index("ix_cafe_world_states_session_id", "cafe_world_states", ["session_id"])
    op.create_index("idx_world_state_time_period", "cafe_world_states", ["time_period"])

    op.create_table(
        "cafe_events",
        uuid_column(),
        sa.Column("event_id", sa.String(128), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("time_period", sa.String(32), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index("ix_cafe_events_event_id", "cafe_events", ["event_id"])
    op.create_index("idx_cafe_events_active_period", "cafe_events", ["is_active", "time_period"])

    op.create_table(
        "encounter_rules",
        uuid_column(),
        sa.Column("time_period", sa.String(32), nullable=False),
        sa.Column("fixed_npc_ids", sa.JSON(), nullable=False),
        sa.Column("random_npc_pool", sa.JSON(), nullable=False),
        sa.Column("random_count", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *timestamps(),
    )
    op.create_index(
        "idx_encounter_rules_period_active",
        "encounter_rules",
        ["time_period", "is_active"],
    )

    op.create_table(
        "dialogue_sessions",
        uuid_column(),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("player_id", sa.String(128), nullable=False),
        sa.Column("selected_npc_id", sa.String(128), nullable=True),
        sa.Column("scene", sa.Text(), nullable=False),
        sa.Column("time_period", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("session_id"),
    )
    op.create_index("ix_dialogue_sessions_session_id", "dialogue_sessions", ["session_id"])
    op.create_index(
        "idx_dialogue_sessions_player_status",
        "dialogue_sessions",
        ["player_id", "status"],
    )

    op.create_table(
        "dialogue_messages",
        uuid_column(),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("player_id", sa.String(128), nullable=False),
        sa.Column("npc_id", sa.String(128), nullable=True),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(128), nullable=True),
        sa.Column("action", sa.String(64), nullable=True),
        sa.Column("emotion", sa.String(32), nullable=True),
        sa.Column("relationship_delta", sa.JSON(), nullable=False),
        sa.Column("state_delta", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("reply_latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_dialogue_messages_session_created",
        "dialogue_messages",
        ["session_id", "created_at"],
    )
    op.create_index(
        "idx_dialogue_messages_player_npc_created",
        "dialogue_messages",
        ["player_id", "npc_id", "created_at"],
    )

    op.create_table(
        "short_term_memories",
        uuid_column(),
        sa.Column("player_id", sa.String(128), nullable=False),
        sa.Column("npc_id", sa.String(128), nullable=False),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.UniqueConstraint("player_id", "npc_id", name="uq_short_memory_player_npc"),
    )

    op.create_table(
        "long_term_memories",
        uuid_column(),
        sa.Column("memory_id", sa.String(128), nullable=False),
        sa.Column("player_id", sa.String(128), nullable=False),
        sa.Column("npc_id", sa.String(128), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.String(32), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("memory_id"),
    )
    op.create_index("ix_long_term_memories_memory_id", "long_term_memories", ["memory_id"])
    op.create_index(
        "idx_long_memories_player_npc_type",
        "long_term_memories",
        ["player_id", "npc_id", "memory_type"],
    )
    op.create_index("idx_long_memories_importance", "long_term_memories", ["importance"])

    op.create_table(
        "coffee_knowledge",
        uuid_column(),
        sa.Column("knowledge_id", sa.String(128), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("knowledge_id"),
    )
    op.create_index("ix_coffee_knowledge_knowledge_id", "coffee_knowledge", ["knowledge_id"])
    op.create_index(
        "idx_coffee_knowledge_category_active",
        "coffee_knowledge",
        ["category", "is_active"],
    )


def downgrade() -> None:
    """删除 CoffeeNpcAgent 首版数据库表。"""
    op.drop_table("coffee_knowledge")
    op.drop_table("long_term_memories")
    op.drop_table("short_term_memories")
    op.drop_table("dialogue_messages")
    op.drop_table("dialogue_sessions")
    op.drop_table("encounter_rules")
    op.drop_table("cafe_events")
    op.drop_table("cafe_world_states")
    op.drop_table("relationships")
    op.drop_table("npcs")
    op.drop_table("players")
