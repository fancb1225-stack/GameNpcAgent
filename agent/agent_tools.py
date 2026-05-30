"""
NpcAgent tools.

This module provides a tool class for a NPC Agent. It follows the same
configuration style as a simple AgentTools class, but all database operations go
through service/db_service layers rather than raw SQL.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime
from uuid import uuid4
from enum import Enum
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel

from db.db_config import Emotion, MemoryType, NpcAction, TimePeriod
from db.db_service import DbService, model_to_dict
from model.dialogue_model import NpcReplyCreate, PlayerMessageCreate
from model.memory_model import LongTermMemoryCreate
from model.relationship_model import RelationshipDelta
from model.world_model import CafeWorldStateUpdate
from service.scene_service import SceneService
from service.dialogue_service import DialogueService
from service.game_document_service import GameDocumentService
from service.long_memory_service import LongMemoryService
from service.npc_service import NpcService
from service.player_service import PlayerService
from service.relationship_service import RelationshipService
from service.short_memory_service import ShortMemoryService
from service.world_state_service import WorldStateService


def _jsonable(value: Any) -> Any:
    """Convert ORM/Pydantic/Enum/datetime objects into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__table__"):
        return _jsonable(model_to_dict(value))
    return value


def _ok(tool_name: str, **payload: Any) -> dict[str, Any]:
    return {"ok": True, "tool_name": tool_name, **_jsonable(payload)}


def _fail(tool_name: str, error: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "tool_name": tool_name, "error": error, **_jsonable(payload)}


class NpcAgentTools:
    """
    Cafe NPC Agent tool set.

    Typical usage:
        tools = NpcAgentTools()
        result = tools.invoke_tool("get_npc_state", {"npc_id": "barista_001"})

    Design rules:
        1. Tool output is stable JSON-like dict.
        2. Tool metadata uses name_for_human/name_for_model/description_for_model.
        3. Tool methods do not execute raw SQL.
        4. Illegal or unknown actions can be checked before being applied.
    """

    def __init__(
        self,
        enable_web_search: bool = False,
        search_api_key: str | None = None,
        llm_service: Any | None = None,
    ) -> None:
        self.db = DbService()
        self.players = PlayerService(self.db)
        self.npcs = NpcService(self.db)
        self.relationships = RelationshipService(self.db)
        self.dialogues = DialogueService(self.db)
        self.short_memories = ShortMemoryService(self.db)
        self.long_memories = LongMemoryService(self.db)
        self.world = WorldStateService(self.db)
        self.game_documents = GameDocumentService(self.db)
        self.scene = SceneService(self.db)
        self.llm_service = llm_service

        self.enable_web_search = enable_web_search
        self.search_api_key = search_api_key or os.getenv("TAVILY_API_KEY") or os.getenv("SEARCH_API_KEY")
        self.toolConfig = self._build_tool_config()
        self._tool_handlers: dict[str, Callable[..., Any]] = {
            "enter_morning_cafe": self.enter_game_world,
            "select_dialogue_npc": self.select_dialogue_npc,
            "get_npc_state": self.get_npc_state,
            "update_npc_emotion": self.update_npc_emotion,
            "move_npc_location": self.move_npc_location,
            "get_player_profile": self.get_player_profile,
            "update_player_profile": self.update_player_profile,
            "get_relationship": self.get_relationship,
            "get_player_relationships": self.get_player_relationships,
            "apply_relationship_delta": self.apply_relationship_delta,
            "get_history_message": self.get_history_message,
            "write_player_message": self.write_player_message,
            "write_npc_reply": self.write_npc_reply,
            "get_short_term_memory": self.get_short_term_memory,
            "get_long_term_memories": self.get_long_term_memories,
            "get_game_setting_context": self.get_game_setting_context,
            "create_long_term_memory": self.create_long_term_memory,
            "create_long_term_memory_from_messages": self.create_long_term_memory_from_messages,
            "get_world_state": self.get_world_state,
            "update_world_state": self.update_world_state,
            "get_cafe_events": self.get_cafe_events,
            "check_action_allowed": self.check_action_allowed,
            "web_search": self.web_search,
            "append_short_term_message": self.append_short_term_message,
            "trim_short_term_memory": self.trim_short_term_memory,
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "NpcAgentTools":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            self.db.rollback()
        self.close()

    def _build_tool_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name_for_human": "进入游戏场景",
                "name_for_model": "enter_game",
                "description_for_model": "为玩家创建或复用画像，并创建会话，返回世界状态、遇见规则和本轮可交互 NPC。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string", "description": "玩家业务 ID"},
                        "nickname": {"type": "string", "description": "玩家昵称，可选"},
                        "session_id": {"type": "string", "description": "指定会话 ID，可选；不传则自动生成"},
                    },
                    "required": ["player_id"],
                },
            },
            {
                "name_for_human": "选择对话 NPC",
                "name_for_model": "select_dialogue_npc",
                "description_for_model": "把当前会话的 selected_npc_id 设置为玩家选择的 NPC。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                    },
                    "required": ["session_id", "npc_id"],
                },
            },
            {
                "name_for_human": "查询 NPC 状态",
                "name_for_model": "get_npc_state",
                "description_for_model": "查询 NPC 的基础设定、当前位置、当前情绪、咖啡偏好和说话风格。NPC 回复前应优先调用。",
                "parameters": {
                    "type": "object",
                    "properties": {"npc_id": {"type": "string"}},
                    "required": ["npc_id"],
                },
            },
            {
                "name_for_human": "更新 NPC 情绪",
                "name_for_model": "update_npc_emotion",
                "description_for_model": "根据状态变化更新 NPC 当前情绪。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "string"},
                        "emotion": {"type": "string", "enum": [item.value for item in Emotion]},
                    },
                    "required": ["npc_id", "emotion"],
                },
            },
            {
                "name_for_human": "移动 NPC 位置",
                "name_for_model": "move_npc_location",
                "description_for_model": "执行 move_location 动作时更新 NPC 所在位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "npc_id": {"type": "string"},
                        "location": {"type": "string", "description": "例如 counter / window_seat / entrance / kitchen"},
                    },
                    "required": ["npc_id", "location"],
                },
            },
            {
                "name_for_human": "查询玩家画像",
                "name_for_model": "get_player_profile",
                "description_for_model": "查询玩家口味偏好、常来时间、常互动 NPC、消费习惯、性格倾向和已知事件。",
                "parameters": {
                    "type": "object",
                    "properties": {"player_id": {"type": "string"}},
                    "required": ["player_id"],
                },
            },
            {
                "name_for_human": "更新玩家画像",
                "name_for_model": "update_player_profile",
                "description_for_model": "根据玩家对话更新玩家画像，只允许更新画像字段，不直接更新关系数值。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "profile_patch": {"type": "object", "description": "要更新的画像字段"},
                    },
                    "required": ["player_id", "profile_patch"],
                },
            },
            {
                "name_for_human": "查询玩家与 NPC 关系",
                "name_for_model": "get_relationship",
                "description_for_model": "查询玩家与 NPC 的五维关系：trust、familiarity、fondness、dislike、stress。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                    },
                    "required": ["player_id", "npc_id"],
                },
            },
            {
                "name_for_human": "查询玩家与全部 NPC 关系",
                "name_for_model": "get_player_relationships",
                "description_for_model": "查询玩家与所有 NPC 的五维关系，用于回答涉及其他 NPC 的问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                    },
                    "required": ["player_id"],
                },
            },
            {
                "name_for_human": "应用关系变化",
                "name_for_model": "apply_relationship_delta",
                "description_for_model": "由规则层确认后，应用玩家与 NPC 的五维关系变化。LLM 只能建议，不能绕过规则直接写入。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "delta": {
                            "type": "object",
                            "properties": {
                                "trust": {"type": "integer"},
                                "familiarity": {"type": "integer"},
                                "fondness": {"type": "integer"},
                                "dislike": {"type": "integer"},
                                "stress": {"type": "integer"},
                            },
                        },
                    },
                    "required": ["player_id", "npc_id", "delta"],
                },
            },
            {
                "name_for_human": "获取与玩家的历史对话",
                "name_for_model": "get_history_message",
                "description_for_model": "查询某玩家、某 NPC、某会话下的最近对话记录，并生成简短 summary。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["player_id"],
                },
            },
            {
                "name_for_human": "写入玩家消息",
                "name_for_model": "write_player_message",
                "description_for_model": "把玩家输入写入 dialogue_messages，并同步写入短期记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["session_id", "player_id", "npc_id", "message"],
                },
            },
            {
                "name_for_human": "写入 NPC 回复",
                "name_for_model": "write_npc_reply",
                "description_for_model": "把 NPC 台词、意图、动作、情绪、关系变化和状态变化写入数据库。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "content": {"type": "string"},
                        "intent": {"type": "string"},
                        "action": {"type": "string", "enum": [item.value for item in NpcAction]},
                        "emotion": {"type": "string", "enum": [item.value for item in Emotion]},
                        "relationship_delta": {"type": "object"},
                        "state_delta": {"type": "object"},
                        "metadata_json": {"type": "object"},
                        "reply_started_at": {"type": "number"},
                    },
                    "required": ["session_id", "player_id", "npc_id", "content"],
                },
            },
            {
                "name_for_human": "查询短期记忆",
                "name_for_model": "get_short_term_memory",
                "description_for_model": "查询玩家与 NPC 最近最多 20 条短期记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                    },
                    "required": ["player_id", "npc_id"],
                },
            },
            {
                "name_for_human": "查询长期记忆",
                "name_for_model": "get_long_term_memories",
                "description_for_model": "查询玩家与 NPC 的长期摘要记忆，按重要性返回。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["player_id"],
                },
            },
            {
                "name_for_human": "创建长期记忆",
                "name_for_model": "create_long_term_memory",
                "description_for_model": "把一段重要对话总结为长期摘要记忆。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "memory_type": {"type": "string", "enum": [item.value for item in MemoryType]},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "source_message_ids": {"type": "array", "items": {"type": "string"}},
                        "metadata_json": {"type": "object"},
                    },
                    "required": ["player_id", "npc_id", "title", "content"],
                },
            },
            {
                "name_for_human": "查询游戏世界状态",
                "name_for_model": "get_world_state",
                "description_for_model": "查询当前事件等世界状态。",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type": "string", "default": "default_morning_session"}},
                    "required": [],
                },
            },
            {
                "name_for_human": "更新游戏世界状态",
                "name_for_model": "update_world_state",
                "description_for_model": "更新天气等世界状态。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "state_patch": {"type": "object"},
                    },
                    "required": ["session_id", "state_patch"],
                },
            },
            {
                "name_for_human": "校验动作是否合法",
                "name_for_model": "check_action_allowed",
                "description_for_model": "校验 LLM 选择的 action 是否属于固定枚举，并检查 move_location、serve_customer 等动作的基础约束。非法 action 应降级为 chat 或 ask_player。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "npc_id": {"type": "string"},
                        "target_location": {"type": "string"},
                    },
                    "required": ["action", "npc_id"],
                },
            },
            {
                "name_for_human": "web 搜索",
                "name_for_model": "web_search",
                "description_for_model": "可选工具。仅当 enable_web_search=True 且存在 Tavily API Key 时，用于查询外部公开信息。游戏内事实应优先使用数据库工具。",
                "parameters": {
                    "type": "object",
                    "properties": {"search_query": {"type": "string"}},
                    "required": ["search_query"],
                },
            },
            {
                "name_for_human": "检索游戏设定",
                "name_for_model": "get_game_setting_context",
                "description_for_model": "按玩家输入检索游戏世界观、规则、地点和任务设定，作为 NPC 对话 RAG 上下文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        ]

    def invoke_tool(self, tool_name: str, arguments: dict[str, Any] | str | None = None) -> Any:
        """Invoke a tool by name. Useful when LLM returns tool name + JSON args."""
        if tool_name not in self._tool_handlers:
            return _fail(tool_name, f"未知工具: {tool_name}")
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return _fail(tool_name, f"工具参数不是合法 JSON: {exc}")
        if not isinstance(arguments, dict):
            return _fail(tool_name, "工具参数必须是 dict 或 JSON object 字符串")
        try:
            return self._tool_handlers[tool_name](**arguments)
        except TypeError as exc:
            return _fail(tool_name, f"工具参数错误: {exc}")
        except Exception as exc:  # keep tool call stable for agent loop
            self.db.rollback()
            return _fail(tool_name, str(exc))

    def get_available_tools(self) -> list[str]:
        return [tool["name_for_model"] for tool in self.toolConfig]

    def get_tool_description(self, tool_name: str) -> str:
        for tool in self.toolConfig:
            if tool["name_for_model"] == tool_name:
                return tool["description_for_model"]
        return "未知工具"

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert toolConfig into OpenAI-style function tool schemas."""
        schemas: list[dict[str, Any]] = []
        for tool in self.toolConfig:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name_for_model"],
                        "description": tool["description_for_model"],
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return schemas

    # ---------- Scene ----------

    def enter_game_world(self, player_id: str, nickname: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        if not player_id:
            return _fail("enter_game_world", "缺少必要参数 player_id")
        result = self.scene.enter_morning_cafe(player_id=player_id, nickname=nickname, session_id=session_id)
        return _ok("enter_game_world", result=result)

    def select_dialogue_npc(self, session_id: str, npc_id: str) -> dict[str, Any]:
        if not session_id or not npc_id:
            return _fail("select_dialogue_npc", "缺少必要参数 session_id 或 npc_id")
        session = self.scene.select_dialogue_npc(session_id=session_id, npc_id=npc_id)
        if session is None:
            return _fail("select_dialogue_npc", "未找到会话", session_id=session_id)
        return _ok("select_dialogue_npc", session=session)

    # ---------- NPC ----------

    def get_npc_state(self, npc_id: str) -> dict[str, Any]:
        if not npc_id:
            return _fail("get_npc_state", "缺少必要参数 npc_id")
        if npc_id == "__list_all__":
            npcs = self.npcs.list_active_npcs()
            return _ok("get_npc_state", npcs=npcs, count=len(npcs))
        npc = self.npcs.get_npc(npc_id)
        if npc is None:
            return _fail("get_npc_state", "未找到 NPC", npc_id=npc_id)
        return _ok("get_npc_state", npc=npc)

    def update_npc_emotion(self, npc_id: str, emotion: str) -> dict[str, Any]:
        if not npc_id or not emotion:
            return _fail("update_npc_emotion", "缺少必要参数 npc_id 或 emotion")
        if emotion not in {item.value for item in Emotion}:
            return _fail("update_npc_emotion", f"非法 emotion: {emotion}")
        npc = self.npcs.update_emotion(npc_id, emotion)
        if npc is None:
            return _fail("update_npc_emotion", "未找到 NPC", npc_id=npc_id)
        return _ok("update_npc_emotion", npc=npc)

    def move_npc_location(self, npc_id: str, location: str) -> dict[str, Any]:
        if not npc_id or not location:
            return _fail("move_npc_location", "缺少必要参数 npc_id 或 location")
        npc = self.npcs.move_location(npc_id, location)
        if npc is None:
            return _fail("move_npc_location", "未找到 NPC", npc_id=npc_id)
        return _ok("move_npc_location", npc=npc)

    # ---------- Player ----------

    def get_player_profile(self, player_id: str) -> dict[str, Any]:
        if not player_id:
            return _fail("get_player_profile", "缺少必要参数 player_id")
        player = self.players.get_player(player_id)
        if player is None:
            return _fail("get_player_profile", "未找到玩家", player_id=player_id)
        return _ok("get_player_profile", player=player)

    def update_player_profile(self, player_id: str, profile_patch: dict[str, Any]) -> dict[str, Any]:
        if not player_id:
            return _fail("update_player_profile", "缺少必要参数 player_id")
        allowed_fields = {
            "nickname",
            "taste_preferences",
            "common_visit_periods",
            "frequent_npc_ids",
            "consumption_habits",
            "personality_tendency",
            "known_events",
            "npc_subjective_notes",
        }
        clean_patch = {key: value for key, value in (profile_patch or {}).items() if key in allowed_fields}
        if not clean_patch:
            return _fail("update_player_profile", "没有可更新的玩家画像字段")
        player = self.db.update_player(player_id, **clean_patch)
        if player is None:
            return _fail("update_player_profile", "未找到玩家", player_id=player_id)
        return _ok("update_player_profile", player=player)

    # ---------- Relationship ----------

    def get_relationship(self, player_id: str, npc_id: str) -> dict[str, Any]:
        if not player_id or not npc_id:
            return _fail("get_relationship", "缺少必要参数 player_id 或 npc_id")
        rel = self.relationships.get_or_create_player_npc_relationship(player_id, npc_id)
        return _ok("get_relationship", relationship=rel)

    def get_player_relationships(self, player_id: str) -> dict[str, Any]:
        if not player_id:
            return _fail("get_player_relationships", "缺少必要参数 player_id")

        # 返回玩家与所有 NPC 的关系，供跨 NPC 对话问题引用。
        relationships = self.relationships.list_player_relationships(player_id)
        return _ok(
            "get_player_relationships",
            player_id=player_id,
            relationships=relationships,
            count=len(relationships),
        )

    def apply_relationship_delta(self, player_id: str, npc_id: str, delta: dict[str, int]) -> dict[str, Any]:
        if not player_id or not npc_id:
            return _fail("apply_relationship_delta", "缺少必要参数 player_id 或 npc_id")
        allowed = {"trust", "familiarity", "fondness", "dislike", "stress"}
        clean_delta = {key: int(value) for key, value in (delta or {}).items() if key in allowed}
        relationship_delta = RelationshipDelta(**clean_delta)
        rel = self.relationships.apply_player_npc_delta(player_id, npc_id, relationship_delta)
        return _ok("apply_relationship_delta", relationship=rel, applied_delta=relationship_delta.as_dict())

    # ---------- Dialogue ----------

    def get_history_message(
        self,
        player_id: str,
        npc_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        if not player_id:
            return _fail("get_history_message", "缺少必要参数 player_id")
        limit = max(1, min(int(limit), 100))
        messages: list[Any]
        if session_id:
            rows = self.dialogues.list_messages(session_id=session_id, limit=limit)
            messages = [row for row in rows if row.player_id == player_id and (not npc_id or row.npc_id in (npc_id, None))]
        else:
            # Use indexed ORM query through DbService generic list path.
            all_rows = self.db.list(self.db._model("dialogue_messages"), filters={"player_id": player_id}, limit=limit, order_by="created_at", desc=True)
            filtered = [row for row in all_rows if not npc_id or row.npc_id in (npc_id, None)]
            messages = list(reversed(filtered))
        summary = "\n".join(f"{getattr(msg, 'role', '')}：{getattr(msg, 'content', '')}" for msg in messages if getattr(msg, "content", None))
        return _ok(
            "get_history_message",
            player_id=player_id,
            npc_id=npc_id,
            session_id=session_id,
            count=len(messages),
            messages=messages,
            summary=summary or "暂无历史对话记录。",
        )

    def write_player_message(self, session_id: str, player_id: str, npc_id: str, message: str) -> dict[str, Any]:
        if not all([session_id, player_id, npc_id, message]):
            return _fail("write_player_message", "缺少必要参数")
        msg = self.dialogues.add_player_message(
            PlayerMessageCreate(session_id=session_id, player_id=player_id, npc_id=npc_id, content=message)
        )
        return _ok("write_player_message", message=msg)

    def write_npc_reply(
        self,
        session_id: str,
        player_id: str,
        npc_id: str,
        content: str,
        intent: str | None = None,
        action: str = "chat",
        emotion: str = "neutral",
        relationship_delta: dict[str, int] | None = None,
        state_delta: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
        reply_started_at: float | None = None,
    ) -> dict[str, Any]:
        if not all([session_id, player_id, npc_id, content]):
            return _fail("write_npc_reply", "缺少必要参数")
        action_check = self.check_action_allowed(action=action, npc_id=npc_id, target_location=(state_delta or {}).get("npc_location"))
        if not action_check.get("allowed"):
            action = "chat"
        if emotion not in {item.value for item in Emotion}:
            emotion = "neutral"
        rel_delta = RelationshipDelta(**{key: int(value) for key, value in (relationship_delta or {}).items() if key in {"trust", "familiarity", "fondness", "dislike", "stress"}})
        msg = self.dialogues.add_npc_reply(
            NpcReplyCreate(
                session_id=session_id,
                player_id=player_id,
                npc_id=npc_id,
                content=content,
                intent=intent,
                action=NpcAction(action),
                emotion=Emotion(emotion),
                relationship_delta=rel_delta,
                state_delta=state_delta or {},
                metadata_json=metadata_json or {},
                reply_started_at=reply_started_at,
            )
        )
        if state_delta and state_delta.get("npc_emotion"):
            self.update_npc_emotion(npc_id, str(state_delta["npc_emotion"]))
        if state_delta and state_delta.get("npc_location"):
            self.move_npc_location(npc_id, str(state_delta["npc_location"]))
        return _ok("write_npc_reply", message=msg, action_after_validation=action)

    # ---------- Memory ----------

    def get_short_term_memory(self, player_id: str, npc_id: str) -> dict[str, Any]:
        if not player_id or not npc_id:
            return _fail("get_short_term_memory", "缺少必要参数 player_id 或 npc_id")
        memory = self.short_memories.get_short_memory(player_id, npc_id)
        if memory is None:
            return _ok("get_short_term_memory", player_id=player_id, npc_id=npc_id, count=0, messages=[])
        return _ok("get_short_term_memory", memory=memory, count=memory.message_count, messages=memory.messages)

    def append_short_term_message(
        self,
        player_id: str,
        npc_id: str,
        message: dict,
        async_archive: bool = False,
    ):
        return self.short_memories.append_short_term_message(
            player_id=player_id,
            npc_id=npc_id,
            message=message,
            async_archive=async_archive,
        )

    def trim_short_term_memory(
        self,
        player_id: str,
        npc_id: str,
        remove_count: int,
    ) -> dict[str, Any]:
        return self.short_memories.trim_short_term_memory(
            player_id=player_id,
            npc_id=npc_id,
            remove_count=remove_count,
        )

    def get_long_term_memories(
        self,
        player_id: str,
        npc_id: Optional[str] = None,
        limit: int = 20,
        query: str | None = None,
    ) -> dict[str, Any]:
        if not player_id:
            return _fail("get_long_term_memories", "缺少必要参数 player_id")
        safe_limit = max(1, min(int(limit), 100))
        if query and hasattr(self.long_memories, "search_long_memories"):
            memories = self.long_memories.search_long_memories(
                player_id=player_id,
                npc_id=npc_id,
                query=query,
                limit=safe_limit,
            )
            return _ok(
                "get_long_term_memories",
                count=len(memories),
                memories=memories,
                query=query,
                search_mode="semantic",
            )
        memories = self.long_memories.list_long_memories(
            player_id=player_id,
            npc_id=npc_id,
            limit=safe_limit,
        )
        return _ok("get_long_term_memories", count=len(memories), memories=memories)

    def get_game_setting_context(self, query: str, limit: int = 5) -> dict[str, Any]:
        if not query:
            return _fail("get_game_setting_context", "缺少必要参数 query")
        result = self.game_documents.search_game_settings(query=query, limit=limit)
        if result.get("ok") is False:
            return _fail("get_game_setting_context", str(result.get("error")))
        return _ok(
            "get_game_setting_context",
            query=result["query"],
            chunks=result["chunks"],
            context=result["context"],
        )

    def create_long_term_memory(
        self,
        player_id: str,
        npc_id: str,
        title: str,
        content: str,
        memory_type: str = "event",
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not all([player_id, npc_id, title, content]):
            return _fail("create_long_term_memory", "缺少必要参数")
        if memory_type not in {item.value for item in MemoryType}:
            memory_type = "event"
        data = LongTermMemoryCreate(
            memory_id=f"mem_{uuid4().hex[:16]}",
            player_id=player_id,
            npc_id=npc_id,
            title=title,
            content=content,
            memory_type=MemoryType(memory_type),
            importance=max(0.0, min(float(importance), 1.0)),
            source_message_ids=source_message_ids or [],
            metadata_json=metadata_json or {},
        )
        memory = self.long_memories.create_long_memory(data)
        return _ok("create_long_term_memory", memory=memory)

    def create_long_term_memory_from_messages(
        self,
        player_id: str,
        npc_id: str,
        messages: list[dict[str, Any]],
        llm_service: Any | None = None,
    ) -> dict[str, Any]:
        if not player_id or not npc_id or not messages:
            return _fail("create_long_term_memory_from_messages", "缺少必要参数")
        return self.long_memories.create_long_term_memory_from_messages(
            player_id=player_id,
            npc_id=npc_id,
            messages=messages,
            llm_service=llm_service or getattr(self, "llm_service", None),
        )

    # ---------- World / Coffee ----------

    def get_world_state(self, session_id: str = "default_morning_session") -> dict[str, Any]:
        state = self.world.get_world_state(session_id=session_id)
        if state is None:
            return _fail("get_world_state", "未找到世界状态", session_id=session_id)
        return _ok("get_world_state", world_state=state)

    def update_world_state(self, session_id: str, state_patch: dict[str, Any]) -> dict[str, Any]:
        if not session_id:
            return _fail("update_world_state", "缺少必要参数 session_id")
        allowed_fields = {"time_period", "customer_flow", "inventory", "today_menu", "weather", "background_music", "seat_occupancy", "current_event_ids"}
        clean_patch = {key: value for key, value in (state_patch or {}).items() if key in allowed_fields}
        if not clean_patch:
            return _fail("update_world_state", "没有可更新的世界状态字段")
        state = self.world.update_world_state(session_id, CafeWorldStateUpdate(**clean_patch))
        if state is None:
            return _fail("update_world_state", "未找到世界状态", session_id=session_id)
        return _ok("update_world_state", world_state=state)

    def get_cafe_events(self, time_period: Optional[str] = None) -> dict[str, Any]:
        if time_period and time_period not in {item.value for item in TimePeriod}:
            return _fail("get_cafe_events", f"非法 time_period: {time_period}")
        events = self.world.list_active_events(time_period=TimePeriod(time_period) if time_period else None)
        return _ok("get_cafe_events", count=len(events), events=events)

    def get_recommendation_context(self, player_id: str, npc_id: str, session_id: str = "default_morning_session") -> dict[str, Any]:
        if not player_id or not npc_id:
            return _fail("get_recommendation_context", "缺少必要参数 player_id 或 npc_id")
        player = self.players.get_player(player_id)
        npc = self.npcs.get_npc(npc_id)
        relationship = self.relationships.get_or_create_player_npc_relationship(player_id, npc_id)
        world_state = self.world.get_world_state(session_id)
        menu = self.world.list_available_menu(session_id)
        coffee_knowledge = self.coffee.list_knowledge(limit=20)
        if player is None:
            return _fail("get_recommendation_context", "未找到玩家", player_id=player_id)
        if npc is None:
            return _fail("get_recommendation_context", "未找到 NPC", npc_id=npc_id)
        return _ok(
            "get_recommendation_context",
            player=player,
            npc=npc,
            relationship=relationship,
            world_state=world_state,
            available_menu=menu,
            coffee_knowledge=coffee_knowledge,
        )

    # ---------- Policy helper ----------

    def check_action_allowed(self, action: str, npc_id: str, target_location: Optional[str] = None) -> dict[str, Any]:
        allowed_actions = {item.value for item in NpcAction}
        if action not in allowed_actions:
            return {"ok": True, "tool_name": "check_action_allowed", "allowed": False, "fallback_action": "chat", "reason": f"非法 action: {action}"}
        npc = self.npcs.get_npc(npc_id)
        if npc is None:
            return {"ok": True, "tool_name": "check_action_allowed", "allowed": False, "fallback_action": "chat", "reason": "NPC 不存在"}
        if action == NpcAction.MOVE_LOCATION.value:
            legal_locations = {"counter", "window_seat", "entrance", "kitchen", "bar", "table_area", "door"}
            if not target_location or target_location not in legal_locations:
                return {"ok": True, "tool_name": "check_action_allowed", "allowed": False, "fallback_action": "chat", "reason": "move_location 目标位置非法或缺失"}
        return {"ok": True, "tool_name": "check_action_allowed", "allowed": True, "fallback_action": None, "reason": "action 合法"}

    # ---------- Optional external search ----------

    def web_search(self, search_query: str) -> dict[str, Any]:
        if not self.enable_web_search:
            return _fail("web_search", "web_search 未启用。游戏内事实请使用数据库工具。")
        if not self.search_api_key:
            return _fail("web_search", "缺少 TAVILY_API_KEY 或 SEARCH_API_KEY")
        if not search_query:
            return _fail("web_search", "缺少必要参数 search_query")
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.search_api_key}", "Content-Type": "application/json"},
                json={"query": search_query, "max_results": 5, "search_depth": "basic"},
                timeout=10,
            )
            if response.status_code != 200:
                return _fail("web_search", f"搜索 API 返回错误：{response.status_code}", response_text=response.text)
            data = response.json()
            results = data.get("results", [])
            return _ok("web_search", query=search_query, count=len(results), results=results[:5])
        except Exception as exc:
            return _fail("web_search", str(exc))
