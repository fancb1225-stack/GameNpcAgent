from __future__ import annotations

from typing import Any


class EvalFakeTools:
    # 为评测提供无数据库副作用的工具替身，保证用例可以在本地和 CI 中稳定运行。
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 记录所有工具调用。
        # 后续可以据此扩展工具调用顺序和副作用评测。
        payload = arguments or {}
        self.calls.append((tool_name, payload))

        # 为不同工具返回最小但语义稳定的上下文。
        if tool_name == "get_npc_state":
            return {
                "ok": True,
                "npc": {
                    "npc_id": payload.get("npc_id"),
                    "name": "林岚",
                    "npc_type": "staff",
                    "current_location": "counter",
                    "current_emotion": "calm",
                },
            }
        if tool_name == "get_player_profile":
            return {
                "ok": True,
                "player": {
                    "player_id": payload.get("player_id"),
                    "taste_preferences": {"likes_latte": True},
                },
            }
        if tool_name == "get_relationship":
            return {
                "ok": True,
                "relationship": {
                    "trust": 10,
                    "familiarity": 20,
                    "fondness": 12,
                    "dislike": 0,
                    "stress": 0,
                },
            }
        if tool_name == "get_world_state":
            return {
                "ok": True,
                "world_state": {
                    "time_period": "morning",
                    "customer_flow": "normal",
                    "weather": "sunny",
                    "today_menu": ["拿铁", "美式", "手冲"],
                },
            }
        if tool_name == "get_short_term_memory":
            return {"ok": True, "messages": [], "count": 0}
        if tool_name == "get_long_term_memories":
            return {"ok": True, "memories": [], "count": 0}
        if tool_name == "get_recommendation_context":
            return {
                "ok": True,
                "available_menu": ["拿铁", "美式", "手冲"],
                "coffee_knowledge": [],
            }
        if tool_name == "check_action_allowed":
            return {"ok": True, "allowed": True}

        # 写入类工具统一返回成功，但不触碰真实数据库。
        return {"ok": True, "tool_name": tool_name}
