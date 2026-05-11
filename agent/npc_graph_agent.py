from typing import Any, Dict, List, Optional, TypedDict


class NpcGraphState(TypedDict, total=False):
    session_id: str
    player_id: str
    npc_id: str
    player_message: str

    npc_state: Dict[str, Any]
    player_profile: Dict[str, Any]
    relationship: Dict[str, Any]
    world_state: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    long_term_memories: List[Dict[str, Any]]
    coffee_knowledge: List[Dict[str, Any]]

    intent: str
    action: str
    decision: Dict[str, Any]
    action_allowed: bool
    action_error: Optional[str]

    npc_reply: str
    relationship_delta: Dict[str, int]
    state_delta: Dict[str, Any]

    memory_write_result: Dict[str, Any]
    errors: List[str]


def load_context(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]

    npc_state = tools.invoke_tool("get_npc_state", {
        "npc_id": state["npc_id"]
    })

    player_profile = tools.invoke_tool("get_player_profile", {
        "player_id": state["player_id"]
    })

    relationship = tools.invoke_tool("get_relationship", {
        "source_type": "player",
        "source_id": state["player_id"],
        "target_type": "npc",
        "target_id": state["npc_id"],
    })

    world_state = tools.invoke_tool("get_world_state", {
        "session_id": state["session_id"]
    })

    return {
        "npc_state": npc_state,
        "player_profile": player_profile,
        "relationship": relationship,
        "world_state": world_state,
    }

def retrieve_memories(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]

    short_memory = tools.invoke_tool("get_short_term_memory", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
    })

    long_memory = tools.invoke_tool("search_long_term_memories", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "query": state["player_message"],
        "limit": 5,
    })

    return {
        "short_term_memory": short_memory,
        "long_term_memories": long_memory.get("memories", []),
    }

def decide_action(state: NpcGraphState) -> Dict[str, Any]:
    llm_service = state.get("llm_service")

    if llm_service is None:
        return {
            "decision": {
                "intent": "chat",
                "action": "chat",
                "emotion": "calm",
                "line_goal": "根据当前上下文自然回复玩家",
                "relationship_delta": {
                    "trust": 0,
                    "familiarity": 1,
                    "fondness": 0,
                    "dislike": 0,
                    "stress": 0,
                },
                "state_delta": {},
                "memory_tags": [],
            },
            "intent": "chat",
            "action": "chat",
        }

    prompt = f"""
你是咖啡厅 NPC 的决策模块。请只输出 JSON。

玩家输入：
{state["player_message"]}

NPC 状态：
{state.get("npc_state")}

玩家画像：
{state.get("player_profile")}

关系：
{state.get("relationship")}

世界状态：
{state.get("world_state")}

短期记忆：
{state.get("short_term_memory")}

长期记忆：
{state.get("long_term_memories")}

合法 action：
chat, recommend_coffee, ask_player, comment_on_npc, move_location, serve_customer, end_dialogue

输出格式：
{{
  "intent": "...",
  "action": "...",
  "emotion": "...",
  "line_goal": "...",
  "relationship_delta": {{
    "trust": 0,
    "familiarity": 0,
    "fondness": 0,
    "dislike": 0,
    "stress": 0
  }},
  "state_delta": {{}},
  "memory_tags": []
}}
"""

    response, _ = llm_service.chat(
        prompt=prompt,
        history=[],
        system_prompt="你是游戏 NPC 决策模块，只输出合法 JSON。",
    )

    import json
    try:
        decision = json.loads(response)
    except Exception:
        decision = {
            "intent": "chat",
            "action": "chat",
            "emotion": "calm",
            "line_goal": "普通回复玩家",
            "relationship_delta": {},
            "state_delta": {},
            "memory_tags": [],
        }

    return {
        "decision": decision,
        "intent": decision.get("intent", "chat"),
        "action": decision.get("action", "chat"),
    }

def validate_action(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]
    decision = state.get("decision", {})
    action = decision.get("action", "chat")

    result = tools.invoke_tool("check_action_allowed", {
        "action": action,
        "npc_id": state["npc_id"],
        "target_npc_id": decision.get("target_npc_id"),
        "target_location": decision.get("state_delta", {}).get("npc_location"),
    })

    if not result.get("ok") or not result.get("allowed", False):
        decision["action"] = "chat"
        decision["line_goal"] = "动作不合法，降级为普通聊天回复。"

        return {
            "decision": decision,
            "action": "chat",
            "action_allowed": False,
            "action_error": result.get("error") or result.get("reason"),
        }

    return {
        "action_allowed": True,
        "action": action,
    }

def route_action(state: NpcGraphState) -> str:
    action = state.get("action", "chat")

    if action == "recommend_coffee":
        return "recommend_coffee"
    if action == "move_location":
        return "move_location"
    if action == "serve_customer":
        return "serve_customer"

    return "generate_line"

def build_recommendation_context(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]

    context = tools.invoke_tool("get_recommendation_context", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "session_id": state["session_id"],
    })

    return {
        "recommendation_context": context,
    }

def generate_line(state: NpcGraphState) -> Dict[str, Any]:
    llm_service = state.get("llm_service")
    decision = state.get("decision", {})

    if llm_service is None:
        return {
            "npc_reply": self_rule_based_reply(state)
        }

    prompt = f"""
你正在扮演咖啡厅 NPC。

NPC 信息：
{state.get("npc_state")}

玩家输入：
{state["player_message"]}

决策：
{decision}

世界状态：
{state.get("world_state")}

长期记忆：
{state.get("long_term_memories")}

咖啡推荐上下文：
{state.get("recommendation_context")}

要求：
1. 只输出 NPC 台词。
2. 不要输出 JSON。
3. 不要越权创造不存在的库存、菜单或事件。
4. 语气符合 NPC 性格。
"""

    reply, _ = llm_service.chat(
        prompt=prompt,
        history=[],
        system_prompt="你是咖啡厅 NPC 台词生成模块。",
    )

    return {"npc_reply": reply.strip()}


def apply_state_change(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]
    decision = state.get("decision", {})

    relationship_delta = decision.get("relationship_delta", {}) or {}
    state_delta = decision.get("state_delta", {}) or {}

    rel_result = tools.invoke_tool("apply_relationship_delta", {
        "source_type": "player",
        "source_id": state["player_id"],
        "target_type": "npc",
        "target_id": state["npc_id"],
        "delta": relationship_delta,
    })

    if state_delta.get("npc_emotion"):
        tools.invoke_tool("update_npc_emotion", {
            "npc_id": state["npc_id"],
            "emotion": state_delta["npc_emotion"],
        })

    if state_delta.get("npc_location"):
        tools.invoke_tool("move_npc_location", {
            "npc_id": state["npc_id"],
            "location": state_delta["npc_location"],
        })

    return {
        "relationship_update_result": rel_result,
        "relationship_delta": relationship_delta,
        "state_delta": state_delta,
    }


def write_memory(state: NpcGraphState) -> Dict[str, Any]:
    tools = state["tools"]

    tools.invoke_tool("write_player_message", {
        "session_id": state["session_id"],
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "content": state["player_message"],
    })

    tools.invoke_tool("write_npc_reply", {
        "session_id": state["session_id"],
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "content": state["npc_reply"],
        "intent": state.get("intent"),
        "action": state.get("action"),
        "emotion": state.get("decision", {}).get("emotion"),
        "relationship_delta": state.get("relationship_delta", {}),
        "state_delta": state.get("state_delta", {}),
    })

    memory_result = tools.invoke_tool("append_short_term_message", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "message": {
            "role": "npc",
            "content": state["npc_reply"],
            "session_id": state["session_id"],
            "intent": state.get("intent"),
            "action": state.get("action"),
        },
        "async_archive": True,
    })

    return {
        "memory_write_result": memory_result,
    }

from langgraph.graph import StateGraph, START, END


def build_npc_graph():
    graph = StateGraph(NpcGraphState)

    graph.add_node("load_context", load_context)
    graph.add_node("retrieve_memories", retrieve_memories)
    graph.add_node("decide_action", decide_action)
    graph.add_node("validate_action", validate_action)
    graph.add_node("build_recommendation_context", build_recommendation_context)
    graph.add_node("generate_line", generate_line)
    graph.add_node("apply_state_change", apply_state_change)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "retrieve_memories")
    graph.add_edge("retrieve_memories", "decide_action")
    graph.add_edge("decide_action", "validate_action")

    graph.add_conditional_edges(
        "validate_action",
        route_action,
        {
            "recommend_coffee": "build_recommendation_context",
            "move_location": "generate_line",
            "serve_customer": "generate_line",
            "generate_line": "generate_line",
        },
    )

    graph.add_edge("build_recommendation_context", "generate_line")
    graph.add_edge("generate_line", "apply_state_change")
    graph.add_edge("apply_state_change", "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile()


class NpcGraphAgent:
    def __init__(self, tools, llm_service=None):
        self.tools = tools
        self.llm_service = llm_service
        self.graph = build_npc_graph()

    def handle_message(
        self,
        session_id: str,
        player_id: str,
        npc_id: str,
        player_message: str,
    ) -> dict:
        state = {
            "session_id": session_id,
            "player_id": player_id,
            "npc_id": npc_id,
            "player_message": player_message,
            "tools": self.tools,
            "llm_service": self.llm_service,
            "errors": [],
        }

        result = self.graph.invoke(state)

        return {
            "ok": True,
            "session_id": session_id,
            "player_id": player_id,
            "npc_id": npc_id,
            "reply": result.get("npc_reply", ""),
            "intent": result.get("intent"),
            "action": result.get("action"),
            "decision": result.get("decision"),
            "relationship_delta": result.get("relationship_delta", {}),
            "memory_write_result": result.get("memory_write_result", {}),
        }