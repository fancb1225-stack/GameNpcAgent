from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


ALLOWED_ACTIONS = {
    "chat",
    "recommend_coffee",
    "ask_player",
    "comment_on_npc",
    "move_location",
    "serve_customer",
    "end_dialogue",
}

ALLOWED_EMOTIONS = {
    "neutral",
    "calm",
    "pleased",
    "happy",
    "curious",
    "worried",
    "annoyed",
    "sad",
    "angry",
    "tired",
}

RELATIONSHIP_FIELDS = {"trust", "familiarity", "fondness", "dislike", "stress"}
DEFAULT_NEXT_ACTION_DELAY_SECONDS = 5.0
COFFEE_ORDER_KEYWORDS = {
    "咖啡",
    "拿铁",
    "美式",
    "手冲",
    "意式",
    "卡布奇诺",
    "摩卡",
    "一杯",
    "点",
    "来杯",
    "来一杯",
    "要一杯",
    "加牛奶",
}


class NpcGraphState(TypedDict, total=False):
    # 保存一次玩家和 NPC 对话所需的基础输入。
    session_id: str
    player_id: str
    npc_id: str
    player_message: str

    # 保存外部依赖，节点通过 tools 和 llm_service 调用业务能力。
    tools: Any
    llm_service: Any

    # 保存上下文读取结果。
    npc_state: Dict[str, Any]
    player_profile: Dict[str, Any]
    relationship: Dict[str, Any]
    world_state: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    long_term_memories: List[Dict[str, Any]]
    recommendation_context: Dict[str, Any]

    # 保存决策和回复结果。
    intent: str
    action: str
    decision: Dict[str, Any]
    action_allowed: bool
    action_error: Optional[str]
    npc_reply: str

    # 保存副作用写入结果。
    relationship_delta: Dict[str, int]
    state_delta: Dict[str, Any]
    relationship_update_result: Dict[str, Any]
    profile_update_result: Dict[str, Any]
    next_action_decision: Dict[str, Any]
    memory_write_result: Dict[str, Any]
    errors: List[str]


def load_context(state: NpcGraphState) -> Dict[str, Any]:
    # 读取 NPC、玩家、关系和世界状态。
    tools = state["tools"]

    return {
        "npc_state": tools.invoke_tool("get_npc_state", {
            "npc_id": state["npc_id"],
        }),
        "player_profile": tools.invoke_tool("get_player_profile", {
            "player_id": state["player_id"],
        }),
        "relationship": tools.invoke_tool("get_relationship", {
            "player_id": state["player_id"],
            "npc_id": state["npc_id"],
        }),
        "world_state": tools.invoke_tool("get_world_state", {
            "session_id": state["session_id"],
        }),
    }


def retrieve_memories(state: NpcGraphState) -> Dict[str, Any]:
    # 读取短期和长期记忆，补足当前对话上下文。
    tools = state["tools"]
    short_memory = tools.invoke_tool("get_short_term_memory", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
    })
    long_memory = tools.invoke_tool("get_long_term_memories", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "query": state.get("player_message", ""),
        "limit": 5,
    })

    return {
        "short_term_memory": short_memory,
        "long_term_memories": long_memory.get("memories", []),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    # 优先解析纯 JSON，失败后从 Markdown 代码块或正文中截取 JSON object。
    if not text:
        return {}

    raw_text = text.strip()
    try:
        data = json.loads(raw_text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S)
    if fenced:
        return json.loads(fenced.group(1))

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw_text[start:end + 1])

    return {}


def _clean_relationship_delta(raw_delta: Any) -> Dict[str, int]:
    # 清洗关系变化字段，限制字段集合和数值范围。
    raw_delta = raw_delta if isinstance(raw_delta, dict) else {}
    clean_delta: Dict[str, int] = {}

    for field_name in RELATIONSHIP_FIELDS:
        try:
            value = int(raw_delta.get(field_name, 0))
        except (TypeError, ValueError):
            value = 0
        clean_delta[field_name] = max(-5, min(5, value))

    return clean_delta


def normalize_decision(data: Any) -> Dict[str, Any]:
    # 将 LLM 或规则输出统一成 Graph 后续节点可安全使用的决策结构。
    data = data if isinstance(data, dict) else {}
    action = str(data.get("action") or "chat")
    emotion = str(data.get("emotion") or "neutral")
    line = str(data.get("line") or data.get("response") or "").strip()

    if action not in ALLOWED_ACTIONS:
        action = "chat"
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"
    if not line:
        line = "嗯，我在听。"

    state_delta = data.get("state_delta")
    memory_tags = data.get("memory_tags")

    return {
        "intent": str(data.get("intent") or "small_talk"),
        "action": action,
        "target_npc_id": data.get("target_npc_id"),
        "emotion": emotion,
        "line_goal": str(data.get("line_goal") or "自然回应玩家"),
        "line": line,
        "relationship_delta": _clean_relationship_delta(data.get("relationship_delta")),
        "state_delta": state_delta if isinstance(state_delta, dict) else {},
        "memory_tags": memory_tags if isinstance(memory_tags, list) else [],
    }


def decide_action(state: NpcGraphState) -> Dict[str, Any]:
    # 只使用 state 注入的 LLM；没有 LLM 时使用稳定规则决策。
    llm_service = state.get("llm_service")

    if llm_service is None:
        decision = normalize_decision({
            "intent": "chat",
            "action": "chat",
            "emotion": "calm",
            "line_goal": "根据当前上下文自然回复玩家",
            "line": rule_based_reply(state),
            "relationship_delta": {"familiarity": 1},
            "state_delta": {},
            "memory_tags": [],
        })
        return {
            "decision": decision,
            "intent": decision["intent"],
            "action": decision["action"],
        }

    # 构造决策提示词，要求 LLM 输出结构化 JSON。
    prompt = json.dumps(
        {
            "task": "根据上下文生成当前 NPC 的动作决策和候选台词。",
            "player_message": state["player_message"],
            "npc_state": state.get("npc_state"),
            "player_profile": state.get("player_profile"),
            "relationship": state.get("relationship"),
            "world_state": state.get("world_state"),
            "short_term_memory": state.get("short_term_memory"),
            "long_term_memories": state.get("long_term_memories"),
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "required_output": {
                "intent": "...",
                "action": "...",
                "emotion": "...",
                "line_goal": "...",
                "line": "...",
                "relationship_delta": {},
                "state_delta": {},
                "memory_tags": [],
            },
        },
        ensure_ascii=False,
        default=str,
    )
    import time
    start_time = time.time()
    response, _ = llm_service.chat(
        prompt=prompt,
        history=[],
        system_prompt="你是游戏 NPC 决策模块，只输出合法 JSON。",
    )
    end_time = time.time()
    print(f"LLM处理耗时：{end_time - start_time}s\n")

    decision = normalize_decision(extract_json_object(response))

    return {
        "decision": decision,
        "intent": decision["intent"],
        "action": decision["action"],
    }


def normalize_decision_node(state: NpcGraphState) -> Dict[str, Any]:
    # 图节点版本的决策清洗，保护后续节点不直接消费脏 LLM 输出。
    decision = normalize_decision(state.get("decision", {}))

    return {
        "decision": decision,
        "intent": decision["intent"],
        "action": decision["action"],
    }


def validate_action(state: NpcGraphState) -> Dict[str, Any]:
    # 校验动作是否合法，非法动作降级为普通聊天。
    tools = state["tools"]
    decision = dict(state.get("decision", {}))
    action = decision.get("action", "chat")

    result = tools.invoke_tool("check_action_allowed", {
        "action": action,
        "npc_id": state["npc_id"],
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
    # 根据校验后的动作决定下一步节点。
    action = state.get("action", "chat")

    if action == "recommend_coffee":
        return "recommend_coffee"
    if action == "move_location":
        return "move_location"
    if action == "serve_customer":
        return "serve_customer"

    return "generate_line"


def build_recommendation_context(state: NpcGraphState) -> Dict[str, Any]:
    # 推荐咖啡前读取推荐上下文。
    tools = state["tools"]
    context = tools.invoke_tool("get_recommendation_context", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "session_id": state["session_id"],
    })

    return {
        "recommendation_context": context,
    }


def rule_based_reply(state: NpcGraphState) -> str:
    # 没有接入 LLM 时提供稳定回复。
    player_message = state.get("player_message", "")
    decision = state.get("decision", {})
    line_goal = decision.get("line_goal") or "自然回应玩家"

    return f"我听到了：{player_message}。{line_goal}"


def generate_line(state: NpcGraphState) -> Dict[str, Any]:
    # 优先使用已规范化决策中的 line，避免重复调用 LLM。
    decision = state.get("decision", {})
    if decision.get("line"):
        return {
            "npc_reply": str(decision["line"]).strip(),
        }

    llm_service = state.get("llm_service")
    if llm_service is None:
        return {
            "npc_reply": rule_based_reply(state),
        }

    # 没有候选台词时，才使用注入的 LLM 生成最终台词。
    prompt = json.dumps(
        {
            "task": "根据决策生成 NPC 最终台词，只输出台词文本。",
            "player_message": state["player_message"],
            "decision": decision,
            "world_state": state.get("world_state"),
            "recommendation_context": state.get("recommendation_context"),
        },
        ensure_ascii=False,
        default=str,
    )
    reply, _ = llm_service.chat(
        prompt=prompt,
        history=[],
        system_prompt="你是咖啡厅 NPC 台词生成模块。",
    )

    return {
        "npc_reply": reply.strip(),
    }


def _looks_like_coffee_order(state: NpcGraphState) -> bool:
    # 根据玩家输入和决策动作判断是否需要安排后续端咖啡动作。
    text = str(state.get("player_message") or "")
    decision = state.get("decision", {})
    action = str(decision.get("action") or state.get("action") or "")

    if action in {"recommend_coffee", "serve_customer"}:
        return True

    return any(keyword in text for keyword in COFFEE_ORDER_KEYWORDS)


def _build_delayed_action_line(state: NpcGraphState) -> str:
    # 生成定时触发时写入的 NPC 后续动作台词。
    return "你点的咖啡好了，我帮你端过来了。请慢用。"


def _run_delayed_next_action(snapshot: Dict[str, Any]) -> None:
    # 定时器触发后写入 NPC 后续动作和短期记忆。
    tools = snapshot["tools"]
    now = datetime.now(timezone.utc).isoformat()
    line = snapshot["line"]

    tools.invoke_tool("write_npc_reply", {
        "session_id": snapshot["session_id"],
        "player_id": snapshot["player_id"],
        "npc_id": snapshot["npc_id"],
        "content": line,
        "intent": "serve_order",
        "action": "serve_customer",
        "emotion": "pleased",
        "relationship_delta": {"familiarity": 1, "fondness": 1},
        "state_delta": {},
        "metadata_json": {
            "source": "delayed_next_action",
            "original_player_message": snapshot["player_message"],
        },
    })
    tools.invoke_tool("append_short_term_message", {
        "player_id": snapshot["player_id"],
        "npc_id": snapshot["npc_id"],
        "message": {
            "role": "npc",
            "content": line,
            "emotion": "pleased",
            "session_id": snapshot["session_id"],
            "intent": "serve_order",
            "action": "serve_customer",
            "timestamp": now,
        },
        "async_archive": True,
    })


def _archive_short_memory_if_needed(
    state: NpcGraphState,
    append_result: Dict[str, Any],
) -> Dict[str, Any] | None:
    # 短期窗口溢出后，才调用长期记忆服务生成摘要并写入长期记忆。
    if not append_result.get("needs_archive"):
        return None

    archive_messages = append_result.get("archive_messages") or []
    if not archive_messages:
        return None

    tools = state["tools"]
    long_result = tools.invoke_tool("create_long_term_memory_from_messages", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "messages": archive_messages,
        "llm_service": state.get("llm_service"),
    })
    if long_result.get("ok"):
        trim_result = tools.invoke_tool("trim_short_term_memory", {
            "player_id": state["player_id"],
            "npc_id": state["npc_id"],
            "remove_count": append_result.get("archive_message_count", len(archive_messages)),
        })
        return {
            "long_term_memory": long_result,
            "trim_short_term_memory": trim_result,
        }

    return {"long_term_memory": long_result}


def decide_next_action(state: NpcGraphState) -> Dict[str, Any]:
    # 判断 NPC 是否需要安排下一步动作，并用定时器延迟执行。
    if not _looks_like_coffee_order(state):
        return {
            "next_action_decision": {
                "scheduled": False,
                "reason": "no_follow_up_action",
            },
        }

    delay_seconds = float(
        state.get("next_action_delay_seconds", DEFAULT_NEXT_ACTION_DELAY_SECONDS)
    )
    line = _build_delayed_action_line(state)
    snapshot = {
        "session_id": state["session_id"],
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "player_message": state["player_message"],
        "tools": state["tools"],
        "line": line,
    }
    timer = threading.Timer(delay_seconds, lambda: _run_delayed_next_action(snapshot))
    timer.daemon = True
    timer.start()

    return {
        "next_action_decision": {
            "scheduled": True,
            "action": "serve_customer",
            "delay_seconds": delay_seconds,
            "line": line,
        },
    }


def apply_state_change(state: NpcGraphState) -> Dict[str, Any]:
    # 应用关系和 NPC 状态变化。
    tools = state["tools"]
    decision = state.get("decision", {})
    relationship_delta = decision.get("relationship_delta", {}) or {}
    state_delta = decision.get("state_delta", {}) or {}

    rel_result = tools.invoke_tool("apply_relationship_delta", {
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
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


def update_player_profile(state: NpcGraphState) -> Dict[str, Any]:
    # 从明确偏好表达中抽取少量玩家画像更新。
    tools = state["tools"]
    text = str(state.get("player_message") or "")
    taste_patch: Dict[str, Any] = {}

    if "拿铁" in text:
        taste_patch["likes_latte"] = True
    if "美式" in text:
        taste_patch["likes_americano"] = True
    if "手冲" in text:
        taste_patch["likes_pour_over"] = True
    if "甜" in text:
        taste_patch["sweetness_hint"] = (
            "likes_sweet"
            if any(word in text for word in ["喜欢甜", "甜一点", "偏甜"])
            else "mentioned_sweetness"
        )
    if "酸" in text:
        taste_patch["acidity_hint"] = "mentioned_acidity"

    if not taste_patch:
        return {
            "profile_update_result": {"ok": True, "skipped": True},
        }

    result = tools.invoke_tool("update_player_profile", {
        "player_id": state["player_id"],
        "profile_patch": {"taste_preferences": taste_patch},
    })

    return {
        "profile_update_result": result,
    }


def write_memory(state: NpcGraphState) -> Dict[str, Any]:
    # 写入玩家消息、NPC 回复和短期记忆。
    tools = state["tools"]
    now = datetime.now(timezone.utc).isoformat()
    decision = state.get("decision", {})
    emotion = decision.get("emotion", "neutral")

    player_message_result = tools.invoke_tool("write_player_message", {
        "session_id": state["session_id"],
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "message": state["player_message"],
    })
    npc_reply_result = tools.invoke_tool("write_npc_reply", {
        "session_id": state["session_id"],
        "player_id": state["player_id"],
        "npc_id": state["npc_id"],
        "content": state["npc_reply"],
        "intent": state.get("intent"),
        "action": state.get("action"),
        "emotion": emotion,
        "relationship_delta": state.get("relationship_delta", {}),
        "state_delta": state.get("state_delta", {}),
    })
    def append_short_memories() -> None:
        # 后台写入短期记忆，避免阻塞后续动作调度。
        player_memory_result = tools.invoke_tool("append_short_term_message", {
            "player_id": state["player_id"],
            "npc_id": state["npc_id"],
            "message": {
                "role": "player",
                "content": state["player_message"],
                "emotion": "neutral",
                "session_id": state["session_id"],
                "timestamp": now,
            },
            "async_archive": False,
        })
        _archive_short_memory_if_needed(state, player_memory_result)

        npc_memory_result = tools.invoke_tool("append_short_term_message", {
            "player_id": state["player_id"],
            "npc_id": state["npc_id"],
            "message": {
                "role": "npc",
                "content": state["npc_reply"],
                "emotion": emotion,
                "session_id": state["session_id"],
                "intent": state.get("intent"),
                "action": state.get("action"),
                "timestamp": now,
            },
            "async_archive": True,
        })
        _archive_short_memory_if_needed(state, npc_memory_result)

    memory_thread = threading.Thread(target=append_short_memories, daemon=True)
    memory_thread.start()

    return {
        "memory_write_result": {
            "player_message": player_message_result,
            "npc_reply": npc_reply_result,
            "async": True,
            "status": "scheduled",
        },
    }


NPC_GRAPH_NODE_HANDLERS = [
    ("load_context", load_context),
    ("retrieve_memories", retrieve_memories),
    ("decide_action", decide_action),
    ("normalize_decision", normalize_decision_node),
    ("validate_action", validate_action),
    ("build_recommendation_context", build_recommendation_context),
    ("generate_line", generate_line),
    ("decide_next_action", decide_next_action),
    ("apply_state_change", apply_state_change),
    ("update_player_profile", update_player_profile),
    ("write_memory", write_memory),
]

NPC_GRAPH_EDGES = [
    (START, "load_context"),
    ("load_context", "retrieve_memories"),
    ("retrieve_memories", "decide_action"),
    ("decide_action", "normalize_decision"),
    ("normalize_decision", "validate_action"),
    ("build_recommendation_context", "generate_line"),
    ("generate_line", "decide_next_action"),
    ("decide_next_action", "apply_state_change"),
    ("apply_state_change", "update_player_profile"),
    ("update_player_profile", "write_memory"),
    ("write_memory", END),
]

NPC_GRAPH_CONDITIONAL_EDGES = {
    "validate_action": {
        "recommend_coffee": "build_recommendation_context",
        "move_location": "generate_line",
        "serve_customer": "generate_line",
        "generate_line": "generate_line",
    },
}


def format_graph_relationships(
    nodes: List[str],
    edges: List[tuple[str, str]],
    conditional_edges: Dict[str, Dict[str, str]],
) -> str:
    # 按固定顺序拼接图节点，方便查看流程。
    lines = ["图节点："]
    lines.extend([f"- {node}" for node in nodes])

    # 打印普通边。
    lines.append("\n普通边：")
    lines.extend([f"{source} -> {target}" for source, target in edges])

    # 打印条件边。
    lines.append("\n条件边：")
    for source, route_map in conditional_edges.items():
        for route_value, target in route_map.items():
            lines.append(f"{source} --[{route_value}]--> {target}")

    return "\n".join(lines)


def describe_npc_graph() -> str:
    # 使用同一份图定义生成说明。
    return format_graph_relationships(
        nodes=[node_name for node_name, _ in NPC_GRAPH_NODE_HANDLERS],
        edges=NPC_GRAPH_EDGES,
        conditional_edges=NPC_GRAPH_CONDITIONAL_EDGES,
    )


def build_npc_graph():
    # 创建真实 LangGraph 状态图。
    graph = StateGraph(NpcGraphState)

    for node_name, handler in NPC_GRAPH_NODE_HANDLERS:
        graph.add_node(node_name, handler)

    # 注册普通边。
    for source, target in NPC_GRAPH_EDGES:
        graph.add_edge(source, target)

    # 注册条件边。
    graph.add_conditional_edges(
        "validate_action",
        route_action,
        NPC_GRAPH_CONDITIONAL_EDGES["validate_action"],
    )

    return graph.compile()


class NpcGraphAgent:
    def __init__(self, tools, llm_service=None):
        # 初始化时编译图，后续请求复用。
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
        # 准备初始状态，并交给 LangGraph 执行。
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
            "profile_update_result": result.get("profile_update_result", {}),
            "next_action_decision": result.get("next_action_decision", {}),
            "memory_write_result": result.get("memory_write_result", {}),
        }


def build_example_agent_state() -> NpcGraphState:
    # 构造一个可直接交给 graph.invoke 的示例状态。
    from agent.coffee_npc_agent_tools import CoffeeNpcAgentTools
    from agent.llm_service import LlmService

    return {
        "session_id": "s001",
        "player_id": "test001",
        "npc_id": "barista_001",
        "player_message": "不好了！咖啡厅门口有人打架！",
        "tools": CoffeeNpcAgentTools(),
        "llm_service": LlmService.getMinimax(),
        "errors": [],
    }


def run_agent_state_example() -> Dict[str, Any]:
    # 使用真实 LangGraph 执行示例 agent_state。
    graph = build_npc_graph()
    agent_state = build_example_agent_state()
    result = graph.invoke(agent_state)
    output = {
        "npc_reply": result.get("npc_reply"),
        "intent": result.get("intent"),
        "action": result.get("action"),
        "decision": result.get("decision"),
        "relationship_delta": result.get("relationship_delta"),
        "profile_update_result": result.get("profile_update_result"),
        "next_action_decision": result.get("next_action_decision"),
        "memory_write_result": result.get("memory_write_result"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=4, default=str))

    return result


def render_npc_graph(output_path: str = "npc_graph") -> str:
    # 使用 Graphviz 绘制有向流程图。
    from graphviz import Digraph

    dot = Digraph("npc_graph", format="png")
    dot.attr(rankdir="LR")

    # 绘制节点。
    for node_name, _ in NPC_GRAPH_NODE_HANDLERS:
        dot.node(node_name, node_name, shape="box")

    dot.node(str(START), "START", shape="oval")
    dot.node(str(END), "END", shape="oval")

    # 绘制普通边和条件边。
    for source, target in NPC_GRAPH_EDGES:
        dot.edge(str(source), str(target))
    for source, route_map in NPC_GRAPH_CONDITIONAL_EDGES.items():
        for label, target in route_map.items():
            dot.edge(source, target, label=label, style="dashed")

    return dot.render(output_path, cleanup=True)


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_agent_state_example()
    end_time = time.time()

    print(f"NPC Agent回复耗时 {end_time - start_time} seconds")

    # 输出graph可视化的png图片
    # render_npc_graph()
