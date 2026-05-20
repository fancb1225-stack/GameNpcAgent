from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {
    "chat",
    "ask_player",
    "comment_on_npc",
    "move_location",
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

DECISION_SYSTEM_PROMPT = """
你是 GameNpcFrame 的 NPC 对话决策 Agent。

你的任务：
1. 理解玩家输入、NPC 设定、世界状态、记忆和游戏设定上下文。
2. 选择一个合法 action，并给出本轮 NPC 最终台词 line。
3. 输出可被程序直接解析的单个 JSON object。

输出要求：
- 只输出一个 JSON object，不要 Markdown，不要解释，不要额外自然语言。
- JSON 字段必须遵守 prompt 中的 decision_contract。
- 缺少信息时不要编造，使用角色口吻承认不知道或提出澄清。

决策原则：
1. action 只能来自 allowed_actions，不能自创 action。
2. emotion 只能来自 allowed_emotions，不能自创 emotion。
3. 事实性内容优先依据 game_setting_context，其次参考长期记忆和短期记忆。
4. NPC 台词必须符合 npc_state 中的身份、性格、职业和说话风格。
5. move_location 必须在 state_delta.npc_location 给出目标地点，否则使用 chat。
6. relationship_delta 只表达本轮轻微变化，单项建议范围为 -3 到 3。
7. end_dialogue 只在玩家明确告别或 NPC 必须结束对话时使用。
""".strip()

DECISION_OUTPUT_SCHEMA = {
    "intent": "玩家意图，例如 greet / ask_lore / ask_advice / small_talk",
    "action": "必须是 allowed_actions 中的一个值",
    "target_npc_id": "当前 NPC ID；没有跨 NPC 目标时保持 null",
    "emotion": "必须是 allowed_emotions 中的一个值",
    "line_goal": "本轮台词目标",
    "line": "NPC 最终对玩家说的话，必须符合角色设定，1 到 3 句话",
    "relationship_delta": {
        "trust": 0,
        "familiarity": 0,
        "fondness": 0,
        "dislike": 0,
        "stress": 0,
    },
    "state_delta": {
        "npc_location": None,
        "npc_emotion": None,
    },
    "memory_tags": [],
}

NPC_GRAPH_NODES = [
    "load_context",
    "retrieve_memories",
    "decide_action",
    "validate_action",
    "generate_line",
    "apply_state_change",
    "update_player_profile",
    "write_memory",
]

NPC_GRAPH_EDGES = [
    (START, "load_context"),
    ("load_context", "retrieve_memories"),
    ("retrieve_memories", "decide_action"),
    ("decide_action", "validate_action"),
    ("validate_action", "generate_line"),
    ("generate_line", "apply_state_change"),
    ("apply_state_change", "update_player_profile"),
    ("update_player_profile", "write_memory"),
    ("write_memory", END),
]


# ---------- State definitions ----------

class NpcInputState(TypedDict, total=False):
    session_id: str
    player_id: str
    npc_id: str
    player_message: str


class NpcOutputState(TypedDict, total=False):
    npc_reply: str
    intent: str
    action: str
    decision: Dict[str, Any]
    relationship_delta: Dict[str, int]
    reply_latency_ms: int


class NpcGraphState(NpcInputState, NpcOutputState, total=False):
    npc_state: Dict[str, Any]
    player_profile: Dict[str, Any]
    relationship: Dict[str, Any]
    world_state: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    long_term_memories: List[Dict[str, Any]]
    game_setting_context: Dict[str, Any]
    action_allowed: bool
    action_error: Optional[str]
    state_delta: Dict[str, Any]
    reply_started_at: float


# ---------- Pure utility functions (no dependency on tools/llm) ----------

def _load_json_object_safely(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    raw_text = text.strip()
    data = _load_json_object_safely(raw_text)
    if data:
        return data

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S)
    if fenced:
        data = _load_json_object_safely(fenced.group(1))
        if data:
            return data

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        data = _load_json_object_safely(raw_text[start:end + 1])
        if data:
            return data

    return {}


def _clean_relationship_delta(raw_delta: Any) -> Dict[str, int]:
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


def _build_decision_prompt(state: NpcGraphState) -> str:
    payload = {
        "task": "npc_dialogue_decision",
        "player_message": state.get("player_message", ""),
        "context": {
            "npc_state": state.get("npc_state"),
            "player_profile": state.get("player_profile"),
            "relationship": state.get("relationship"),
            "world_state": state.get("world_state"),
            "game_setting_context": state.get("game_setting_context"),
            "short_term_memory": state.get("short_term_memory"),
            "long_term_memories": state.get("long_term_memories"),
        },
        "decision_contract": {
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "allowed_emotions": sorted(ALLOWED_EMOTIONS),
            "relationship_fields": sorted(RELATIONSHIP_FIELDS),
            "output_schema": DECISION_OUTPUT_SCHEMA,
        },
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _rule_based_reply(state: NpcGraphState) -> str:
    player_message = state.get("player_message", "")
    decision = state.get("decision", {})
    line_goal = decision.get("line_goal") or "自然回应玩家"
    return f"我听到了：{player_message}。{line_goal}"


def _archive_short_memory_if_needed(
    tools: Any,
    llm_service: Any,
    player_id: str,
    npc_id: str,
    append_result: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not append_result.get("needs_archive"):
        return None

    archive_messages = append_result.get("archive_messages") or []
    if not archive_messages:
        return None

    long_result = tools.invoke_tool("create_long_term_memory_from_messages", {
        "player_id": player_id,
        "npc_id": npc_id,
        "messages": archive_messages,
        "llm_service": llm_service,
    })
    if long_result.get("ok"):
        trim_result = tools.invoke_tool("trim_short_term_memory", {
            "player_id": player_id,
            "npc_id": npc_id,
            "remove_count": append_result.get("archive_message_count", len(archive_messages)),
        })
        return {
            "long_term_memory": long_result,
            "trim_short_term_memory": trim_result,
        }

    return {"long_term_memory": long_result}


# ---------- Graph agent ----------

class NpcGraphAgent:
    def __init__(self, tools, llm_service=None):
        self.tools = tools
        self.llm_service = llm_service
        self.graph = self._build_graph()

    def _build_graph(self):
        tools = self.tools
        llm_service = self.llm_service

        def load_context(state: NpcGraphState) -> Dict[str, Any]:
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
            short_memory = tools.invoke_tool("get_short_term_memory", {
                "player_id": state["player_id"],
                "npc_id": state["npc_id"],
            })
            long_memory = tools.invoke_tool("get_long_term_memories", {
                "player_id": state["player_id"],
                "npc_id": state["npc_id"],
                "query": state.get("player_message", ""),
                "limit": 8,
            })
            game_setting = tools.invoke_tool("get_game_setting_context", {
                "query": state.get("player_message", ""),
                "limit": 5,
            })

            return {
                "short_term_memory": short_memory,
                "long_term_memories": long_memory.get("memories", []),
                "game_setting_context": game_setting,
            }

        def decide_action(state: NpcGraphState) -> Dict[str, Any]:
            logger.info(
                "[decide_action] llm_service=%s, type=%s",
                llm_service,
                type(llm_service).__name__ if llm_service else "None",
            )

            if llm_service is None:
                decision = normalize_decision({
                    "intent": "chat",
                    "action": "chat",
                    "emotion": "calm",
                    "line_goal": "根据当前上下文自然回复玩家",
                    "line": _rule_based_reply(state),
                    "relationship_delta": {"familiarity": 1},
                    "state_delta": {},
                    "memory_tags": [],
                })
                return {
                    "decision": decision,
                    "intent": decision["intent"],
                    "action": decision["action"],
                }

            prompt = _build_decision_prompt(state)
            start_time = time.time()
            try:
                response, _ = llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt=DECISION_SYSTEM_PROMPT,
                )
            except Exception:
                logger.exception("[decide_action] LLM 调用异常")
                fallback_goal = "Agent 调用失败，使用规则兜底回复。"
                fallback_state = {**state, "decision": {"line_goal": fallback_goal}}
                decision = normalize_decision({
                    "intent": "chat",
                    "action": "chat",
                    "emotion": "calm",
                    "line_goal": fallback_goal,
                    "line": _rule_based_reply(fallback_state),
                    "relationship_delta": {"familiarity": 1},
                    "state_delta": {},
                    "memory_tags": [],
                })
                return {
                    "decision": decision,
                    "intent": decision["intent"],
                    "action": decision["action"],
                }

            elapsed = time.time() - start_time
            logger.info("[decide_action] LLM 耗时 %.2fs, 返回(前300字): %s", elapsed, (response or "")[:300])

            parsed_decision = extract_json_object(response)
            if not parsed_decision:
                parsed_decision = {
                    "intent": "chat",
                    "action": "chat",
                    "emotion": "calm",
                    "line_goal": "LLM 输出格式异常，使用规则兜底回复。",
                    "line": _rule_based_reply(state),
                    "relationship_delta": {"familiarity": 0},
                    "state_delta": {},
                    "memory_tags": [],
                }

            decision = normalize_decision(parsed_decision)
            return {
                "decision": decision,
                "intent": decision["intent"],
                "action": decision["action"],
            }

        def validate_action(state: NpcGraphState) -> Dict[str, Any]:
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

        def generate_line(state: NpcGraphState) -> Dict[str, Any]:
            decision = state.get("decision", {})
            if decision.get("line"):
                return {"npc_reply": str(decision["line"]).strip()}

            if llm_service is None:
                return {"npc_reply": _rule_based_reply(state)}

            prompt = json.dumps(
                {
                    "task": "根据决策生成 NPC 最终台词，只输出台词文本。",
                    "player_message": state["player_message"],
                    "decision": decision,
                    "world_state": state.get("world_state"),
                },
                ensure_ascii=False,
                default=str,
            )
            reply, _ = llm_service.chat(
                prompt=prompt,
                history=[],
                system_prompt="你是 NPC 台词生成模块。",
            )
            return {"npc_reply": reply.strip()}

        def apply_state_change(state: NpcGraphState) -> Dict[str, Any]:
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

            logger.debug("[apply_state_change] rel_result=%s", rel_result)
            return {
                "relationship_delta": relationship_delta,
                "state_delta": state_delta,
            }

        def update_player_profile(state: NpcGraphState) -> Dict[str, Any]:
            text = str(state.get("player_message") or "")
            if not text.strip():
                return {}

            if llm_service is None:
                return {}

            prompt = json.dumps({
                "task": (
                    "从对话中提取玩家画像更新。"
                    "只提取玩家明确表达的偏好或自身事实。"
                    "返回 JSON 对象，字段名对应画像字段，无更新则返回 {}。"
                ),
                "player_message": text,
                "current_profile": state.get("player_profile"),
                "updatable_fields": [
                    "taste_preferences", "common_visit_periods", "frequent_npc_ids",
                    "consumption_habits", "personality_tendency", "known_events",
                    "npc_subjective_notes",
                ],
            }, ensure_ascii=False, default=str)

            try:
                response, _ = llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt="你是玩家画像提取模块。只输出 JSON 对象，或 {}。",
                )
                patch = extract_json_object(response)
                profile_patch = patch.get("profile_patch", patch) if patch else {}
            except Exception:
                logger.exception("[update_player_profile] LLM 提取失败")
                return {}

            if not profile_patch:
                return {}

            result = tools.invoke_tool("update_player_profile", {
                "player_id": state["player_id"],
                "profile_patch": profile_patch,
            })
            logger.info("[update_player_profile] result=%s", result)
            return {}

        def write_memory(state: NpcGraphState) -> Dict[str, Any]:
            now = datetime.now(timezone.utc).isoformat()
            decision = state.get("decision", {})
            emotion = decision.get("emotion", "neutral")
            player_id = state["player_id"]
            npc_id = state["npc_id"]
            session_id = state["session_id"]

            tools.invoke_tool("write_player_message", {
                "session_id": session_id,
                "player_id": player_id,
                "npc_id": npc_id,
                "message": state["player_message"],
            })
            tools.invoke_tool("write_npc_reply", {
                "session_id": session_id,
                "player_id": player_id,
                "npc_id": npc_id,
                "content": state["npc_reply"],
                "intent": state.get("intent"),
                "action": state.get("action"),
                "emotion": emotion,
                "relationship_delta": state.get("relationship_delta", {}),
                "state_delta": state.get("state_delta", {}),
                "reply_started_at": state.get("reply_started_at"),
            })

            player_mem_result = tools.invoke_tool("append_short_term_message", {
                "player_id": player_id,
                "npc_id": npc_id,
                "message": {
                    "role": "player",
                    "content": state["player_message"],
                    "emotion": "neutral",
                    "session_id": session_id,
                    "timestamp": now,
                },
                "async_archive": False,
            })
            _archive_short_memory_if_needed(
                tools, llm_service, player_id, npc_id, player_mem_result,
            )

            npc_mem_result = tools.invoke_tool("append_short_term_message", {
                "player_id": player_id,
                "npc_id": npc_id,
                "message": {
                    "role": "npc",
                    "content": state["npc_reply"],
                    "emotion": emotion,
                    "session_id": session_id,
                    "intent": state.get("intent"),
                    "action": state.get("action"),
                    "timestamp": now,
                },
                "async_archive": True,
            })
            _archive_short_memory_if_needed(
                tools, llm_service, player_id, npc_id, npc_mem_result,
            )

            return {}

        # Build graph
        graph = StateGraph(NpcGraphState, input=NpcInputState, output=NpcOutputState)

        graph.add_node("load_context", load_context)
        graph.add_node("retrieve_memories", retrieve_memories)
        graph.add_node("decide_action", decide_action)
        graph.add_node("validate_action", validate_action)
        graph.add_node("generate_line", generate_line)
        graph.add_node("apply_state_change", apply_state_change)
        graph.add_node("update_player_profile", update_player_profile)
        graph.add_node("write_memory", write_memory)

        for source, target in NPC_GRAPH_EDGES:
            graph.add_edge(source, target)

        return graph.compile()

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
            "reply_started_at": time.perf_counter(),
        }
        result = self.graph.invoke(state)
        reply_latency_ms = int(round((time.perf_counter() - state["reply_started_at"]) * 1000))

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
            "reply_latency_ms": reply_latency_ms,
        }


# ---------- Graph description / visualization ----------

def describe_npc_graph() -> str:
    lines = ["图节点："]
    lines.extend([f"- {node}" for node in NPC_GRAPH_NODES])
    lines.append("\n边：")
    lines.extend([f"{source} -> {target}" for source, target in NPC_GRAPH_EDGES])
    return "\n".join(lines)


def render_npc_graph(output_path: str = "npc_graph") -> str:
    from graphviz import Digraph

    dot = Digraph("npc_graph", format="png")
    dot.attr(rankdir="LR")

    for node_name in NPC_GRAPH_NODES:
        dot.node(node_name, node_name, shape="box")

    dot.node(str(START), "START", shape="oval")
    dot.node(str(END), "END", shape="oval")

    for source, target in NPC_GRAPH_EDGES:
        dot.edge(str(source), str(target))

    return dot.render(output_path, cleanup=True)


if __name__ == "__main__":
    import time
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )

    from agent.agent_tools import NpcAgentTools
    from agent.llm_service import LlmService

    tools = NpcAgentTools()
    llm = LlmService.getLLM()

    agent = NpcGraphAgent(tools=tools, llm_service=llm)

    start_time = time.time()
    result = agent.handle_message(
        session_id="test_session_001",
        player_id="test_player_001",
        npc_id="barista_001",
        player_message="你好，今天有什么推荐吗？",
    )
    elapsed = time.time() - start_time

    print("\n========== NPC 回复测试 ==========")
    print(f"耗时: {elapsed:.2f}s")
    print(f"NPC 回复: {result.get('reply')}")
    print(f"意图: {result.get('intent')}")
    print(f"动作: {result.get('action')}")
    print(f"决策: {json.dumps(result.get('decision', {}), ensure_ascii=False, indent=2)}")
    print(f"关系变化: {result.get('relationship_delta')}")
    print("===================================\n")

    tools.close()
