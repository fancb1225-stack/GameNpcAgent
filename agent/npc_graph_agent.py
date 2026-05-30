from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
LLM_CONTEXT_OMIT_FIELDS = {"id", "created_at", "updated_at", "create_time", "update_time"}

_DEFAULT_BACKGROUND_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="npc_background",
)

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
8. 涉及游戏设定或 NPC 设定的事实，只能依据 context 中的 npc_directory、
   player_npc_relationships、game_setting_context、记忆和当前状态。
9. 不能编造不存在的人名、身份、关系、地点、同行安排或任务；缺少依据时必须明确说不确定。
""".strip()

GAME_SETTING_RETRIEVAL_SYSTEM_PROMPT = """
你是 GameNpcFrame 的 ReAct 检索规划节点。

你的任务：
1. 判断回复玩家消息前，是否必须查询游戏设定知识库。
2. 如果需要检索，只能输出 get_game_setting_context 工具调用 JSON。
3. 如果不需要检索，明确输出 need_retrieval=false。

需要检索的情况：
- 玩家询问世界观、地点、势力、历史、人物背景、规则、专有名词等设定事实。
- 玩家要求 NPC 解释与游戏设定相关的来源、关系、传闻或背景。

不需要检索的情况：
- 普通寒暄、情绪回应、闲聊、承接上下文即可回答的问题。
- 玩家询问当前 NPC/玩家关系、短期记忆或当前场景中已经给出的信息。

输出要求：
- 只输出一个 JSON object，不要 Markdown，不要解释。
- 需要检索时使用格式：
  {
      "need_retrieval": true,
      "tool_call": {
          "tool_name": "get_game_setting_context",
          "arguments": {"query": "检索关键词", "limit": 5}
      },
      "reason": "简短原因"
  }
- 不需要检索时使用格式：
  {"need_retrieval": false, "reason": "简短原因"}
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
    "plan_game_setting_retrieval",
    "retrieve_game_setting",
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
    ("retrieve_memories", "plan_game_setting_retrieval"),
    ("retrieve_game_setting", "decide_action"),
    ("decide_action", "validate_action"),
    ("validate_action", "generate_line"),
    ("generate_line", "apply_state_change"),
    ("apply_state_change", "update_player_profile"),
    ("update_player_profile", "write_memory"),
    ("write_memory", END),
]

NPC_GRAPH_CONDITIONAL_EDGES = {
    "plan_game_setting_retrieval": {
        "retrieve": "retrieve_game_setting",
        "skip": "decide_action",
    }
}


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
    npc_directory: Dict[str, Any]
    player_npc_relationships: Dict[str, Any]
    world_state: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    long_term_memories: List[Dict[str, Any]]
    game_setting_context: Dict[str, Any]
    game_setting_retrieval_plan: Dict[str, Any]
    game_setting_retrieval_error: Optional[str]
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


def compact_llm_context(value: Any) -> Any:
    """递归移除 LLM 不需要的数据库元字段，降低上下文体积。"""

    # 字典类型按键过滤，只保留对模型判断有用的业务字段。
    if isinstance(value, dict):
        return {
            key: compact_llm_context(item)
            for key, item in value.items()
            if str(key) not in LLM_CONTEXT_OMIT_FIELDS
        }

    # 列表/元组保持原顺序，对内部对象继续压缩。
    if isinstance(value, (list, tuple)):
        return [compact_llm_context(item) for item in value]

    return value


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


def normalize_game_setting_retrieval_plan(data: Any) -> Dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    need_retrieval = bool(data.get("need_retrieval", False))
    reason = str(data.get("reason") or "").strip()

    # 不需要检索时，保持显式计划，方便图路由和后续异步化观察。
    if not need_retrieval:
        return {
            "need_retrieval": False,
            "tool_call": None,
            "reason": reason or "模型判断无需检索游戏设定",
            "error": None,
        }

    tool_call = data.get("tool_call") if isinstance(data.get("tool_call"), dict) else {}
    tool_name = str(tool_call.get("tool_name") or "").strip()
    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    query = str(arguments.get("query") or "").strip()

    # ReAct 工具边界只允许游戏设定检索，拒绝模型编造或越权工具调用。
    if tool_name != "get_game_setting_context":
        return {
            "need_retrieval": False,
            "tool_call": None,
            "reason": reason,
            "error": f"非法工具调用: {tool_name or 'missing'}",
        }
    if not query:
        return {
            "need_retrieval": False,
            "tool_call": None,
            "reason": reason,
            "error": "游戏设定检索 query 为空",
        }

    try:
        limit = int(arguments.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5

    return {
        "need_retrieval": True,
        "tool_call": {
            "tool_name": "get_game_setting_context",
            "arguments": {
                "query": query,
                "limit": max(1, min(10, limit)),
            },
        },
        "reason": reason or "模型判断需要检索游戏设定",
        "error": None,
    }


def _build_game_setting_retrieval_prompt(state: NpcGraphState) -> str:
    payload = {
        "task": "plan_game_setting_retrieval",
        "player_message": state.get("player_message", ""),
        "context": {
            "npc_state": state.get("npc_state"),
            "player_profile": state.get("player_profile"),
            "relationship": state.get("relationship"),
            "npc_directory": state.get("npc_directory"),
            "player_npc_relationships": state.get("player_npc_relationships"),
            "world_state": state.get("world_state"),
            "short_term_memory": state.get("short_term_memory"),
            "long_term_memories": state.get("long_term_memories"),
        },
        "available_tool": {
            "tool_name": "get_game_setting_context",
            "arguments_schema": {
                "query": "用于检索游戏设定的中文关键词或问题",
                "limit": "1 到 10 的整数，默认 5",
            },
        },
    }
    return json.dumps(compact_llm_context(payload), ensure_ascii=False, default=str)


def _build_decision_prompt(state: NpcGraphState) -> str:
    payload = {
        "task": "npc_dialogue_decision",
        "player_message": state.get("player_message", ""),
        "context": {
            "npc_state": state.get("npc_state"),
            "player_profile": state.get("player_profile"),
            "relationship": state.get("relationship"),
            "npc_directory": state.get("npc_directory"),
            "player_npc_relationships": state.get("player_npc_relationships"),
            "world_state": state.get("world_state"),
            "game_setting_context": state.get("game_setting_context"),
            "game_setting_retrieval_plan": state.get("game_setting_retrieval_plan"),
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
    return json.dumps(compact_llm_context(payload), ensure_ascii=False, default=str)


def _rule_based_reply(state: NpcGraphState) -> str:
    player_message = state.get("player_message", "")
    decision = state.get("decision", {})
    line_goal = decision.get("line_goal") or "自然回应玩家"
    return f"我听到了：{player_message}。{line_goal}"


def _extract_player_profile_patch_from_messages(
    tools: Any,
    llm_service: Any,
    player_id: str,
    current_profile: Dict[str, Any] | None,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if llm_service is None or not messages:
        return {}

    # 用与长期记忆相同的近期对话片段提取玩家画像，避免只看单轮消息造成误判。
    prompt = json.dumps(compact_llm_context({
        "task": (
            "从近期对话中提取玩家画像更新。"
            "只提取玩家明确表达的偏好、习惯、自身事实或稳定倾向。"
            "返回 JSON 对象，字段名对应画像字段，无更新则返回 {}。"
        ),
        "source_messages": messages,
        "current_profile": current_profile,
        "updatable_fields": [
            "taste_preferences", "common_visit_periods", "frequent_npc_ids",
            "consumption_habits", "personality_tendency", "known_events",
            "npc_subjective_notes",
        ],
    }), ensure_ascii=False, default=str)

    try:
        response, _ = llm_service.chat(
            prompt=prompt,
            history=[],
            system_prompt="你是玩家画像提取模块。只输出 JSON 对象，或 {}。",
        )
        patch = extract_json_object(response)
        profile_patch = patch.get("profile_patch", patch) if patch else {}
    except Exception:
        logger.exception("[update_player_profile] 后台画像提取失败")
        return {}

    if not profile_patch:
        return {}

    # 画像工具只接收画像补丁，近期消息只用于 LLM 提取阶段。
    result = tools.invoke_tool("update_player_profile", {
        "player_id": player_id,
        "profile_patch": profile_patch,
    })
    logger.info("[update_player_profile] async result=%s", result)
    return result


def _archive_memory_and_update_profile(
    tools: Any,
    llm_service: Any,
    player_id: str,
    npc_id: str,
    current_profile: Dict[str, Any] | None,
    archive_messages: List[Dict[str, Any]],
    append_result: Dict[str, Any],
) -> Dict[str, Any] | None:
    # 后台任务内部并发执行长期记忆总结和画像更新，避免阻塞玩家消息返回。
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="npc_archive") as executor:
        long_future = executor.submit(
            tools.invoke_tool,
            "create_long_term_memory_from_messages",
            {
                "player_id": player_id,
                "npc_id": npc_id,
                "messages": archive_messages,
                "llm_service": llm_service,
            },
        )
        profile_future = executor.submit(
            _extract_player_profile_patch_from_messages,
            tools,
            llm_service,
            player_id,
            current_profile,
            archive_messages,
        )

        long_result = long_future.result()
        profile_result = profile_future.result()

    if long_result.get("ok"):
        trim_result = tools.invoke_tool("trim_short_term_memory", {
            "player_id": player_id,
            "npc_id": npc_id,
            "remove_count": append_result.get("archive_message_count", len(archive_messages)),
        })
        return {
            "long_term_memory": long_result,
            "player_profile": profile_result,
            "trim_short_term_memory": trim_result,
        }

    return {
        "long_term_memory": long_result,
        "player_profile": profile_result,
    }


def _schedule_archive_short_memory_if_needed(
    tools: Any,
    llm_service: Any,
    background_executor: Any,
    player_id: str,
    npc_id: str,
    current_profile: Dict[str, Any] | None,
    append_result: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not append_result.get("needs_archive"):
        return None

    archive_messages = append_result.get("archive_messages") or []
    if not archive_messages:
        return None

    # 只调度后台任务，长期记忆和画像更新不参与本轮消息返回链路。
    future = background_executor.submit(
        _archive_memory_and_update_profile,
        tools,
        llm_service,
        player_id,
        npc_id,
        current_profile,
        archive_messages,
        append_result,
    )
    return {"scheduled": True, "future": future}


# ---------- Graph agent ----------

class NpcGraphAgent:
    def __init__(
        self,
        tools,
        llm_service=None,
        background_executor=None,
        defer_memory_write: bool = False,
    ):
        self.tools = tools
        self.llm_service = llm_service
        self.background_executor = background_executor or _DEFAULT_BACKGROUND_EXECUTOR
        self.defer_memory_write = defer_memory_write
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
                "npc_directory": tools.invoke_tool("get_npc_state", {
                    "npc_id": "__list_all__",
                }),
                "player_npc_relationships": tools.invoke_tool("get_player_relationships", {
                    "player_id": state["player_id"],
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

            return {
                "short_term_memory": short_memory,
                "long_term_memories": long_memory.get("memories", []),
            }

        def plan_game_setting_retrieval(state: NpcGraphState) -> Dict[str, Any]:
            if llm_service is None:
                plan = normalize_game_setting_retrieval_plan({
                    "need_retrieval": False,
                    "reason": "无 LLM 服务，跳过游戏设定检索规划",
                })
                return {"game_setting_retrieval_plan": plan}

            prompt = _build_game_setting_retrieval_prompt(state)
            try:
                response, _ = llm_service.chat(
                    prompt=prompt,
                    history=[],
                    system_prompt=GAME_SETTING_RETRIEVAL_SYSTEM_PROMPT,
                )
            except Exception:
                logger.exception("[plan_game_setting_retrieval] LLM 调用异常")
                plan = normalize_game_setting_retrieval_plan({
                    "need_retrieval": False,
                    "reason": "检索规划失败，跳过游戏设定检索",
                })
                return {
                    "game_setting_retrieval_plan": plan,
                    "game_setting_retrieval_error": "检索规划 LLM 调用失败",
                }

            plan = normalize_game_setting_retrieval_plan(extract_json_object(response))
            result: Dict[str, Any] = {"game_setting_retrieval_plan": plan}
            if plan.get("error"):
                result["game_setting_retrieval_error"] = plan["error"]
            return result

        def route_game_setting_retrieval(state: NpcGraphState) -> str:
            plan = state.get("game_setting_retrieval_plan", {})
            return "retrieve" if plan.get("need_retrieval") else "skip"

        def retrieve_game_setting(state: NpcGraphState) -> Dict[str, Any]:
            plan = state.get("game_setting_retrieval_plan", {})
            tool_call = plan.get("tool_call") if isinstance(plan.get("tool_call"), dict) else {}
            arguments = (
                tool_call.get("arguments")
                if isinstance(tool_call.get("arguments"), dict)
                else {}
            )

            # 工具调用已经在规划阶段校验，这里只执行白名单内的检索动作。
            if tool_call.get("tool_name") != "get_game_setting_context":
                return {
                    "game_setting_context": {},
                    "game_setting_retrieval_error": "缺少合法的游戏设定检索工具调用",
                }

            game_setting = tools.invoke_tool("get_game_setting_context", {
                "query": arguments.get("query", ""),
                "limit": arguments.get("limit", 5),
            })
            return {"game_setting_context": game_setting}

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
            logger.info(
                "[decide_action] LLM 耗时 %.2fs, 返回(前300字): %s",
                elapsed,
                (response or "")[:300],
            )

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
                compact_llm_context({
                    "task": "根据决策生成 NPC 最终台词，只输出台词文本。",
                    "player_message": state["player_message"],
                    "decision": decision,
                    "world_state": state.get("world_state"),
                }),
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
            # 画像更新已迁移到短期记忆归档后台任务，保证与长期记忆使用同一批近期对话。
            return {}

        def write_memory(state: NpcGraphState) -> Dict[str, Any]:
            if self.defer_memory_write:
                # Director 场景下把记忆写入放入后台，便于并发执行下一步动作判断。
                self.background_executor.submit(write_memory_now, dict(state))
                return {"memory_write_scheduled": True}

            return write_memory_now(state)

        def write_memory_now(state: NpcGraphState) -> Dict[str, Any]:
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
            _schedule_archive_short_memory_if_needed(
                tools,
                llm_service,
                self.background_executor,
                player_id,
                npc_id,
                state.get("player_profile"),
                player_mem_result,
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
            _schedule_archive_short_memory_if_needed(
                tools,
                llm_service,
                self.background_executor,
                player_id,
                npc_id,
                state.get("player_profile"),
                npc_mem_result,
            )

            return {}

        # Build graph
        graph = StateGraph(NpcGraphState, input=NpcInputState, output=NpcOutputState)

        graph.add_node("load_context", load_context)
        graph.add_node("retrieve_memories", retrieve_memories)
        graph.add_node("plan_game_setting_retrieval", plan_game_setting_retrieval)
        graph.add_node("retrieve_game_setting", retrieve_game_setting)
        graph.add_node("decide_action", decide_action)
        graph.add_node("validate_action", validate_action)
        graph.add_node("generate_line", generate_line)
        graph.add_node("apply_state_change", apply_state_change)
        graph.add_node("update_player_profile", update_player_profile)
        graph.add_node("write_memory", write_memory)

        for source, target in NPC_GRAPH_EDGES:
            graph.add_edge(source, target)
        graph.add_conditional_edges(
            "plan_game_setting_retrieval",
            route_game_setting_retrieval,
            NPC_GRAPH_CONDITIONAL_EDGES["plan_game_setting_retrieval"],
        )

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
    lines.append("\n条件边：")
    for source, mapping in NPC_GRAPH_CONDITIONAL_EDGES.items():
        lines.extend([f"{source} --{route}--> {target}" for route, target in mapping.items()])
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
    for source, mapping in NPC_GRAPH_CONDITIONAL_EDGES.items():
        for route, target in mapping.items():
            dot.edge(str(source), str(target), label=route, style="dashed")

    return dot.render(output_path, cleanup=True)


if __name__ == "__main__":
    # import time
    # import logging
    #
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    #     force=True,
    # )
    #
    # from agent.agent_tools import NpcAgentTools
    # from agent.llm_service import LlmService
    #
    # tools = NpcAgentTools()
    # llm = LlmService.getLLM()
    #
    # agent = NpcGraphAgent(tools=tools, llm_service=llm)
    #
    # start_time = time.time()
    # result = agent.handle_message(
    #     session_id="test_session_001",
    #     player_id="test_player_001",
    #     npc_id="barista_001",
    #     player_message="你好，今天有什么推荐吗？",
    # )
    # elapsed = time.time() - start_time
    #
    # print("\n========== NPC 回复测试 ==========")
    # print(f"耗时: {elapsed:.2f}s")
    # print(f"NPC 回复: {result.get('reply')}")
    # print(f"意图: {result.get('intent')}")
    # print(f"动作: {result.get('action')}")
    # print(f"决策: {json.dumps(result.get('decision', {}), ensure_ascii=False, indent=2)}")
    # print(f"关系变化: {result.get('relationship_delta')}")
    # print("===================================\n")
    #
    # tools.close()

    render_npc_graph("npc_graph")
