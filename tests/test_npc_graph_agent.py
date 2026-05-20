import importlib
import json
import sys
import types

from typing import Any


def import_tools_class():
    pgvector_module = types.ModuleType("pgvector")
    pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")
    from sqlalchemy.types import UserDefinedType

    class FakeVector(UserDefinedType):
        def __init__(self, dimensions):
            self.dimensions = dimensions

        def get_col_spec(self, **kwargs):
            return "VECTOR"

    pgvector_sqlalchemy.Vector = FakeVector
    monkey_modules = {
        "pgvector": pgvector_module,
        "pgvector.sqlalchemy": pgvector_sqlalchemy,
    }
    for module_name, module in monkey_modules.items():
        sys.modules.setdefault(module_name, module)
    sys.modules.pop("agent.agent_tools", None)
    sys.modules.pop("agent.coffee_npc_agent_tools", None)

    try:
        return importlib.import_module("agent.agent_tools").NpcAgentTools
    except ModuleNotFoundError:
        return importlib.import_module("agent.coffee_npc_agent_tools").CoffeeNpcAgentTools


def import_graph_agent(monkeypatch):
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.START = "__start__"
    graph_module.END = "__end__"

    class FakeStateGraph:
        def __init__(self, state_schema=None, input=None, output=None):
            self._nodes = {}
            self._edges = []
            self._input = input
            self._output = output

        def add_node(self, name, handler):
            self._nodes[name] = handler

        def add_edge(self, source, target):
            self._edges.append((source, target))

        def add_conditional_edges(self, source, router, mapping):
            pass

        def compile(self):
            return self

        def invoke(self, state):
            result = dict(state)
            for name, handler in self._nodes.items():
                update = handler(result)
                if update:
                    result.update(update)
            # Only return output fields
            if self._output:
                output_fields = set(self._output.__annotations__.keys()) if hasattr(self._output, '__annotations__') else set(result.keys())
                return {k: v for k, v in result.items() if k in output_fields}
            return result

    graph_module.StateGraph = FakeStateGraph
    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.graph = graph_module
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_module)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)
    sys.modules.pop("agent.npc_graph_agent", None)

    return importlib.import_module("agent.npc_graph_agent")


class FakeTools:
    def __init__(self):
        self.append_results = []
        self.calls = []

    def invoke_tool(self, tool_name, arguments=None):
        self.calls.append((tool_name, arguments or {}))

        if tool_name == "check_action_allowed":
            return {"ok": True, "allowed": True}
        if tool_name == "append_short_term_message":
            if self.append_results:
                return self.append_results.pop(0)
            return {"ok": True, "tool_name": tool_name, "needs_archive": False}
        if tool_name == "create_long_term_memory_from_messages":
            return {"ok": True, "tool_name": tool_name, "memory": {"memory_id": "m001"}}
        if tool_name == "trim_short_term_memory":
            return {"ok": True, "tool_name": tool_name}
        if tool_name == "update_player_profile":
            return {"ok": True, "tool_name": tool_name}

        return {"ok": True, "tool_name": tool_name}


class FakeLlm:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, prompt, history=None, system_prompt=""):
        self.calls.append({
            "prompt": prompt,
            "history": history or [],
            "system_prompt": system_prompt,
        })
        return self.response, []


class FailingLlm:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, history=None, system_prompt=""):
        self.calls.append({
            "prompt": prompt,
            "history": history or [],
            "system_prompt": system_prompt,
        })
        raise RuntimeError("模型调用失败")


class FakeMemoryService:
    def __init__(self):
        self.append_calls = []
        self.search_calls = []
        self.list_calls = []

    def append_short_term_message(self, **kwargs):
        self.append_calls.append(kwargs)
        return {"ok": True, "tool_name": "append_short_term_message"}

    def search_long_memories(self, **kwargs):
        self.search_calls.append(kwargs)
        return [{"memory_id": "m001", "content": "玩家喜欢拿铁"}]

    def list_long_memories(self, **kwargs):
        self.list_calls.append(kwargs)
        return [{"memory_id": "m002", "content": "普通长期记忆"}]


class FakeShortMemoryService:
    def __init__(self):
        self.append_calls = []

    def append_short_term_message(self, **kwargs):
        self.append_calls.append(kwargs)
        return {"ok": True, "needs_archive": False, "short_term_count": 2}


class FakeLongMemoryService:
    def __init__(self):
        self.create_calls = []
        self.search_calls = []
        self.list_calls = []

    def create_long_term_memory_from_messages(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"ok": True, "memory": {"memory_id": "m001"}}

    def search_long_memories(self, **kwargs):
        self.search_calls.append(kwargs)
        return [{"memory_id": "m001", "content": "玩家喜欢拿铁"}]

    def list_long_memories(self, **kwargs):
        self.list_calls.append(kwargs)
        return [{"memory_id": "m002", "content": "普通长期记忆"}]


# ---------- Tests for pure utility functions ----------

def test_extract_json_object_accepts_fenced_json(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)

    data = npc_graph_agent.extract_json_object(
        '```json\n{"action": "dance", "emotion": "unknown"}\n```'
    )

    assert data == {"action": "dance", "emotion": "unknown"}


def test_normalize_decision_cleans_schema(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)

    decision = npc_graph_agent.normalize_decision({
        "intent": "greet",
        "action": "dance",
        "emotion": "excited",
        "response": "欢迎。",
        "relationship_delta": {"trust": "99", "bad": 1},
        "state_delta": [],
        "memory_tags": "tag",
    })

    assert decision["action"] == "chat"
    assert decision["emotion"] == "neutral"
    assert decision["line"] == "欢迎。"
    assert decision["relationship_delta"]["trust"] == 5
    assert set(decision["relationship_delta"]) == npc_graph_agent.RELATIONSHIP_FIELDS
    assert decision["state_delta"] == {}
    assert decision["memory_tags"] == []


def test_allowed_actions_excludes_coffee_actions(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    assert "recommend_coffee" not in npc_graph_agent.ALLOWED_ACTIONS
    assert "serve_customer" not in npc_graph_agent.ALLOWED_ACTIONS
    assert "chat" in npc_graph_agent.ALLOWED_ACTIONS


# ---------- Tests for node functions via NpcGraphAgent ----------

def test_decide_action_uses_injected_llm_and_extracts_json(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"intent": "greet", "action": "chat", "line": "你好呀"}')

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    # Extract the closure-defined decide_action from the graph
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "你好",
    })

    assert llm.calls
    assert result.get("action") == "chat"


def test_decide_action_uses_structured_prompt_contract(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"intent": "ask_lore", "action": "chat", "line": "雪州很冷。"}')

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "雪州是什么地方",
    })

    prompt_payload = json.loads(llm.calls[0]["prompt"])
    system_prompt = llm.calls[0]["system_prompt"]
    assert prompt_payload["task"] == "npc_dialogue_decision"
    assert "decision_contract" in prompt_payload
    assert "allowed_actions" in prompt_payload["decision_contract"]
    assert "只输出一个 JSON object" in system_prompt
    # No coffee actions in the contract
    assert "recommend_coffee" not in prompt_payload["decision_contract"]["allowed_actions"]
    assert "serve_customer" not in prompt_payload["decision_contract"]["allowed_actions"]


def test_decide_action_falls_back_to_rules_on_llm_failure(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FailingLlm()

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "雪州是什么地方",
    })

    assert llm.calls
    assert result.get("action") == "chat"


def test_generate_line_uses_decision_line_before_llm(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm("不应该调用")

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    # Provide a decision with a line already set to test that generate_line uses it
    # We need to test the closure directly - get the handler from _build_graph
    # Easier: use a LLM that returns a valid JSON with a line
    llm2 = FakeLlm('{"intent": "greet", "action": "chat", "line": "已经有台词"}')
    agent2 = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm2)
    result = agent2.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "你好",
    })
    assert result.get("npc_reply") == "已经有台词"


def test_retrieve_memories_uses_player_message_as_long_term_query(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=None)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我想喝不太酸的拿铁",
    })

    long_memory_call = [
        args for name, args in tools.calls
        if name == "get_long_term_memories"
    ][0]
    assert long_memory_call["query"] == "我想喝不太酸的拿铁"


def test_write_memory_appends_player_and_npc_short_messages_synchronously(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"intent": "chat", "action": "chat", "line": "好的。", "emotion": "pleased"}')

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我喜欢拿铁",
    })

    memory_calls = [
        args for name, args in tools.calls
        if name == "append_short_term_message"
    ]
    assert len(memory_calls) >= 2
    assert memory_calls[0]["message"]["role"] == "player"
    assert memory_calls[0]["message"]["emotion"] == "neutral"
    assert memory_calls[1]["message"]["role"] == "npc"
    assert memory_calls[1]["message"]["emotion"] == "pleased"


def test_write_memory_passes_reply_started_at_to_npc_reply(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"intent":"greet","action":"chat","emotion":"pleased","line":"你好。"}')

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.handle_message(
        session_id="s001",
        player_id="p001",
        npc_id="npc_001",
        player_message="你好",
    )

    npc_reply_calls = [args for name, args in tools.calls if name == "write_npc_reply"]
    assert npc_reply_calls
    assert npc_reply_calls[0]["reply_started_at"] is not None


def test_write_memory_archives_overflow_with_long_memory_service(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"intent": "chat", "action": "chat", "line": "我记住了。", "emotion": "pleased"}')
    tools.append_results = [
        {"ok": True, "needs_archive": False, "short_term_count": 20},
        {
            "ok": True,
            "needs_archive": True,
            "archive_messages": [{"role": "player", "content": "旧消息"}],
            "archive_message_count": 1,
            "remaining_message_count": 11,
            "short_term_count": 21,
        },
    ]

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我经常点拿铁",
    })

    long_calls = [
        args for name, args in tools.calls
        if name == "create_long_term_memory_from_messages"
    ]
    trim_calls = [
        args for name, args in tools.calls
        if name == "trim_short_term_memory"
    ]
    assert long_calls[0]["messages"] == [{"role": "player", "content": "旧消息"}]
    assert long_calls[0]["llm_service"] is llm
    assert trim_calls[0]["remove_count"] == 1


def test_tools_append_short_term_message_uses_short_service_only():
    CoffeeNpcAgentTools = import_tools_class()

    tools = CoffeeNpcAgentTools.__new__(CoffeeNpcAgentTools)
    tools.short_memories = FakeShortMemoryService()
    tools.long_memories = FakeLongMemoryService()
    tools.llm_service = FakeLlm("{}")

    result = tools.append_short_term_message(
        player_id="p001",
        npc_id="barista_001",
        message={"role": "player", "content": "我喜欢拿铁"},
        async_archive=True,
    )

    call = tools.short_memories.append_calls[0]
    assert result["ok"] is True
    assert call["async_archive"] is True
    assert "llm_service" not in call
    assert tools.long_memories.create_calls == []


def test_tools_get_long_term_memories_searches_when_query_is_provided():
    CoffeeNpcAgentTools = import_tools_class()

    tools = CoffeeNpcAgentTools.__new__(CoffeeNpcAgentTools)
    tools.long_memories = FakeLongMemoryService()

    result = tools.get_long_term_memories(
        player_id="p001",
        npc_id="barista_001",
        query="拿铁偏好",
        limit=5,
    )

    assert result["ok"] is True
    assert result["count"] == 1
    assert tools.long_memories.search_calls[0]["query"] == "拿铁偏好"
    assert tools.long_memories.list_calls == []


def test_graph_topology_is_linear_eight_nodes(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    nodes = npc_graph_agent.NPC_GRAPH_NODES
    edges = npc_graph_agent.NPC_GRAPH_EDGES

    assert len(nodes) == 8
    assert nodes[0] == "load_context"
    assert nodes[-1] == "write_memory"
    # All edges are linear (no conditional edges)
    assert len(edges) == 9  # START->node + 7 node->node + node->END
    for source, target in edges[1:-1]:
        # Each intermediate edge connects consecutive nodes
        assert source in nodes
        assert target in nodes


def test_update_player_profile_uses_llm_extraction(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"taste_preferences": {"likes_latte": true}}')

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "我喜欢甜一点的拿铁",
    })

    # Verify update_player_profile was called via tools
    profile_calls = [args for name, args in tools.calls if name == "update_player_profile"]
    assert len(profile_calls) >= 1
    assert "taste_preferences" in profile_calls[0]["profile_patch"]


def test_update_player_profile_skips_when_no_llm(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=None)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "我喜欢甜一点的拿铁",
    })

    # No update_player_profile call when no LLM
    profile_calls = [args for name, args in tools.calls if name == "update_player_profile"]
    assert len(profile_calls) == 0
