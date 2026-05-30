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
            self._conditional_edges = {}
            self._input = input
            self._output = output

        def add_node(self, name, handler):
            self._nodes[name] = handler

        def add_edge(self, source, target):
            self._edges.append((source, target))

        def add_conditional_edges(self, source, router, mapping):
            self._conditional_edges[source] = (router, mapping)

        def compile(self):
            return self

        def invoke(self, state):
            result = dict(state)
            current = graph_module.START
            while True:
                if current == graph_module.END:
                    break

                # 按显式边推进测试图，模拟 LangGraph 的路由语义。
                next_nodes = [target for source, target in self._edges if source == current]
                if not next_nodes:
                    break

                current = next_nodes[0]
                if current == graph_module.END:
                    break

                handler = self._nodes[current]
                update = handler(result)
                if update:
                    result.update(update)

                if current in self._conditional_edges:
                    router, mapping = self._conditional_edges[current]
                    route_key = router(result)
                    current = mapping[route_key]
                    if current == graph_module.END:
                        break
                    handler = self._nodes[current]
                    update = handler(result)
                    if update:
                        result.update(update)
            # Only return output fields
            if self._output:
                output_fields = (
                    set(self._output.__annotations__.keys())
                    if hasattr(self._output, '__annotations__')
                    else set(result.keys())
                )
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
        if tool_name == "get_player_relationships":
            return {
                "ok": True,
                "tool_name": tool_name,
                "relationships": [
                    {
                        "id": "rel-row-001",
                        "target_id": "brother_lu_qingya",
                        "trust": 6,
                        "familiarity": 7,
                        "fondness": 4,
                        "dislike": 0,
                        "stress": 1,
                        "created_at": "2026-05-20T10:00:00",
                        "updated_at": "2026-05-20T10:01:00",
                    }
                ],
            }
        if tool_name == "get_npc_state" and (arguments or {}).get("npc_id") == "__list_all__":
            return {
                "ok": True,
                "tool_name": tool_name,
                "npcs": [
                    {
                        "id": "npc-row-001",
                        "npc_id": "brother_lu_qingya",
                        "name": "陆青崖",
                        "role": "玩家的师兄弟",
                        "background": "玩家常称其为陆师兄。",
                        "create_time": "2026-05-20T10:00:00",
                        "update_time": "2026-05-20T10:01:00",
                    }
                ],
                "count": 1,
            }

        return {"ok": True, "tool_name": tool_name}


class DeferredExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        return {"scheduled": True}


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


class SequenceFakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, prompt, history=None, system_prompt=""):
        self.calls.append({
            "prompt": prompt,
            "history": history or [],
            "system_prompt": system_prompt,
        })
        if not self.responses:
            return "{}", []
        return self.responses.pop(0), []


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
    llm = SequenceFakeLlm([
        '{"need_retrieval": false, "reason": "本测试只验证决策契约"}',
        '{"intent": "ask_lore", "action": "chat", "line": "雪州很冷。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "雪州是什么地方",
    })

    # 第二次模型调用进入决策节点，验证原有决策契约仍然稳定。
    prompt_payload = json.loads(llm.calls[1]["prompt"])
    system_prompt = llm.calls[1]["system_prompt"]
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


def test_react_planner_skips_game_setting_when_not_needed(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        '{"need_retrieval": false, "reason": "普通寒暄不需要设定"}',
        '{"intent": "greet", "action": "chat", "line": "早，风雪还没停。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "早啊",
    })

    game_setting_calls = [
        args for name, args in tools.calls
        if name == "get_game_setting_context"
    ]
    planner_payload = json.loads(llm.calls[0]["prompt"])
    assert game_setting_calls == []
    assert planner_payload["task"] == "plan_game_setting_retrieval"
    assert result.get("npc_reply") == "早，风雪还没停。"


def test_react_planner_retrieves_game_setting_with_model_query(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        json.dumps({
            "need_retrieval": True,
            "tool_call": {
                "tool_name": "get_game_setting_context",
                "arguments": {"query": "玄京 雪州 关系", "limit": 4},
            },
            "reason": "玩家询问世界设定事实",
        }, ensure_ascii=False),
        '{"intent": "ask_lore", "action": "chat", "line": "玄京与雪州向来隔着风雪与旧怨。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "玄京和雪州到底是什么关系？",
    })

    game_setting_calls = [
        args for name, args in tools.calls
        if name == "get_game_setting_context"
    ]
    assert game_setting_calls == [{"query": "玄京 雪州 关系", "limit": 4}]


def test_react_planner_rejects_illegal_tool_call(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        json.dumps({
            "need_retrieval": True,
            "tool_call": {
                "tool_name": "web_search",
                "arguments": {"query": "玄京"},
            },
            "reason": "错误地请求外部工具",
        }, ensure_ascii=False),
        '{"intent": "ask_lore", "action": "chat", "line": "这事我不敢乱说。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "玄京是什么？",
    })

    blocked_calls = [
        args for name, args in tools.calls
        if name == "web_search" or name == "get_game_setting_context"
    ]
    assert blocked_calls == []


def test_load_context_includes_npc_directory_and_player_relationships(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=None)
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "master_shen_zhaowei",
        "player_message": "陆师兄要和我一起下山吗？",
    })

    assert result.get("action") == "chat"
    assert ("get_npc_state", {"npc_id": "__list_all__"}) in tools.calls
    assert ("get_player_relationships", {"player_id": "p001"}) in tools.calls


def test_retrieval_planner_prompt_contains_npc_directory(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        json.dumps({
            "need_retrieval": True,
            "tool_call": {
                "tool_name": "get_game_setting_context",
                "arguments": {"query": "陆青崖 陆师兄 下山", "limit": 5},
            },
            "reason": "玩家询问其他 NPC 设定",
        }, ensure_ascii=False),
        '{"intent": "ask_lore", "action": "chat", "line": "我先确认陆青崖的安排。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "master_shen_zhaowei",
        "player_message": "陆师兄要和我一起下山吗？",
    })

    planner_payload = json.loads(llm.calls[0]["prompt"])
    assert planner_payload["context"]["npc_directory"]["npcs"][0]["name"] == "陆青崖"
    assert (
        planner_payload["context"]["player_npc_relationships"]["relationships"][0]
        ["target_id"] == "brother_lu_qingya"
    )


def test_decision_prompt_forbids_answering_without_setting_evidence(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        '{"need_retrieval": false, "reason": "只验证决策约束"}',
        '{"intent": "ask_lore", "action": "chat", "line": "我不确定。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "master_shen_zhaowei",
        "player_message": "陆师兄要和我一起下山吗？",
    })

    decision_payload = json.loads(llm.calls[1]["prompt"])
    system_prompt = llm.calls[1]["system_prompt"]
    assert "npc_directory" in decision_payload["context"]
    assert "player_npc_relationships" in decision_payload["context"]
    assert "不能编造" in system_prompt
    assert "游戏设定或 NPC 设定" in system_prompt


def test_llm_prompts_strip_database_metadata_fields(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        '{"need_retrieval": false, "reason": "只验证上下文压缩"}',
        '{"intent": "chat", "action": "chat", "line": "我知道了。"}',
    ])

    agent = npc_graph_agent.NpcGraphAgent(tools=tools, llm_service=llm)
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "master_shen_zhaowei",
        "player_message": "陆师兄是谁？",
    })

    # 所有进入 LLM 的 prompt 都不应携带数据库行元数据。
    forbidden_fields = {"id", "created_at", "updated_at", "create_time", "update_time"}
    for call in llm.calls:
        payload = json.loads(call["prompt"])
        serialized = json.dumps(payload, ensure_ascii=False)
        for field_name in forbidden_fields:
            assert f'"{field_name}"' not in serialized


def test_profile_extraction_prompt_strips_database_metadata(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"taste_preferences": {"likes_tea": true}}')
    messages = [
        {
            "id": "msg-row-001",
            "role": "player",
            "content": "我喜欢清茶。",
            "created_at": "2026-05-20T10:00:00",
            "update_time": "2026-05-20T10:01:00",
        }
    ]

    npc_graph_agent._extract_player_profile_patch_from_messages(
        tools=tools,
        llm_service=llm,
        player_id="p001",
        current_profile={
            "id": "player-row-001",
            "nickname": "少侠",
            "updated_at": "2026-05-20T10:02:00",
        },
        messages=messages,
    )

    payload = json.loads(llm.calls[0]["prompt"])
    serialized = json.dumps(payload, ensure_ascii=False)
    assert '"id"' not in serialized
    assert '"created_at"' not in serialized
    assert '"updated_at"' not in serialized
    assert '"update_time"' not in serialized


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


def test_write_memory_schedules_overflow_archive_without_blocking(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = SequenceFakeLlm([
        '{"need_retrieval": false, "reason": "本测试只验证后台归档"}',
        '{"intent": "chat", "action": "chat", "line": "我记住了。", "emotion": "pleased"}',
        '{"taste_preferences": {"likes_latte": true}}',
    ])
    executor = DeferredExecutor()
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

    agent = npc_graph_agent.NpcGraphAgent(
        tools=tools,
        llm_service=llm,
        background_executor=executor,
    )
    agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我经常点拿铁",
    })

    # 返回前只调度后台任务，不执行长期记忆写入，也不更新画像。
    assert len(executor.submissions) == 1
    assert [
        name for name, _ in tools.calls
        if name in {"create_long_term_memory_from_messages", "update_player_profile"}
    ] == []

    fn, args, kwargs = executor.submissions[0]
    fn(*args, **kwargs)

    long_calls = [
        args for name, args in tools.calls
        if name == "create_long_term_memory_from_messages"
    ]
    profile_calls = [
        args for name, args in tools.calls
        if name == "update_player_profile"
    ]
    trim_calls = [
        args for name, args in tools.calls
        if name == "trim_short_term_memory"
    ]
    assert long_calls[0]["messages"] == [{"role": "player", "content": "旧消息"}]
    assert long_calls[0]["llm_service"] is llm
    profile_prompt = json.loads(llm.calls[-1]["prompt"])
    assert profile_prompt["source_messages"] == [{"role": "player", "content": "旧消息"}]
    assert "taste_preferences" in profile_calls[0]["profile_patch"]
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


def test_graph_topology_has_react_rag_branch(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    nodes = npc_graph_agent.NPC_GRAPH_NODES
    edges = npc_graph_agent.NPC_GRAPH_EDGES

    assert len(nodes) == 10
    assert nodes[0] == "load_context"
    assert nodes[-1] == "write_memory"
    assert "plan_game_setting_retrieval" in nodes
    assert "retrieve_game_setting" in nodes
    assert ("retrieve_memories", "plan_game_setting_retrieval") in edges
    assert ("retrieve_game_setting", "decide_action") in edges
    assert npc_graph_agent.NPC_GRAPH_CONDITIONAL_EDGES == {
        "plan_game_setting_retrieval": {
            "retrieve": "retrieve_game_setting",
            "skip": "decide_action",
        }
    }


def test_update_player_profile_waits_for_short_memory_archive(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"taste_preferences": {"likes_latte": true}}')
    executor = DeferredExecutor()

    agent = npc_graph_agent.NpcGraphAgent(
        tools=tools,
        llm_service=llm,
        background_executor=executor,
    )
    result = agent.graph.invoke({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "npc_001",
        "player_message": "我喜欢甜一点的拿铁",
    })

    # 没有触发短期记忆归档时，不单独阻塞式更新玩家画像。
    profile_calls = [args for name, args in tools.calls if name == "update_player_profile"]
    assert profile_calls == []
    assert executor.submissions == []


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
