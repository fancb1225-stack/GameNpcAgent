import importlib
import sys
import types


def import_tools_class():
    # 给测试环境补一个轻量 pgvector 替身，避免工具层导入时连接真实扩展。
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
    sys.modules.pop("agent.coffee_npc_agent_tools", None)

    return importlib.import_module("agent.coffee_npc_agent_tools").CoffeeNpcAgentTools


def import_graph_agent(monkeypatch):
    # 使用最小 LangGraph 替身，让测试聚焦节点业务逻辑。
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.START = "__start__"
    graph_module.END = "__end__"
    graph_module.StateGraph = object
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)
    sys.modules.pop("agent.npc_graph_agent", None)

    return importlib.import_module("agent.npc_graph_agent")


class FakeTools:
    def __init__(self):
        self.append_results = []
        # 记录所有工具调用，方便断言节点行为。
        self.calls = []

    def invoke_tool(self, tool_name, arguments=None):
        # 返回最小稳定结果，避免测试触碰真实数据库。
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


class FakeThread:
    started = []

    def __init__(self, target, daemon=False):
        # 保存线程参数，测试中立即执行目标函数。
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        # 立即执行，避免测试依赖真实后台线程调度。
        self.started = True
        FakeThread.started.append(self)
        self.target()


class FakeTimer:
    scheduled = []

    def __init__(self, interval, function):
        # 保存定时器参数，不立即执行，便于断言延迟调度。
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False

    def start(self):
        # 只记录已启动，不真实等待。
        self.started = True
        FakeTimer.scheduled.append(self)


class FakeLlm:
    def __init__(self, response):
        # 保存模型回复，并记录调用次数。
        self.response = response
        self.calls = []

    def chat(self, prompt, history=None, system_prompt=""):
        # 模拟 LLM chat 接口。
        self.calls.append({
            "prompt": prompt,
            "history": history or [],
            "system_prompt": system_prompt,
        })
        return self.response, []


class FakeMemoryService:
    def __init__(self):
        # 记录工具层透传到记忆服务的参数，避免测试触碰真实数据库。
        self.append_calls = []
        self.search_calls = []
        self.list_calls = []

    def append_short_term_message(self, **kwargs):
        # 模拟短期记忆写入结果，重点验证 llm_service 和 async_archive 是否保留。
        self.append_calls.append(kwargs)
        return {"ok": True, "tool_name": "append_short_term_message"}

    def search_long_memories(self, **kwargs):
        # 模拟长期记忆语义检索，重点验证 query 是否被使用。
        self.search_calls.append(kwargs)
        return [{"memory_id": "m001", "content": "玩家喜欢拿铁"}]

    def list_long_memories(self, **kwargs):
        # 模拟长期记忆兜底列表查询，确保没有 query 时仍可工作。
        self.list_calls.append(kwargs)
        return [{"memory_id": "m002", "content": "普通长期记忆"}]


class FakeShortMemoryService:
    def __init__(self):
        # 记录短期记忆服务调用，确认它不接收 LLM。
        self.append_calls = []

    def append_short_term_message(self, **kwargs):
        self.append_calls.append(kwargs)
        return {"ok": True, "needs_archive": False, "short_term_count": 2}


class FakeLongMemoryService:
    def __init__(self):
        # 记录长期记忆服务调用，确认创建和检索都走长期服务。
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


def test_decide_action_uses_injected_llm_and_extracts_json(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    llm = FakeLlm('说明：{"intent": "greet", "action": "chat", "line": "你好呀"}')

    result = npc_graph_agent.decide_action({
        "player_message": "你好",
        "llm_service": llm,
    })

    assert llm.calls
    assert result["decision"]["line"] == "你好呀"
    assert result["action"] == "chat"


def test_generate_line_uses_decision_line_before_llm(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    llm = FakeLlm("不应该调用")

    result = npc_graph_agent.generate_line({
        "player_message": "你好",
        "llm_service": llm,
        "decision": {"line": "已经有台词"},
    })

    assert result["npc_reply"] == "已经有台词"
    assert llm.calls == []


def test_retrieve_memories_uses_player_message_as_long_term_query(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()

    result = npc_graph_agent.retrieve_memories({
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我想喝不太酸的拿铁",
        "tools": tools,
    })

    long_memory_call = [
        args for name, args in tools.calls
        if name == "get_long_term_memories"
    ][0]
    assert long_memory_call["query"] == "我想喝不太酸的拿铁"
    assert result["long_term_memories"] == []


def test_write_memory_appends_player_and_npc_short_messages(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    monkeypatch.setattr(npc_graph_agent.threading, "Thread", FakeThread)
    FakeThread.started = []

    result = npc_graph_agent.write_memory({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我喜欢拿铁",
        "npc_reply": "好的。",
        "intent": "chat",
        "action": "chat",
        "decision": {"emotion": "pleased"},
        "relationship_delta": {},
        "state_delta": {},
        "tools": tools,
    })

    memory_calls = [
        args for name, args in tools.calls
        if name == "append_short_term_message"
    ]
    assert len(memory_calls) == 2
    assert memory_calls[0]["message"]["role"] == "player"
    assert memory_calls[0]["message"]["emotion"] == "neutral"
    assert memory_calls[1]["message"]["role"] == "npc"
    assert memory_calls[1]["message"]["emotion"] == "pleased"
    assert result["memory_write_result"]["async"] is True
    assert result["memory_write_result"]["status"] == "scheduled"
    assert FakeThread.started[0].daemon is True


def test_write_memory_does_not_pass_llm_to_short_memory(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm("{}")
    monkeypatch.setattr(npc_graph_agent.threading, "Thread", FakeThread)
    FakeThread.started = []

    npc_graph_agent.write_memory({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我经常点拿铁",
        "npc_reply": "我记住了。",
        "intent": "chat",
        "action": "chat",
        "decision": {"emotion": "pleased"},
        "relationship_delta": {},
        "state_delta": {},
        "tools": tools,
        "llm_service": llm,
    })

    memory_calls = [
        args for name, args in tools.calls
        if name == "append_short_term_message"
    ]
    assert "llm_service" not in memory_calls[0]
    assert "llm_service" not in memory_calls[1]
    assert memory_calls[1]["async_archive"] is True


def test_write_memory_archives_overflow_with_long_memory_service(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm("{}")
    monkeypatch.setattr(npc_graph_agent.threading, "Thread", FakeThread)
    FakeThread.started = []
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

    npc_graph_agent.write_memory({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我经常点拿铁",
        "npc_reply": "我记住了。",
        "intent": "chat",
        "action": "chat",
        "decision": {"emotion": "pleased"},
        "relationship_delta": {},
        "state_delta": {},
        "tools": tools,
        "llm_service": llm,
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


def test_update_player_profile_detects_taste_keywords(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()

    result = npc_graph_agent.update_player_profile({
        "player_id": "p001",
        "player_message": "我喜欢甜一点的拿铁",
        "tools": tools,
    })

    assert result["profile_update_result"]["ok"] is True
    call = tools.calls[-1]
    assert call[0] == "update_player_profile"
    assert call[1]["profile_patch"]["taste_preferences"]["likes_latte"] is True
    assert call[1]["profile_patch"]["taste_preferences"]["sweetness_hint"] == "likes_sweet"


def test_decide_next_action_schedules_serving_coffee(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    monkeypatch.setattr(npc_graph_agent.threading, "Timer", FakeTimer)
    FakeTimer.scheduled = []

    result = npc_graph_agent.decide_next_action({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "我要一杯拿铁",
        "decision": {"action": "chat", "emotion": "pleased"},
        "tools": tools,
    })

    assert result["next_action_decision"]["scheduled"] is True
    assert result["next_action_decision"]["action"] == "serve_customer"
    assert FakeTimer.scheduled[0].interval == npc_graph_agent.DEFAULT_NEXT_ACTION_DELAY_SECONDS
    assert FakeTimer.scheduled[0].daemon is True


def test_decide_next_action_skips_when_no_order(monkeypatch):
    npc_graph_agent = import_graph_agent(monkeypatch)
    tools = FakeTools()
    monkeypatch.setattr(npc_graph_agent.threading, "Timer", FakeTimer)
    FakeTimer.scheduled = []

    result = npc_graph_agent.decide_next_action({
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "barista_001",
        "player_message": "今天天气不错",
        "decision": {"action": "chat"},
        "tools": tools,
    })

    assert result["next_action_decision"]["scheduled"] is False
    assert FakeTimer.scheduled == []
