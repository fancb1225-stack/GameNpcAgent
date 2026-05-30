import importlib
import json
import sys
import types


class FakeGraphAgent:
    instances = []

    def __init__(self, tools, llm_service=None, background_executor=None, defer_memory_write=False):
        # 记录构造参数，确认 Director 把依赖传给图 Agent。
        self.tools = tools
        self.llm_service = llm_service
        self.background_executor = background_executor
        self.defer_memory_write = defer_memory_write
        self.calls = []
        FakeGraphAgent.instances.append(self)

    def handle_message(self, session_id, player_id, npc_id, player_message):
        # 返回固定结果，确认 Director.talk 直接走图 Agent 入口。
        self.calls.append({
            "session_id": session_id,
            "player_id": player_id,
            "npc_id": npc_id,
            "player_message": player_message,
        })
        return {
            "ok": True,
            "reply": "图 Agent 回复",
            "decision": {"action": "chat"},
        }


class FakeTools:
    def __init__(self):
        # 记录工具调用，避免测试访问真实数据库。
        self.calls = []

    def invoke_tool(self, tool_name, arguments=None):
        # Director 显式传入 npc_id 时会同步选择 NPC。
        self.calls.append((tool_name, arguments or {}))
        return {"ok": True, "tool_name": tool_name}


class FakeLlm:
    def __init__(self, response='{"need_next_action": false, "reason": "结束"}'):
        self.response = response
        self.calls = []

    def chat(self, prompt, history=None, system_prompt=""):
        self.calls.append({
            "prompt": prompt,
            "history": history or [],
            "system_prompt": system_prompt,
        })
        return self.response, []


class DeferredExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        return {"scheduled": True}


def import_director_with_fake_graph_agent(monkeypatch):
    # 注入假的图 Agent 模块，避免测试依赖真实 LangGraph 和数据库。
    fake_graph_module = types.ModuleType("agent.npc_graph_agent")
    fake_graph_module.NpcGraphAgent = FakeGraphAgent
    monkeypatch.setitem(sys.modules, "agent.npc_graph_agent", fake_graph_module)
    sys.modules.pop("agent.director_agent", None)
    FakeGraphAgent.instances = []

    return importlib.import_module("agent.director_agent")


def test_director_routes_messages_to_npc_graph_agent(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm()
    director = director_agent.DirectorAgent(tools=tools, llm=llm)

    result = director.talk(
        player_id="p001",
        session_id="s001",
        npc_id="passerby_001",
        message="你好",
    )

    assert result["reply"] == "图 Agent 回复"
    assert len(FakeGraphAgent.instances) == 1
    assert FakeGraphAgent.instances[0].tools is tools
    assert FakeGraphAgent.instances[0].llm_service is llm
    assert FakeGraphAgent.instances[0].defer_memory_write is True
    assert FakeGraphAgent.instances[0].calls == [{
        "session_id": "s001",
        "player_id": "p001",
        "npc_id": "passerby_001",
        "player_message": "你好",
    }]


def test_director_keeps_none_llm_for_graph_agent(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()

    def fail_get_llm():
        # 未显式启用 LLM 时，Director 不应自行创建模型。
        raise AssertionError("不应该创建默认 LLM")

    monkeypatch.setattr(director_agent.LlmService, "getLLM", fail_get_llm)
    director = director_agent.DirectorAgent(tools=tools, llm=None)
    graph_agent = director.get_npc_agent("passerby_001")

    assert graph_agent.llm_service is None


def test_director_creates_default_llm_when_llm_argument_is_omitted(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    default_llm = FakeLlm()

    monkeypatch.setattr(director_agent.LlmService, "getLLM", lambda: default_llm)
    director = director_agent.DirectorAgent(tools=tools)
    graph_agent = director.get_npc_agent("passerby_001")

    assert graph_agent.llm_service is default_llm


def test_director_llm_appends_one_next_action_reply(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm(json.dumps({
        "need_next_action": True,
        "line": "等等，还有一件事。",
        "reason": "NPC 主动补充关键提醒",
    }, ensure_ascii=False))
    executor = DeferredExecutor()
    director = director_agent.DirectorAgent(
        tools=tools,
        llm=llm,
        background_executor=executor,
    )

    result = director.talk(
        player_id="p001",
        session_id="s001",
        npc_id="passerby_001",
        message="我要走了",
    )

    prompt_payload = json.loads(llm.calls[0]["prompt"])
    assert result["reply"] == "图 Agent 回复\n等等，还有一件事。"
    assert result["next_action"]["triggered"] is True
    assert result["next_action"]["reply"] == "等等，还有一件事。"
    assert prompt_payload["recent_messages"] == [
        {"role": "player", "content": "我要走了"},
        {"role": "npc", "content": "图 Agent 回复"},
    ]


def test_director_ends_flow_when_llm_says_no_next_action(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm('{"need_next_action": false, "reason": "无需补充"}')
    director = director_agent.DirectorAgent(tools=tools, llm=llm)

    result = director.talk(
        player_id="p001",
        session_id="s001",
        npc_id="passerby_001",
        message="你好",
    )

    assert result["reply"] == "图 Agent 回复"
    assert result["next_action"]["triggered"] is False
    assert len(llm.calls) == 1


def test_director_next_action_lock_allows_only_one_active_trigger(monkeypatch):
    director_agent = import_director_with_fake_graph_agent(monkeypatch)
    tools = FakeTools()
    llm = FakeLlm(json.dumps({
        "need_next_action": True,
        "line": "我只能补充一次。",
    }, ensure_ascii=False))
    director = director_agent.DirectorAgent(tools=tools, llm=llm)
    lock = director._next_action_locks[("s001", "passerby_001")]
    lock.acquire()
    try:
        result = director.talk(
            player_id="p001",
            session_id="s001",
            npc_id="passerby_001",
            message="继续",
        )
    finally:
        lock.release()

    assert result["reply"] == "图 Agent 回复"
    assert result["next_action"]["triggered"] is False
    assert result["next_action"]["skipped_reason"] == "next_action_locked"
    assert llm.calls == []
