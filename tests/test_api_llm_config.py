import importlib
import sys
import types

from sqlalchemy.types import UserDefinedType

from agent.llm_service import LlmService
from agent.llm_service import DeepSeekV4Flash


class FakeLlm:
    pass


class FakeVector(UserDefinedType):
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def get_col_spec(self, **kwargs):
        return "VECTOR"


def import_api_main(monkeypatch):
    pgvector_module = types.ModuleType("pgvector")
    pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")
    pgvector_sqlalchemy.Vector = FakeVector
    monkeypatch.setitem(sys.modules, "pgvector", pgvector_module)
    monkeypatch.setitem(sys.modules, "pgvector.sqlalchemy", pgvector_sqlalchemy)

    # 使用最小 LangGraph 替身，避免测试依赖真实图运行时。
    graph_module = types.ModuleType("langgraph.graph")
    graph_module.START = "__start__"
    graph_module.END = "__end__"
    graph_module.StateGraph = object
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)
    sys.modules.pop("api.main", None)
    sys.modules.pop("agent.director_agent", None)
    sys.modules.pop("agent.npc_graph_agent", None)

    return importlib.import_module("api.main")


def test_api_creates_llm_by_default_when_not_explicitly_disabled(monkeypatch):
    api_main = import_api_main(monkeypatch)
    default_llm = FakeLlm()

    monkeypatch.delenv("ENABLE_LLM", raising=False)
    monkeypatch.setattr(LlmService, "getLLM", lambda: default_llm)

    assert api_main._create_llm_if_enabled() is default_llm


def test_api_keeps_llm_disabled_when_env_is_false(monkeypatch):
    api_main = import_api_main(monkeypatch)

    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setattr(
        LlmService,
        "getLLM",
        lambda: (_ for _ in ()).throw(AssertionError("不应该创建 LLM")),
    )

    assert api_main._create_llm_if_enabled() is None


def test_default_llm_uses_environment_configuration(monkeypatch):
    # 验证 Docker 注入的环境变量会传递给默认 LLM 客户端。
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    llm = LlmService.getLLM()

    assert isinstance(llm, DeepSeekV4Flash)
    assert llm.api_key == "test-api-key"
    assert llm.base_url == "https://example.test/v1"
    assert llm.model == "test-model"
