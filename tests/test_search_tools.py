import asyncio
import sys
import types
from types import SimpleNamespace

from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

utils_module_name = "custom_components.ha_ragent.src.utils"
previous_utils_module = sys.modules.get(utils_module_name)
utils_stub = types.ModuleType(utils_module_name)
utils_stub.get_tool_description = lambda *_args: ""
sys.modules[utils_module_name] = utils_stub
try:
    from custom_components.ha_ragent.src.homeassistant.tools.search_tools import (
        RAGentSemanticSearchTool,
    )
finally:
    if previous_utils_module is None:
        sys.modules.pop(utils_module_name, None)
    else:
        sys.modules[utils_module_name] = previous_utils_module


def test_model_search_query_can_search_outside_user_request() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(
        latest_request="turn off the light strip",
        area="Bedroom Jonas",
        floor="2nd Floor",
        candidates=[{"name": "light.strip", "friendly_name": "Light Strip"}],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": "find a ventilation control tool"},
    )))

    assert query.startswith("Search intent: find a ventilation control tool")
    assert "Default area when the request has no explicit location: Bedroom Jonas" in query
    assert "Default floor when the request has no explicit location: 2nd Floor" in query
    assert "Current candidate: light.strip | Light Strip" in query


def test_user_context_is_fallback_without_model_search_query() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._contextual_query = "Current request: turn off a light"

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert query == "Current request: turn off a light"


def test_search_context_signature_accepts_empty_keyword_context() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)

    tool.set_search_context()

    assert tool._contextual_query == ""
    assert tool._candidate_context == []


def test_contextual_fallback_includes_trusted_location() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(
        latest_request="turn off the lights",
        area="Kitchen",
        floor="Ground floor",
        candidates=[],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert "Default area when the request has no explicit location: Kitchen" in query
    assert "Default floor when the request has no explicit location: Ground floor" in query


def test_device_search_ignores_model_guessed_concepts() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(latest_request="turn on the bathroom lights")

    query = tool._device_search_query(
        "lights bathroom area switch",
        "fallback query",
    )

    assert query == "turn on the bathroom lights"


def test_refresh_does_not_restore_pruned_candidates() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    candidates = [
        {"name": "light.kitchen"},
        {"name": "light.dining"},
    ]
    tool.set_search_context(candidates=candidates)

    tool.prune_candidates({"LIGHT.KITCHEN"})
    tool.refresh_candidates(candidates)

    assert tool._candidate_context == [{"name": "light.dining"}]


class _FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def async_embed_text(self, _config: dict, _query: str) -> list[float]:
        self.queries.append(_query)
        return [1.0, 0.0]


class _FakeToolVectorDatabase:
    def __init__(self, tools: list[LlmTool]) -> None:
        self.tools = tools

    async def async_retrieve_scored_objects(self, *_args) -> list[ScoredResult[LlmTool]]:
        return [
            ScoredResult(tool, 1.0 - (index * 0.1), index + 1)
            for index, tool in enumerate(self.tools)
        ]

    async def async_get_lexical_objects(self, *_args) -> list[LlmTool]:
        return self.tools


def test_semantic_tool_search_filters_capability_before_vector_rank() -> None:
    timer = LlmTool(name="HassTimerCancel", description="Cancel a timer")
    set_light = LlmTool(name="HassLightSet", description="Set light brightness")
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off")
    vector_database = _FakeToolVectorDatabase([timer, set_light, turn_off])
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn off the kitchen light",
        candidates=[{"name": "light.kitchen", "domain": ["light"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={"search_query": "switch off lights", "scope": "tools"},
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in result["candidate_tools"]]
    assert candidate_names[0] == "HassTurnOff"
    assert set(candidate_names) == {"HassTurnOff", "HassLightSet", "HassTimerCancel"}


def test_tool_search_uses_trusted_action_and_resolved_switch_domain() -> None:
    light_set = LlmTool(name="HassLightSet", description="Set light brightness")
    broadcast = LlmTool(name="HassBroadcast", description="Broadcast a message")
    turn_on = LlmTool(name="HassTurnOn", description="Turn a device on")
    embedder = _FakeEmbedder()
    vector_database = _FakeToolVectorDatabase([light_set, broadcast, turn_on])
    entry = SimpleNamespace(embedder_backend=embedder, vector_db_backend=vector_database)
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{
            "name": "switch.bathroom_heater",
            "friendly_name": "Bathroom heater",
            "domain": ["switch"],
        }],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={
                    "search_query": "lights bathroom area switch toggle",
                    "scope": "tools",
                },
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in result["candidate_tools"]]
    assert candidate_names[0] == "HassTurnOn"
    assert candidate_names == ["HassTurnOn"]
    assert result["candidate_tools"][0]["canonical_action"] == "on"
    assert result["candidate_tools"][0]["ranking_signals"]["action_intent"] == 1.0
    assert result["tool_search_confidence"] == "high"
    assert "canonical action: on" in result["tool_search_query"]
    assert "supported domains: switch" in result["tool_search_query"]
    assert "lights bathroom area" not in result["tool_search_query"]
    assert embedder.queries == [result["tool_search_query"]]


def test_weak_tool_result_returns_explicit_fallback_signal() -> None:
    vector_database = _FakeToolVectorDatabase([
        LlmTool(name="HassBroadcast", description="Broadcast a message"),
    ])
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "heater toggle", "scope": "tools"})
        )
    )

    assert [candidate["name"] for candidate in result["candidate_tools"]] == [
        "HassBroadcast"
    ]
    assert result["candidate_devices"] == [
        {"name": "switch.bathroom_heater", "domain": ["switch"]}
    ]
    assert result["tool_search_status"] == "weak_candidates"
    assert result["tool_search_confidence"] == "low"
    assert result["fallback_required"] is True
    assert "do not invent a tool" in result["tool_search_message"]


def test_empty_tool_index_returns_no_tools_fallback() -> None:
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=_FakeToolVectorDatabase([]),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "heater", "scope": "tools"})
        )
    )

    assert result["candidate_tools"] == []
    assert result["tool_search_status"] == "no_tools_found"
    assert result["tool_search_confidence"] == "none"
    assert result["fallback_required"] is True
