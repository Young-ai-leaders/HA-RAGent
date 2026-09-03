import asyncio
import sys
import types
from types import SimpleNamespace

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
        recent_requests=["turn on the bedroom lamp"],
        area="Bedroom Jonas",
        floor="2nd Floor",
        candidates=[{"name": "light.strip", "friendly_name": "Light Strip"}],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": "find a ventilation control tool"},
    )))

    assert query == "find a ventilation control tool"


def test_user_context_is_fallback_without_model_search_query() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._contextual_query = "Current request: turn off a light"

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert query == "Current request: turn off a light"


def test_contextual_fallback_includes_trusted_location() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(
        latest_request="turn off the lights",
        recent_requests=[],
        area="Kitchen",
        floor="Ground floor",
        candidates=[],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert "Default area when the request has no explicit location: Kitchen" in query
    assert "Default floor when the request has no explicit location: Ground floor" in query
