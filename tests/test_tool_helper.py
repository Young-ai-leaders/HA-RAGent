import json
from unittest.mock import Mock

import pytest

from custom_components.ha_ragent.src.homeassistant.helpers.tool_helper import ToolHelper
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup

def test_parse_tool_call_preserves_friendly_name() -> None:
    """A friendly-name target must survive nested argument parsing."""
    hass = Mock()
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": "{\\"name\\":\\"Light Strip\\",\\"area\\":\\"Bedroom Jonas\\"}"}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_name == "HassTurnOn"
    assert calls[0].tool_args == {
        "name": "Light Strip",
        "area": "Bedroom Jonas",
    }
    hass.states.get.assert_not_called()

def test_parse_tool_call_preserves_apostrophes_in_json_strings() -> None:
    """Valid JSON strings containing apostrophes must not be rewritten."""
    helper = ToolHelper(Mock())
    response = """```homeassistant
{"tool": "HassRememberFact", "arguments": {"memory": "My brother's name is Elias."}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {"memory": "My brother's name is Elias."}


def test_unknown_tool_arguments_are_preserved_exactly_from_live_schema() -> None:
    helper = ToolHelper(Mock())
    arguments = {
        "name": "raw.vendor_target",
        "friendly_name": "schema-owned value",
        "original_name": "schema-owned original",
        "options": {"mode": "turbo", "levels": [1, 2]},
    }
    response = f'''```homeassistant
{{"tool": "VendorExecute", "arguments": {json.dumps(arguments)}}}
```'''

    calls = helper.parse_tool_calls(response, {"VendorExecute": ToolMetadata()})
    execution_call = helper.to_home_assistant_tool_call(calls[0], ToolMetadata())

    assert execution_call.tool_args == arguments


def test_completed_request_rejects_unintended_extra_action() -> None:
    tools = [
        LlmTool(name="HassTurnOn", description="Turn a device on"),
        LlmTool(name="HassTurnOff", description="Turn a device off"),
    ]
    outstanding = ["on"]

    completed = ToolHelper.requested_action_for_tool("HassTurnOn", tools, outstanding)
    outstanding.remove(completed)

    assert ToolHelper.requested_action_for_tool("HassTurnOff", tools, outstanding) is None


def test_explicit_full_intent_authorizes_matching_target() -> None:
    call = Mock(
        tool_args={
            "original_name": "light.bathroom_ceiling",
            "friendly_name": "Bathroom ceiling light",
            "area": "Bathroom",
            "domain": ["light"],
        },
    )
    candidates = [{
        "name": "light.bathroom_ceiling",
        "friendly_name": "Bathroom ceiling light",
        "area": "Bathroom",
        "domain": ["light"],
    }]

    allowed, error = ToolHelper.authorize_requested_target(
        call,
        ToolMetadata(is_domain_aware=True, is_area_aware=True),
        "turn on the bathroom ceiling light",
        candidates,
        "on",
    )

    assert allowed
    assert error == ""


def test_ambiguous_target_is_not_authorized_by_retrieval_order() -> None:
    call = Mock(tool_args={"name": "Bathroom ceiling light", "area": "Bathroom"})
    candidates = [
        {
            "name": "light.bathroom_ceiling",
            "friendly_name": "Bathroom ceiling light",
            "area": "Bathroom",
            "domain": ["light"],
        },
        {
            "name": "light.bathroom_mirror",
            "friendly_name": "Bathroom mirror light",
            "area": "Bathroom",
            "domain": ["light"],
        },
    ]

    allowed, error = ToolHelper.authorize_requested_target(
        call,
        ToolMetadata(is_domain_aware=True, is_area_aware=True),
        "turn on the bathroom light",
        candidates,
        "on",
    )

    assert not allowed
    assert "ambiguous" in error


def test_elliptical_request_exposes_and_authorizes_prior_action_in_new_location() -> None:
    group = TargetGroup(
        entities=("light.kitchen",),
        areas=("Kitchen",),
        domains=("light",),
        action="HassTurnOn",
    )
    resolved = RetrievalHelper.resolve_followup_query(
        "in the bathroom",
        ContinuityContext(target_groups=[(group, 0.9)]),
    )
    tools = [
        LlmTool(name="HassTurnOff", description="Turn a device off"),
        LlmTool(name="HassTurnOn", description="Turn a device on"),
    ]
    candidates = [{
        "name": "light.bathroom_ceiling",
        "friendly_name": "Bathroom ceiling light",
        "area": "Bathroom",
        "domain": ["light"],
    }]
    call = Mock(tool_args={"name": "Bathroom ceiling light", "area": "Bathroom"})

    action = ToolHelper.requested_action_for_tool("HassTurnOn", tools, ["on"])
    allowed, _ = ToolHelper.authorize_requested_target(
        call,
        ToolMetadata(is_domain_aware=True, is_area_aware=True),
        resolved,
        candidates,
        action,
    )

    assert action == "on"
    assert RetrievalHelper.rank_tools_for_query(tools, resolved, candidates)[0].name == "HassTurnOn"
    assert allowed

def test_parse_tool_call_uses_device_class_as_missing_domain() -> None:
    """A device class supplies the domain when the model omitted it."""
    hass = Mock()
    hass.states.get.return_value = None
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"name": "Kitchen Window", "device_class": ["window"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {
        "name": "Kitchen Window",
        "device_class": ["window"],
    }

def test_parse_tool_call_prefers_explicit_domain() -> None:
    """An explicit domain takes precedence over the device class fallback."""
    hass = Mock()
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"area": "Living Room", "domain": ["light"], "device_class": ["switch"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {
        "area": "Living Room",
        "domain": ["light"],
        "device_class": ["switch"],
    }

def test_parse_tool_call_converts_entity_id_and_resolves_friendly_name() -> None:
    """An entity-id target is normalized to its configured friendly name."""
    hass = Mock()
    hass.states.get.return_value = Mock(
        attributes={"friendly_name": "Bedroom Ceiling Light"}
    )
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"entity_id": "light.bedroom_ceiling"}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args == {
        "friendly_name": "Bedroom Ceiling Light",
        "original_name": "light.bedroom_ceiling",
    }
    hass.states.get.assert_called_once_with("light.bedroom_ceiling")

def test_parse_tool_call_finds_domain_from_entity_id() -> None:
    """A full entity ID supplies the missing domain."""
    hass = Mock()
    hass.states.get.return_value = None
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"entity_id": "switch.bedroom_fan"}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args == {
        "friendly_name": "switch.bedroom_fan",
        "original_name": "switch.bedroom_fan",
    }
    hass.states.get.assert_called_once_with("switch.bedroom_fan")

def test_parse_tool_call_preserves_name_without_domain_aware_metadata() -> None:
    """A name is not resolved when domain-aware metadata is unavailable."""
    hass = Mock()
    hass.states.get.return_value = Mock(
        attributes={"friendly_name": "Bedroom 2 Ceiling Light"}
    )
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"name": "bedroom_2_ceiling_light", "domain": ["light"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args["name"] == "bedroom_2_ceiling_light"
    assert "original_name" not in calls[0].tool_args
    hass.states.get.assert_not_called()

def test_tool_call_signature_ignores_argument_order() -> None:
    """Equivalent calls have the same signature regardless of key order."""
    helper = ToolHelper(Mock())
    first = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 1"},
    )
    second = Mock(
        tool_name="HassTurnOn",
        tool_args={"area": "Bedroom 1", "domain": ["light"]},
    )

    assert helper.tool_call_signature(first) == helper.tool_call_signature(second)

def test_tool_call_signature_distinguishes_targets() -> None:
    """Calls to different targets remain independently executable."""
    helper = ToolHelper(Mock())
    bedroom_one = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 1"},
    )
    bedroom_two = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 2"},
    )

    assert helper.tool_call_signature(bedroom_one) != helper.tool_call_signature(
        bedroom_two
    )


def test_identical_failed_retry_ignores_argument_order() -> None:
    helper = ToolHelper(Mock())
    first = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom"},
    )
    retry = Mock(
        tool_name="HassTurnOn",
        tool_args={"area": "Bedroom", "domain": ["light"]},
    )

    failed = {helper.tool_call_signature(first): {"success": False}}

    assert helper.is_identical_failed_retry(retry, failed)


def test_semantic_search_signature_normalizes_query_and_scope() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "  TURN   ON lights ", "scope": "TOOLS"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"scope": "tools", "search_query": "turn on LIGHTS"},
    )

    assert ToolHelper.tool_call_signature(first) == ToolHelper.tool_call_signature(second)


def test_semantic_search_signature_reuses_action_aliases() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "switch off kitchen lights", "scope": "tools"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "power off the kitchen light", "scope": "tools"},
    )

    assert ToolHelper.tool_call_signature(first) == ToolHelper.tool_call_signature(second)


def test_semantic_search_signature_reuses_weak_power_rewrites() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "heater bathroom switch toggle", "scope": "tools"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "heater bathroom on/off", "scope": "tools"},
    )

    assert ToolHelper.tool_call_signature(first) == ToolHelper.tool_call_signature(second)


def test_exposed_tool_name_normalizes_only_known_namespace_variants() -> None:
    exposed = {"HassTurnOn", "ha_ragent__HassSemanticSearch"}

    assert ToolHelper.resolve_exposed_tool_name("switch__HassTurnOn", exposed) == "HassTurnOn"
    assert ToolHelper.resolve_exposed_tool_name("HassSemanticSearch", exposed) == (
        "ha_ragent__HassSemanticSearch"
    )
    assert ToolHelper.resolve_exposed_tool_name("switch__HassSwitchToggle", exposed) is None


def test_discovered_tools_are_converted_for_next_iteration() -> None:
    existing_names = {"HassSemanticSearch"}
    discovered = ToolHelper.discovered_tools(
        {
            "candidate_tools": [{
                "name": "HassTurnOn",
                "description": "Turn on a target",
                "parameters": {"properties": {"name": {"type": "string"}}},
                "metadata": {"family": "power", "is_domain_aware": True},
            }],
        },
        existing_names,
    )

    assert [tool.name for tool in discovered] == ["HassTurnOn"]
    assert discovered[0].metadata.family == "power"
    assert discovered[0].metadata.is_domain_aware is True
    assert "HassTurnOn" in existing_names


def test_discovered_unknown_tool_keeps_live_schema_and_generic_metadata() -> None:
    parameters = {"type": "object", "properties": {"mode": {"enum": ["quiet"]}}}
    discovered = ToolHelper.discovered_tools(
        {"candidate_tools": [{
            "name": "VendorExecute",
            "description": "Execute a vendor operation",
            "parameters": parameters,
        }]},
        set(),
    )

    assert discovered[0].parameters is parameters
    assert discovered[0].metadata == ToolMetadata()

def test_validate_tool_call_target_rejects_area_only_device_call() -> None:
    """Domain-aware tools cannot target every entity in an area implicitly."""
    call = Mock(
        tool_name="HassTurnOn",
        tool_args={"area": "Bedroom 1"},
    )

    with pytest.raises(ValueError, match="requires a combination"):
        ToolHelper(Mock()).block_broad_tool_calls(call, ToolMetadata(is_domain_aware=True))

@pytest.mark.parametrize(
    "tool_args",
    [
        {"name": "Bedroom 1 Ceiling Light"},
        {"domain": ["light"]},
        {},
    ],
)
def test_validate_tool_call_target_rejects_unscoped_device_call(
    tool_args: dict,
) -> None:
    """Domain-aware tools require both a target and a location scope."""
    call = Mock(tool_name="HassTurnOn", tool_args=tool_args)

    with pytest.raises(ValueError, match="requires a combination"):
        ToolHelper(Mock()).block_broad_tool_calls(call, ToolMetadata(is_domain_aware=True))

@pytest.mark.parametrize(
    "tool_args",
    [
        {"name": "Bedroom 1 Ceiling Light", "area": "Bedroom 1"},
        {"area": "Bedroom 1", "domain": ["light"]},
        {"name": "Bedroom 1 Ceiling Light", "floor": "Ground Floor"},
    ],
)
def test_validate_tool_call_target_accepts_scoped_device_call(
    tool_args: dict,
) -> None:
    """A name or domain provides the required device scope."""
    call = Mock(tool_name="HassTurnOn", tool_args=tool_args)

    ToolHelper(Mock()).block_broad_tool_calls(call, ToolMetadata(is_domain_aware=True))

def test_validate_tool_call_target_ignores_non_domain_tool() -> None:
    """Tools without device-domain targeting may validly use an area alone."""
    call = Mock(
        tool_name="HassBroadcast",
        tool_args={"area": "Bedroom 1"},
    )

    ToolHelper(Mock()).block_broad_tool_calls(call, ToolMetadata(is_domain_aware=False))
