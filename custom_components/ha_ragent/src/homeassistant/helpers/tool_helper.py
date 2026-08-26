import logging
import json
from typing import Any, Dict, List

try:
    from homeassistant.core import HomeAssistant, JsonObjectType
    from homeassistant.helpers.llm import ToolInput
    from homeassistant.helpers import area_registry, device_registry, entity_registry, floor_registry
except ImportError:
    from custom_components.ha_ragent.src.mock import (
        MockHomeAssistant as HomeAssistant,
        MockToolInput as ToolInput,
    )
    area_registry = device_registry = floor_registry = None
    entity_registry = None
    JsonObjectType = dict[str, Any]

from custom_components.ha_ragent.src.const import TOOL_REGEX_PATTERN
from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata

_logger = logging.getLogger(__name__)

class ToolHelper:
    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    def _save_json_load(self, json_string: str) -> dict | None:
        """Safely load a JSON string into a dictionary."""
        if not isinstance(json_string, str):
            return None

        json_string = json_string.replace("'", '"').strip()

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            _logger.debug(f"Failed to parse JSON: {e}")
            return None

    def _tool_string_to_dict(self, tool_string: str) -> dict | None:
        """Convert a tool string to a dictionary."""
        if not isinstance(tool_string, str):
            return None

        tool_string = tool_string.strip()

        tool_json = None
        first_brace = tool_string.find('{')
        last_brace = tool_string.rfind('}')

        if first_brace >= 0 and last_brace > first_brace:
            json_str = tool_string[first_brace:last_brace + 1]
            tool_json = self._save_json_load(json_str)
                
        return tool_json

    def _parse_domain(self, parameters: dict[str, Any]) -> None:
        """Parse the domain from the parameters and set it in the parameters dictionary."""
        domain = parameters.get("domain")

        if "device_class" in parameters:
            device_class = parameters.pop("device_class")
            if not domain:
                domain = device_class

        if domain:
            parameters["domain"] = domain

    def _parse_parameters(self, parameters: dict[str, Any], tool_metadata: ToolMetadata | None) -> None:
        """Parse and normalize tool parameters."""
        self._parse_domain(parameters)

        name = None
        domain = parameters.get("domain")
        has_domain = "domain" in parameters and tool_metadata and tool_metadata.is_domain_aware
        has_area = "area" in parameters and tool_metadata and tool_metadata.is_area_aware
        has_floor = "floor" in parameters and tool_metadata and tool_metadata.is_area_aware
        
        if "name" in parameters:
            name = parameters["name"]
        elif "entity_id" in parameters:
            name = parameters.pop("entity_id")

        if not isinstance(name, str) or ("." not in name and not domain):
            return

        if not has_domain and "." in name:
            domain = name.split(".", 1)[0]
            parameters["domain"] = domain

        lookup_domain = domain[0] if isinstance(domain, list) and domain else domain
        temp_name = name if "." in name else f"{lookup_domain}.{name}"
        state = self._hass.states.get(temp_name)
        if state:
            name = state.attributes.get("friendly_name", name)

        entity_reg = entity_registry.async_get(self._hass) if entity_registry else None
        entity_entry = entity_reg.async_get(temp_name) if entity_reg else None

        aliases = []
        if entity_entry:
            aliases = entity_registry.async_get_entity_aliases(self._hass, entity_entry)

        if aliases:
            name = aliases[0]

        if name:
            parameters["name"] = name

        area_id = entity_entry.area_id if entity_entry else None
        if not area_id and entity_entry and entity_entry.device_id and device_registry:
            device = device_registry.async_get(self._hass).async_get_device(entity_entry.device_id)
            area_id = device.area_id if device else None
        if not area_id or not area_registry:
            return

        area = area_registry.async_get(self._hass).async_get_area(area_id)
        if not area:
            return

        if has_area:
            parameters.setdefault("area", area.name)

        if has_floor and area.floor_id and floor_registry:
            floor = floor_registry.async_get(self._hass).async_get_floor(area.floor_id)
            if floor:
                parameters.setdefault("floor", floor.name)

    def parse_tool_calls(self, llm_response: str, tool_metadata_dic: Dict[str, ToolMetadata] | None = None) -> List[ToolInput]:
        """Parse tool calls from LLM response."""
        parsed_calls = []
        tool_metadata_dic = tool_metadata_dic or {}
        
        for match in TOOL_REGEX_PATTERN.finditer(llm_response):
            tool_json = self._tool_string_to_dict(match.group(1))

            if tool_json is None:
                _logger.debug(f"Failed to parse tool call from LLM response: {match.group(1)}")
                continue

            tool_name = tool_json.get("tool")
            if not tool_name:
                _logger.debug(f"Tool name missing in tool call: {tool_json}")
                continue

            parameters = tool_json.get("arguments")
            if isinstance(parameters, str):
                parameters = self._save_json_load(parameters)

            if not isinstance(parameters, dict):
                _logger.debug(f"Empty tool arguments: {tool_json.get('arguments')}")
                continue

            self._parse_parameters(parameters, tool_metadata_dic.get(tool_name))
            parsed_call = ToolInput(tool_name=tool_name, tool_args=parameters)
            _logger.debug(f"Parsed tool call: name={parsed_call.tool_name}, arguments={parsed_call.tool_args}")
            parsed_calls.append(parsed_call)

        return parsed_calls

    def tool_call_signature(self, tool_call: ToolInput) -> str:
        """Return a stable signature for a tool name and its arguments."""
        return json.dumps(
            {
                "tool": tool_call.tool_name,
                "arguments": tool_call.tool_args,
            },
            sort_keys=True
        )

    def block_broad_tool_calls(self, tool_call: ToolInput, metadata: ToolMetadata) -> None:
        """Reject broad calls using the tool's target metadata."""
        if not metadata or not metadata.is_domain_aware:
            return

        arguments = tool_call.tool_args
        if not isinstance(arguments, dict):
            _logger.warning(f"Tool arguments are not a dictionary: {arguments!r}")
            return

        name = arguments.get("name")
        domain = arguments.get("domain")
        area = arguments.get("floor") or arguments.get("area")
        if (name or domain) and area:
            return

        raise ValueError(f"Device tool {tool_call.tool_name} requires a combination of name and area or domain and area; received arguments={arguments!r}")

    def parse_tool_results(self, tool_result: JsonObjectType) -> Dict[str, Any]:
        """Parse tool results from LLM response."""
        if not isinstance(tool_result, dict):
            return {"result": tool_result}

        data = tool_result.get("data", {})
        success = data.get("success", [])
        failed = data.get("failed", [])
        parsed_result: Dict[str, Any] = {}
        
        success_ids = [x["id"] for x in success if x.get("type") == "entity"]
        if success_ids:
            parsed_result["success"] = success_ids

        failed_ids = [x["id"] for x in failed if x.get("type") == "entity"]
        if failed_ids:
            parsed_result["failed"] = failed_ids

        if parsed_result:
            return parsed_result

        return tool_result
