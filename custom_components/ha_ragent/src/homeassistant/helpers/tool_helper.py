import logging
import json
from typing import Any, Dict, List, Tuple

try:
    from homeassistant.core import HomeAssistant, JsonObjectType
    from homeassistant.helpers.llm import ToolInput
    from homeassistant.helpers import area_registry, device_registry, entity_registry, floor_registry
    from homeassistant.helpers.entity_registry import RegistryEntry as EntityEntry
except ImportError:
    from custom_components.ha_ragent.src.mock import (
        MockHomeAssistant as HomeAssistant,
        MockToolInput as ToolInput,
    )
    area_registry = device_registry = floor_registry = None
    entity_registry = None
    JsonObjectType = dict[str, Any]
    EntityEntry = Any

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

        json_string = json_string.strip()

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

    def _parse_original_name(self, original_name: str | None, parameters: dict[str, Any]) -> str | None:
        """Parse the original name from parameters."""
        original_name = parameters.get("name", None)

        if "entity_id" in parameters:
            original_name = parameters.pop("entity_id")

        return original_name

    def _parse_domain(self, original_name: str | None, is_domain_aware: bool, parameters: dict[str, Any]) -> List[str] | None:
        """Parse the domain from the parameters and set it in the parameters dictionary."""
        domain = parameters.get("domain", None)

        if not is_domain_aware:
            return None

        if domain:
            return domain if isinstance(domain, list) else [domain]

        if original_name and "." in original_name:
            domain = original_name.split(".", 1)[0]
            return [domain]

        return None

    def _parse_friendly_name_and_entity_entry(self, original_name: str | None, domain: List[str] | None) -> Tuple[str | None, EntityEntry | None]:
        """Parse the friendly name from the parameters and set it in the parameters dictionary."""
        friendly_name = original_name

        if isinstance(original_name, str) and "." not in original_name and domain:
            temp_name = f"{domain[0]}.{original_name}"
        else:
            temp_name = original_name

        state = self._hass.states.get(temp_name)
        if state:
            friendly_name = state.attributes.get("friendly_name", original_name)

        entity_reg = entity_registry.async_get(self._hass) if entity_registry else None
        entity_entry = entity_reg.async_get(temp_name) if entity_reg else None

        aliases = []
        if entity_entry:
            aliases = entity_registry.async_get_entity_aliases(self._hass, entity_entry)

        if aliases:
            friendly_name = aliases[0]

        return friendly_name, entity_entry

    def _parse_area_and_floor(self, entity_entry: EntityEntry | None, original_area: str | None, original_floor: str | None) -> Tuple[str | None, str | None]:
        """Parse area and floor from the parameters and set them in the parameters dictionary."""
        area = None
        floor = None
        area_id = entity_entry.area_id if entity_entry else None

        if not area_id and entity_entry and entity_entry.device_id and device_registry:
            device = device_registry.async_get(self._hass).async_get_device(entity_entry.device_id)
            area_id = device.area_id if device else None

        if area_id and area_registry:
            area = area_registry.async_get(self._hass).async_get_area(area_id)
            if area and area.floor_id and floor_registry:
                floor = floor_registry.async_get(self._hass).async_get_floor(area.floor_id)

        area_name = original_area if isinstance(original_area, str) and original_area else area.name if area and area.name else None
        floor_name = original_floor if isinstance(original_floor, str) and original_floor else floor.name if floor and floor.name else None
        return (area_name, floor_name)

    def _parse_parameters(self, parameters: dict[str, Any], tool_metadata: ToolMetadata | None) -> None:
        """Parse and normalize tool parameters."""
        is_domain_aware = tool_metadata.is_domain_aware if tool_metadata else False
        is_area_aware = tool_metadata.is_area_aware if tool_metadata else False

        original_name = self._parse_original_name(parameters.get("name"), parameters)
        domain = self._parse_domain(original_name, is_domain_aware, parameters)
        area = parameters.get("area", None)
        floor = parameters.get("floor", None)

        if not isinstance(original_name, str) or ("." not in original_name and not isinstance(domain, list)):
            return

        friendly_name, entity_entry = self._parse_friendly_name_and_entity_entry(original_name, domain)
        area_name, floor_name = self._parse_area_and_floor(entity_entry, area, floor)

        parameters["original_name"] = original_name
        parameters["friendly_name"] = friendly_name

        if is_domain_aware:
            parameters["domain"] = domain

        if is_area_aware:
            if area_name:
                parameters["area"] = area_name
            if floor_name:
                parameters["floor"] = floor_name

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
            parsed_calls.append(parsed_call)

        return parsed_calls

    @staticmethod
    def tool_call_signature(tool_call: ToolInput) -> str:
        """Return a stable signature for a tool name and its arguments."""
        return json.dumps(
            {
                "tool": tool_call.tool_name,
                "arguments": tool_call.tool_args,
            },
            sort_keys=True
        )

    @staticmethod
    def is_identical_failed_retry(tool_call: ToolInput, failed_signatures: set[str] | dict[str, Any]) -> bool:
        """Return whether the same canonical call has already failed."""
        return ToolHelper.tool_call_signature(tool_call) in failed_signatures

    @staticmethod
    def block_broad_tool_calls(tool_call: ToolInput, metadata: ToolMetadata) -> None:
        """Reject broad calls using the tool's target metadata."""
        if not metadata or not metadata.is_domain_aware:
            return

        arguments = tool_call.tool_args
        if not isinstance(arguments, dict):
            raise ValueError(f"Invalid tool arguments for {tool_call.tool_name} follow the expected tool signature.")

        name = arguments.get("name")
        domain = arguments.get("domain")
        area = arguments.get("floor") or arguments.get("area")
        
        if (name or domain) and area:
            return

        raise ValueError(f"Device tool {tool_call.tool_name} requires a combination: name plus area/floor for one device, or domain plus area/floor without name for an all/plural/category target. Never enumerate a whole category.")

    @staticmethod
    def parse_tool_results(tool_result: JsonObjectType) -> Dict[str, Any]:
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

    @staticmethod
    def to_home_assistant_tool_call(tool_call: ToolInput) -> ToolInput:
        """Create the Home Assistant call with parser metadata removed."""
        args = dict(tool_call.tool_args)
        args.pop("original_name", None)
        friendly_name = args.pop("friendly_name", None)
        if friendly_name is not None:
            args["name"] = friendly_name
        return ToolInput(id=tool_call.id, tool_name=tool_call.tool_name, tool_args=args)

    @staticmethod
    def to_history_tool_call(tool_call: ToolInput) -> ToolInput:
        """Create the history call with the original name restored."""
        args = dict(tool_call.tool_args)
        args.pop("friendly_name", None)
        original_name = args.pop("original_name", None)
        if original_name is not None:
            args["name"] = original_name
        return ToolInput(id=tool_call.id, tool_name=tool_call.tool_name, tool_args=args)
