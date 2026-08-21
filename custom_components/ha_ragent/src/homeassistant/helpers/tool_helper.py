import logging
import json
from typing import Any, Dict, List

try:
    from homeassistant.core import HomeAssistant, JsonObjectType
    from homeassistant.helpers.llm import ToolInput
except ImportError:
    from custom_components.ha_ragent.src.mock import (
        MockHomeAssistant as HomeAssistant,
        MockToolInput as ToolInput,
    )
    JsonObjectType = dict[str, Any]

from custom_components.ha_ragent.src.const import TOOL_REGEX_PATTERN

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

    def _parse_domain(self, parameters: dict[str, Any]) -> Any:
        domain = parameters.get("domain")

        if "device_class" in parameters:
            device_class = parameters.pop("device_class")
            if not domain:
                domain = device_class

        return domain

    def _parse_name(self, parameters: dict[str, Any], domain: str | None) -> str | None:
        name = None
        
        if "name" in parameters:
            name = parameters["name"]
        elif "entity_id" in parameters:
            name = parameters.pop("entity_id")

        if isinstance(name, str) and ("." in name or domain):
            temp_name = name if "." in name else f"{domain}.{name}"
            state = self._hass.states.get(temp_name)
            if state:
                name = state.attributes.get("friendly_name", name)

        return name

    def parse_tool_calls(self, llm_response: str) -> List[ToolInput]:
        """Parse tool calls from LLM response."""
        parsed_calls = []
        
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

            domain = self._parse_domain(parameters)
            first_domain = domain[0] if isinstance(domain, (list, tuple)) and domain else None
            name = self._parse_name(parameters, first_domain)

            if domain:
                parameters["domain"] = domain

            if name:
                parameters["name"] = name

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

    def validate_tool_call_target(self, tool_call: ToolInput, is_domain_aware: bool) -> None:
        """Reject broad device calls without a name, domain, or device class."""
        if not is_domain_aware:
            return

        arguments = tool_call.tool_args
        if not isinstance(arguments, dict):
            raise ValueError(f"Device tool {tool_call.tool_name} received invalid arguments: {arguments!r}")

        name = arguments.get("name")
        domain = arguments.get("domain")
        device_class = arguments.get("device_class")
        if name or domain or device_class:
            return

        raise ValueError(f"Device tool {tool_call.tool_name} requires a non-empty name, domain, or device_class; received arguments={arguments!r}")

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
