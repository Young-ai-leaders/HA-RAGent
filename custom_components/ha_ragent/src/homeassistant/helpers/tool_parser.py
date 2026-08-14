import logging
import json
from typing import Any, Dict, List

from homeassistant.core import HomeAssistant, JsonObjectType
from homeassistant.helpers.llm import ToolInput

from custom_components.ha_ragent.src.const import TOOL_REGEX_PATTERN

_logger = logging.getLogger(__name__)

class ToolParser:
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

            if "name" in parameters:
                name = parameters.pop("name")
                if "." in name:
                    state = self._hass.states.get(name)
                    parameters["name"] = state.attributes.get("friendly_name") if state else name

            if "entity_id" in parameters:
                entity_id = parameters.pop("entity_id")
                state = self._hass.states.get(entity_id)
                parameters["name"] = state.attributes.get("friendly_name") if state else entity_id

            if "device_class" in parameters:
                device_class = parameters.pop("device_class")
                if "domain" not in parameters:
                    parameters["domain"] = device_class

            parsed_calls.append(ToolInput(tool_name=tool_name, tool_args=parameters))

        return parsed_calls

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
