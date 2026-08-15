from unittest.mock import Mock

from custom_components.ha_ragent.src.homeassistant.helpers.tool_parser import ToolParser


def test_parse_tool_call_preserves_friendly_name() -> None:
    """A friendly-name target must survive nested argument parsing."""
    hass = Mock()
    parser = ToolParser(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": "{\\"name\\":\\"Light Strip\\",\\"area\\":\\"Bedroom Jonas\\"}"}
```"""

    calls = parser.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_name == "HassTurnOn"
    assert calls[0].tool_args == {
        "name": "Light Strip",
        "area": "Bedroom Jonas",
    }
    hass.states.get.assert_not_called()
