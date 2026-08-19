from __future__ import annotations

from datetime import datetime, timedelta
import logging

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    RAGENT_PLANNED_ACTION_TOOL_NAME,
)

_logger = logging.getLogger(__name__)

RAGENT_SCHEDULED_REQUEST_PREFIX = (
    "Execute this action now. It was previously scheduled, so do not schedule "
    "it again: "
)


class RAGentPlannedActionTool(llm.Tool):
    name = RAGENT_PLANNED_ACTION_TOOL_NAME
    description = (
        "Schedule an action once after a delay. Use this tool when the user says to "
        "schedule an action or gives a future delay such as 'in 2 minutes'. Do not "
        "execute the action now. description must be an immediate command containing "
        "only the action and target. Never include scheduling or time wording such as "
        "'schedule', 'scheduled', 'later', 'in 2 minutes', or 'at 8 PM' in description. "
        "Example: for 'schedule the bathroom light in 2 minutes', use description "
        "'turn on the bathroom light' and minutes 2. After success, call no other "
        "action tool."
    )
    parameters = vol.Schema(
        {
            vol.Required("description"): str,
            vol.Required("minutes"): vol.All(
                vol.Coerce(float),
                vol.Range(min=0.01, max=10080),
            ),
        }
    )

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        agent_id: str,
        context: Context,
        language: str | None,
        device_id: str | None,
    ) -> None:
        self.hass = hass
        self.agent_id = agent_id
        self.context = context
        self.language = language
        self.device_id = device_id

    async def _async_execute(self, _now: datetime, description: str) -> None:
        """Send the due action back through the originating conversation agent."""
        prompt = f"{RAGENT_SCHEDULED_REQUEST_PREFIX}{description}"
        try:
            result = await conversation.async_converse(
                hass=self.hass,
                text=prompt,
                conversation_id=None,
                context=self.context,
                language=self.language,
                agent_id=self.agent_id,
                device_id=self.device_id,
            )
            if result.response.error_code is not None:
                _logger.error(f"Planned action failed for agent {self.agent_id}: {description}. Error: {result.response.error_code}")
            else:
                _logger.info(f"Executed planned action for agent {self.agent_id}: {description}")
        except Exception:
            _logger.exception(f"Failed to execute planned action for agent {self.agent_id}: {description}")

    async def async_call(self, tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        """Schedule the requested action without blocking the conversation."""
        description = str(tool_input.tool_args.get("description", "")).strip()
        if not description:
            return {"error": "description must not be empty"}

        try:
            minutes = float(tool_input.tool_args.get("minutes", 0))
        except (TypeError, ValueError):
            return {"error": "minutes must be a number"}

        if not 1 <= minutes <= 1440:
            return {"error": "minutes must be between 1 and 1440"}

        execute_at = dt_util.utcnow() + timedelta(minutes=minutes)
        local_execute_at = dt_util.as_local(execute_at)
        human_execute_at = local_execute_at.strftime("%A, %B %d, %Y at %H:%M %Z").replace(" 0", " ")

        async def execute(now: datetime) -> None:
            await self._async_execute(now, description)

        async_call_later(self.hass, minutes * 60, execute)
        return {
            "success": True,
            "description": description,
            "minutes": minutes,
            "execute_at": human_execute_at,
        }
