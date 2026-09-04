from __future__ import annotations

from datetime import timedelta
from typing import Any

try:
    from homeassistant.components import conversation
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.util import dt as dt_util
except ImportError:
    from custom_components.ha_ragent.src.mock import conversation, dt_util

    ConversationInput = Any

from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.models.turn_context import TurnContext

class HistoryManager:
    def __init__(self, runtime_options: dict[str, Any]) -> None:
        self._runtime_options = runtime_options
        self._message_history: list[conversation.Content] = []

    @property
    def message_history(self) -> list[conversation.Content]:
        """Return the active Home Assistant message history."""
        return self._message_history

    def _select_retained_turns(self, chat_log: conversation.ChatLog) -> list[list[conversation.Content]]:
        """Select complete conversation turns within the configured retention limits."""
        remember_time_minutes = self._runtime_options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES)
        remember_num_interactions = self._runtime_options.get(CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS, DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS)

        if not remember_time_minutes and not remember_num_interactions:
            return []

        # Home Assistant has already appended the current user input to the log.
        # It is added explicitly when the prompt history is built below.
        raw_history = list(chat_log.content)[:-1]
        turns: list[list[conversation.Content]] = []

        for message in raw_history:
            if isinstance(message, conversation.SystemContent):
                continue
            elif isinstance(message, conversation.UserContent):
                turns.append([message])
            elif turns:
                turns[-1].append(message)

        if remember_time_minutes:
            now = dt_util.utcnow()
            cutoff = now - timedelta(minutes=remember_time_minutes)
            turns = [turn for turn in turns if getattr(turn[0], "created_at", now) >= cutoff]

        if remember_num_interactions and len(turns) > remember_num_interactions:
            turns = turns[-remember_num_interactions:]

        return turns

    def select_retained_history(self, chat_log: conversation.ChatLog) -> list[conversation.Content]:
        """Return retained history as a flat message list."""
        return [message for turn in self._select_retained_turns(chat_log) for message in turn]

    @staticmethod
    def _add_values(target: set[str], value: object) -> None:
        if isinstance(value, str) and value:
            target.add(value)
        elif isinstance(value, (list, tuple, set)):
            target.update(str(item) for item in value if isinstance(item, (str, int, float)))

    @classmethod
    def _collect_tool_result(
        cls,
        result: object,
        entities: set[str],
        tools: set[str],
        areas: set[str],
        domains: set[str],
        device_classes: set[str],
        ambiguous_entities: set[str],
    ) -> None:
        if not isinstance(result, dict):
            return
        cls._add_values(entities, result.get("success"))
        devices = result.get("devices")
        if isinstance(devices, list):
            candidate_ids: list[str] = []
            for device in devices:
                if not isinstance(device, dict):
                    continue
                name = device.get("name") or device.get("entity_id")
                cls._add_values(entities, name)
                if isinstance(name, str):
                    candidate_ids.append(name)
                cls._add_values(areas, device.get("area"))
                cls._add_values(domains, device.get("domain"))
                cls._add_values(device_classes, device.get("device_class"))
            if len(candidate_ids) > 1:
                ambiguous_entities.update(candidate_ids)
        result_tools = result.get("tools")
        if isinstance(result_tools, list):
            for tool in result_tools:
                if isinstance(tool, dict):
                    cls._add_values(tools, tool.get("name"))

    def structured_turn_contexts(self, chat_log: conversation.ChatLog) -> list[TurnContext]:
        """Extract language-independent context from retained completed turns."""
        contexts: list[TurnContext] = []
        for turn in self._select_retained_turns(chat_log):
            user_message = next(
                (message for message in turn if isinstance(message, conversation.UserContent)),
                None,
            )
            if user_message is None:
                continue
            entities: set[str] = set()
            tools: set[str] = set()
            areas: set[str] = set()
            domains: set[str] = set()
            device_classes: set[str] = set()
            actions: set[str] = set()
            ambiguous_entities: set[str] = set()

            for message in turn:
                if isinstance(message, conversation.AssistantContent):
                    for tool_call in getattr(message, "tool_calls", None) or []:
                        tool_name = str(getattr(tool_call, "tool_name", "") or "")
                        if tool_name:
                            tools.add(tool_name)
                            actions.add(tool_name)
                        arguments = getattr(tool_call, "tool_args", None) or {}
                        if isinstance(arguments, dict):
                            self._add_values(entities, arguments.get("name"))
                            self._add_values(areas, arguments.get("area"))
                            self._add_values(areas, arguments.get("floor"))
                            self._add_values(domains, arguments.get("domain"))
                            self._add_values(device_classes, arguments.get("device_class"))
                            self._add_values(actions, arguments.get("action"))
                elif isinstance(message, conversation.ToolResultContent):
                    tool_name = str(getattr(message, "tool_name", "") or "")
                    if tool_name:
                        tools.add(tool_name)
                        actions.add(tool_name)
                    self._collect_tool_result(
                        getattr(message, "tool_result", None),
                        entities,
                        tools,
                        areas,
                        domains,
                        device_classes,
                        ambiguous_entities,
                    )

            created_at = getattr(user_message, "created_at", None)
            timestamp = created_at.timestamp() if hasattr(created_at, "timestamp") else None
            text = str(getattr(user_message, "content", "") or "").strip()
            key = "\x1f".join((text, *sorted(entities), *sorted(tools)))
            contexts.append(TurnContext(
                key=key,
                text=text,
                entities=tuple(sorted(entities)),
                tools=tuple(sorted(tools)),
                areas=tuple(sorted(areas)),
                domains=tuple(sorted(domains)),
                device_classes=tuple(sorted(device_classes)),
                actions=tuple(sorted(actions)),
                ambiguous_entities=tuple(sorted(ambiguous_entities)),
                created_at=timestamp,
            ))
        return contexts

    def filter_prompt_history(self, chat_log: conversation.ChatLog) -> list[conversation.Content]:
        """Return prompt history while compacting semantic-search results."""
        prompt_history: list[conversation.Content] = []

        for message in self.select_retained_history(chat_log):
            if isinstance(message, (conversation.UserContent, conversation.AssistantContent)):
                prompt_history.append(message)
            elif isinstance(message, conversation.ToolResultContent):
                prompt_history.append(MessageHelper.compact_tool_result(message))

        return prompt_history

    def build_prompt_history(self, chat_log: conversation.ChatLog, user_input: ConversationInput, system_prompt_content: str) -> list[conversation.Content]:
        """Build model history with system prompt first and current user last."""
        self._message_history = [
            conversation.SystemContent(content=system_prompt_content)
        ]
        self._message_history.extend(self.filter_prompt_history(chat_log))
        self._message_history.append(conversation.UserContent(content=user_input.text))
        return self._message_history

    def append_message(self, message: conversation.Content) -> None:
        """Append content to the active history."""
        self._message_history.append(message)

    def persist_chat_history(self, chat_log: conversation.ChatLog) -> None:
        """Persist the active normalized history to Home Assistant."""
        chat_log.content = self._message_history
