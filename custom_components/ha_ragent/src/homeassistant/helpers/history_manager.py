from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper

class HistoryManager:
    def __init__(self, runtime_options: dict[str, Any]) -> None:
        self._runtime_options = runtime_options
        self._message_history: list[conversation.Content] = []

    @property
    def message_history(self) -> list[conversation.Content]:
        """Return the active Home Assistant message history."""
        return self._message_history

    def select_retained_history(self, chat_log: conversation.ChatLog) -> list[conversation.Content]:
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

        return [message for turn in turns for message in turn]

    def filter_prompt_history(self, chat_log: conversation.ChatLog) -> list[conversation.Content]:
        """Return prompt history while compacting semantic-search results."""
        prompt_history: list[conversation.Content] = []

        for message in self.select_retained_history(chat_log):
            if isinstance(message, (conversation.UserContent, conversation.AssistantContent)):
                prompt_history.append(message)
            elif isinstance(message, conversation.ToolResultContent):
                prompt_history.append(MessageHelper.compact_tool_result(message))

        return prompt_history

    def retrieval_texts(self, chat_log: conversation.ChatLog) -> list[str]:
        """Select retained history and convert it into retrieval text."""
        retrieval_texts = [MessageHelper.message_to_retrieval_text(message) for message in self.select_retained_history(chat_log)]
        return [text for text in retrieval_texts if text]

    def recent_user_requests(self, chat_log: conversation.ChatLog) -> list[str]:
        """Return retained user text without assistant or tool-result noise."""
        return [
            message.content.strip()
            for message in self.select_retained_history(chat_log)
            if isinstance(message, conversation.UserContent) and message.content.strip()
        ]

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
