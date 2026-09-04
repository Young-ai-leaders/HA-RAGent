"""Helpers for converting and cleaning conversation messages."""

from __future__ import annotations

import json
try:
    from homeassistant.components import conversation
except ImportError:
    from custom_components.ha_ragent.src.mock import conversation

from custom_components.ha_ragent.src.const import (
    RAGENT_PREFIXED_TOOL_NAMES_BY_NAME,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
    TOOL_REGEX_PATTERN,
)
from custom_components.ha_ragent.src.models.chat.chat_message import (
    ChatFunction,
    ChatMessage,
    ChatToolCall,
    ChatToolFailure,
)


class MessageHelper:
    @staticmethod
    def tool_result_succeeded(result: object) -> bool:
        """Return whether a tool result represents a success."""
        if isinstance(result, dict):
            if result.get("success"):
                return True
            if any(result.get(key) for key in ("error", "errors", "failed")):
                return False
            return True

        success = getattr(result, "success", None)
        if success is not None:
            return bool(success)
        return not any(getattr(result, key, None) for key in ("error", "errors", "failed"))

    @staticmethod
    def create_repeated_tool_result_message(agent_id: str | None, tool_call_id: str | None, tool_name: str, previous_result: object) -> conversation.ToolResultContent:
        """Return the original result for a repeated tool call."""
        success_value = previous_result.get("success")
        if success_value is None:
            success = not any(previous_result.get(key) for key in ("error", "errors"))
        else:
            success = bool(success_value)
        result = {
            "success": success,
            "already_executed": True,
        }
        for error_key in ("error", "errors"):
            if result["success"] is False and error_key in previous_result:
                result[error_key] = previous_result[error_key]

        return conversation.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=result
        )

    @staticmethod
    def create_tool_failure_message(agent_id: str | None, tool_call_id: str | None, tool_name: str, error: Exception) -> conversation.ToolResultContent:
        """Create a tool-result message for a failed tool call."""
        error_value = "Unknown error ensure you follow the tool call format." if isinstance(error, KeyError) and error.args else str(error)
        failure = ChatToolFailure(
            success=False,
            tool=tool_name,
            error=error_value,
        )
        
        return conversation.ToolResultContent(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=failure,
        )

    @staticmethod
    def compact_tool_result(tool_message: conversation.ToolResultContent) -> conversation.ToolResultContent:
        """Compact semantic-search results stored in prompt history."""
        if tool_message.tool_name != RAGENT_PREFIXED_TOOL_NAMES_BY_NAME[RAGENT_SEMANTIC_SEARCH_TOOL_NAME]:
            return tool_message

        return conversation.ToolResultContent(
            agent_id=tool_message.agent_id,
            tool_call_id=tool_message.tool_call_id,
            tool_name=tool_message.tool_name,
            tool_result={"success": True},
        )

    @staticmethod
    def message_to_retrieval_text(message: conversation.Content) -> str:
        """Convert Home Assistant content into retrieval text."""
        if isinstance(
            message,
            (conversation.UserContent, conversation.AssistantContent),
        ):
            return (message.content or "").strip()

        if isinstance(message, conversation.ToolResultContent):
            return f"{message.tool_name} {message.tool_result}".strip()

        return ""

    @staticmethod
    def message_to_chat_messages(messages: list[conversation.Content]) -> list[ChatMessage]:
        """Convert Home Assistant content into canonical backend messages."""
        formatted_messages: list[ChatMessage] = []

        for message in messages:
            if isinstance(message, conversation.SystemContent):
                formatted_messages.append(
                    ChatMessage(role="system", content=message.content)
                )
            elif isinstance(message, conversation.UserContent):
                formatted_messages.append(
                    ChatMessage(role="user", content=message.content)
                )
            elif isinstance(message, conversation.AssistantContent):
                assistant_message = ChatMessage(
                    role="assistant",
                    content=message.content or "",
                )
                if message.tool_calls:
                    assistant_message["tool_calls"] = [
                        ChatToolCall(
                            id=tool_call.id,
                            type="function",
                            function=ChatFunction(
                                name=tool_call.tool_name,
                                arguments=tool_call.tool_args,
                            ),
                        )
                        for tool_call in message.tool_calls
                    ]
                formatted_messages.append(assistant_message)
            elif isinstance(message, conversation.ToolResultContent):
                tool_message = ChatMessage(
                    role="tool",
                    content=message.tool_result,
                    tool_name=message.tool_name
                )
                if message.tool_call_id:
                    tool_message["tool_call_id"] = message.tool_call_id
                formatted_messages.append(tool_message)

        return formatted_messages

    @staticmethod
    def clean_assistant_content(assistant_content: str, has_tool_calls: bool) -> str:
        """Remove the internal fenced tool representation from stored content."""
        return TOOL_REGEX_PATTERN.sub("", assistant_content).strip() if has_tool_calls else assistant_content
