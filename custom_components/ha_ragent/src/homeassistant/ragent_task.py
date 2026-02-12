from __future__ import annotations

from json import JSONDecodeError
import logging

from homeassistant.components import ai_task, conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.json import json_loads

from .ragent_entity import RAGentEntity
from .ragent import RAGent

from ..const import (
    CONF_PROMPT,
    CONF_SELECTED_LANGUAGE,
    DEFAULT_PROMPT,
    DEFAULT_SELECTED_LANGUAGE,
)

from ..models.enums import ResultTypes

_logger = logging.getLogger(__name__)

class RAGentTaskEntity(ai_task.AITaskEntity, RAGentEntity):
    """Ollama AI Task entity."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Ollama AI Task entity."""
        super().__init__(*args, **kwargs)
        self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""

        extraction_method = ResultTypes.NONE

        try:
            raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
            selected_language = self.runtime_options.get(CONF_SELECTED_LANGUAGE, DEFAULT_SELECTED_LANGUAGE)

            message_history = chat_log.content[:]

            if not message_history or not isinstance(message_history[0], conversation.SystemContent):
                system_prompt_content = RAGent.build_prompt_template(selected_language, raw_prompt)
                system_prompt = conversation.SystemContent(content=system_prompt_content)
                message_history.insert(0, system_prompt)

            _logger.debug("Generating response for %s...", task.name)

            formatted_messages = []
            for msg in message_history:
                if isinstance(msg, conversation.SystemContent):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, conversation.UserContent):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, conversation.AssistantContent):
                    formatted_messages.append({"role": "assistant", "content": msg.content or ""})

            llm_backend = self.entry.llm_backend
            assistant_content = ""
            async for chunk in llm_backend.async_send_chat_request(formatted_messages):
                assistant_content += chunk

            text = assistant_content

            if not task.structure:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=text,
                )
            
            if extraction_method == ResultTypes.NONE:
                raise HomeAssistantError("Task structure provided but no extraction method was specified!")
            elif extraction_method == ResultTypes.STRUCTURED_OUTPUT:
                try:
                    data = json_loads(text)
                except JSONDecodeError as err:
                    _logger.error(
                        "Failed to parse JSON response: %s. Response: %s",
                        err,
                        text,
                    )
                    raise HomeAssistantError("Error with Local LLM structured response") from err
            elif extraction_method == ResultTypes.TOOL:
                try:
                    data = chat_log.content[-1].tool_calls[0].tool_args
                except (IndexError, AttributeError) as err:
                    _logger.error(
                        "Failed to extract tool arguments from response: %s. Response: %s",
                        err,
                        text,
                    )
                    raise HomeAssistantError("Error with Local LLM tool response") from err
            else:
                raise ValueError()

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )
        except Exception as err:
            _logger.exception("Unhandled exception while running AI Task '%s'", task.name)
            raise HomeAssistantError(f"Unhandled error while running AI Task '{task.name}'") from err
