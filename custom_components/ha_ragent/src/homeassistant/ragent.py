from __future__ import annotations

import logging
from typing import Any, List, Tuple
from ..backends.database.base_backend import ABaseDbBackend
from ..backends.embedder.base_backend import ABaseEmbedder
from ..backends.llm.base_backend import ALlmBaseBackend
from .ragent_entity import RAGentEntity

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm

from ..const import (
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    PERSONA_PROMPTS,
    CURRENT_DATE_PROMPT,
    DEVICES_PROMPT,
    SERVICES_PROMPT,
    TOOLS_PROMPT,
    AREA_PROMPT,
    USER_INSTRUCTION
)

_logger = logging.getLogger(__name__)

class RAGent(ConversationEntity, AbstractConversationAgent, RAGentEntity):
    vector_db_backend: ABaseDbBackend
    embedder_backend: ABaseEmbedder
    llm_backend: ALlmBaseBackend

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        super().__init__(hass, entry, subentry)

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
            refresh_system_prompt = self.runtime_options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)
            remember_conversation = self.runtime_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
            remember_num_interactions = self.runtime_options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)
            max_tool_call_iterations = self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)
            llm_api: llm.APIInstance | None = None
            if self.runtime_options.get(CONF_LLM_HASS_API):
                try:
                    llm_api = await llm.async_get_api(
                        self.hass,
                        self.runtime_options[CONF_LLM_HASS_API],
                        llm_context=user_input.as_llm_context(DOMAIN)
                    )
                except HomeAssistantError as err:
                    _logger.error("Error getting LLM API: %s", err)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        f"Error preparing LLM API: {err}",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )
                
            # ensure this chat log has the LLM API instance
            chat_log.llm_api = llm_api

            if remember_conversation:
                message_history = chat_log.content[:]
            else:
                message_history = []

            # trim message history before processing if necessary
            if remember_num_interactions and len(message_history) > (remember_num_interactions * 2) + 1:
                new_message_history = [message_history[0]] # copy system prompt
                new_message_history.extend(message_history[1:][-(remember_num_interactions * 2):])

            # re-generate prompt if necessary
            if len(message_history) == 0 or refresh_system_prompt:
                try:
                    # Build the system prompt from the template
                    system_prompt_content = RAGent.build_prompt_template(
                        self.supported_languages if isinstance(self.supported_languages, str) else "en",
                        raw_prompt
                    )
                    system_prompt = conversation.SystemContent(content=system_prompt_content)
                except TemplateError as err:
                    _logger.error("Error rendering prompt: %s", err)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        f"Sorry, I had a problem with my template: {err}",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )

                if len(message_history) == 0:
                    message_history.append(system_prompt)
                else:
                    message_history[0] = system_prompt

            # RAG Step 1: Embed user input
            _logger.debug("RAG Step 1: Embedding user input: %s", user_input.text)
            query_embedding = await self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), user_input.text)
            
            # RAG Step 2: Query best matching devices using vector search
            retrieved_devices = []
            if query_embedding:
                _logger.debug("RAG Step 2: Querying vector database for similar devices")
                collection_name = f"devices_{self.subentry_id}"
                try:
                    retrieved_devices = await self.entry.vector_db_backend.async_retrieve_devices(
                        dict(self.subentry.data),
                        collection_name=collection_name,
                        query_embedding=query_embedding,
                        top_k=10  # Retrieve top 10 most relevant devices
                    )
                    _logger.info("Retrieved %d relevant devices from vector database", len(retrieved_devices))
                except Exception as e:
                    _logger.error("Error retrieving devices from vector DB: %s", e, exc_info=True)
            
            # RAG Step 3: Build context with retrieved devices
            devices_context = ""
            if retrieved_devices:
                devices_context = "\n\nRelevant devices for this query:\n"
                for device in retrieved_devices:
                    devices_context += f"- {device.name} ({device.device_type})"
                    if device.area_name:
                        devices_context += f" in {device.area_name}"
                    if device.device_tags:
                        devices_context += f" [tags: {', '.join(device.device_tags)}]"
                    devices_context += "\n"
                _logger.debug("Built devices context with %d devices", len(retrieved_devices))

            tool_calls: List[Tuple[llm.ToolInput, Any]] = []
            # if max tool calls is 0 then we expect to generate the response & tool call in one go
            for idx in range(max(1, max_tool_call_iterations)):
                _logger.debug(f"Generating response for {user_input.text=}, iteration {idx+1}/{max_tool_call_iterations}")
                
                # Get the LLM backend from entry
                llm_backend = self.entry.llm_backend
                
                # Format messages for the LLM
                formatted_messages = []
                for msg in message_history:
                    if isinstance(msg, conversation.SystemContent):
                        formatted_messages.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, conversation.UserContent):
                        formatted_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, conversation.AssistantContent):
                        formatted_messages.append({"role": "assistant", "content": msg.content or ""})
                
                # Add current user input if not already in history
                # RAG Step 3 (continued): Augment user input with retrieved device context
                if not formatted_messages or formatted_messages[-1]["role"] != "user":
                    user_message_with_context = user_input.text
                    if devices_context:
                        user_message_with_context = user_input.text + devices_context
                    formatted_messages.append({"role": "user", "content": user_message_with_context})
                
                # Stream from the LLM backend
                last_generation_had_tool_calls = False
                try:
                    assistant_content = ""
                    async for chunk in llm_backend.async_send_chat_request(dict(self.subentry.data), formatted_messages):
                        assistant_content += chunk
                    
                    # Create assistant message and add to history
                    message = conversation.AssistantContent(
                        agent_id=user_input.agent_id,
                        content=assistant_content,
                        tool_calls=[]  # TODO: parse tool calls from response if using GBNF grammar
                    )
                    message_history.append(message)
                    _logger.debug("Added message to history: %s", message)
                    
                    if message.tool_calls and len(message.tool_calls) > 0:
                        last_generation_had_tool_calls = True
                    else:
                        last_generation_had_tool_calls = False
                        
                except Exception as err:
                    _logger.exception("There was a problem talking to the backend")
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                        f"Sorry, there was a problem talking to the backend: {repr(err)}",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )

                # If not multi-turn, break after first tool call
                # also break if no tool calls were made
                if not last_generation_had_tool_calls:
                    break

                # return an error if we run out of attempt without succeeding
                if idx == max_tool_call_iterations - 1 and max_tool_call_iterations > 0:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                        f"Sorry, I ran out of attempts to handle your request",
                    )
                    return ConversationResult(
                        response=intent_response, conversation_id=user_input.conversation_id
                    )
                
            # generate intent response to Home Assistant
            intent_response = intent.IntentResponse(language=user_input.language)
            if len(tool_calls) > 0:
                str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls]
                tools_str = '\n'.join(str_tools)
                intent_response.async_set_card(
                    title="Changes",
                    content=f"Ran the following tools:\n{tools_str}"
                )

            has_speech = False
            for i in range(1, len(message_history)):
                cur_msg = message_history[-1 * i]
                if isinstance(cur_msg, conversation.AssistantContent) and cur_msg.content:
                    intent_response.async_set_speech(cur_msg.content)
                    has_speech = True
                    break

            if not has_speech:
                intent_response.async_set_speech("I don't have anything to say right now")
                _logger.debug(message_history)

            return ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )


    @staticmethod
    def build_prompt_template(selected_language: str, prompt_template_template: str):
        persona = PERSONA_PROMPTS.get(selected_language, PERSONA_PROMPTS["en"])
        current_date = CURRENT_DATE_PROMPT.get(selected_language, CURRENT_DATE_PROMPT["en"])
        devices = DEVICES_PROMPT.get(selected_language, DEVICES_PROMPT["en"])
        services = SERVICES_PROMPT.get(selected_language, SERVICES_PROMPT["en"])
        tools = TOOLS_PROMPT.get(selected_language, TOOLS_PROMPT["en"])
        area = AREA_PROMPT.get(selected_language, AREA_PROMPT["en"])
        user_instruction = USER_INSTRUCTION.get(selected_language, USER_INSTRUCTION["en"])

        prompt_template_template = prompt_template_template.replace("<persona>", persona)
        prompt_template_template = prompt_template_template.replace("<current_date>", current_date)
        prompt_template_template = prompt_template_template.replace("<devices>", devices)
        prompt_template_template = prompt_template_template.replace("<services>", services)
        prompt_template_template = prompt_template_template.replace("<tools>", tools)
        prompt_template_template = prompt_template_template.replace("<area>", area)
        prompt_template_template = prompt_template_template.replace("<user_instruction>", user_instruction)

        return prompt_template_template
