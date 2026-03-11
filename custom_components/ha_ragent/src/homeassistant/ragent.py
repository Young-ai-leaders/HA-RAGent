from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Tuple, Dict

try:
    from voluptuous_openapi import convert
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False
    convert = None

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, JsonObjectType
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.template import Template
from homeassistant.helpers.llm import ToolInput, APIInstance

from .ragent_entity import RAGentEntity
from .ragent_config_entry import RAGentConfigEntry
from ..models.device import Device

from ..const import (
    CONF_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    PERSONA_PROMPTS,
    CURRENT_DATE_PROMPT,
    DEVICES_PROMPT,
    AREAS_PROMPT,
    USER_INSTRUCTION,
    DEVICE_CONTROL_PROMPT
)

from ..utils import (
    get_placeholder_translation,
    clean_device_attributes
)

_logger = logging.getLogger(__name__)

class RAGent(ConversationEntity, AbstractConversationAgent, RAGentEntity):
    """RAG-based conversation agent for Home Assistant."""
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry, subentry: ConfigSubentry) -> None:
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

    async def _async_embed_query(self, user_input: ConversationInput) -> bool:
        _logger.debug("RAG Step 1: Embedding user input: %s", user_input.text)
        query_embedding = None
        try:
            query_embedding = await self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), user_input.text)
            _logger.debug(f"User input embedded successfully, embedding shape: {len(query_embedding)}"),
        except Exception as e:
            _logger.error("Error embedding user input: %s", e, exc_info=True)
        
        return query_embedding

    async def _async_retrieve_devices(self, query_embedding: List[float], n_devices: int) -> List[Device]:
        _logger.debug("RAG Step 2: Querying vector database for similar devices")
        collection_name = f"devices_{self.subentry_id}"
        _logger.debug(f"Collection name: {collection_name}, Query embedding dimension: {len(query_embedding)}")
        try:
            retrieved_devices = await self.entry.vector_db_backend.async_retrieve_devices(
                dict(self.subentry.data),
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=n_devices
            )
            _logger.debug("Retrieved %d relevant devices from vector database (collection: %s)", len(retrieved_devices), collection_name)
        except Exception as e:
            _logger.error("Error retrieving devices from vector DB: %s", e, exc_info=True)
        
        return retrieved_devices

    async def _async_render_template(self, template_str: str, devices: List[Device], tools: List[str], icl_examples: List[str]) -> str:
        """Render a Jinja2 template string with the given context."""
        try:
            template = Template(template_str, self.hass)
            rendered = template.async_render({
                "device_list": devices,
                "area_list": list(set(device.area_name for device in devices if device.area_name)),
                "icl_examples": icl_examples
            })
            return rendered
        except TemplateError as e:
            _logger.error("Template rendering error: %s", e, exc_info=True)
            raise e

    async def _async_get_message_history(self, chat_log: conversation.ChatLog, user_input: ConversationInput, devices: List[Device]) -> List[conversation.Content]:
        """Build the prompt for the LLM, including retrieved device context."""
        raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)
        remember_conversation = self.runtime_options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)
        remember_num_interactions = self.runtime_options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)

        try:
            prompt_template = RAGent.build_base_prompt_template(user_input.language, raw_prompt)
            system_prompt_content = await self._async_render_template(prompt_template, devices, [], [])
            system_prompt = conversation.SystemContent(content=system_prompt_content)
        except Exception as err:
            _logger.error("Error rendering prompt: %s", err, exc_info=True)
            return None

        if remember_conversation:
            message_history = chat_log.content[:]
        else:
            message_history = []

        # trim message history before processing if necessary
        if remember_num_interactions and len(message_history) > (remember_num_interactions * 2) + 1:
            new_message_history = [message_history[0]] # copy system prompt
            new_message_history.extend(message_history[1:][-(remember_num_interactions * 2):])
            message_history = new_message_history

        if len(message_history) == 0:
            message_history.append(system_prompt)
        else:
            message_history[0] = system_prompt

        message_history.append(conversation.UserContent(content=user_input.text))

        # log the system prompt for debugging
        if message_history and len(message_history) > 0:
            msg = message_history[0]
            if isinstance(msg, conversation.SystemContent):
                _logger.debug("System prompt:\n%s", msg.content)
        
        return message_history

    def _get_tool_list(self, llm_api: APIInstance) -> List[Dict]:
        if not llm_api or not HAS_VOLUPTUOUS_OPENAPI:
            return []

        tools = []
        try:
            for tool in llm_api.tools:
                if tool.name == "GetLiveContext":
                    continue

                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                    }
                }

                if hasattr(tool, 'parameters') and tool.parameters:
                    try:
                        tool_def["function"]["parameters"] = convert(tool.parameters, custom_serializer=llm_api.custom_serializer)
                    except Exception as param_err:
                        _logger.debug("Could not convert parameters for tool %s: %s", tool.name, param_err)
                        tool_def["function"]["parameters"] = {}
                else:
                    tool_def["function"]["parameters"] = {}
                
                tools.append(tool_def)
        except Exception as err:
            _logger.warning("Failed to format tools for Ollama: %s", err)

        return tools

    def _parse_tool_calls(self, response_text: str) -> List[dict]:
        """Parse tool calls from LLM response."""
        parsed_calls = []
        
        _logger.debug("Parsing tool calls from LLM response.")
        homeassistant_pattern = r'```homeassistant\s*(.*?)\s*```'
        for match in re.finditer(homeassistant_pattern, response_text, re.DOTALL):
            try:
                content = match.group(1).strip()
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_str = content[first_brace:last_brace + 1]
                    tool_json = json.loads(json_str)
                    
                    parameters = tool_json.get("arguments", {})
                    if "device_class" in parameters:
                        parameters["domain"] = parameters.pop("device_class")

                    parsed_calls.append({
                        "name": tool_json.get("tool"),
                        "parameters": parameters
                    })

                    _logger.debug("Parsed tool call from homeassistant block: %s", tool_json)
            except (json.JSONDecodeError, AttributeError) as e:
                _logger.warning("Failed to parse homeassistant block JSON: %s", e)

        return parsed_calls
    
    def _parse_tool_results(self, tool_result: JsonObjectType) -> Dict[str, List[str]]:
        """Parse tool results from LLM response."""
        data = tool_result.get("data", {})
        success = data.get("success", [])
        failed = data.get("failed", [])
        tool_result = {}
        
        success_ids = [x["id"] for x in success if x.get("type") == "entity"]
        if success_ids:
            tool_result["success"] = success_ids

        failed_ids = [x["id"] for x in failed if x.get("type") == "entity"]
        if failed_ids:
            tool_result["failed"] = failed_ids

        return tool_result

    async def _async_prompt_model(self, llm_api: llm.APIInstance, user_input: ConversationInput, message_history: List[conversation.Content]) -> ConversationResult:
        """Process a prompt through the RAGent."""
        max_tool_call_iterations = self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)

        tool_calls: List[Tuple[llm.ToolInput, Any]] = []
        for idx in range(max(1, max_tool_call_iterations)):
            _logger.debug(f"Generating response for {user_input.text}, iteration {idx + 1}/{max_tool_call_iterations}.")
            
            formatted_messages = []
            for msg in message_history:
                if isinstance(msg, conversation.SystemContent):
                    formatted_messages.append({"role": "SYSTEM", "content": msg.content})
                elif isinstance(msg, conversation.UserContent):
                    formatted_messages.append({"role": "USER", "content": msg.content})
                elif isinstance(msg, conversation.AssistantContent):
                    formatted_messages.append({"role": "ASSISTANT", "content": msg.content})
                elif isinstance(msg, conversation.ToolResultContent):
                    formatted_messages.append({"role": "TOOL", "content": "{" + f"name: {msg.tool_name}, result: {msg.tool_result}" + "}"})

            tool_calls_in_iteration = []
            try:
                _logger.info(f"Sending prompt to LLM (Iteration {idx + 1}/{max_tool_call_iterations}).")

                full_prompt = []
                for msg in formatted_messages:
                    full_prompt.append(f"{msg.get('role')}: {msg.get('content')}")

                _logger.debug("Full prompt sent to the LLM:\n%s", "\n".join(full_prompt))
                
                assistant_content = ""

                tool_list = self._get_tool_list(llm_api)
                async for chunk in self.entry.llm_backend.async_send_chat_request(dict(self.subentry.data), formatted_messages, tool_list):
                    assistant_content += chunk

                _logger.debug("LLM response: %s", assistant_content)
                
                tool_calls_in_iteration = self._parse_tool_calls(assistant_content)
                
                message = conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=assistant_content,
                    tool_calls=tool_calls_in_iteration
                )
                message_history.append(message)
                
                if tool_calls_in_iteration and len(tool_calls_in_iteration) > 0:
                    _logger.info("Executing %d tool calls", len(tool_calls_in_iteration))
                    
                    for tool_call in tool_calls_in_iteration:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("parameters", {})
                        _logger.debug(f"Executing tool: {tool_name} with args: {tool_args}.")
                        
                        tool_input = ToolInput(tool_name=tool_name, tool_args=tool_args)
                        try:
                            if llm_api:
                                tool_result = await llm_api.async_call_tool(tool_input)
                                _logger.debug(f"Tool result: {tool_result}.")
                                
                                tool_calls.append((tool_input, tool_result))
                                
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_input.id,
                                    tool_name=tool_name,
                                    tool_result=self._parse_tool_results(tool_result)
                                )
                                message_history.append(tool_result_msg)
                            else:
                                _logger.warning("LLM API not available, skipping tool execution for tool: %s", tool_name)
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_input.id,
                                    tool_name=tool_name,
                                    tool_result="Tool calling is not active on this instance instruct the user to activate it manually."
                                )
                                message_history.append(tool_result_msg)

                        except Exception as tool_err:
                            tool_result_msg = conversation.ToolResultContent(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_input.id,
                                tool_name=tool_name,
                                tool_result=f"Tool '{tool_name}' failed with error: {str(tool_err)}"
                            )
                            message_history.append(tool_result_msg)
                    
            except Exception as err:
                _logger.error(f"There was a problem talking to the backend: {err}")
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, there was a problem talking to the backend.")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

            if not llm_api or not tool_calls_in_iteration or len(tool_calls_in_iteration) == 0:
                break

            if idx == max_tool_call_iterations - 1 and max_tool_call_iterations > 0 and tool_calls_in_iteration:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, I ran out of attempts to handle your request")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
            
        intent_response = intent.IntentResponse(language=user_input.language)
        if len(tool_calls) > 0:
            str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls]
            tools_str = '\n'.join(str_tools)
            intent_response.async_set_card(title="Changes", content=f"Ran the following tools:\n{tools_str}")

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

        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
        

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process the user request"""
        try:
            with (
                chat_session.async_get_chat_session(self.hass, user_input.conversation_id) as session,
                conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                llm_api: llm.APIInstance | None = None

                if self.runtime_options.get(CONF_LLM_HASS_API) != "none":
                    try:
                        llm_api = await llm.async_get_api(
                            self.hass,
                            self.runtime_options[CONF_LLM_HASS_API],
                            llm_context=user_input.as_llm_context(DOMAIN)
                        )
                    except HomeAssistantError as err:
                        _logger.error("Error getting LLM API: %s", err)
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Error preparing LLM API.")
                        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
                    
                # ensure this chat log has the LLM API instance
                chat_log.llm_api = llm_api

                query_embedding = await self._async_embed_query(user_input)
                if not query_embedding:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to embed user input.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                retrieved_devices = await self._async_retrieve_devices(query_embedding, n_devices=10)
                if not retrieved_devices:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to retrieve relevant devices.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                device_list = []
                for device in retrieved_devices:
                    st = self.hass.states.get(device.id)
                    if st is None:
                        continue

                    device.state = st.state
                    device.attributes = clean_device_attributes(st.attributes)
                    device_list.append(device)

                message_history = await self._async_get_message_history(chat_log, user_input, device_list)
                if not message_history:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Template rendering failed.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
                
                return await self._async_prompt_model(llm_api, user_input, message_history)
        except Exception as err:
            _logger.error("Unexpected error in async_process: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, an unexpected error occurred.")
            return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

    @staticmethod
    def build_base_prompt_template(selected_language: str, prompt_template: str):
        """Build prompt template with RAG-augmented device context."""
        prompt_template = prompt_template.replace("<persona>", get_placeholder_translation(PERSONA_PROMPTS, selected_language))
        prompt_template = prompt_template.replace("<current_date>", get_placeholder_translation(CURRENT_DATE_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<devices>", get_placeholder_translation(DEVICES_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<areas>", get_placeholder_translation(AREAS_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<device_control_prompt>", get_placeholder_translation(DEVICE_CONTROL_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<user_instruction>", get_placeholder_translation(USER_INSTRUCTION, selected_language))
        
        return prompt_template