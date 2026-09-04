from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, List, Tuple

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.template import Template
from homeassistant.helpers.llm import LLMContext
from homeassistant.helpers import area_registry as ar, device_registry as dr, floor_registry as fr
from probatio import to_openapi

from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import HistoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.homeassistant.helpers.tool_helper import ToolHelper
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.chat_message import ChatMessage
from custom_components.ha_ragent.src.models.memory import Memory
from custom_components.ha_ragent.src.models.turn_context import ContinuityContext, TurnContext

from custom_components.ha_ragent.src.homeassistant.ragent_entity import RAGentEntity
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.homeassistant.ragent_api import (
    RAGentAugmentedAPIInstance,
    resolve_llm_api_id,
)
from custom_components.ha_ragent.src.models.device import Device

from custom_components.ha_ragent.src.const import (
    CONF_NUM_DEVICES_TO_EXTRACT,
    CONF_NUM_TOOLS_TO_EXTRACT,
    CONF_NUM_MEMORIES_TO_EXTRACT,
    CONF_PROMPT,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_NUM_MEMORIES_TO_EXTRACT,
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DOMAIN,
    PERSONA_PROMPTS,
    DEVICES_PROMPT,
    AREAS_PROMPT,
    MAX_RETRIES_PROMPT,
    INSTRUCTION_PROMPT,
    MEMORIES_CONTEXT_PROMPT,
    CONF_SELECTED_LANGUAGE,
    DEFAULT_SELECTED_LANGUAGE,
    CONF_ALLOW_QUESTIONS,
    DEFAULT_ALLOW_QUESTIONS,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    RAGENT_SCHEDULED_REQUEST_PREFIX,
    RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES,
)

from custom_components.ha_ragent.src.utils import (
    get_placeholder_translation,
    clean_device_attributes
)

_logger = logging.getLogger(__name__)

class RAGent(ConversationEntity, AbstractConversationAgent, RAGentEntity):
    """RAG-based conversation agent for Home Assistant."""
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry, subentry: ConfigSubentry) -> None:
        super().__init__(hass, entry, subentry)
        self._history_contexts: dict[str, dict[str, tuple[TurnContext, float]]] = {}
        self._history_vectors: dict[tuple[str, str], list[float]] = {}

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

    async def _async_embed_query(self, user_input: ConversationInput) -> list[float] | None:
        """Embed the current request without language-specific heuristics."""
        retrieval_text = RetrievalHelper.build_retrieval_text(user_input.text)

        try:
            embedding = await self.entry.embedder_backend.async_embed_text(dict(self.subentry.data), retrieval_text)
        except Exception as err:
            _logger.error(f"Error embedding user input with history: {err}", exc_info=True)
            return None

        return embedding or None

    async def _async_build_continuity_context(self, history_manager: HistoryManager, chat_log: conversation.ChatLog, conversation_id: str | None, query_embedding: list[float]) -> ContinuityContext:
        """Retrieve short-term structured and longer-term semantic history."""
        extracted_contexts = history_manager.structured_turn_contexts(chat_log)
        cache_scope = str(conversation_id or "")
        now = time.time()

        if cache_scope:
            stored = self._history_contexts.setdefault(cache_scope, {})
            for context in extracted_contexts:
                stored[context.key] = (context, now)
            stored = {
                key: value
                for key, value in stored.items()
                if now - value[1] <= 300.0
            }
            stored = dict(list(stored.items())[-10:])
            self._history_contexts[cache_scope] = stored
            contexts = [context for context, _ in stored.values()]
        else:
            contexts = extracted_contexts

        missing = [
            context
            for context in contexts
            if (cache_scope, context.key) not in self._history_vectors
        ]

        async def embed_context(context: TurnContext) -> tuple[TurnContext, list[float] | None]:
            try:
                vector = await self.entry.embedder_backend.async_embed_text(
                    dict(self.subentry.data),
                    context.to_embedding_text(),
                )
                return context, vector or None
            except Exception as err:
                _logger.debug("Failed to embed semantic history turn: %s", err)
                return context, None

        if missing:
            for context, vector in await asyncio.gather(*(embed_context(context) for context in missing)):
                if vector:
                    self._history_vectors[(cache_scope, context.key)] = vector

        active_keys = {(cache_scope, context.key) for context in contexts}
        self._history_vectors = {
            key: vector
            for key, vector in self._history_vectors.items()
            if key[0] != cache_scope or key in active_keys
        }
        vectors = {
            context.key: self._history_vectors.get((cache_scope, context.key), [])
            for context in contexts
        }
        selected = RetrievalHelper.select_history_contexts(
            contexts,
            vectors,
            query_embedding,
            max_age_seconds=300.0,
        )
        return RetrievalHelper.build_continuity_context(selected)

    async def _async_retrieve_devices(self, query_embedding: List[float], query: str, n_devices: int, continuity: ContinuityContext) -> List[Device]:
        """Retrieve relevant devices from vector database based on query embedding."""
        if n_devices <= 0:
            return []
        collection_name = f"devices_{self.subentry_id}"
        try:
            candidate_limit = RetrievalHelper.adaptive_candidate_limit(n_devices)
            scored_devices, all_devices = await asyncio.gather(
                self.entry.vector_db_backend.async_retrieve_scored_objects(
                    DeviceEmbedding,
                    dict(self.subentry.data),
                    collection_name,
                    query_embedding,
                    candidate_limit,
                ),
                self.entry.vector_db_backend.async_list_objects(
                    DeviceEmbedding,
                    dict(self.subentry.data),
                    collection_name,
                ),
            )
        except Exception as e:
            _logger.error(f"Error retrieving devices from vector DB: {e}", exc_info=True)
            return []

        return RetrievalHelper.rank_scored_candidates(
            scored_devices,
            all_devices,
            query,
            lambda device: device.id,
            lambda device: (
                device.id,
                device.friendly_name,
                *(device.aliases or []),
                device.area_name,
                device.floor_name,
                *(device.domain or []),
                device.device_class,
                *(device.device_labels or []),
            ),
            n_devices,
            metadata_score=lambda device: RetrievalHelper.field_match_score(
                query,
                (
                    device.area_name,
                    device.floor_name,
                    *(device.domain or []),
                    device.device_class,
                    *(device.device_labels or []),
                ),
            ) + continuity.device_score(device),
        )

    async def _async_retrieve_tools(self, query_embedding: List[float], query: str, n_tools: int, continuity: ContinuityContext) -> List[LlmTool]:
        """Retrieve relevant tools from vector database based on query embedding."""
        if n_tools <= 0:
            return []
        collection_name = f"tools_{self.subentry_id}"
        try:
            candidate_limit = RetrievalHelper.adaptive_candidate_limit(n_tools)
            scored_tools, all_tools = await asyncio.gather(
                self.entry.vector_db_backend.async_retrieve_scored_objects(
                    LlmToolEmbedding,
                    dict(self.subentry.data),
                    collection_name,
                    query_embedding,
                    candidate_limit,
                ),
                self.entry.vector_db_backend.async_list_objects(
                    LlmToolEmbedding,
                    dict(self.subentry.data),
                    collection_name,
                ),
            )
        except Exception as e:
            _logger.error(f"Error retrieving tools from vector DB: {e}", exc_info=True)
            return []

        return RetrievalHelper.rank_scored_candidates(
            scored_tools,
            all_tools,
            query,
            lambda tool: tool.name,
            lambda tool: (tool.name, tool.description),
            min(candidate_limit, n_tools * 2),
            metadata_score=lambda tool: RetrievalHelper.field_match_score(
                query,
                ((tool.parameters or {}).get("properties") or {}).keys(),
            ) + continuity.tool_score(tool),
        )

    async def _async_retrieve_memories(self, query_embedding: List[float], n_memories: int) -> List[Memory]:
        """Retrieve relevant persistent memories for this agent."""
        try:
            return await MemoryManager(
                self.hass,
                self.entry_id,
                self.subentry_id,
            ).async_recall(query_embedding, n_memories)
        except Exception as err:
            _logger.error(f"Error retrieving memories from vector DB: {err}", exc_info=True)
            return []

    async def _async_render_system_prompt(self, devices: List[Device], memories: List[Memory], area: ar.AreaEntry, floor: fr.FloorEntry) -> str | None:
        """Render the system prompt with retrieved device context."""
        raw_prompt = self.runtime_options.get(CONF_PROMPT, DEFAULT_PROMPT)

        try:
            template = Template(raw_prompt, self.hass)
            rendered_prompt = template.async_render({
                "device_list": devices,
                "memory_list": memories,
                "area_list": list(set(device.area_name for device in devices if device.area_name)),
                "area_name": area.name if area else None,
                "floor_name": floor.name if floor else None,
                "max_retries": self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS),
            })
            return rendered_prompt
        except Exception as err:
            _logger.error(f"Error rendering prompt: {err}", exc_info=True)
            return None

    def _convert_api_tool(self, api_tool: Any, llm_api: llm.APIInstance | None) -> LlmTool | None:
        """Convert a Home Assistant LLM tool into the local tool schema."""
        tool_name = getattr(api_tool, "name", None)
        if not tool_name:
            return None

        parameters = {}
        if hasattr(api_tool, "parameters") and api_tool.parameters:
            try:
                parameters = to_openapi(api_tool.parameters, custom_serializer=llm_api.custom_serializer if llm_api else None)
                if not isinstance(parameters, dict):
                    _logger.warning(f"Could not convert parameters for tool {tool_name}: converter returned {type(parameters).__name__}")
                    parameters = {}
            except Exception as err:
                _logger.warning(f"Could not convert parameters for tool {tool_name}: {err}")
                parameters = {}

        return LlmTool(
            name=tool_name,
            description=getattr(api_tool, "description", ""),
            parameters=parameters,
            metadata={},
        )

    def _ensure_required_tools_exposed(self, tool_list: List[LlmTool], llm_api: llm.APIInstance | None) -> List[LlmTool]:
        """Expose required tools before tools selected by semantic retrieval."""
        if not llm_api or not hasattr(llm_api, "tools"):
            return tool_list

        required_names = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        required_tools = [tool for tool in tool_list if tool.name in required_names]
        searched_tools = [tool for tool in tool_list if tool.name not in required_names]
        seen_tool_names = {tool.name for tool in tool_list}

        for api_tool in llm_api.tools:
            tool_name = getattr(api_tool, "name", None)
            if tool_name not in required_names or tool_name in seen_tool_names:
                continue

            converted_tool = self._convert_api_tool(api_tool, llm_api)
            if converted_tool:
                required_tools.append(converted_tool)
                seen_tool_names.add(tool_name)

        return [*required_tools, *searched_tools]

    @staticmethod
    def _exclude_prohibited_scheduled_request_tools(tool_list: List[LlmTool], scheduled_request: bool) -> List[LlmTool]:
        """Exclude scheduling-management tools from due scheduled actions."""
        if not scheduled_request:
            return tool_list
        
        prohibited = set(RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES)
        return [tool for tool in tool_list if tool.name not in prohibited]

    def _get_current_device_location(self, llm_context: LLMContext) -> ar.AreaEntry | None:
        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

        return area, floor

    async def _async_prompt_model(
        self,
        llm_api: llm.APIInstance,
        user_input: ConversationInput,
        tool_list: List[LlmTool],
        chat_log: conversation.ChatLog,
        history_manager: HistoryManager,
        scheduled_request: bool = False
    ) -> ConversationResult:
        """Process a prompt through the RAGent."""
        tool_helper = ToolHelper(self.hass)
        max_tool_call_iterations = self.runtime_options.get(CONF_MAX_TOOL_CALL_ITERATIONS, DEFAULT_MAX_TOOL_CALL_ITERATIONS)

        tool_calls_overall: List[Tuple[llm.ToolInput, Any]] = []
        executed_tool_calls: set[str] = set()
        tool_call_results: dict[str, Any] = {}
        tool_metadata_dict: dict[str, dict[str, Any]] = {tool.name: tool.metadata for tool in tool_list if tool.metadata}
        formatted_messages: list[ChatMessage] = []
        formatted_index = 0

        for idx in range(max(1, max_tool_call_iterations)):
            iteration_start = time.perf_counter()
            formatted_messages.extend(MessageHelper.message_to_chat_messages(history_manager.message_history[formatted_index:]))
            formatted_index = len(history_manager.message_history)

            tool_calls_in_iteration = []
            try:
                _logger.debug(f"Sending prompt to LLM (Iteration {idx + 1}/{max_tool_call_iterations}).")
                if _logger.isEnabledFor(logging.DEBUG):
                    _logger.debug(f"Full messages sent to the LLM:\n{json.dumps(formatted_messages, ensure_ascii=False, indent=2, default=str)}")
                
                content_chunks = []
                async for chunk in self.entry.llm_backend.async_send_chat_request(dict(self.subentry.data), formatted_messages, tool_list):
                    content_chunks.append(chunk)
                assistant_content = "".join(content_chunks)
                _logger.debug(f"RAGent timing: LLM iteration {idx + 1}: {time.perf_counter() - iteration_start:.3f}s")

                _logger.debug(f"RAW LLM response: {assistant_content}")
                
                helper_start = time.perf_counter()
                tool_calls_in_iteration = tool_helper.parse_tool_calls(assistant_content, tool_metadata_dict)
                _logger.debug(f"RAGent timing: parse_tool_calls: {time.perf_counter() - helper_start:.3f}s")

                helper_start = time.perf_counter()
                message_content = MessageHelper.clean_assistant_content(assistant_content, bool(tool_calls_in_iteration))
                _logger.debug(f"RAGent timing: clean_assistant_content: {time.perf_counter() - helper_start:.3f}s")

                helper_start = time.perf_counter()
                history_tool_calls = [tool_helper.to_history_tool_call(call) for call in tool_calls_in_iteration]
                _logger.debug(f"RAGent timing: to_history_tool_call: {time.perf_counter() - helper_start:.3f}s")

                message = conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=message_content,
                    tool_calls=history_tool_calls
                )
                history_manager.append_message(message)
                
                if tool_calls_in_iteration and len(tool_calls_in_iteration) > 0:                    
                    for tool_call in tool_calls_in_iteration:
                        tool_name = tool_call.tool_name
                        tool_args = tool_call.tool_args

                        try:
                            if llm_api:
                                helper_start = time.perf_counter()
                                tool_helper.block_broad_tool_calls(tool_call, tool_metadata_dict.get(tool_name))
                                _logger.debug(f"RAGent timing: block_broad_tool_calls ({tool_name}): {time.perf_counter() - helper_start:.3f}s")

                                helper_start = time.perf_counter()
                                tool_call_signature = tool_helper.tool_call_signature(tool_call)
                                _logger.debug(f"RAGent timing: tool_call_signature ({tool_name}): {time.perf_counter() - helper_start:.3f}s")

                                if tool_call_signature in executed_tool_calls:
                                    _logger.debug(f"Returning already-executed result for repeated tool call: {tool_name} with arguments {tool_args}")
                                    tool_result_msg = MessageHelper.create_repeated_tool_result_message(
                                        agent_id=user_input.agent_id,
                                        tool_call_id=tool_call.id,
                                        tool_name=tool_name,
                                        previous_result=tool_call_results[tool_call_signature],
                                    )
                                    history_manager.append_message(tool_result_msg)
                                    continue

                                executed_tool_calls.add(tool_call_signature)
                                tool_start = time.perf_counter()
                                execution_call = tool_helper.to_home_assistant_tool_call(tool_call)
                                tool_result = await llm_api.async_call_tool(execution_call)
                                tool_call_results[tool_call_signature] = tool_result
                                tool_calls_overall.append((tool_call, tool_result))                                
                                parsed_tool_result = tool_helper.parse_tool_results(tool_result)
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    tool_result=parsed_tool_result
                                )
                                history_manager.append_message(tool_result_msg)
                                _logger.debug(f"RAGent timing: tool {tool_name}: {time.perf_counter() - tool_start:.3f}s")
                            else:
                                _logger.warning(f"LLM API not available, skipping tool execution for tool: {tool_name}")
                                tool_result_msg = conversation.ToolResultContent(
                                    agent_id=user_input.agent_id,
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_name,
                                    tool_result="Tool calling is not active on this instance instruct the user to activate it manually."
                                )
                                history_manager.append_message(tool_result_msg)

                        except Exception as tool_err:
                            _logger.debug(f"Tool {tool_name} failed; passing the failure back to the model: {tool_err}")
                            tool_result_msg = MessageHelper.create_tool_failure_message(
                                agent_id=user_input.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                error=tool_err,
                            )
                            history_manager.append_message(tool_result_msg)

            except Exception as err:
                _logger.error(f"There was a problem talking to the backend: {err}")
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, there was a problem talking to the backend.")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

            history_manager.persist_chat_history(chat_log)

            if not tool_calls_in_iteration:
                break

            if idx + 1 == max_tool_call_iterations:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, I ran out of attempts to handle your request")
                return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
            
        intent_response = intent.IntentResponse(language=user_input.language)
        if len(tool_calls_overall) > 0:
            str_tools = [f"{input.tool_name}({', '.join(str(x) for x in input.tool_args.values())})" for input, response in tool_calls_overall]
            tools_str = '\n'.join(str_tools)
            intent_response.async_set_card(title="Changes", content=f"Ran the following tools:\n{tools_str}")

        has_speech = False
        continue_conversation = False
        for cur_msg in reversed(history_manager.message_history[1:]):
            if isinstance(cur_msg, conversation.AssistantContent) and cur_msg.content:
                speech = cur_msg.content.strip()

                if (
                    self.runtime_options.get(CONF_ALLOW_QUESTIONS, DEFAULT_ALLOW_QUESTIONS)
                    and speech.endswith(("?", ";", "？"))
                ):
                    continue_conversation = True

                intent_response.async_set_speech(speech)
                has_speech = True
                break

        if not has_speech:
            intent_response.async_set_speech("I don't have anything to say right now")

        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id, continue_conversation=continue_conversation)
        

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process the user request"""
        timing_start = time.perf_counter()
        timing_mark = timing_start

        def log_timing(stage: str, details: str | None = None) -> None:
            nonlocal timing_mark
            now = time.perf_counter()
            detail_text = f"{details} in " if details else ""
            _logger.debug(f"RAGent {stage}: {detail_text}{now - timing_mark:.3f}s (total {now - timing_start:.3f}s)")
            timing_mark = now

        scheduled_request = user_input.text.startswith(RAGENT_SCHEDULED_REQUEST_PREFIX)
        if scheduled_request:
            user_input.text = user_input.text.removeprefix(RAGENT_SCHEDULED_REQUEST_PREFIX)
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
                            resolve_llm_api_id(self.runtime_options[CONF_LLM_HASS_API]),
                            llm_context=user_input.as_llm_context(DOMAIN),
                        )
                        if isinstance(llm_api, RAGentAugmentedAPIInstance):
                            llm_api.set_conversation_agent_id(user_input.agent_id)
                            llm_api.set_search_scope(self.entry_id,self.subentry_id)

                    except HomeAssistantError as err:
                        _logger.error(f"Error getting LLM API: {err}")
                        intent_response = intent.IntentResponse(language=user_input.language)
                        intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Error preparing LLM API.")
                        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                    log_timing("timing: LLM API setup")
                    
                # ensure this chat log has the LLM API instance
                chat_log.llm_api = llm_api

                history_manager = HistoryManager(
                    runtime_options=self.runtime_options,
                )
                query_embedding = await self._async_embed_query(user_input)
                embedding_dimension = len(query_embedding) if query_embedding else 0
                log_timing(f"Step 1 embedded retrieval query with shape {embedding_dimension}")
                if not query_embedding:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to embed user input.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                continuity = await self._async_build_continuity_context(
                    history_manager,
                    chat_log,
                    user_input.conversation_id,
                    query_embedding,
                )

                retrieve_devices_task = self._async_retrieve_devices(query_embedding, user_input.text, n_devices=self.runtime_options.get(CONF_NUM_DEVICES_TO_EXTRACT, DEFAULT_NUM_DEVICES_TO_EXTRACT), continuity=continuity)
                retrieve_tools_task = self._async_retrieve_tools(query_embedding, user_input.text, n_tools=self.runtime_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT), continuity=continuity) if llm_api else None
                retrieve_memories_task = self._async_retrieve_memories(query_embedding, n_memories=self.runtime_options.get(CONF_NUM_MEMORIES_TO_EXTRACT, DEFAULT_NUM_MEMORIES_TO_EXTRACT))

                if retrieve_tools_task:
                    retrieved_devices, retrieved_tools, retrieved_memories = await asyncio.gather(
                        retrieve_devices_task,
                        retrieve_tools_task,
                        retrieve_memories_task,
                    )
                else:
                    retrieved_devices, retrieved_memories = await asyncio.gather(
                        retrieve_devices_task,
                        retrieve_memories_task,
                    )
                    retrieved_tools = []
                log_timing(f"Step 2 retrieved {len(retrieved_devices)} devices, {len(retrieved_tools)} tools and {len(retrieved_memories)} memories")

                retrieved_tools = RetrievalHelper.rerank_tools_for_devices(
                    retrieved_tools,
                    retrieved_devices,
                    self.runtime_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT),
                )

                retrieved_tools = self._ensure_required_tools_exposed(retrieved_tools, llm_api)
                retrieved_tools = self._exclude_prohibited_scheduled_request_tools(retrieved_tools, scheduled_request)

                if not retrieved_tools and llm_api:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to retrieve relevant tools.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                device_list = []
                for device in retrieved_devices:
                    st = self.hass.states.get(device.id)
                    if st is None:
                        continue

                    device.state = st.state
                    device.attributes = clean_device_attributes(st.attributes)
                    device_list.append(device)
                
                area, floor = self._get_current_device_location(user_input.as_llm_context(DOMAIN))

                if isinstance(llm_api, RAGentAugmentedAPIInstance):
                    llm_api.set_search_context(
                        latest_request=user_input.text,
                        area=area.name if area else "",
                        floor=floor.name if floor else "",
                        candidates=[
                            {
                                "name": device.id,
                                "friendly_name": device.friendly_name,
                                "aliases": device.aliases,
                                "area": device.area_name,
                                "floor": device.floor_name,
                                "domain": device.domain,
                                "device_class": device.device_class,
                            }
                            for device in device_list
                        ],
                    )

                system_prompt_content = await self._async_render_system_prompt(
                    device_list,
                    retrieved_memories,
                    area,
                    floor,
                )
                log_timing("timing: system prompt rendering")
                if not system_prompt_content:
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Template rendering failed.")
                    return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

                history_manager.build_prompt_history(
                    chat_log,
                    user_input,
                    system_prompt_content,
                )

                result = await self._async_prompt_model(
                    llm_api,
                    user_input,
                    retrieved_tools,
                    chat_log,
                    history_manager,
                    scheduled_request
                )
                log_timing("model and tool processing")
                return result
        except Exception as err:
            _logger.error(f"Unexpected error in async_process: {err}")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, f"Sorry, an unexpected error occurred.")
            return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)

    @staticmethod
    def build_base_prompt_template(selected_language: str, prompt_template: str):
        """Build base prompt template from constants in specified language."""
        prompt_template = prompt_template.replace("<persona_prompt>", get_placeholder_translation(PERSONA_PROMPTS, selected_language))
        prompt_template = prompt_template.replace("<area_prompt>", get_placeholder_translation(AREAS_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<devices_prompt>", get_placeholder_translation(DEVICES_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<memories_context_prompt>", get_placeholder_translation(MEMORIES_CONTEXT_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<max_retries_prompt>", get_placeholder_translation(MAX_RETRIES_PROMPT, selected_language))
        prompt_template = prompt_template.replace("<instruction_prompt>", get_placeholder_translation(INSTRUCTION_PROMPT, selected_language))

        return prompt_template
