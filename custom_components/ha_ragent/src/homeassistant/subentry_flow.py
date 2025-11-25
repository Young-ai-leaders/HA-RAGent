import logging
import os
from typing import Any
import voluptuous as vol

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.config_entries import (
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult
) 

from ..const import (
    CONF_LLM_BACKEND_TYPE,
    CONF_LLM_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_GBNF_GRAMMAR_ENABLED,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_IN_CONTEXT_LEARNING_ENABLED,
    CONF_IN_CONTEXT_LEARNING_FILE,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_REQUEST_TIMEOUT,
    CONF_SELECTED_LANGUAGE,

    DEFAULT_GBNF_GRAMMAR_FILE,
    DEFAULT_IN_CONTEXT_LEARNING_FILE,
    DEFAULT_OPTIONS,
    DEFAULT_PROMPT,
)

from ..utils import (
    try_parse_int
)

from .ui_schemas import (
    pick_remote_model_schema,
    ragent_config_option_schema
)

from .ragent_client import RAGentClient, RAGentConfigEntry

_logger = logging.getLogger(__name__)

class RagentSubentryFlowHandler(ConfigSubentryFlow):
    def __init__(self) -> None:
        super().__init__()

        self.model_config: dict[str, Any] = {}
        self.download_task = None
        self.download_error = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    @property
    def _client(self) -> RAGentClient:
        entry: RAGentConfigEntry = self._get_entry()
        return entry.runtime_data

    async def async_step_pick_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        schema = vol.Schema({})
        errors = {}
        description_placeholders = {}
        entry = self._get_entry()

        schema = pick_remote_model_schema(await entry.runtime_data.async_get_available_models())

        if user_input and "result" not in user_input:
            self.model_config.update(user_input)
            return await self.async_step_model_parameters()

        return self.async_show_form(
            step_id="pick_model",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=False,
        )

    async def async_step_model_parameters(
        self, user_input: dict[str, Any] | None = None,
    ) -> SubentryFlowResult:
        errors = {}
        description_placeholders = {}
        entry = self._get_entry()
        backend_type = entry.data[CONF_LLM_BACKEND_TYPE]

        if CONF_PROMPT not in self.model_config:
            selected_language = self.model_config.get(
                CONF_SELECTED_LANGUAGE, entry.options.get(CONF_SELECTED_LANGUAGE, "en")
            )

            selected_default_options = {**DEFAULT_OPTIONS}

            selected_default_options[CONF_PROMPT] = RAGentClient.build_prompt_template(
                selected_language, str(selected_default_options.get(CONF_PROMPT, DEFAULT_PROMPT))
            )
            
            self.model_config = {**selected_default_options, **self.model_config}

        schema = vol.Schema(
            ragent_config_option_schema(
                self.hass,
                entry.options.get(CONF_SELECTED_LANGUAGE, "en"),
                self.model_config,
                backend_type,
                self._subentry_type,
            )
        )

        if user_input:
            if not user_input.get(CONF_REFRESH_SYSTEM_PROMPT):
                errors["base"] = "sys_refresh_caching_enabled"

            if user_input.get(CONF_GBNF_GRAMMAR_ENABLED):
                filename = user_input.get(CONF_GBNF_GRAMMAR_FILE, DEFAULT_GBNF_GRAMMAR_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_gbnf_file"
                    description_placeholders["filename"] = filename

            if user_input.get(CONF_IN_CONTEXT_LEARNING_ENABLED):
                filename = user_input.get(CONF_IN_CONTEXT_LEARNING_FILE, DEFAULT_IN_CONTEXT_LEARNING_FILE)
                if not os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                    errors["base"] = "missing_icl_file"
                    description_placeholders["filename"] = filename

            for key in (
                CONF_REMEMBER_NUM_INTERACTIONS,
                CONF_MAX_TOOL_CALL_ITERATIONS,
                CONF_CONTEXT_LENGTH,
                CONF_MAX_TOKENS,
                CONF_REQUEST_TIMEOUT,
             ):
                if key in user_input:
                    user_input[key] = try_parse_int(user_input[key], user_input.get(key) or 0)
            
            if len(errors) == 0:
                try:
                    schema(user_input)
                    self.model_config.update(user_input)

                    if self.model_config.get(CONF_LLM_HASS_API) == "none":
                        self.model_config.pop(CONF_LLM_HASS_API, None)
                    
                    return await self.async_step_finish()
                except Exception:
                    _logger.exception("An unknown error has occurred!")
                    errors["base"] = "unknown"

        return self.async_show_form(
            step_id="model_parameters",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._is_new:
            return self.async_create_entry(
                title=self.model_config.get(CONF_LLM_MODEL, "Model"),
                data=self.model_config,
            )
        else:
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), data=self.model_config
            )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        
        if not self.model_config:
            self.model_config = {}
        
        return await self.async_step_pick_model(user_input)
    
    async_step_init = async_step_user
        
    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None):
        if not self.model_config:
            self.model_config = dict(self._get_reconfigure_subentry().data)

        return await self.async_step_model_parameters(user_input)
