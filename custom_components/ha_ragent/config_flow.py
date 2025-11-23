from __future__ import annotations

import logging
from typing import Any
import voluptuous as vol

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.config_entries import (
    ConfigEntriesFlowManager,
    ConfigFlow,
    ConfigFlowResult
)
from homeassistant.helpers import llm

from .src.const import (
    BACKEND_TO_CLASS,
    BACKEND_TYPE_OPTIONS,
    
    CONF_BACKEND_TYPE,
    CONF_BACKEND_PATH,
    CONF_SELECTED_LANGUAGE,
    
    DEFAULT_BACKEND_TYPE,
    DEFAULT_LANGUAGE,
    DOMAIN,

    LLM_API_ID,
    SELECTED_LANGUAGE_OPTIONS
)

from .src.homeassistant.ui_schemas import (
    remote_connection_schema,
    pick_backend_schema
)

from .src.utils import (
    is_valid_host
)

_logger = logging.getLogger(__name__)

class RagentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Local LLM Conversation."""
    VERSION = 1

    client_config: dict[str, Any] = {}
    flow_step: str = "init"

    @property
    def flow_manager(self) -> ConfigEntriesFlowManager:
        """Return the correct flow manager."""
        return self.hass.config_entries.flow
            
    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initialization step."""
        match self.flow_step:
            case "init": return await self._init_flow()
            case "configure_backend": return await self._configure_backend(user_input)
            case "connect_to_backend": return await self._connect_to_backend_async(user_input)
            case _: raise AbortFlow("Uknown config flow step.")
                
    async def _init_flow(self) -> ConfigFlowResult:
        """Registers the llm api in home assitant if not already present."""
        if not any([x.id == LLM_API_ID for x in llm.async_get_apis(self.hass)]):
            #llm.async_register_api(self.hass, HomeLLMAPI(self.hass))

            self.flow_step = "configure_backend"
            return self.async_show_form(
                step_id="user", 
                data_schema=self._pick_backend_schema(), 
                last_step=False
            )
            
        return self.async_abort(reason="already_configured")
            
    async def _configure_backend(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input:
            self.client_config.update(user_input)
            self.flow_step = "connect_to_backend"
            return self.async_show_form(
                step_id="user", 
                data_schema=remote_connection_schema(self.client_config[CONF_BACKEND_TYPE]),
                last_step=True
            )
        return self.async_show_form(
            step_id="user", 
            data_schema=pick_backend_schema(
                backend_type=self.client_config.get(CONF_BACKEND_TYPE),
                selected_language=self.client_config.get(CONF_SELECTED_LANGUAGE)), 
            last_step=False)
        
    async def _connect_to_backend_async(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors = {}
        description_placeholders = {}
        if user_input:
            self.client_config.update(user_input)
            hostname = user_input.get(CONF_HOST, "")
            if not is_valid_host(hostname):
                errors["base"] = "invalid_hostname"
                description_placeholders["exception"] = "The provided hostname could not be resolved to an IP address."
            else:
                connect_err = await BACKEND_TO_CLASS[self.client_config[CONF_BACKEND_TYPE]].async_validate_connection(self.hass, self.client_config)
                if connect_err:
                    errors["base"] = "failed_to_connect"
                    description_placeholders["exception"] = str(connect_err)
                else:
                    return await self._step_finish_async()
            
        return self.async_show_form(
            step_id="user", 
            data_schema=remote_connection_schema(
                self.client_config[CONF_BACKEND_TYPE],
                host=self.client_config.get(CONF_HOST),
                port=self.client_config.get(CONF_PORT),
                ssl=self.client_config.get(CONF_SSL)), 
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=True
        )
        
    async def _step_finish_async(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        backend = self.client_config[CONF_BACKEND_TYPE]
        title = BACKEND_TO_CLASS[backend].get_name(self.client_config)
        _logger.debug(f"Creating provider with config: {self.client_config}")

        return self.async_create_entry(
            title=title,
            description="A Large Language Model Chat Agent",
            data={CONF_BACKEND_TYPE: backend},
            options=self.client_config,
        )