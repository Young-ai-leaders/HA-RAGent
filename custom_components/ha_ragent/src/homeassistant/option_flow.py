from asyncio import Task
import logging
from typing import Any

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.config_entries import (
    ConfigFlowResult,
    OptionsFlow,
)

from ..const import (
    BACKEND_TO_CLASS, 
    CONF_LLM_BACKEND_TYPE,
    DEFAULT_LLM_BACKEND_TYPE
)

from .ui_schemas import (
    remote_connection_schema
)

_logger = logging.getLogger(__name__)

class RagentOptionsFlow(OptionsFlow):
    def __init__(self):
        super().__init__()
        self.model_config: dict[str, Any] | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors = {}
        description_placeholders = {}

        backend_type = self.config_entry.data.get(CONF_LLM_BACKEND_TYPE, DEFAULT_LLM_BACKEND_TYPE)
        client_config = dict(self.config_entry.options)

        if user_input is not None:
            client_config.update(user_input)

            connect_err = await BACKEND_TO_CLASS[backend_type].async_validate_connection(self.hass, client_config)

            if not connect_err:
                return self.async_create_entry(data=client_config)
            else:
                errors["base"] = "failed_to_connect"
                description_placeholders["exception"] = str(connect_err)

        schema = remote_connection_schema(
            backend_type=backend_type,
            host=client_config.get(CONF_HOST),
            port=client_config.get(CONF_PORT),
            ssl=client_config.get(CONF_SSL),
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )