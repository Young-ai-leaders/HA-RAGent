import logging
from typing import Any

from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.config_entries import (
    ConfigFlowResult,
    OptionsFlow,
)

from ..const import (
    CONF_EMBEDDING_BACKEND_SECTION,
    CONF_LLM_BACKEND_SECTION,
    CONF_LLM_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_VECTOR_DB_PASSWORD,
    CONF_VECTOR_DB_SECTION,
    CONF_VECTOR_DB_USERNAME
)

from ..utils import embedding_backend_to_class, embedding_backend_to_class, llm_backend_to_class

from .ui_schemas import (
    ui_schema_backend_connections
)

_logger = logging.getLogger(__name__)

class RagentOptionsFlow(OptionsFlow):
    def __init__(self):
        super().__init__()
        self.model_config: dict[str, Any] | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors = {}
        description_placeholders = {}
        client_config = dict(self.config_entry.options)

        if user_input is not None:
            client_config.update(user_input)
            connect_err = None

            if not connect_err:
                connect_err = await embedding_backend_to_class(client_config[CONF_EMBEDDING_BACKEND_TYPE]).async_validate_connection(self.hass, client_config)
                
            if not connect_err:
                connect_err = await llm_backend_to_class(client_config[CONF_LLM_BACKEND_TYPE]).async_validate_connection(self.hass, client_config)

            if not connect_err:
                return self.async_create_entry(data=client_config)
            else:
                errors["base"] = "failed_to_connect"
                description_placeholders["exception"] = str(connect_err)

        schema = ui_schema_backend_connections(
            vector_db_backend_type=client_config[CONF_VECTOR_DB_BACKEND_TYPE],
            embedding_backend_type=client_config[CONF_EMBEDDING_BACKEND_TYPE],
            llm_backend_type=client_config[CONF_LLM_BACKEND_TYPE],
            vector_db_username=client_config[CONF_VECTOR_DB_SECTION].get(CONF_VECTOR_DB_USERNAME),
            vector_db_password=client_config[CONF_VECTOR_DB_SECTION].get(CONF_VECTOR_DB_PASSWORD),
            vector_db_host=client_config[CONF_VECTOR_DB_SECTION].get(CONF_HOST),
            vector_db_port=client_config[CONF_VECTOR_DB_SECTION].get(CONF_PORT),
            vector_db_ssl=client_config[CONF_VECTOR_DB_SECTION].get(CONF_SSL),
            embedding_host=client_config[CONF_EMBEDDING_BACKEND_SECTION].get(CONF_HOST),
            embedding_port=client_config[CONF_EMBEDDING_BACKEND_SECTION].get(CONF_PORT),
            embedding_ssl=client_config[CONF_EMBEDDING_BACKEND_SECTION].get(CONF_SSL),
            llm_host=client_config[CONF_LLM_BACKEND_SECTION].get(CONF_HOST),
            llm_port=client_config[CONF_LLM_BACKEND_SECTION].get(CONF_PORT),
            llm_ssl=client_config[CONF_LLM_BACKEND_SECTION].get(CONF_SSL))
        
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )