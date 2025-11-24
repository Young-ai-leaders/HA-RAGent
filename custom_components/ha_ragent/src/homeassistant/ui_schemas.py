import logging
from typing import Any
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL, CONF_LLM_HASS_API, UnitOfTime
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TemplateSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    BooleanSelector,
    BooleanSelectorConfig,
)

from ..const import (
    BACKEND_LLM_TYPE_OPTIONS,
    CONF_LLM_BACKEND_TYPE,
    CONF_LLM_MODEL,
    CONF_CONTEXT_LENGTH,
    CONF_GBNF_GRAMMAR_ENABLED,
    CONF_GBNF_GRAMMAR_FILE,
    CONF_IN_CONTEXT_LEARNING_ENABLED,
    CONF_IN_CONTEXT_LEARNING_FILE,
    CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_OLLAMA_JSON_MODE,
    CONF_OLLAMA_KEEP_ALIVE_MIN,
    CONF_PROMPT,
    CONF_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_NUM_INTERACTIONS,
    CONF_REQUEST_TIMEOUT,
    CONF_SELECTED_LANGUAGE,
    CONF_TEMPERATURE,
    CONF_K_TOP,
    CONF_P_MIN,
    CONF_P_TOP,
    CONF_P_TYPICAL,

    DEFAULT_LLM_BACKEND_TYPE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_IN_CONTEXT_LEARNING_ENABLED,
    DEFAULT_IN_CONTEXT_LEARNING_FILE,
    DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALL_ITERATIONS,
    DEFAULT_OLLAMA_JSON_MODE,
    DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    DEFAULT_PROMPT,
    DEFAULT_REFRESH_SYSTEM_PROMPT,
    DEFAULT_REMEMBER_CONVERSATION,
    DEFAULT_REMEMBER_NUM_INTERACTIONS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SELECTED_LANGUAGE,
    DEFAULT_TEMPERATURE,
    DEFAULT_K_TOP,
    DEFAULT_P_MIN,
    DEFAULT_P_TOP,
    DEFAULT_P_TYPICAL,

    SELECTED_LANGUAGE_OPTIONS,
)

from ..utils import (
    get_value
)

from .ragent_client import RAGent

_logger = logging.getLogger(__name__)

def pick_backend_schema(backend_type=None, selected_language=None) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                CONF_LLM_BACKEND_TYPE,
                default=get_value(backend_type, DEFAULT_LLM_BACKEND_TYPE)
            ): SelectSelector(SelectSelectorConfig(
                options=BACKEND_LLM_TYPE_OPTIONS,
                translation_key=CONF_LLM_BACKEND_TYPE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
            vol.Required(
                CONF_SELECTED_LANGUAGE, 
                default=get_value(selected_language, DEFAULT_SELECTED_LANGUAGE)
            ): SelectSelector(SelectSelectorConfig(
                options=SELECTED_LANGUAGE_OPTIONS,
                translation_key=CONF_SELECTED_LANGUAGE,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )

def remote_connection_schema(backend_type: str, host=None, port=None, ssl=None):
    if backend_type not in BACKEND_LLM_TYPE_OPTIONS:
        raise AbortFlow("Uknown backend type.")
        
    default_port = 11434
    return vol.Schema(
        {
            vol.Required(CONF_HOST, default=host if host else ""): str,
            vol.Optional(CONF_PORT, default=port if port else default_port): int,
            vol.Required(CONF_SSL, default=ssl if ssl else False): bool,
        }
    )

def pick_remote_model_schema(available_models: list[str], chat_model: str | None = None):
    _logger.debug(f"available models: {available_models}")
    return vol.Schema(
        {
            vol.Required(CONF_LLM_MODEL, default=chat_model if chat_model else available_models[0]): SelectSelector(SelectSelectorConfig(
                options=available_models,
                custom_value=True,
                multiple=False,
                mode=SelectSelectorMode.DROPDOWN,
            )),
        }
    )


def ragent_config_option_schema(
    hass: HomeAssistant,
    language: str,
    options: dict[str, Any],
    backend_type: str, 
    subentry_type: str,
) -> dict:

    default_prompt = RAGent.build_prompt_template(language, DEFAULT_PROMPT)

    result: dict = {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options.get(CONF_PROMPT, default_prompt)},
            default=options.get(CONF_PROMPT, default_prompt),
        ): TemplateSelector(),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)},
            default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        ): NumberSelector(NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_IN_CONTEXT_LEARNING_ENABLED,
            description={"suggested_value": options.get(CONF_IN_CONTEXT_LEARNING_ENABLED)},
            default=DEFAULT_IN_CONTEXT_LEARNING_ENABLED,
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_IN_CONTEXT_LEARNING_FILE,
            description={"suggested_value": options.get(CONF_IN_CONTEXT_LEARNING_FILE)},
            default=DEFAULT_IN_CONTEXT_LEARNING_FILE,
        ): str,
        vol.Required(
            CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
            description={"suggested_value": options.get(CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES)},
            default=DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
        ): NumberSelector(NumberSelectorConfig(min=1, max=16, step=1)),
        vol.Required(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=DEFAULT_MAX_TOKENS,
        ): NumberSelector(NumberSelectorConfig(min=1, max=8192, step=1)),
        vol.Required(
            CONF_CONTEXT_LENGTH,
            description={"suggested_value": options.get(CONF_CONTEXT_LENGTH)},
            default=DEFAULT_CONTEXT_LENGTH,
        ): NumberSelector(NumberSelectorConfig(min=512, max=1_048_576, step=512)),
        vol.Required(
            CONF_K_TOP,
            description={"suggested_value": options.get(CONF_K_TOP)},
            default=DEFAULT_K_TOP,
        ): NumberSelector(NumberSelectorConfig(min=1, max=256, step=1)),
        vol.Required(
            CONF_P_TOP,
            description={"suggested_value": options.get(CONF_P_TOP)},
            default=DEFAULT_P_TOP,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
         vol.Required(
            CONF_P_MIN,
            description={"suggested_value": options.get(CONF_P_MIN)},
            default=DEFAULT_P_MIN,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Required(
            CONF_P_TYPICAL,
            description={"suggested_value": options.get(CONF_P_TYPICAL)},
            default=DEFAULT_P_TYPICAL,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Required(
            CONF_OLLAMA_JSON_MODE,
            description={"suggested_value": options.get(CONF_OLLAMA_JSON_MODE)},
            default=DEFAULT_OLLAMA_JSON_MODE,
        ): BooleanSelector(BooleanSelectorConfig()),
        vol.Required(
            CONF_REQUEST_TIMEOUT,
            description={"suggested_value": options.get(CONF_REQUEST_TIMEOUT)},
            default=DEFAULT_REQUEST_TIMEOUT,
        ): NumberSelector(NumberSelectorConfig(min=5, max=900, step=1, unit_of_measurement=UnitOfTime.SECONDS, mode=NumberSelectorMode.BOX)),
        vol.Required(
            CONF_OLLAMA_KEEP_ALIVE_MIN,
            description={"suggested_value": options.get(CONF_OLLAMA_KEEP_ALIVE_MIN)},
            default=DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
        ): NumberSelector(NumberSelectorConfig(min=-1, max=1440, step=1, unit_of_measurement=UnitOfTime.MINUTES, mode=NumberSelectorMode.BOX)),
    }

    if subentry_type == "conversation":
        apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label="No control",
                value="none",
            )
        ]
        apis.extend(
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(hass)
        )
        result.update({
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=apis)),
            vol.Optional(
                CONF_REFRESH_SYSTEM_PROMPT,
                description={"suggested_value": options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT)},
                default=options.get(CONF_REFRESH_SYSTEM_PROMPT, DEFAULT_REFRESH_SYSTEM_PROMPT),
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Optional(
                CONF_REMEMBER_CONVERSATION,
                description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION)},
                default=options.get(CONF_REMEMBER_CONVERSATION, DEFAULT_REMEMBER_CONVERSATION),
            ): BooleanSelector(BooleanSelectorConfig()),
            vol.Optional(
                CONF_REMEMBER_NUM_INTERACTIONS,
                description={"suggested_value": options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS)},
                default=options.get(CONF_REMEMBER_NUM_INTERACTIONS, DEFAULT_REMEMBER_NUM_INTERACTIONS),
            ): NumberSelector(NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)),
            vol.Optional(
                CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
                description={"suggested_value": options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION)},
                default=options.get(CONF_REMEMBER_CONVERSATION_TIME_MINUTES, DEFAULT_REMEMBER_CONVERSATION),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1440, mode=NumberSelectorMode.BOX)),
            vol.Required(
                CONF_MAX_TOOL_CALL_ITERATIONS,
                description={"suggested_value": options.get(CONF_MAX_TOOL_CALL_ITERATIONS)},
                default=DEFAULT_MAX_TOOL_CALL_ITERATIONS,
            ): int,
        })

    global_order = [
        # general
        CONF_LLM_HASS_API,
        CONF_PROMPT,
        CONF_CONTEXT_LENGTH,
        CONF_MAX_TOKENS,
        # sampling parameters
        CONF_TEMPERATURE,
        CONF_P_TOP,
        CONF_P_MIN,
        CONF_P_TYPICAL,
        CONF_K_TOP,
        # tool calling/reasoning
        CONF_MAX_TOOL_CALL_ITERATIONS,
        CONF_GBNF_GRAMMAR_ENABLED,
        CONF_GBNF_GRAMMAR_FILE,
        # integration specific options
        CONF_REFRESH_SYSTEM_PROMPT,
        CONF_REMEMBER_CONVERSATION,
        CONF_REMEMBER_NUM_INTERACTIONS,
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
        CONF_IN_CONTEXT_LEARNING_ENABLED,
        CONF_IN_CONTEXT_LEARNING_FILE,
        CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
        # backend specific options
        CONF_OLLAMA_KEEP_ALIVE_MIN,
        CONF_OLLAMA_JSON_MODE,
    ]

    result = { k: v for k, v in sorted(result.items(), key=lambda item: global_order.index(item[0]) if item[0] in global_order else 9999) }

    return result