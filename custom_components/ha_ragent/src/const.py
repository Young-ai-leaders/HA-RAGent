from .llm_backends.ollama_backend import OllamaBackend

DOMAIN = "HA-RAGent"
LLM_API_ID = "ha-ragent-api"
INTEGRATION_VERSION = "0.1.0"

SERVICE_TOOL_NAME = "HassCallService"
SERVICE_TOOL_ALLOWED_SERVICES = ["turn_on", "turn_off", "toggle", "press", "increase_speed", "decrease_speed", "open_cover", "close_cover", "stop_cover", "lock", "unlock", "start", "stop", "return_to_base", "pause", "cancel", "add_item", "set_temperature", "set_humidity", "set_fan_mode", "set_hvac_mode", "set_preset_mode"]
SERVICE_TOOL_ALLOWED_DOMAINS = ["light", "switch", "button", "fan", "cover", "lock", "media_player", "climate", "vacuum", "todo", "timer", "script"]

ALLOWED_SERVICE_CALL_ARGUMENTS = ["rgb_color", "brightness", "temperature", "humidity", "fan_mode", "hvac_mode", "preset_mode", "item", "duration" ]

CONF_BACKEND_TYPE = "model_backend"
CONF_BACKEND_PATH = "model_backend_path"
CONF_SELECTED_LANGUAGE = "selected_language"

BACKEND_TYPE_OLLAMA = "ollama"

DEFAULT_BACKEND_TYPE = BACKEND_TYPE_OLLAMA
DEFAULT_LANGUAGE = "en"

SELECTED_LANGUAGE_OPTIONS = [ "en", "de"]
BACKEND_TYPE_OPTIONS = [ 
    BACKEND_TYPE_OLLAMA 
]

BACKEND_TO_CLASS = {
    BACKEND_TYPE_OLLAMA: OllamaBackend
}
