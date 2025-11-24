from .llm_backends.ollama_backend import OllamaBackend

#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "HA-RAGent"
LLM_API_ID = "ha-ragent-api"
INTEGRATION_VERSION = "0.1.0"

#-----------------------------------------------
# Language constants
#-----------------------------------------------
CONF_SELECTED_LANGUAGE = "selected_language"

SELECTED_LANGUAGE_OPTIONS = [ "en", "de" ]

DEFAULT_LANGUAGE = "en"

#-----------------------------------------------
# Service Tool constants
#-----------------------------------------------
SERVICE_TOOL_NAME = "HassCallService"
SERVICE_TOOL_ALLOWED_SERVICES = ["turn_on", "turn_off", "toggle", "press", "increase_speed", "decrease_speed", "open_cover", "close_cover", "stop_cover", "lock", "unlock", "start", "stop", "return_to_base", "pause", "cancel", "add_item", "set_temperature", "set_humidity", "set_fan_mode", "set_hvac_mode", "set_preset_mode"]
SERVICE_TOOL_ALLOWED_DOMAINS = ["light", "switch", "button", "fan", "cover", "lock", "media_player", "climate", "vacuum", "todo", "timer", "script"]

ALLOWED_SERVICE_CALL_ARGUMENTS = ["rgb_color", "brightness", "temperature", "humidity", "fan_mode", "hvac_mode", "preset_mode", "item", "duration" ]

#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------

#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"

BACKEND_LLM_TYPE_OLLAMA = "ollama"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA 
]
BACKEND_TO_CLASS = {
    BACKEND_LLM_TYPE_OLLAMA: OllamaBackend
}

DEFAULT_LLM_BACKEND_TYPE = BACKEND_LLM_TYPE_OLLAMA

#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_CONTEXT_LENGTH = "rag_context_length"
CONF_GBNF_GRAMMAR_ENABLED = "rag_gbnf_grammar_enabled"
CONF_GBNF_GRAMMAR_FILE = "rag_gbnf_grammar_file"

CONF_IN_CONTEXT_LEARNING_ENABLED = "rag_in_context_learning_enabled"
CONF_IN_CONTEXT_LEARNING_FILE = "rag_in_context_learning_file"
CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES = "rag_in_context_learning_num_examples"

CONF_GBNF_GRAMMAR_ENABLED = "rag_gbnf_grammar_enabled"
CONF_IN_CONTEXT_LEARNING_ENABLED = "rag_in_context_learning_enabled"

CONF_MAX_TOKENS = "rag_max_tokens"
CONF_MAX_TOOL_CALL_ITERATIONS = "rag_max_tool_call_iterations"

CONF_OLLAMA_JSON_MODE = "rag_ollama_json_mode"
CONF_OLLAMA_KEEP_ALIVE_MIN = "rag_ollama_keep_alive_min"
CONF_PROMPT = "rag_prompt"
CONF_REFRESH_SYSTEM_PROMPT = "rag_refresh_system_prompt"

CONF_REMEMBER_CONVERSATION = "rag_remember_conversation"
CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_NUM_INTERACTIONS = "rag_remember_num_interactions"
CONF_REQUEST_TIMEOUT = "rag_request_timeout"
CONF_SELECTED_LANGUAGE = "rag_selected_language"

CONF_TEMPERATURE = 0.7
CONF_K_TOP = 40
CONF_P_MIN = 0.1
CONF_P_TOP = 0.9
CONF_P_TYPICAL = 1.0

PERSONA_PROMPTS = {
    "de": "Du bist \"YAIL\", ein hilfreicher KI-Assistent, der die Geräte in einem Haus steuert. Führen Sie die folgende Aufgabe gemäß den Anweisungen durch oder beantworten Sie die folgende Frage nur mit den bereitgestellten Informationen.",
    "en": "You are 'YAIL', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.",
}
CURRENT_DATE_PROMPT = {
    "de": """{% set day_name = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"] %}{% set month_name = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"] %}Die aktuelle Uhrzeit und das aktuelle Datum sind {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "en": """The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}"""
}
DEVICES_PROMPT = {
    "de": "Geräte",
    "en": "Devices",
}
SERVICES_PROMPT = {
    "de": "Dienste",
    "en": "Services"
}
TOOLS_PROMPT = {
    "de": "Werkzeuge",
    "en": "Tools"
}
AREA_PROMPT = {
    "de": "Bereich",
    "en": "Area"
}
USER_INSTRUCTION = {
    "de": "Benutzeranweisung",
    "en": "User instruction"
}


DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_IN_CONTEXT_LEARNING_ENABLED = True
DEFAULT_IN_CONTEXT_LEARNING_FILE = "default_icl_examples.txt"
DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES = 3

DEFAULT_GBNF_GRAMMAR_ENABLED = False
DEFAULT_GBNF_GRAMMAR_FILE = "default_grammar.gbnf"

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_OLLAMA_JSON_MODE = True
DEFAULT_OLLAMA_KEEP_ALIVE_MIN = 5
DEFAULT_PROMPT = "default_prompt"
DEFAULT_REFRESH_SYSTEM_PROMPT = False
DEFAULT_REMEMBER_CONVERSATION = False
DEFAULT_REMEMBER_NUM_INTERACTIONS = 10
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_SELECTED_LANGUAGE = "en"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_K_TOP = 40
DEFAULT_P_MIN = 0.1
DEFAULT_P_TOP = 0.9
DEFAULT_P_TYPICAL = 1.0

#-----------------------------------------------
# Default override options for new entries
#-----------------------------------------------
DEFAULT_OPTIONS = {
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_K_TOP: DEFAULT_K_TOP,
    CONF_P_TOP: DEFAULT_P_TOP,
    CONF_P_MIN: DEFAULT_P_MIN,
    CONF_P_TYPICAL: DEFAULT_P_TYPICAL,
    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
    CONF_GBNF_GRAMMAR_ENABLED: DEFAULT_GBNF_GRAMMAR_ENABLED,
    CONF_REFRESH_SYSTEM_PROMPT: DEFAULT_REFRESH_SYSTEM_PROMPT,
    CONF_REMEMBER_CONVERSATION: DEFAULT_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS: DEFAULT_REMEMBER_NUM_INTERACTIONS,
    CONF_IN_CONTEXT_LEARNING_ENABLED: DEFAULT_IN_CONTEXT_LEARNING_ENABLED,
    CONF_IN_CONTEXT_LEARNING_FILE: DEFAULT_IN_CONTEXT_LEARNING_FILE,
    CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES: DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
    CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
    CONF_OLLAMA_KEEP_ALIVE_MIN: DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
    CONF_OLLAMA_JSON_MODE: DEFAULT_OLLAMA_JSON_MODE
}