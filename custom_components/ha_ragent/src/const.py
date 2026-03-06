#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "ha_ragent"
RAGENT_LLM_API_ID = "ha_ragent_api"
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


#-----------------------------------------------
# Vector database backend constants
#-----------------------------------------------
CONF_VECTOR_DB_BACKEND_TYPE = "rag_vector_db_backend"
CONF_VECTOR_DB_NAME = "rag_vector_db_name"
CONF_VECTOR_DB_USERNAME = "rag_vector_db_username"
CONF_VECTOR_DB_PASSWORD = "rag_vector_db_password"
CONF_VECTOR_DB_HOST = "rag_vector_db_host"
CONF_VECTOR_DB_PORT = "rag_vector_db_port"
CONF_VECTOR_DB_SSL = "rag_vector_db_ssl"

BACKEND_VECTOR_DB_TYPE_MONGODB = "mongodb"

BACKEND_VECTOR_DB_TYPE_OPTIONS = [ 
    BACKEND_VECTOR_DB_TYPE_MONGODB 
]

DEFAULT_VECTOR_DB_BACKEND_TYPE = BACKEND_VECTOR_DB_TYPE_MONGODB
DEFAULT_VECTOR_DB_NAME = "ha_ragent_db"

#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------
CONF_EMBEDDING_BACKEND_TYPE = "rag_embedding_backend"
CONF_EMBEDDING_MODEL = "rag_embedding_model"
CONF_EMBEDDING_HOST = "rag_embedding_host"
CONF_EMBEDDING_PORT = "rag_embedding_port"
CONF_EMBEDDING_SSL = "rag_embedding_ssl"

BACKEND_EMBEDDING_TYPE_OLLAMA = "ollama"

BACKEND_EMBEDDING_TYPE_OPTIONS = [ 
    BACKEND_EMBEDDING_TYPE_OLLAMA 
]

DEFAULT_EMBEDDING_BACKEND_TYPE = BACKEND_EMBEDDING_TYPE_OLLAMA

#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"
CONF_LLM_HOST = "rag_llm_host"
CONF_LLM_PORT = "rag_llm_port"
CONF_LLM_SSL = "rag_llm_ssl"

BACKEND_LLM_TYPE_OLLAMA = "ollama"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA 
]

DEFAULT_LLM_BACKEND_TYPE = BACKEND_LLM_TYPE_OLLAMA

#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_CONTEXT_LENGTH = "rag_context_length"

CONF_IN_CONTEXT_LEARNING_ENABLED = "rag_in_context_learning_enabled"
CONF_IN_CONTEXT_LEARNING_FILE = "rag_in_context_learning_file"
CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES = "rag_in_context_learning_num_examples"

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

CONF_TEMPERATURE = "rag_temperature"
CONF_K_TOP = "rag_k_top"
CONF_P_MIN = "rag_p_min"
CONF_P_TOP = "rag_p_top"
CONF_P_TYPICAL = "rag_p_typical"

PERSONA_PROMPTS = {
    "de": "Du bist \"YAIL\", ein hilfreicher KI-Assistent, der die Geräte in einem Haus steuert. Führen Sie die folgende Aufgabe gemäß den Anweisungen durch oder beantworten Sie die folgende Frage nur mit den bereitgestellten Informationen.",
    "en": "You are 'YAIL', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.",
}
CURRENT_DATE_PROMPT = {
    "de": """{% set day_name = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"] %}{% set month_name = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"] %}Die aktuelle Uhrzeit und das aktuelle Datum sind {{ (as_timestamp(now()) | timestamp_custom("%H:%M", local=True)) }} {{ day_name[now().weekday()] }}, {{ now().day }} {{ month_name[now().month -1]}} {{ now().year }}.""",
    "en": """The current time and date is {{ (as_timestamp(now()) | timestamp_custom("%I:%M %p on %A %B %d, %Y", True, "")) }}"""
}
DEVICES_PROMPT = {
    "de": "## Verfügbare Geräte:",
    "en": "## Available Devices:",
}
AREA_PROMPT = {
    "de": "## Verfügbar Bereich:",
    "en": "## Available Areas:"
}
USER_INSTRUCTION = {
    "de": "## Benutzeranweisung:",
    "en": "## User instruction:"
}


DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_IN_CONTEXT_LEARNING_ENABLED = True
DEFAULT_IN_CONTEXT_LEARNING_FILE = "default_icl_examples.txt"
DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES = 3

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_OLLAMA_JSON_MODE = True
DEFAULT_OLLAMA_KEEP_ALIVE_MIN = 5
DEFAULT_PROMPT = """<persona>
<current_date>

<devices>
{% for device in device_list %}
- { "entity_id": "{{ device.id }}", "entity_name": "{{ device.name }}", "entity_domain": "{{ device.device_type }}", "entity_area": "{{ device.area_name }}" }
{% endfor %}

# Device Control Rules

## Device Control Instructions

When a user asks you to control a device:
1. **Find matching devices** from the device list by **name or area**.  
   - If the user specifies an area (e.g., "bedroom 1") and multiple devices match, **generate a separate tool call for each device** in that area.  
2. **Use the exact `entity_id`** from the device list (for example: `light.bedroom_1_ceiling_light`, `switch.kitchen_lamp`).  
3. **Only use areas listed** in the device list (`entity_area`).  
4. **Respond with ONLY the tool call code block(s)** — no explanations, extra text, or commentary.  
5. **Do NOT nest `arguments` inside another `arguments`.**  
6. **Each JSON tool call must contain exactly two top-level fields:** `tool` and `arguments`.  
7. **If multiple devices must be controlled:**
   - Return **one JSON object per tool call**.  
   - **Do not combine multiple entity_ids in one JSON object.**  
   - The LLM may return **multiple JSON blocks** consecutively if needed, one per device.  
8. **Tool call format example:**
```homeassistant
{"tool": "HassTurnOff", "arguments": {"name": "light.bedroom_1_ceiling_light", "area": "Bedroom 1", "domain": ["light"]}}
{"tool": "HassTurnOff", "arguments": {"name": "light.bedroom_1_bedside_lamp", "area": "Bedroom 1", "domain": ["light"]}}

Use **this exact format** (Note: use "name" for the entity ID):
```homeassistant
{"tool": "ToolName", "arguments": {"name": "entity_id", "area": "entity_area", "domain": ["entity_domain"]}}
```

Examples:
```homeassistant
{"tool": "HassTurnOn", "arguments": {"name": "switch.living_room", "area": "living_room", "domain": ["switch"]}}
{"tool": "HassTurnOff", "arguments": {"name": "fan.kitchen", "area": "kitchen", "domain": ["fan"]}}
{"tool": "HassLightSet", "arguments": {"name": "light.living_room_ceiling", "area": "living_room", "domain": ["light"], "brightness": 50}}
```

<user_instruction>
Ouput the tool call(s) needed to complete the user's instruction based on the available devices and the rules above.
"""
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