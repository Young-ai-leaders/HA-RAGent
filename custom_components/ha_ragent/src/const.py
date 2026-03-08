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

CONF_OLLAMA_KEEP_ALIVE_MIN = "rag_ollama_keep_alive_min"
CONF_PROMPT = "rag_prompt"

CONF_REMEMBER_CONVERSATION = "rag_remember_conversation"
CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_NUM_INTERACTIONS = "rag_remember_num_interactions"
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
AREAS_PROMPT = {
    "de": "## Verfügbar Bereich:",
    "en": "## Available Areas:"
}
DEVICE_CONTROL_PROMPT = {
    "de": """## Geräte Steuerungsanweisungen:
Wenn du ein Gerät steuerst folge diesen Anweisungen:
1. Geräteauflösung
   - Suchkriterien: Identifiziere Zielgeräte anhand des exakten Namens oder des angegebenen Bereichs.
   - Bereichserweiterung: Wenn ein Benutzer einen Bereich anspricht (z. B. „Wohnzimmer“), musst du jedes Gerät in diesem Bereich identifizieren und für jedes einzelne einen eigenen Tool-Aufruf erzeugen.
   - ID-Zuordnung: Verwende ausschließlich die **entity_id** (z. B. `light.desk_lamp`), die in der Geräteliste angegeben ist.
2. **Struktur der Tool-Aufrufe**
   - Atomarität: Jede Geräteaktion muss ein eigener, unabhängiger Tool-Aufruf sein.
   - Kein Batching: Kombiniere niemals mehrere **entity_ids** in einem einzigen JSON-Objekt.
   - Parameterbereinigung: Nimm die **device_class** nicht in den Tool-Aufruf auf. Verwende nur die erforderlichen Argumente (**name**, **area**, **domain**).
3. **Strenges Ausgabeformat**
   - Gib **ein gültiges JSON-Objekt pro Tool-Aufruf** oder eine **Textantwort für den Benutzer** zurück.
   - Halte ein **1:1-Verhältnis** zwischen der Anzahl der Zielgeräte und der Anzahl der erzeugten Tool-Aufrufe ein.""",

    "en": """## Device Control Instructions:
When controlling a device follow these steps:
1. Device Resolution
    - Search Criteria: Identify target devices using the exact name or the specified area.
    - Area Expansion: If a user targets an area (e.g., "Living Room"), you must identify every device within that area and generate a dedicated tool call for each one.
    - ID Mapping: Use only the entity_id (e.g., light.desk_lamp) provided in the device list.
2. Tool Call Structure
    - Atomicity: Every device action must be its own independent tool call.
    - No Batching: Never combine multiple entity_ids into a single JSON object.
    - Parameter Stripping: Do not include the device_class in the tool call. Only use the required arguments (name, area, domain).
    - To execute multiple tool calls in one response use a format like this
        ```homeassistant {"tool": "HassTurnOff", "arguments": {"name": "light.bedroom_1_ceiling_light", "area": "Bedroom 1", "domain": ["light"]}} ```
        ```homeassistant {"tool": "HassTurnOff", "arguments": {"name": "light.bedroom_1_bedside_lamp", "area": "Bedroom 1", "domain": ["light"]}} ```
3. Strict Output Format
    3.1 Answering with tool calls:
        - Return a valid JSON object for each tool call.
        - Encapsulate each JSON object within ```homeassistant ``` tags to clearly delineate tool calls from text responses.
        - When the task is completed and no further tool calls are needed, return a clear text response to the user indicating that the action has been completed.
    3.2 Answering with text:
        - If the user's request cannot be fulfilled with tool calls alone, provide a clear and concise text response to the user.
        - Always use the friendly_name attribute when referring to devices in text responses, never the entity_id.
        - Do not include the room if it is part of the device's friendly name to avoid redundancy (e.g., "Living Room Lamp" instead of "Living Room Living Room Lamp")."""
}
USER_INSTRUCTION = {
    "de": "## Benutzeranweisung:",
    "en": "## User instruction:"
}

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

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
- { "name": "{{ device.id }}", "friendly_name": "{{ device.name }}", "domain": {{ device.domain | tojson }}, "area": "{{ device.area_name }}", "services": {{ device.services | tojson }}  "device_class": ["{{ device.domain | tojson }}"], "state": {{ device.state }}, "attributes": {{ device.attributes | tojson }} }
{% endfor %}

<areas>
{% for area in area_list %}
- "{{ area }}"
{% endfor %}

<device_control_prompt>

<user_instruction>
"""

DEFAULT_REMEMBER_CONVERSATION = False
DEFAULT_REMEMBER_NUM_INTERACTIONS = 10
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
    CONF_REMEMBER_CONVERSATION: DEFAULT_REMEMBER_CONVERSATION,
    CONF_REMEMBER_NUM_INTERACTIONS: DEFAULT_REMEMBER_NUM_INTERACTIONS,
    CONF_IN_CONTEXT_LEARNING_ENABLED: DEFAULT_IN_CONTEXT_LEARNING_ENABLED,
    CONF_IN_CONTEXT_LEARNING_FILE: DEFAULT_IN_CONTEXT_LEARNING_FILE,
    CONF_IN_CONTEXT_LEARNING_NUM_EXAMPLES: DEFAULT_IN_CONTEXT_LEARNING_NUM_EXAMPLES,
    CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
    CONF_OLLAMA_KEEP_ALIVE_MIN: DEFAULT_OLLAMA_KEEP_ALIVE_MIN,
}