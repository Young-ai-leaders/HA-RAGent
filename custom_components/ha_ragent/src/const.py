import re

#-----------------------------------------------
# General constants
#-----------------------------------------------
DOMAIN = "ha_ragent"
RAGENT_LLM_API_ID = "ha_ragent_api"
RAGENT_LLM_API_NAME = "HA RAGent"
RAGENT_SEMANTIC_SEARCH_TOOL_NAME = "HassSemanticSearch"
INTEGRATION_VERSION = "0.3.0"

STARTUP_EMBEDDING_RUNNING_FLAG = "ha_ragent_startup_embedding_running"

#-----------------------------------------------
# Language constants
#-----------------------------------------------
CONF_SELECTED_LANGUAGE = "selected_language"

SELECTED_LANGUAGE_OPTIONS = [ "en", "de" ]

DEFAULT_LANGUAGE = "en"

#-----------------------------------------------
# Service Tool constants
#-----------------------------------------------
RAGENT_TIMER_DEVICE_ID = "ha_ragent_timer_device_a03a100a-81ca-415d"

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
BACKEND_VECTOR_DB_TYPE_CHROMA = "chromadb"
BACKEND_VECTOR_DB_TYPE_FAISS = "faiss"

BACKEND_VECTOR_DB_TYPE_OPTIONS = [ 
    BACKEND_VECTOR_DB_TYPE_MONGODB,
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_VECTOR_DB_TYPE_FAISS
]

DEFAULT_VECTOR_DB_BACKEND_TYPE = BACKEND_VECTOR_DB_TYPE_FAISS
DEFAULT_VECTOR_DB_NAME = "ha_ragent_db"

#-----------------------------------------------
# Embedding backend constants
#-----------------------------------------------
RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS = 12000
RAGENT_EMBEDDING_TRUNCATE_RETRIES = 3
RAGENT_EMBEDDING_BATCH_SIZE = 16

CONF_EMBEDDING_BACKEND_TYPE = "rag_embedding_backend"
CONF_EMBEDDING_MODEL = "rag_embedding_model"
CONF_EMBEDDING_HOST = "rag_embedding_host"
CONF_EMBEDDING_PORT = "rag_embedding_port"
CONF_EMBEDDING_SSL = "rag_embedding_ssl"
CONF_EMBEDDING_API_KEY = "rag_embedding_api_key"

BACKEND_EMBEDDING_TYPE_OLLAMA = "ollama"
BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE = "openai_compatible"

BACKEND_EMBEDDING_TYPE_OPTIONS = [ 
    BACKEND_EMBEDDING_TYPE_OLLAMA,
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
]

EMBEDDING_BACKENDS_WITH_API_KEY = [
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
]

DEFAULT_EMBEDDING_BACKEND_TYPE = BACKEND_EMBEDDING_TYPE_OLLAMA

#-----------------------------------------------
# Chat backend constants
#-----------------------------------------------
RAGENT_CHAT_TRUNCATE_MAX_CHARS = 12000
RAGENT_CHAT_TRUNCATE_RETRIES = 3

CONF_LLM_BACKEND_TYPE = "rag_llm_backend"
CONF_LLM_MODEL = "rag_llm_model"
CONF_LLM_HOST = "rag_llm_host"
CONF_LLM_PORT = "rag_llm_port"
CONF_LLM_SSL = "rag_llm_ssl"
CONF_LLM_API_KEY = "rag_llm_api_key"

BACKEND_LLM_TYPE_OLLAMA = "ollama"
BACKEND_LLM_TYPE_OPENAI_COMPATIBLE = "openai_compatible"

BACKEND_LLM_TYPE_OPTIONS = [ 
    BACKEND_LLM_TYPE_OLLAMA,
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
]

LLM_BACKENDS_WITH_API_KEY = [
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
]

DEFAULT_LLM_BACKEND_TYPE = BACKEND_LLM_TYPE_OLLAMA

#-----------------------------------------------
# Prompt configuration constants
#----------------------------------------------
CONF_NUM_DEVICES_TO_EXTRACT = "rag_num_devices_to_extract"
CONF_NUM_TOOLS_TO_EXTRACT = "rag_num_tools_to_extract"
CONF_CONTEXT_LENGTH = "rag_context_length"

CONF_MAX_TOKENS = "rag_max_tokens"
CONF_MAX_TOOL_CALL_ITERATIONS = "rag_max_tool_call_iterations"

CONF_PROMPT = "rag_prompt"

CONF_ENABLE_MODEL_THINKING = "rag_enable_model_thinking"
CONF_ALLOW_AUTO_EMBEDDING = "rag_allow_auto_embedding"

CONF_REMEMBER_CONVERSATION_TIME_MINUTES = "rag_remember_conversation_time_minutes"
CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS = "rag_remember_conversation_num_interactions"
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
    "de": """Bereichsanweisungen:
{% if area_name %}
- Aktueller Standort: Du befindest dich physisch im {{ area_name }}{% if floor_name %} ({{ floor_name }} Stock){% endif %}.
- Standardverhalten: Wenn der Benutzer eine Gerätekategorie angibt (z. B. „die Lichter“), ohne einen Raum zu nennen, ziele NUR auf die Geräte im {{ area_name }} ab.
{% else %}
- KRITISCH: Du hast keine Erlaubnis, einen Raum zu erraten oder das gesamte Haus anzusprechen.
- Wenn der Benutzer keinen Raum angibt, MUSST du um Klarstellung bitten.
{% endif %}""",
    "en": """Area Instructions:
{% if area_name %}
- Current Location: You are physically located in the {{ area_name }}{% if floor_name %} ({{ floor_name }} floor){% endif %}.
- Default Behavior: If the user specifies a device category (e.g., "the lights") without naming a room, target ONLY the devices within the {{ area_name }}.
{% else %}
- CRITICAL: You do not have permission to guess a room or target the entire house.
- If the user does not name a room, you MUST ask for clarification.
{% endif %}"""
}

DEVICE_CONTROL_PROMPT = {
    "de": """## Anweisungen zur Gerätesteuerung:

1. Auflösen

- Priorisiere die neueste Benutzernachricht; verwende frühere Nachrichten nur als Kontext für Referenzen.
- Folgeanweisungen sind neue Aktionen.
- Löse Geräte nur anhand von bekanntem Kontext oder Tool-Ergebnissen auf.
- Erfinde, errate, konstruiere oder leite niemals einen `name` ab.
- Verwende einen `name` nur, wenn er ausdrücklich im Kontext vorhanden ist oder von einem Tool zurückgegeben wurde.
- Bereich + Kategorie bedeutet alle passenden Geräte in diesem Bereich.
- Ordne Aktionen exakt zu: an → an, aus → aus, umschalten → umschalten.
- Korrigiere offensichtliche Tippfehler oder Speech-to-Text-Fehler, wenn die Absicht klar ist.
- Steuere niemals irrelevante Geräte oder Geräte außerhalb des angeforderten Bereichs.

2. Suchen

- Verwende die semantische Suche, wenn die angeforderte Zielmenge anhand der verfügbaren Geräte nicht vollständig aufgelöst werden kann.
- Verwende die semantische Suche bei ungenauen Namen, natürlichsprachlichen Bezeichnungen, Bereichen, Tippfehlern, Kategorien oder möglichen Mehrfachtreffern.
- Leite niemals einen `name` aus einem Anzeigenamen ab.
- Verwende nur exakte `name`-Werte, die von der semantischen Suche zurückgegeben wurden oder bereits im Kontext vorhanden sind.
- Wenn genau ein eindeutiger Treffer gefunden wird, führe die Aktion ohne Rückfrage aus.
- Frage nur nach, wenn mehrere widersprüchliche Ziele übrig bleiben.

3. Ausführen

- Prüfe vor jedem Steuerungsaufruf, dass der `name` aus dem Kontext oder einem Tool-Ergebnis stammt.
- Wenn kein gültiger `name` verfügbar ist, suche statt die Aktion auszuführen.
- Bevorzuge dedizierte Ein-/Aus-Tools gegenüber allgemeinen Tools zum Setzen eines Zustands.
- Bei mehreren Geräten gib pro Gerät einen `homeassistant`-Block aus.
- Führe eindeutige Befehle direkt aus.

4. Antworten

- Zuerst Tool-Aufrufe, danach die Antwort.
- Behaupte niemals einen Erfolg ohne erfolgreiches Tool-Ergebnis.
- Bei Teilerfolgen gib an, was funktioniert hat und was fehlgeschlagen ist.
- Erwähne bei Folgeanweisungen nur die neueste Aktion.
- Halte Antworten kurz und verwende benutzerfreundliche Gerätenamen, niemals technische IDs.
""",
    "en": """## Device Control Instructions:

1. Resolve

- Prioritize the latest user message; use earlier context only for references.
- Follow-up commands are new actions.
- Resolve devices only from known context or tool results.
- Never invent, guess, construct, or infer an `name`.
- Use an `name` only if it was explicitly provided in context or returned by a tool.
- Area + category means all matching devices in that area.
- Map actions exactly: on → on, off → off, toggle → toggle.
- Correct obvious typos/STT errors when intent is clear.
- Never control unrelated devices or devices outside the requested area.

2. Search

- Use semantic search when the requested target set cannot be fully resolved from available devices.
- Use semantic search for fuzzy names, natural-language names, areas, typos, categories, or possible multiple matches.
- Never derive an `name` from a friendly name.
- Use only exact `name`s returned by semantic search or already present in context.
- If one clear match is found, execute it without asking.
- Ask only if multiple conflicting targets remain.

3. Execute

- Before every control call, verify that the `name` came from context or a tool result.
- If no valid `name` is available, search instead of executing.
- Prefer dedicated on/off tools over generic state-setting tools.
- For multiple devices, emit one homeassistant block per device.
- Execute clear commands directly.

4. Respond

- Tool calls first, response second.
- Never claim success without a successful tool result.
- For partial failures, state what succeeded and failed.
- For follow-ups, mention only the newest action.
- Keep responses brief and use friendly names, never technical IDs.
"""
}

CONVERSATION_PRIORITY_PROMPT = {
    "de": """Die neueste Benutzernachricht hat Priorität. Direkte Folgeanweisungen sind auszuführende Befehle, keine Bitte um Bestätigung. Antworte bei Folgeanweisungen nur über die neueste Aktion. Nutze die semantische Suche als Auflösungshilfe bei ungenauen Namen, Bereichsreferenzen oder offensichtlichen Speech-to-Text-Fehlern, aber nicht zum Vorschlagen von Optionen. Wenn genau ein plausibles Ziel übrig bleibt, handle selbstständig. Simuliere keine erfolgreiche Gerätesteuerung: gib Tool-Aufrufe aus, frage nur bei echter Unklarheit nach oder antworte auf Basis echter Tool-Ergebnisse.""",
    "en": """The latest user message has priority. Direct follow-up commands should be executed, not turned into confirmation questions. For follow-up commands, respond only about the newest action. Use semantic search as a resolution aid for fuzzy names, area-based references, or obvious speech-to-text mistakes, but not to preview options. If exactly one plausible target remains, act on it confidently. For clear on or off commands, choose the semantically correct action rather than a similar tool with different default behavior. Do not simulate successful device control: emit tool calls, ask only when genuinely unclear, or respond from real tool results.""",
}

MAX_RETRIES_PROMPT = {
    "de": """Du hast maximal {{ max_retries}} Antwortversuche zur Verfügung.""",
    "en": """You have a maximum of {{ max_retries }} response attempts."""
}

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

TOOL_REGEX_PATTERN = re.compile(r"```homeassistant\s*(.*?)\s*```", re.DOTALL)

DEFAULT_NUM_DEVICES_TO_EXTRACT = 4
DEFAULT_NUM_TOOLS_TO_EXTRACT = 4
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_PROMPT = """<persona_prompt>
<current_date_prompt>
<area_prompt>

<device_control_prompt>

<max_retries_prompt>
<conversation_priority_prompt>

<devices_prompt>
{% for device in device_list %}
- { "name": "{{ device.id }}", "friendly_name": "{{ device.name }}", "aliases": {{ device.aliases | tojson }}, "domain": {{ device.domain | tojson }}, "area": "{{ device.area_name }}", "device_class": {{ device.domain | tojson }}, "state": {{ device.state }} }
{% endfor %}
"""

DEFAULT_ENABLE_MODEL_THINKING = False
DEFAULT_ALLOW_AUTO_EMBEDDING = True
DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES = 5
DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS = 10
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
    CONF_ALLOW_AUTO_EMBEDDING: DEFAULT_ALLOW_AUTO_EMBEDDING,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES: DEFAULT_REMEMBER_CONVERSATION_TIME_MINUTES,
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: DEFAULT_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_CONTEXT_LENGTH: DEFAULT_CONTEXT_LENGTH,
    CONF_NUM_DEVICES_TO_EXTRACT: DEFAULT_NUM_DEVICES_TO_EXTRACT,
}
