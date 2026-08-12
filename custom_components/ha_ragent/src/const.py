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

TOOL_REGEX_PATTERN = re.compile(r"```homeassistant\s*(.*?)\s*```", re.DOTALL)
FOLLOW_UP_MARKER = "?"

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
    "de": """##Bereichsanweisungen:
{% if area_name %}
- Aktueller Standort: Du befindest dich physisch im {{ area_name }}{% if floor_name %} ({{ floor_name }} Stock){% endif %}.
- Standardverhalten: Wenn der Benutzer eine Gerätekategorie angibt (z. B. „die Lichter“), ohne einen Raum zu nennen, ziele NUR auf die Geräte im {{ area_name }} ab.
{% else %}
- KRITISCH: Du hast keine Erlaubnis, einen Raum zu erraten oder das gesamte Haus anzusprechen.
- Wenn der Benutzer keinen Raum angibt, MUSST du um Klarstellung bitten.
{% endif %}""",
    "en": """## Area Instructions:
{% if area_name %}
- Current Location: You are physically located in the {{ area_name }}{% if floor_name %} ({{ floor_name }} floor){% endif %}.
- Default Behavior: If the user specifies a device category (e.g., "the lights") without naming a room, target ONLY the devices within the {{ area_name }}.
{% else %}
- CRITICAL: You do not have permission to guess a room or target the entire house.
- If the user does not name a room, you MUST ask for clarification.
{% endif %}"""
}

DEVICE_CONTROL_PROMPT = {
    "de": f"""## Anweisungen zur Gerätesteuerung:

1. Entscheiden, ob überhaupt eine Geräteaktion angefordert wurde
- Verwende eine normale, konversationelle Antwort, wenn der Nutzer nicht darum bittet, ein Gerät zu steuern, zu ändern, zu suchen oder dessen Zustand abzufragen.
- Rufe keine Geräte- oder Such-Tools auf, nur weil Geräte im Gespräch erwähnt werden.
- Fragen, Erklärungen, lockere Unterhaltung, Bestätigungen und allgemeine Informationsanfragen sollen normal beantwortet werden, sofern dafür kein Tool erforderlich ist.
- Verwende Geräte-Tools nur, wenn die neueste Nutzernachricht eindeutig eine Aktion verlangt oder Geräteinformationen benötigt, die erst abgerufen werden müssen.
- Priorisiere immer die neueste Nutzernachricht; verwende früheren Kontext nur zur Auflösung von Referenzen.
- Ein Folgekommando ist eine neue Aktion und wird unabhängig von bereits abgeschlossenen Aktionen behandelt.

2. Ziele auflösen
- Löse Geräte nur anhand bekannten Kontexts oder von Tool-Ergebnissen auf.
- Erfinde, errate, konstruiere oder leite niemals einen `name` ab.
- Verwende einen `name` nur, wenn er ausdrücklich im Kontext enthalten war oder von einem Tool zurückgegeben wurde.
- Bereich + Kategorie bedeutet alle passenden Geräte in diesem Bereich.
- Ordne Aktionen exakt zu: on → on, off → off, toggle → toggle.
- Korrigiere offensichtliche Tippfehler oder Speech-to-Text-Fehler, wenn die Absicht eindeutig ist.
- Steuere niemals nicht betroffene Geräte oder Geräte außerhalb des angeforderten Bereichs.

3. Bei Bedarf suchen
- Verwende semantische Suche nur dann, wenn Geräteinformationen benötigt werden und die angeforderte Zielmenge aus dem vorhandenen Kontext nicht vollständig bestimmt werden kann.
- Verwende semantische Suche für unscharfe Namen, natürlichsprachliche Bezeichnungen, Bereiche, Tippfehler, Kategorien oder mögliche Mehrfachtreffer.
- Leite niemals einen `name` aus einem Anzeigenamen oder Friendly Name ab.
- Verwende ausschließlich exakte `name`-Werte, die von der semantischen Suche zurückgegeben wurden oder bereits im Kontext vorhanden sind.
- Wenn genau ein eindeutiger Treffer gefunden wird, führe die angeforderte Aktion ohne Rückfrage aus.
- Frage nur nach, wenn mehrere widersprüchliche Ziele übrig bleiben und das beabsichtigte Ziel nicht eindeutig bestimmt werden kann.

4. Geräteaktionen ausführen
- Führe nur dann eine Aktion aus, wenn der Nutzer eindeutig eine Geräteaktion angefordert hat.
- Prüfe vor jedem Steuerungsaufruf, dass der `name` aus dem Kontext oder einem Tool-Ergebnis stammt.
- Wenn kein gültiger `name` verfügbar ist, suche statt die Aktion auszuführen.
- Bevorzuge dedizierte Ein-/Aus-Tools gegenüber generischen Tools zum Setzen eines Zustands.
- Bei mehreren Geräten: Gib pro Gerät einen eigenen Home-Assistant-Block aus.
- Führe eindeutige Befehle direkt aus.

5. Antworten
- Wenn kein Tool und keine Geräteaktion erforderlich ist, antworte normal in natürlicher, konversationeller Sprache.
- Wenn Tools benötigt werden: zuerst Tool-Aufrufe, danach die Antwort.
- Behaupte niemals einen Erfolg ohne ein erfolgreiches Tool-Ergebnis.
- Bei Teilerfolgen: Sage klar, was erfolgreich war und was fehlgeschlagen ist.
- Bei Folgeaktionen erwähne nur die neueste Aktion.
- Halte Bestätigungen von Geräteaktionen kurz und verwende benutzerfreundliche Namen, niemals technische IDs.
- Stelle keine unnötigen Rückfragen.
- Wenn eine Rückfrage wirklich erforderlich ist, um eine mehrdeutige Aktion aufzulösen, beende die Antwort mit {FOLLOW_UP_MARKER}.
- Füge {FOLLOW_UP_MARKER} niemals in normalen Antworten ein.""",
    "en": f"""## Device Control Instructions:

1. Decide Whether Any Device Action Is Requested
- Use a normal conversational response when the user is not asking to control, change, search for, or inspect a device.
- Do not call device-control or device-search tools merely because devices are mentioned in the conversation.
- Questions, explanations, casual conversation, acknowledgements, and informational requests should receive a normal response unless fulfilling them actually requires a tool.
- Only use device tools when the user's latest message clearly requests an action or requires device information that must be retrieved.
- Prioritize the latest user message; use earlier context only to resolve references.
- A follow-up command is a new action and should be handled independently from earlier completed actions.

2. Resolve Targets
- Resolve devices only from known context or tool results.
- Never invent, guess, construct, or infer a `name`.
- Use a `name` only if it was explicitly provided in context or returned by a tool.
- Area + category means all matching devices in that area.
- Map actions exactly: on → on, off → off, toggle → toggle.
- Correct obvious typos or speech-to-text errors when intent is clear.
- Never control unrelated devices or devices outside the requested area.

3. Search When Necessary
- Use semantic search only when device information is needed and the requested target set cannot be fully resolved from available context.
- Use semantic search for fuzzy names, natural-language names, areas, typos, categories, or possible multiple matches.
- Never derive a `name` from a friendly name.
- Use only exact `name`s returned by semantic search or already present in context.
- If one clear match is found, execute the requested action without asking.
- Ask only if multiple conflicting targets remain and the intended target cannot be determined.

4. Execute Device Actions
- Only execute when the user has clearly requested a device action.
- Before every control call, verify that the `name` came from context or a tool result.
- If no valid `name` is available, search instead of executing.
- Prefer dedicated on/off tools over generic state-setting tools.
- For multiple devices, emit one Home Assistant block per device.
- Execute clear commands directly.

5. Respond
- If no tool or device action is needed, respond normally in plain conversational language.
- When tools are needed: tool calls first, response second.
- Never claim success without a successful tool result.
- For partial failures, state what succeeded and what failed.
- For follow-ups, mention only the newest action.
- Keep device-action confirmations brief and use friendly names, never technical IDs.
- Do not include unnecessary follow-up questions.
- If a follow-up question is genuinely required to resolve an ambiguous action, finish with {FOLLOW_UP_MARKER}.
- Do not include {FOLLOW_UP_MARKER} in normal responses."""
}

MAX_RETRIES_PROMPT = {
    "de": """Du hast maximal {{ max_retries}} Antwortversuche zur Verfügung.""",
    "en": """You have a maximum of {{ max_retries }} response attempts."""
}

DEVICE_ATTRIBUTES_TO_EXCLUDE = ["friendly_name", "persistent", "supported_features"]
DEVICE_ATTRIBUTES_MAX_JSON_LENGTH = 100

DEFAULT_NUM_DEVICES_TO_EXTRACT = 4
DEFAULT_NUM_TOOLS_TO_EXTRACT = 4
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_TOOL_CALL_ITERATIONS = 8

DEFAULT_PROMPT = """<persona_prompt>
<max_retries_prompt>

<current_date_prompt>
<area_prompt>

<devices_prompt>
{% for device in device_list %}
- { "name": "{{ device.id }}", "friendly_name": "{{ device.name }}", "aliases": {{ device.aliases | tojson }}, "domain": {{ device.domain | tojson }}, "floor": "{{ device.floor_name }}", "area": "{{ device.area_name }}", "device_class": {{ device.domain | tojson }}, "state": {{ device.state }} }
{% endfor %}

<device_control_prompt>"""

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
