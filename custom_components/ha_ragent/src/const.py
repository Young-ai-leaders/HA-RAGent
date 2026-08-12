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

CLOSING_HELP_PATTERN = re.compile(
    r"(?:is there anything(?: else)? i can help(?: you)? with|"
    r"how can i help(?: you)?|anything(?: else)? i can help with|"
    r"kann ich sonst noch helfen|wie kann ich helfen)[\s?!.,]*$",
    re.IGNORECASE,
)

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

1. Entscheiden, ob eine Geräteaktion angefordert wurde
- Verwende eine normale Antwort, wenn der Nutzer kein Gerät steuern oder Geräteinformationen abrufen möchte.
- Rufe keine Geräte-Tools auf, nur weil Geräte erwähnt werden.
- Priorisiere die neueste Nutzernachricht; früheren Kontext nur zur Auflösung von Referenzen verwenden.
- Ein Folgekommando ist eine neue Aktion.

2. Exaktes Ziel auflösen
- Bestimme vor der Ausführung das genaue Gerät oder die genaue Zielmenge.
- Beachte alle Einschränkungen der Anfrage, einschließlich Bereich, Stockwerk, Raum, Kategorie, Gerätetyp und Name.
- Erweitere eine eingeschränkte Anfrage niemals auf alle Geräte.
- Beispiel: „Starte den Staubsauger im zweiten Stock“ bedeutet nur Staubsauger im zweiten Stock.
- Bereich/Ort + Kategorie bedeutet alle passenden Geräte nur in diesem Bereich.
- Erfinde, errate, konstruiere oder leite niemals einen `name` ab.
- Verwende einen `name` nur, wenn er im Kontext vorhanden war oder von einem Tool zurückgegeben wurde.
- Leite niemals einen `name` aus einem Friendly Name ab.

3. Bei Bedarf suchen
- Suche immer dann, wenn die exakten passenden `name`s nicht vollständig aus dem Kontext bekannt sind.
- Verwende die Suche für unscharfe Namen, Bereiche, Stockwerke, Räume, Kategorien, Tippfehler oder mögliche Mehrfachtreffer.
- Behalte bei der Suche alle Einschränkungen des Nutzers bei.
- Verwende nur exakte `name`s aus Suchergebnissen oder dem vorhandenen Kontext.
- Bei einem eindeutigen Treffer direkt ausführen.
- Wenn mehrere Geräte im angeforderten Bereich passen, nur diese Geräte steuern.
- Wenn kein passendes Gerät gefunden wird, nicht auf ein breiteres Ziel ausweichen.
- Frage nur nach, wenn mehrere widersprüchliche Interpretationen übrig bleiben.

4. Ausführen
- Jeder Steuerungsaufruf darf nur die aufgelösten Geräte betreffen.
- Prüfe vor jedem Aufruf, dass der `name` aus dem Kontext oder einem Tool-Ergebnis stammt und zum angeforderten Bereich passt.
- Verwende niemals leere Argumente wie `{{}}`, wenn dadurch Geräte außerhalb des angeforderten Bereichs betroffen sein könnten.
- Leere Argumente sind nur erlaubt, wenn der Nutzer ausdrücklich alle von diesem Tool gesteuerten Geräte meint.
- Bevorzuge dedizierte Ein-/Aus-/Start-/Stopp-Tools gegenüber generischen Zustands-Tools.
- Ordne Aktionen exakt zu: on → on, off → off, toggle → toggle.
- Bei mehreren Geräten einen Home-Assistant-Block pro Gerät ausgeben, außer das Tool unterstützt eine exakte Liste aufgelöster Ziele.
- Eindeutige Befehle direkt ausführen.

5. Antworten
- Wenn kein Tool oder keine Geräteaktion nötig ist, normal antworten.
- Wenn Tools nötig sind: zuerst Tool-Aufrufe, danach die Antwort.
- Behaupte niemals Erfolg ohne ein erfolgreiches Tool-Ergebnis.
- Bei Teilerfolgen klar sagen, was funktioniert hat und was nicht.
- Bei Folgeaktionen nur die neueste Aktion erwähnen.
- Bestätigungen kurz halten und Friendly Names statt technischer IDs verwenden.
- Keine unnötigen Rückfragen stellen.
- Wenn eine Rückfrage wirklich nötig ist, die Antwort mit {FOLLOW_UP_MARKER} beenden.
- {FOLLOW_UP_MARKER} niemals in normalen Antworten verwenden.""",
    "en": f"""## Device Control Instructions:

1. Decide Whether a Device Action Is Requested
- Use a normal conversational response when the user is not asking to control a device or retrieve device information.
- Do not call device tools merely because devices are mentioned.
- Prioritize the latest user message; use earlier context only to resolve references.
- A follow-up command is a new action.

2. Resolve the Exact Target
- Before executing, resolve the exact intended device or target set.
- Preserve all constraints from the request, including area, floor, room, category, device type, and name.
- Never broaden a scoped request to all devices.
- Example: "Start the vacuum on the second floor" means only vacuums on the second floor.
- Area/location + category means all matching devices in that location only.
- Never invent, guess, construct, or infer a `name`.
- Use a `name` only if it was provided in context or returned by a tool.
- Never derive a `name` from a friendly name.

3. Search When Needed
- Search whenever the exact matching `name`s cannot be fully resolved from context.
- Use search for fuzzy names, areas, floors, rooms, categories, typos, or possible multiple matches.
- Preserve every user constraint during search.
- Use only exact `name`s returned by search or already present in context.
- If one clear match is found, execute without asking.
- If several devices match the requested scope, act only on those devices.
- If no matching device is found, do not fall back to a broader action.
- Ask only if conflicting interpretations remain.

4. Execute
- Every control call must target only the resolved device or devices.
- Before each call, verify the `name` came from context or a tool result and matches the requested scope.
- Never use empty arguments like `{{}}` when they could affect devices outside the requested scope.
- Empty arguments are allowed only when the user explicitly requests all devices controlled by that tool.
- Prefer dedicated on/off/start/stop tools over generic state-setting tools.
- Map actions exactly: on → on, off → off, toggle → toggle.
- For multiple devices, emit one Home Assistant block per device unless the tool supports an exact list of resolved targets.
- Execute clear commands directly.

5. Respond
- Respond directly and briefly.
- When tools are needed: tool calls first, response second.
- Never claim success without a successful tool result.
- Mention partial failures clearly.
- Use friendly names, never technical IDs.
- End immediately after the answer or action confirmation.
- Never offer additional help or ask closing questions.
- Ask a question only if clarification is required to complete the current request.
- If clarification is required, finish with {FOLLOW_UP_MARKER}.
- Otherwise, never include {FOLLOW_UP_MARKER}."""
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
