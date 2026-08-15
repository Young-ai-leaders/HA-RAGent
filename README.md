<img src="https://raw.github.com/youngaileaderslinz/HA-RAGent/main/custom_components/ha_ragent/brand/logo.png" alt="HA-RAGent logo" title="HA-RAGent" align="right" height="80" />

# HA-RAGent (Home Assistant Retrieval‑Augmented‑Generation Agent)
HA‑RAGent is a custom component that wraps an LLM and a vector database to let you talk to your smart home. Instead of hard‑coding every possible command, the agent embeds your question, looks up the most relevant devices, and then either replies in natural language or emits “tool calls” that turn into real service calls inside Home Assistant.

This is particularly useful on self‑hosted installs, where you deliberately keep the model’s prompt window small to keep responses snappy. As soon as you move past a dozen or so entities, a plain conversation agent has to dump the entire device list into every prompt. A large or growing device set quickly blows out the context window and drags performance. Additionally, smaller models struggle even more, getting confused by the noise and sometimes emitting seemingly random tool calls.

## Disclaimers
### Default System Prompt
Changes to the default system prompt apply only to newly created RAGent entries. Existing entries retain the prompt saved in their configuration. To use the latest default prompt with an existing entry, copy it into the entry’s **System Prompt** field and save the configuration or recreate the entry.

### OpenAI-Compatible Backends
OpenAI-compatible backends have currently been tested only with llamaccp. Compatibility with other providers is not guaranteed, so test the selected backend thoroughly before using it in production.

## Installation
### HACS (recommended)
If HACS is installed on your system use this link to directly go to the install page:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=youngaileaderslinz&repository=HA-RAGent)

### Manual
To install this integration manually you have to download the repository [HA-RAGent.zip](https://github.com/youngaileaderslinz/HA-RAGent/archive/refs/heads/main.zip) and extract its contents to `config/custom_components/ha_ragent` directory.

## Configuration
### Using UI
[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_ragent)

From the Home Assistant front page go to `Configuration` and then select `Devices & Services` from the list.
Use the `Add Integration` button in the bottom right to add a new integration called `Home Assistant RAG Agent`.

### Add Service Entry:
**Select Backends:**
- `Vector database backend`
    - **FAISS (Local DB)** stores embeddings locally and is the default, simplest setup
    - **MongoDB** stores embeddings in an external MongoDB instance
    - **ChromaDB** stores embeddings in an external ChromaDB server
- `Embedding backend`
    - **Ollama** requires an external Ollama instance and an installed embedding model ([find embedding models](https://ollama.com/search?c=embedding))
    - **OpenAI Compatible** works with APIs that expose an OpenAI-style embeddings endpoint
- `LLM backend`
    - **Ollama** requires an external Ollama instance and an installed chat model with tool support ([find tool-capable models](https://ollama.com/search?c=tools))
    - **OpenAI Compatible** works with APIs that expose an OpenAI-style chat completions endpoint
- `Language`
    - **English** used in order to setup the default prompt
    - **German** used in order to setup the default prompt

**Setup Connections:**

`Vector Database Options`
- **Database Username** optional database username, currently relevant for MongoDB
- **Database Password** optional database password, currently relevant for MongoDB
- **Vector DB Hostname** hostname or IP of the vector database server, used for MongoDB and ChromaDB
- **Vector DB Port** port of the vector database server, used for MongoDB and ChromaDB
- **Use HTTPS** enables SSL/TLS for the vector database connection when supported by the selected backend
- **Database Name** can be left as is or changed (Must be unique for each instance when multiple instances of HA-RAGent are configured. The default name is already unique.)

`Embedding Backend Options`
- **Embedding Hostname** hostname or IP of the embedding API server
- **Embedding Port** port of the embedding API server
- **Use HTTPS** enables SSL/TLS for the embedding API connection
- **Embedding API Key** optional bearer token for OpenAI-compatible embedding APIs

`LLM Backend Options`
- **LLM Hostname** hostname or IP of the LLM API server
- **LLM Port** port of the LLM API server
- **Use HTTPS** enables SSL/TLS for the LLM API connection
- **LLM API Key** optional bearer token for OpenAI-compatible LLM APIs

### Add AI RAGent Entry:
**Pick one of the configured services**
- The name contains database, embedding and llm backend

**Pick Models**
- `Embedding Model`
    - **Only shows downloaded models** that can be used for ebedding generation
- `LLM Model`
    - **Only shows downloaded models** that can be used as LLM model
- `Allow Auto Embedding`
    - Automatically embeds exposed devices and tools for this AI RAGent during Home Assistant startup and on config entry reload.

**Fine Tuning**
- `LLM Home Assistant API`
    - **No Control** means the model is not allowed to control devices
    - **Assist** allows the model to control devices and exposes Home Assistant tools
- `System Prompt`
    - The prompt that is sent to the model.
- `Enable Model Thinking`
    - Controls wheter model is allowed to think (when speed is of the essence keep the default)
- `Number of Devices`
    - Controls how many devices are retrieved and sent to the LLM
- `Number of Tools`
    - Controls how many tools are retrieved and sent to the LLM
- `Context Lenght`
    - Controls the context lenght of the LLM
- `Maximum Tokens`
    - Controls the maximum number of tokens the LLM is allowed to generate
- `Temperature`
    - Controls how much the LLM halucinates.
- `Maximum Tool Call Iterations`
    - Controls how often the LLM is allowed to perform tool calls per request (**Important note:** one response can call multiple tools the LLM can respond up to 8 times per default)
- `Conversation Memory Interactions`
    - Controls how many past user interactions are kept in memory for context.
- `Conversation Memory Duration`
    - Controls how long the assistant will retain conversation history in minutes.

### Available Prompt Variables
The **System Prompt** is rendered as a Home Assistant Jinja template for every request. The following variables are passed to it:

- `device_list`
    - The retrieved device candidates whose entities currently exist in Home Assistant. Each device provides `id`, `name`, `area_name`, `floor_name`, `domain`, `device_labels`, `services`, `aliases`, `state` and `attributes`.
- `area_list`
    - A list of the distinct, non-empty area names found in `device_list`. It contains only areas associated with the retrieved candidates, not every area in Home Assistant.
- `area_name`
    - The area of the device through which the conversation was started, or `None` when no area is available.
- `floor_name`
    - The floor of the device through which the conversation was started, or `None` when no floor is available.
- `max_retries`
    - The configured maximum number of tool-call iterations.

## Services
HA-RAGent registers the following Home Assistant services for each conversation entity created by the integration:

- `ha_ragent.embed_subentry`
    - Rebuilds device and tool embeddings for the selected AI RAGent subentry.
- `ha_ragent.preload_models`
    - Preloads the embedding model and LLM for the selected AI RAGent subentry.
- `ha_ragent.unload_models`
    - Unloads the embedding model and LLM for the selected AI RAGent subentry to free resources.

All three services target the HA-RAGent conversation entity, so you can run them from Developer Tools by selecting the specific assistant instance you want to manage.

## Help and Contribution
**Found a bug?** <br>
Open an issue and I’ll take a look. ([open issue](https://github.com/youngaileaderslinz/HA-RAGent/issues))

**Want to add a feature or otherwise improve the code?** <br>
Send a pull request (or drop a quick issue first so we can chat about it).

**How to start?** <br>
Setup development environment ([see more](https://github.com/youngaileaderslinz/HA-RAGent/blob/main/dev/DEV_SETUP.md))
