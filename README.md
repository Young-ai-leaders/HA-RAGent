<img src="https://raw.github.com/Young-ai-leaders/HA-RAGent/main/custom_components/ha_ragent/brand/logo.png" alt="HA-RAGent logo" title="HA-RAGent" align="right" height="80" />

# HA-RAGent (Home Assistant Retrieval‑Augmented‑Generation Agent)
HA‑RAGent is a custom component that wraps an LLM and a vector database to let you talk to your smart home. Instead of hard‑coding every possible command, the agent embeds your question, looks up the most relevant devices, and then either replies in natural language or emits “tool calls” that turn into real service calls inside Home Assistant.

This is particularly useful on self‑hosted installs, where you deliberately keep the model’s prompt window small to keep responses snappy. As soon as you move past a dozen or so entities, a plain conversation agent has to dump the entire device list into every prompt. A large or growing device set quickly blows out the context window and drags performance. Additionally, smaller models struggle even more, getting confused by the noise and sometimes emitting seemingly random tool calls.

## Installation
### HACS (recommended)
This integration is available in HACS (Home Assistant Community Store).

1. Install HACS if you don't have it already
2. Open HACS in Home Assistant
3. Go to any of the sections (integrations, frontend, automation)
4. Click on the 3 dots in the top right corner
5. Select "Custom repositories"
6. Add following URL to the repository `https://github.com/Young-ai-leaders/HA-RAGent`
7. Select Integration as category.
8. Click the "ADD" button
9. Search for "Home Assistant RAG Agent"
10. Click the "Download" button

### Manual

To install this integration manually you have to download the repository [HA-RAGent.zip](https://github.com/Young-ai-leaders/HA-RAGent/archive/refs/heads/main.zip) and extract its contents to `config/custom_components/ha_ragent` directory.

## Configuration

### Using UI

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_ragent)

From the Home Assistant front page go to `Configuration` and then select `Devices & Services` from the list.
Use the `Add Integration` button in the bottom right to add a new integration called `Home Assistant RAG Agent`.


## Help and Contribution
**Found a bug?** <br>
Open an issue and I’ll take a look.

**Want to add a feature or otherwise improve the code?** <br>
Send a pull request (or drop a quick issue first so we can chat about it).

**How to start?** <br>
Setup development environment ([See more](https://github.com/Young-ai-leaders/HA-RAGent/blob/main/dev/DEV_SETUP.md))