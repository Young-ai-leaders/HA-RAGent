from types import SimpleNamespace

import faiss
from custom_components.ha_ragent.src.backends.database.faiss_backend import (
    FaissDbBackend,
)
from custom_components.ha_ragent.src.const import CONF_VECTOR_DB_NAME, INSTRUCTION_PROMPT
from custom_components.ha_ragent.src.mock import MockHomeAssistant
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding


def _backend(tmp_path) -> FaissDbBackend:
    hass = MockHomeAssistant()
    hass.config = SimpleNamespace(path=lambda name: str(tmp_path / name))
    return FaissDbBackend(hass, {CONF_VECTOR_DB_NAME: "test"})


def test_faiss_uses_cosine_similarity(tmp_path) -> None:
    backend = _backend(tmp_path)
    collection = "tools"
    aligned = LlmTool(name="aligned", description="", parameters={})
    nearby = LlmTool(name="nearby", description="", parameters={})

    backend._save_device_embeddings(collection, [
        LlmToolEmbedding(aligned, [10.0, 0.0]),
        LlmToolEmbedding(nearby, [1.0, 1.0]),
    ])

    assert backend._indices[collection].metric_type == faiss.METRIC_INNER_PRODUCT
    result = backend._query_devices(collection, [1.0, 0.0], 1)
    assert result[0]["name"] == "aligned"


def test_tool_embedding_omits_parameter_schema() -> None:
    tool = LlmTool(
        name="HassTurnOff",
        description="Turn off a Home Assistant device.",
        parameters={"properties": {"area": {"type": "string"}}},
    )

    assert tool.to_embedding_text() == (
        "Tool name: HassTurnOff | description: Turn off a Home Assistant device."
    )


def test_recovery_instruction_searches_for_missing_action_tool() -> None:
    assert "A matching device candidate alone is insufficient" in INSTRUCTION_PROMPT["en"]
    assert "scope `tools`" in INSTRUCTION_PROMPT["en"]
