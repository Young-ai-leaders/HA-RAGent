from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tests

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.database.chromadb_backend import ChromaDbBackend
from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.backends.database.mongodb_backend import MongoDbBackend
from custom_components.ha_ragent.src.const import CONF_VECTOR_DB_NAME
from custom_components.ha_ragent.src.mock import MockHomeAssistant
from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.memory import Memory
from custom_components.ha_ragent.src.models.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata
from tests.mocks import (
    MOCK_CHROMADB_CONNECTION_USER_INPUT,
    MOCK_CHROMADB_CONNECTION_USER_INPUT_INVALID,
    MOCK_FAISS_DB_CONFIG,
    MOCK_MONGODB_CONNECTION_USER_INPUT,
    MOCK_MONGODB_CONNECTION_USER_INPUT_INVALID,
)


EMBEDDING_DIMENSION = 3


@dataclass(frozen=True)
class DatabaseCase:
    """Configuration for one vector database backend test case."""

    backend_class: type[ABaseDbBackend]
    connection_input: dict[str, Any]
    invalid_connection_input: dict[str, Any] | None = None


DATABASE_BACKENDS = [
    DatabaseCase(FaissDbBackend, MOCK_FAISS_DB_CONFIG),
    DatabaseCase(
        MongoDbBackend,
        MOCK_MONGODB_CONNECTION_USER_INPUT,
        MOCK_MONGODB_CONNECTION_USER_INPUT_INVALID,
    ),
    DatabaseCase(
        ChromaDbBackend,
        MOCK_CHROMADB_CONNECTION_USER_INPUT,
        MOCK_CHROMADB_CONNECTION_USER_INPUT_INVALID,
    ),
]


@pytest.fixture(params=DATABASE_BACKENDS, ids=lambda case: case.backend_class.__name__)
def database_case(request: pytest.FixtureRequest) -> DatabaseCase:
    """Provide every vector database backend from the central backend list."""
    return request.param


@pytest.fixture
def hass(tmp_path: Path) -> MockHomeAssistant:
    """Provide a mock Home Assistant with isolated local storage."""
    instance = MockHomeAssistant()
    instance.config = SimpleNamespace(path=lambda name: str(tmp_path / name))
    return instance


@pytest.fixture(autouse=True)
def suppress_logging() -> Any:
    """Suppress expected connection and retry errors in this test module."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


def _run_async_test(hass: MockHomeAssistant, test: Awaitable[None]) -> None:
    """Run an async test and close mock resources before its loop exits."""

    async def run() -> None:
        try:
            await test
        finally:
            await hass.async_close()

    asyncio.run(run())


def _create_backend(
    database_case: DatabaseCase,
    hass: MockHomeAssistant,
) -> tuple[ABaseDbBackend, dict[str, Any], str]:
    test_id = uuid4().hex[:12]
    config = {
        **database_case.connection_input,
        CONF_VECTOR_DB_NAME: f"ha_ragent_test_{test_id}",
    }
    return database_case.backend_class(hass, config), config, test_id


async def _async_wait_for_retrieval(
    backend: ABaseDbBackend,
    object_type: type,
    config: dict[str, Any],
    collection_name: str,
    query_embedding: list[float],
) -> list[Any]:
    """Allow MongoDB Atlas Search time to make a new vector index queryable."""
    for _ in range(20):
        objects = await backend.async_retrieve_objects(
            object_type,
            config,
            collection_name,
            query_embedding,
            top_k=1,
        )
        if objects:
            return objects
        await asyncio.sleep(0.5)
    return []


async def _async_test_connection(
    database_case: DatabaseCase,
    backend: ABaseDbBackend,
    hass: MockHomeAssistant,
    config: dict[str, Any],
) -> None:
    validation_error = await backend.async_validate_connection(hass, config)
    assert validation_error is None, validation_error

    if database_case.invalid_connection_input is not None:
        invalid_config = {
            **database_case.invalid_connection_input,
            CONF_VECTOR_DB_NAME: config[CONF_VECTOR_DB_NAME],
        }
        validation_error = await backend.async_validate_connection(hass, invalid_config)
        assert validation_error is not None


async def _async_test_object_round_trips(
    backend: ABaseDbBackend,
    config: dict[str, Any],
    collection_prefix: str,
) -> None:
    device_collection = f"{collection_prefix}_devices"
    tool_collection = f"{collection_prefix}_tools"
    device = Device(
        id="sensor.bathroom_temperature",
        friendly_name="Bathroom temperature",
        area_name="Bathroom",
        floor_name="1st Floor",
        domain=["sensor"],
        device_labels=["Climate"],
        services=["turn_on", "turn_off"],
        aliases=["Bath temperature"],
        device_class="temperature",
        unit_of_measurement="°C",
    )
    tool = LlmTool(
        name="HassTurnOn",
        description="Turns on a Home Assistant device.",
        metadata=ToolMetadata(is_domain_aware=True, is_area_aware=True),
        parameters={
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "floor": {"type": "string"},
                "domain": {"type": "array", "items": {"type": "string"}},
            },
        },
    )

    await backend.async_ensure_collection_exists(
        config, device_collection, EMBEDDING_DIMENSION
    )
    await backend.async_ensure_collection_exists(
        config, tool_collection, EMBEDDING_DIMENSION
    )
    await backend.async_save_objects(
        config,
        device_collection,
        [DeviceEmbedding(device, [1.0, 0.0, 0.0])],
    )
    await backend.async_save_objects(
        config,
        tool_collection,
        [LlmToolEmbedding(tool, [0.0, 1.0, 0.0])],
    )

    assert await backend.async_list_objects(
        DeviceEmbedding, config, device_collection
    ) == [device]
    assert await backend.async_list_objects(
        LlmToolEmbedding, config, tool_collection
    ) == [tool]

    retrieved_devices = await _async_wait_for_retrieval(
        backend,
        DeviceEmbedding,
        config,
        device_collection,
        [1.0, 0.0, 0.0],
    )
    retrieved_tools = await _async_wait_for_retrieval(
        backend,
        LlmToolEmbedding,
        config,
        tool_collection,
        [0.0, 1.0, 0.0],
    )
    assert retrieved_devices == [device]
    assert retrieved_tools == [tool]

    scored_devices = await backend.async_retrieve_scored_objects(
        DeviceEmbedding,
        config,
        device_collection,
        [1.0, 0.0, 0.0],
        top_k=1,
    )
    assert scored_devices[0].item == device
    assert 0.0 <= scored_devices[0].score <= 1.0
    assert scored_devices[0].rank == 1

    await backend.async_reset_collection(
        config, device_collection, EMBEDDING_DIMENSION
    )
    assert await backend.async_list_objects(
        DeviceEmbedding, config, device_collection
    ) == []


async def _async_test_memory_lifecycle(
    backend: ABaseDbBackend,
    config: dict[str, Any],
    collection_name: str,
) -> None:
    first = Memory(
        "1111111111111111",
        "The bathroom is on the first floor.",
        "2026-09-03T10:00:00+00:00",
    )
    updated_first = Memory(
        first.id,
        "The bathroom is on 1st Floor.",
        first.created_at,
    )
    second = Memory(
        "2222222222222222",
        "The reading lamp is preferred.",
        "2026-09-03T10:01:00+00:00",
    )

    await backend.async_ensure_collection_exists(
        config, collection_name, EMBEDDING_DIMENSION
    )
    await backend.async_upsert_objects(
        config,
        collection_name,
        "memory_id",
        [
            MemoryEmbedding(first, [1.0, 0.0, 0.0]),
            MemoryEmbedding(second, [0.0, 1.0, 0.0]),
        ],
    )
    await backend.async_upsert_objects(
        config,
        collection_name,
        "memory_id",
        [MemoryEmbedding(updated_first, [1.0, 0.0, 0.0])],
    )

    memories = await backend.async_list_objects(
        MemoryEmbedding, config, collection_name
    )
    assert sorted(memories, key=lambda memory: memory.id) == [updated_first, second]

    await backend.async_increment_memory_retrieval_counts(
        config, collection_name, [updated_first.id]
    )
    memories = await backend.async_list_objects(
        MemoryEmbedding, config, collection_name
    )
    memories_by_id = {memory.id: memory for memory in memories}
    assert memories_by_id[updated_first.id].retrieval_count == 1
    assert memories_by_id[second.id].retrieval_count == 0

    assert await backend.async_delete_objects(
        config, collection_name, "memory_id", [updated_first.id]
    ) == 1
    assert await backend.async_list_objects(
        MemoryEmbedding, config, collection_name
    ) == [second]


async def _async_test_backend_lifecycle(
    database_case: DatabaseCase,
    backend: ABaseDbBackend,
    hass: MockHomeAssistant,
    config: dict[str, Any],
    test_id: str,
) -> None:
    collections = [
        f"ragent_{test_id}_devices",
        f"ragent_{test_id}_tools",
        f"ragent_{test_id}_memories",
    ]
    try:
        await _async_test_connection(database_case, backend, hass, config)
        await _async_test_object_round_trips(
            backend, config, f"ragent_{test_id}"
        )
        await _async_test_memory_lifecycle(backend, config, collections[2])
    finally:
        for collection_name in collections:
            await backend.async_cleanup_collection(config, collection_name)


def test_database_backend_lifecycle(
    database_case: DatabaseCase,
    hass: MockHomeAssistant,
) -> None:
    """Exercise persistence and retrieval for every vector database backend."""
    backend, config, test_id = _create_backend(database_case, hass)
    _run_async_test(
        hass,
        _async_test_backend_lifecycle(
            database_case,
            backend,
            hass,
            config,
            test_id,
        ),
    )
