"""Minimal Home Assistant substitutes used when Home Assistant is unavailable."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import Mock
from types import SimpleNamespace

import aiohttp

from custom_components.ha_ragent.src.const import CONF_LLM_HASS_API

class MockLlmTool:
    """Minimal substitute for Home Assistant's LLM tool base class."""


llm = SimpleNamespace(Tool=MockLlmTool)


class MockDateTimeUtil:
    """Minimal substitute for Home Assistant's datetime helpers."""

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)


dt_util = MockDateTimeUtil()


@dataclass
class MockToolInput:
    tool_name: str
    tool_args: dict[str, Any]


class MockContent:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class MockSystemContent(MockContent):
    pass


class MockUserContent(MockContent):
    pass


class MockAssistantContent(MockContent):
    pass


class MockToolResultContent(MockContent):
    pass


conversation = SimpleNamespace(
    Content=MockContent,
    SystemContent=MockSystemContent,
    UserContent=MockUserContent,
    AssistantContent=MockAssistantContent,
    ToolResultContent=MockToolResultContent,
)


class MockHomeAssistant:
    def __init__(self) -> None:
        self.async_add_executor_job = Mock(side_effect=self._run_executor_job)
        self._client_session: aiohttp.ClientSession | None = None

    async def _run_executor_job(self, target: Callable[..., Any], *args: Any) -> Any:
        """Run a function in the executor."""
        return target(*args)

    async def async_close(self) -> None:
        """Close resources owned by the Home Assistant test substitute."""
        if self._client_session is not None:
            await self._client_session.close()
            self._client_session = None


def async_get_clientsession(hass: MockHomeAssistant) -> aiohttp.ClientSession:
    """Get the client session for the Home Assistant test substitute."""
    if hass._client_session is None:
        hass._client_session = aiohttp.ClientSession()
    return hass._client_session
