from typing import Any, Callable
from unittest.mock import Mock
import aiohttp

class MockHomeAssistant:
    """Small Home Assistant substitute covering the backend's API surface."""
    def __init__(self) -> None:
        self.async_add_executor_job = Mock(side_effect=self._run_executor_job)
        self._client_session: aiohttp.ClientSession | None = None

    async def _run_executor_job(self, target: Callable[..., Any], *args: Any) -> Any:
        return target(*args)


def async_get_clientsession(hass: MockHomeAssistant) -> aiohttp.ClientSession:
    if hass._client_session is None:
        hass._client_session = aiohttp.ClientSession()
    return hass._client_session
