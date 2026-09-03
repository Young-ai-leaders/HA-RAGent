from types import SimpleNamespace

from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import (
    HistoryManager,
    conversation,
)


def test_retrieval_texts_include_only_user_queries() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 0,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="  turn off the light strip  "),
        conversation.AssistantContent(
            agent_id="agent",
            content="I'll turn it off.",
            tool_calls=[],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="call",
            tool_name="HassCancelAllPlannedActions",
            tool_result={"success": True},
        ),
        conversation.UserContent(content="turn on the light strip"),
    ])

    assert manager.retrieval_texts(chat_log) == ["turn off the light strip"]
