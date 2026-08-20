from types import SimpleNamespace

from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import (
    MessageHelper,
    conversation,
)

ToolInput = SimpleNamespace

def test_create_tool_failure_message() -> None:
    message = MessageHelper.create_tool_failure_message(
        "agent", "call-1", "HassTurnOn", RuntimeError("failed"),
    )

    assert message.tool_result == {
        "success": False,
        "tool": "HassTurnOn",
        "error": "failed",
    }

def test_compact_tool_result_only_compacts_semantic_search() -> None:
    search = conversation.ToolResultContent(
        agent_id="agent", tool_call_id="call-1",
        tool_name="HassSemanticSearch", tool_result={"devices": ["large"]},
    )
    other = conversation.ToolResultContent(
        agent_id="agent", tool_call_id="call-2",
        tool_name="HassTurnOn", tool_result={"success": False},
    )

    assert MessageHelper.compact_tool_result(search).tool_result == {"success": True}
    assert MessageHelper.compact_tool_result(other) is other

def test_message_to_retrieval_text() -> None:
    assert MessageHelper.message_to_retrieval_text(
        conversation.UserContent(content="  turn on light  ")
    ) == "turn on light"
    assert MessageHelper.message_to_retrieval_text(
        conversation.ToolResultContent(
            agent_id="agent", tool_call_id="call", tool_name="HassTurnOn",
            tool_result={"success": True},
        )
    ) == "HassTurnOn {'success': True}"
    assert MessageHelper.message_to_retrieval_text(object()) == ""

def test_message_to_chat_messages() -> None:
    messages = MessageHelper.message_to_chat_messages([
        conversation.SystemContent(content="system"),
        conversation.UserContent(content="user"),
        conversation.AssistantContent(
            agent_id="agent", content="assistant",
            tool_calls=[ToolInput(id="call", tool_name="HassTurnOn", tool_args={"name": "light.x"})],
        ),
        conversation.ToolResultContent(
            agent_id="agent", tool_call_id="call", tool_name="HassTurnOn",
            tool_result={"success": True},
        ),
    ])

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert messages[3]["tool_call_id"] == "call"

def test_clean_assistant_content() -> None:
    content = "Done\n```homeassistant\n{\"tool\": \"HassTurnOn\"}\n```"

    assert MessageHelper.clean_assistant_content(content, True) == "Done"
    assert MessageHelper.clean_assistant_content(content, False) == content

def test_repeated_success_result_contains_only_success_fields() -> None:
    message = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-1", "HassTurnOn",
        {"success": True, "description": "light", "error": "stale"},
    )

    assert message.tool_result == {
        "success": True,
        "already_executed": True,
    }

def test_repeated_failure_preserves_error() -> None:
    message = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-2", "HassTurnOn",
        {"success": False, "error": "target not found", "details": "ignored"},
    )

    assert message.tool_result == {
        "success": False,
        "already_executed": True,
        "error": "target not found",
    }
