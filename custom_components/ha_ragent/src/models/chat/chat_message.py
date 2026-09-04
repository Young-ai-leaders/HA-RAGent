from typing import Any, Literal, NotRequired, TypedDict

class ChatFunction(TypedDict):
    name: str
    arguments: dict[str, Any] | str

class ChatToolCall(TypedDict):
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: ChatFunction

class ChatToolFailure(TypedDict):
    success: Literal[False]
    tool: str
    error: str

class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: NotRequired[list[ChatToolCall]]
    tool_name: NotRequired[str]
    tool_call_id: NotRequired[str]
