from enum import StrEnum

class ResultTypeEnum(StrEnum):
    NONE = "none"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL = "tool"