from enum import IntEnum, StrEnum, auto


class VarType(StrEnum):
    INTEGER = auto()
    BYTE = auto()
    BYTE_ARRAY = auto()


class ShiftDirection(IntEnum):
    LEFT = auto()
    RIGHT = auto()


class ComparisonOperator(StrEnum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class BrainFogUserDataType(StrEnum):
    VAR = "var"
    INT = "int"
    BYTE = "byte"
    BYTE_ARRAY = "byte[]"


class OpCode(StrEnum):
    VAR = auto()
    READ = auto()
    PRINT = auto()
    SET = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    RAW = auto()
    IF = auto()
    ENDIF = auto()
