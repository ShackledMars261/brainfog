from enum import IntEnum, StrEnum, auto


class VarType(StrEnum):
    INTEGER = auto()
    BYTE = auto()
    BYTE_ARRAY = auto()
    BOOL = auto()


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
    BOOL_AND = "and"
    BOOL_OR = "or"
    BOOL_NAND = "nand"
    BOOL_NOR = "nor"
    BOOL_XOR = "xor"
    BOOL_XNOR = "xnor"


class UserDataType(StrEnum):
    VAR = "var"
    INT = "int"
    BYTE = "byte"
    BYTE_ARRAY = "byte[]"
    BOOL = "bool"

    @classmethod
    def _missing_(cls, value):
        return cls.BYTE_ARRAY


class BooleanValue(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"


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
