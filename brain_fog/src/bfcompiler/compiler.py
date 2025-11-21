import copy
import statistics
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

from .enums import (
    BooleanValue,
    ComparisonOperator,
    OpCode,
    ShiftDirection,
    UserDataType,
    VarType,
)


class Variable(ABC):
    def __init__(
        self,
        type: VarType,
        index: int = -1,
        length: int = 1,
    ):
        self.type = type
        self.index = index
        self.length = length

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(type={self.type}, value={self.value}, index={self.index}, length={self.length})"

    @property
    @abstractmethod
    def value(self):
        pass

    def get_shift_instruction(self, direction: ShiftDirection) -> str:
        return ("<" if direction == ShiftDirection.LEFT else ">") * self.index


class IntegerVariable(Variable):
    def __init__(self, index: int = -1, initial_value: int = 0):
        super().__init__(type=VarType.INTEGER, index=index, length=1)
        self._value: int = initial_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: int):
        if not isinstance(new_value, int):
            try:
                new_value: int = int(new_value)
            except Exception:
                raise ValueError("Value must be able to be converted to an integer")
        self._value = new_value

    @value.deleter
    def value(self):
        del self._value


class BooleanVariable(Variable):
    def __init__(self, index: int = -1, initial_value: int = 0):
        super().__init__(type=VarType.BOOL, index=index, length=1)
        self._value: int = initial_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: int | bool = True):
        if isinstance(new_value, bool):
            new_value: int = 1 if new_value else 0
        if not isinstance(new_value, int):
            try:
                new_value: int = int(new_value)
            except Exception:
                raise ValueError("Value must be able to be converted to an integer")

    @value.deleter
    def value(self):
        return self._value


class ByteVariable(Variable):
    def __init__(self, index: int = -1, initial_value: str | int = 0):
        super().__init__(type=VarType.BYTE, index=index, length=1)
        self._value: int = (
            ord(initial_value) if isinstance(initial_value, str) else initial_value
        )

    @property
    def value(self):
        return chr(self._value)

    @value.setter
    def value(self, new_value: str):
        self._value = ord(new_value)

    @value.deleter
    def value(self):
        del self._value

    @property
    def value_int(self):
        return self._value

    @value_int.setter
    def value_int(self, new_value: int):
        self._value = new_value

    @value_int.deleter
    def value_int(self):
        del self._value


class ByteArrayVariable(Variable):
    def __init__(
        self, index: int = -1, length: int = 10, initial_value: List[int] | None = None
    ):
        if initial_value is None:
            initial_value = [0 for _ in range(length)]
        else:
            temp_initial_value = [
                i for index, i in enumerate(initial_value) if index < length
            ]
            initial_value = copy.deepcopy(temp_initial_value)
        super().__init__(type=VarType.BYTE_ARRAY, index=index, length=length)
        self._value: List[int] = initial_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: int, index: int):
        self._value[index] = new_value

    @value.deleter
    def value(self):
        del self._value


class BrainFogCompiler:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.variables: Dict[str, Variable] = {}
        self.opcodes: Dict[str, Callable] = {}
        self.instructions: List[
            Tuple[str, int]
        ] = []  # [(instructions, pointer movement (i.e. -2 if shifting 2 spots left overall (None if unknown)))]
        self.output_str: str = ""
        self.current_pointer_location: int = 0
        self.current_block_depth: int = 0

        self.reserved_cells: Dict[str, int] = {}
        self.var_offset: int = len(self.reserved_cells)
        self.bool_in_code: bool = False

    def __print(self, *args, **kwargs) -> None:
        """Wrapper for print"""
        if self.debug:
            print(*args, **kwargs)

    def compile_from_file(self, filepath: str) -> str:
        """Compiles a Brain Fog file (.bfg) to Brainfuck

        Args:
            filepath (str): The path to the Brain Fog file

        Returns:
            str: The compiled Brainfuck
        """
        with open(filepath, "r") as f:
            data = f.read()
            f.close()

        return self.__compile_from_string(data)

    def __compile_from_string(self, input: str) -> str:
        """Compiles a string containing Brain Fog to Brainfuck

        Args:
            input (str): The incoming Brain Fog string

        Returns:
            str: The compiled Brainfuck
        """
        data: List[str] = input.split("\n")
        lines: List[str] = []

        block_depth: int = 0

        for line in data:
            line = line.strip()

            line = line.split("//", 1)[0]  # remove comments

            line = line.strip()

            if line == "":  # remove empty lines
                continue

            if "bool" in line.lower():
                if not self.bool_in_code:
                    self.__print("Boolean Detected! Enabling Boolean Support")
                    self.bool_in_code = True

            if line.startswith("if"):
                self.__reserve_new_cell(f"IF_{block_depth}")
                block_depth += 1

            if line.startswith("endif"):
                block_depth -= 1

            self.__reserve_cells_for_line(line)

            lines.append(line)

        self.__parse_lines(lines)

        return self.__compile_instructions()

    def __reserve_cells_for_line(self, line: str) -> None:
        line_parts: List[str] = line.split(" ")
        opcode: str = OpCode[line_parts[0].upper()]
        args: List[str] = line_parts[1:]
        match opcode:
            case OpCode.VAR:  # reserved_cells used: none
                pass
            case OpCode.READ:  # reserved_cells used: none
                pass
            case OpCode.PRINT:  # reserved_cells used: none
                pass
            case OpCode.SET:  # reserved_cells used: (COPYING, GENERAL_TEMP_1, GENERAL_TEMP_2) if BOOL in code
                if self.bool_in_code:
                    self.__reserve_cells(
                        ["COPYING", "GENERAL_TEMP_1", "GENERAL_TEMP_2"]
                    )
            case OpCode.ADD:  # reserved_cells used: COPYING
                self.__reserve_cells(["COPYING"])
            case OpCode.SUB:  # reserved_cells used: COPYING
                self.__reserve_cells(["COPYING"])
            case OpCode.MUL:  # reserved_cells used: COPYING, MULT_1, MULT_2
                self.__reserve_cells(["COPYING", "MULT_1", "MULT_2"])
            case (
                OpCode.DIV
            ):  # reserved_cells used: COPYING, DIVMOD_1, DIVMOD_2, DIVMOD_3, DIVMOD_4
                self.__reserve_cells(
                    ["COPYING", "DIVMOD_1", "DIVMOD_2", "DIVMOD_3", "DIVMOD_4"]
                )
            case (
                OpCode.MOD
            ):  # reserved_cells used: COPYING, DIVMOD_1, DIVMOD_2, DIVMOD_3, DIVMOD_4
                self.__reserve_cells(
                    ["COPYING", "DIVMOD_1", "DIVMOD_2", "DIVMOD_3", "DIVMOD_4"]
                )
            case OpCode.POW:  # reserved_cells used: COPYING, MULT_1, MULT_2, POW_1
                self.__reserve_cells(["COPYING", "MULT_1", "MULT_2", "POW_1"])
            case OpCode.RAW:  # reserved_cells used: none
                pass
            case OpCode.IF:  # reserved_cells used: COPYING, IF_{X} (already handled), COMPARISON_INPUT_1, COMPARISON_INPUT_2, MISC (Check comparison operators)
                self.__reserve_cells(
                    ["COPYING", "COMPARISON_INPUT_1", "COMPARISON_INPUT_2"]
                )
                self.__reserve_cells_for_if_begin(args)
            case OpCode.ENDIF:  # reserved_cells used: IF_{X} (already handled)
                pass

    def __reserve_cells_for_if_begin(self, args: List[str]) -> None:
        comparison_operator: ComparisonOperator = ComparisonOperator(args[2])
        match comparison_operator:
            case ComparisonOperator.GT:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_1, COMPARISON_TEMP_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.LT:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_1, COMPARISON_TEMP_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.GTE:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_1, COMPARISON_TEMP_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_2",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.LTE:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_1, COMPARISON_TEMP_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_2",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.EQUAL:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.NOT_EQUAL:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_1, COMPARISON_TEMP_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_2",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                    ]
                )
            case ComparisonOperator.BOOL_AND:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )
            case ComparisonOperator.BOOL_OR:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )
            case ComparisonOperator.BOOL_NAND:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )
            case ComparisonOperator.BOOL_NOR:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )
            case ComparisonOperator.BOOL_XOR:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )
            case ComparisonOperator.BOOL_XNOR:  # reserved_cells used: COPYING, COMPARISON_INPUT_1, COMPARISON_INPUT_2, COMPARISON_TEMP_3, COMPARISON_TEMP_4, COMPARISON_TEMP_5, COMPARISON_TEMP_6, GENERAL_TEMP_1
                self.__reserve_cells(
                    [
                        "COPYING",
                        "COMPARISON_INPUT_1",
                        "COMPARISON_INPUT_2",
                        "COMPARISON_TEMP_1",
                        "COMPARISON_TEMP_3",
                        "COMPARISON_TEMP_4",
                        "COMPARISON_TEMP_5",
                        "COMPARISON_TEMP_6",
                        "GENERAL_TEMP_1",
                    ]
                )

    def __reserve_cells(self, keys: List[str]) -> None:
        for key in keys:
            self.__reserve_new_cell(key)

    def __reserve_new_cell(self, key: str) -> None:
        if key not in self.reserved_cells.keys():
            self.__print(f"Reserving cell {key}: {self.var_offset}")
            self.reserved_cells[key] = self.var_offset
            self.var_offset = len(self.reserved_cells)

    def __optimize_instruction(self, instruction: str) -> str:
        output: str = copy.deepcopy(instruction)

        while not (len(output) == len(self.__optimize_instruction_step(output))):
            output = self.__optimize_instruction_step(output)

        return output

    def __optimize_instruction_step(self, instruction: str) -> str:
        # remove redundant shifts
        instruction = instruction.replace("<>", "")
        instruction = instruction.replace("><", "")

        # remove redundant increment and decrements
        instruction = instruction.replace("+-", "")
        instruction = instruction.replace("-+", "")

        # remove redundant cell clearings
        instruction = instruction.replace("[-][-]", "[-]")

        return instruction

    def __compile_instructions(self) -> str:
        output: str = ""

        for instruction, position_shift in self.instructions:
            if position_shift is None:
                print(
                    'Detected use of "raw" OpCode. Make sure you return the pointer to the same cell it starts at.'
                )

            output += instruction

        start_len: int = len(output)
        output = self.__optimize_instruction(output)
        end_len: int = len(output)

        self.__print(f"Optimized!\nStart Length: {start_len}\nEnd Length: {end_len}")

        return output

    def __parse_lines(self, lines: List[str]) -> None:
        for index, line in enumerate(lines):
            line_parts: List[str] = line.split(" ")
            opcode: str = OpCode[line_parts[0].upper()]
            args: List[str] = line_parts[1:]
            self.__print(f"{opcode}: {args}")
            match opcode:
                case OpCode.VAR:
                    self.__create_variable(args)
                case OpCode.READ:
                    self.__read_to_variable(args)
                case OpCode.PRINT:
                    self.__print_variable(args)
                case OpCode.SET:
                    self.__set_variable(args)
                case OpCode.ADD:
                    self.__add_variables(args)
                case OpCode.SUB:
                    self.__sub_variables(args)
                case OpCode.MUL:
                    self.__mul_variables(args)
                case OpCode.DIV:
                    self.__div_variables(args)
                case OpCode.MOD:
                    self.__mod_variables(args)
                case OpCode.POW:
                    self.__pow_variables(args)
                case OpCode.RAW:
                    self.__run_raw_bf_string(args)
                case OpCode.IF:
                    self.__begin_if_statement(args)
                case OpCode.ENDIF:
                    self.__end_if_statement(args)

        self.__print(f"Current Block Depth: {self.current_block_depth}")

        self.__print("Variables:", end="")
        if len(self.variables) == 0:
            self.__print(" None")
        else:
            self.__print()
        for name, var in self.variables.items():
            self.__print(f" - {name}: {var}")

        self.__print("Reserved Cells:", end="")
        if len(self.reserved_cells) == 0:
            self.__print(" None")
        else:
            self.__print()
        for name, cell_index in self.reserved_cells.items():
            self.__print(f" - {name}: {cell_index}")

    def __end_if_statement(self, args) -> None:
        self.__print(f"END_IF: getting IF_{self.current_block_depth - 1}")
        temp_index_1: int = self.reserved_cells[
            f"IF_{self.current_block_depth - 1}"
        ]  # stores result of comparison (x)
        instruction: str = ""
        instruction += self.__clear_cell(temp_index_1)
        instruction += ">" * temp_index_1
        instruction += "]"
        instruction += "<" * temp_index_1

        self.__print(instruction)
        self.current_block_depth -= 1
        self.instructions.append((instruction, 0))

    def __begin_if_statement(self, args) -> None:
        value_1_type: UserDataType = UserDataType(args[0][1:])
        value_1_value_str: str = args[1]
        value_1_value: int
        comparison_operator: ComparisonOperator = ComparisonOperator(args[2])
        value_2_type: UserDataType = UserDataType(args[3])
        value_2_value_str: str = args[4][:-1]
        self.__print(f"BEGIN_IF: getting IF_{self.current_block_depth}")
        temp_index_1: int = self.reserved_cells[
            f"IF_{self.current_block_depth}"
        ]  # stores result of comparison (x)
        temp_index_2: int = self.reserved_cells["COMPARISON_INPUT_1"]
        temp_index_3: int = self.reserved_cells["COMPARISON_INPUT_2"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                temp_index_1,
                temp_index_2,
                temp_index_3,
            ]
        )

        match value_1_type:
            case UserDataType.INT:
                value_1_value = int(value_1_value_str)
                instruction += self.__set_cell_to_value(temp_index_2, value_1_value)
            case UserDataType.BOOL:
                value_1_value = (
                    1
                    if BooleanValue(value_1_value_str.upper()) == BooleanValue.TRUE
                    else 0
                )
                instruction += self.__set_cell_to_value(temp_index_2, value_1_value)
            case UserDataType.BYTE:
                value_1_value = ord(value_1_value_str[1])
                instruction += self.__set_cell_to_value(temp_index_2, value_1_value)
            case UserDataType.BYTE_ARRAY:
                value_1_value = statistics.mean(
                    [ord(character) for character in value_1_value_str[1:-1]]
                )
                instruction += self.__set_cell_to_value(temp_index_2, value_1_value)
            case UserDataType.VAR:
                providing_var: Variable = self.variables[value_1_value_str]
                instruction += self.__copy_cell(providing_var.index, temp_index_2)

        match value_2_type:
            case UserDataType.INT:
                value_2_value = int(value_2_value_str)
                instruction += self.__set_cell_to_value(temp_index_3, value_2_value)
            case UserDataType.BOOL:
                value_2_value = (
                    1
                    if BooleanValue(value_2_value_str.upper()) == BooleanValue.TRUE
                    else 0
                )
                instruction += self.__set_cell_to_value(temp_index_3, value_2_value)
            case UserDataType.BYTE:
                value_2_value = ord(value_2_value_str[1])
                instruction += self.__set_cell_to_value(temp_index_3, value_2_value)
            case UserDataType.BYTE_ARRAY:
                value_2_value = statistics.mean(
                    [ord(character) for character in value_2_value_str[1:-1]]
                )
                instruction += self.__set_cell_to_value(temp_index_3, value_2_value)
            case UserDataType.VAR:
                providing_var: Variable = self.variables[value_2_value_str]
                instruction += self.__copy_cell(providing_var.index, temp_index_3)

        match comparison_operator:
            case ComparisonOperator.GT:
                instruction += self.__gt_variables(temp_index_1)
            case ComparisonOperator.LT:
                instruction += self.__lt_variables(temp_index_1)
            case ComparisonOperator.GTE:
                instruction += self.__gte_variables(temp_index_1)
            case ComparisonOperator.LTE:
                instruction += self.__lte_variables(temp_index_1)
            case ComparisonOperator.EQUAL:
                instruction += self.__equal_variables(temp_index_1)
            case ComparisonOperator.NOT_EQUAL:
                instruction += self.__not_equal_variables(temp_index_1)
            case ComparisonOperator.BOOL_AND:
                instruction += self.__bool_and_variables(temp_index_1)
            case ComparisonOperator.BOOL_OR:
                instruction += self.__bool_or_variables(temp_index_1)
            case ComparisonOperator.BOOL_NAND:
                instruction += self.__bool_nand_variables(temp_index_1)
            case ComparisonOperator.BOOL_NOR:
                instruction += self.__bool_nor_variables(temp_index_1)
            case ComparisonOperator.BOOL_XOR:
                instruction += self.__bool_xor_variables(temp_index_1)
            case ComparisonOperator.BOOL_XNOR:
                instruction += self.__bool_xnor_variables(temp_index_1)

        instruction += ">" * temp_index_1
        instruction += "["
        instruction += "<" * temp_index_1

        self.__print(instruction)
        self.current_block_depth += 1
        self.instructions.append((instruction, 0))

    def __bool_xnor_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_3"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(left_index)
        instruction += self.__convert_cell_to_boolean(right_index)
        instruction += self.__bool_xor_variables(output_index)
        instruction += self.__copy_cell(output_index, left_index)
        instruction += self.__clear_cells([right_index, output_index])
        instruction += self.__bool_not_variable(output_index)
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __bool_xor_variables(self, output_index: int) -> str:
        x_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        y_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        z_index: int = output_index
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells([x_storage_index, y_storage_index, z_index])
        instruction += self.__copy_cell(x_index, x_storage_index)
        instruction += self.__copy_cell(y_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(x_index)
        instruction += self.__convert_cell_to_boolean(y_index)
        instruction += ">" * x_index
        instruction += "["
        instruction += "<" * x_index
        instruction += ">" * y_index
        instruction += "-"
        instruction += "<" * y_index
        instruction += ">" * x_index
        instruction += "-]"
        instruction += "<" * x_index
        instruction += ">" * y_index
        instruction += "["
        instruction += "<" * y_index
        instruction += ">" * z_index
        instruction += "+"
        instruction += "<" * z_index
        instruction += ">" * y_index
        instruction += "[-]]"
        instruction += "<" * y_index

        instruction += self.__copy_cell(x_storage_index, x_index)
        instruction += self.__copy_cell(y_storage_index, y_index)

        return instruction

    def __bool_nand_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_3"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(left_index)
        instruction += self.__convert_cell_to_boolean(right_index)
        instruction += self.__bool_and_variables(output_index)
        instruction += self.__copy_cell(output_index, left_index)
        instruction += self.__clear_cells([right_index, output_index])
        instruction += self.__bool_not_variable(output_index)
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __bool_nor_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_3"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(left_index)
        instruction += self.__convert_cell_to_boolean(right_index)
        instruction += self.__bool_or_variables(output_index)
        instruction += self.__copy_cell(output_index, left_index)
        instruction += self.__clear_cells([right_index, output_index])
        instruction += self.__bool_not_variable(output_index)
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __bool_or_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(left_index)
        instruction += self.__convert_cell_to_boolean(right_index)
        instruction += ">" * left_index
        instruction += "["
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "+"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "-]"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "[[-]"
        instruction += "<" * right_index
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * right_index
        instruction += "]"
        instruction += "<" * right_index
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __bool_and_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += self.__convert_cell_to_boolean(left_index)
        instruction += self.__convert_cell_to_boolean(right_index)
        instruction += ">" * left_index
        instruction += "[-"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "[-"
        instruction += "<" * right_index
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * right_index
        instruction += "]"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "]"
        instruction += "<" * left_index
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, y_storage_index)

        return instruction

    def __gt_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        temp_index_1: int = self.reserved_cells["COMPARISON_TEMP_1"]
        temp_index_2: int = self.reserved_cells["COMPARISON_TEMP_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [temp_index_1, temp_index_2, x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += ">" * left_index
        instruction += "["
        instruction += "<" * left_index
        instruction += ">" * temp_index_1
        instruction += "+"
        instruction += "<" * temp_index_1
        instruction += ">" * right_index
        instruction += "[-"
        instruction += "<" * right_index
        instruction += self.__clear_cell(temp_index_1)
        instruction += ">" * temp_index_2
        instruction += "+"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "]"
        instruction += "<" * right_index
        instruction += ">" * temp_index_1
        instruction += "[-"
        instruction += "<" * temp_index_1
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * temp_index_1
        instruction += "]"
        instruction += "<" * temp_index_1
        instruction += ">" * temp_index_2
        instruction += "[-"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "+"
        instruction += "<" * right_index
        instruction += ">" * temp_index_2
        instruction += "]"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "-"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "-]"
        instruction += "<" * left_index
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __lt_variables(self, output_index: int) -> str:
        right_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        left_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        temp_index_1: int = self.reserved_cells["COMPARISON_TEMP_1"]
        temp_index_2: int = self.reserved_cells["COMPARISON_TEMP_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [temp_index_1, temp_index_2, x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += ">" * left_index
        instruction += "["
        instruction += "<" * left_index
        instruction += ">" * temp_index_1
        instruction += "+"
        instruction += "<" * temp_index_1
        instruction += ">" * right_index
        instruction += "[-"
        instruction += "<" * right_index
        instruction += self.__clear_cell(temp_index_1)
        instruction += ">" * temp_index_2
        instruction += "+"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "]"
        instruction += "<" * right_index
        instruction += ">" * temp_index_1
        instruction += "[-"
        instruction += "<" * temp_index_1
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * temp_index_1
        instruction += "]"
        instruction += "<" * temp_index_1
        instruction += ">" * temp_index_2
        instruction += "[-"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "+"
        instruction += "<" * right_index
        instruction += ">" * temp_index_2
        instruction += "]"
        instruction += "<" * temp_index_2
        instruction += ">" * right_index
        instruction += "-"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "-]"
        instruction += "<" * left_index
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __gte_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        temp_index_0: int = self.reserved_cells["COMPARISON_TEMP_1"]
        temp_index_1: int = self.reserved_cells["COMPARISON_TEMP_2"]
        gt_output_index: int = self.reserved_cells[
            "COMPARISON_TEMP_3"
        ]  # stores the output of GT
        equal_output_index: int = self.reserved_cells[
            "COMPARISON_TEMP_4"
        ]  # stores the output of EQUAL
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                temp_index_0,
                temp_index_1,
                gt_output_index,
                equal_output_index,
                output_index,
                x_storage_index,
                y_storage_index,
            ]
        )
        instruction += self.__gt_variables(gt_output_index)
        instruction += self.__clear_cells([temp_index_0, temp_index_1])
        instruction += self.__equal_variables(equal_output_index)
        instruction += self.__copy_cell(gt_output_index, left_index)
        instruction += self.__copy_cell(equal_output_index, right_index)
        instruction += self.__or_variables(output_index)

        return instruction

    def __lte_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        temp_index_0: int = self.reserved_cells["COMPARISON_TEMP_1"]
        temp_index_1: int = self.reserved_cells["COMPARISON_TEMP_2"]
        lt_output_index: int = self.reserved_cells[
            "COMPARISON_TEMP_3"
        ]  # stores the output of LT
        equal_output_index: int = self.reserved_cells[
            "COMPARISON_TEMP_4"
        ]  # stores the output of EQUAL
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                temp_index_0,
                temp_index_1,
                lt_output_index,
                equal_output_index,
                output_index,
                x_storage_index,
                y_storage_index,
            ]
        )
        instruction += self.__lt_variables(lt_output_index)
        instruction += self.__clear_cells([temp_index_0, temp_index_1])
        instruction += self.__equal_variables(equal_output_index)
        instruction += self.__copy_cell(lt_output_index, left_index)
        instruction += self.__copy_cell(equal_output_index, right_index)
        instruction += self.__or_variables(output_index)

        return instruction

    def __or_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += ">" * left_index
        instruction += "["
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "+"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "-]"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "[[-]"
        instruction += "<" * right_index
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * right_index
        instruction += "]"
        instruction += "<" * right_index
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __equal_variables(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        right_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_6"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [x_storage_index, y_storage_index, output_index]
        )
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += self.__copy_cell(right_index, y_storage_index)
        instruction += ">" * left_index
        instruction += "[-"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "-"
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "]+"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "["
        instruction += "<" * right_index
        instruction += ">" * left_index
        instruction += "-"
        instruction += "<" * left_index
        instruction += ">" * right_index
        instruction += "[-]]"
        instruction += "<" * right_index
        instruction += self.__copy_cell(left_index, output_index)
        instruction += self.__copy_cell(x_storage_index, left_index)
        instruction += self.__copy_cell(y_storage_index, right_index)

        return instruction

    def __not_equal_variables(self, output_index: int) -> str:
        x_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        y_index: int = self.reserved_cells["COMPARISON_INPUT_2"]
        temp_index_0: int = self.reserved_cells[
            "COMPARISON_TEMP_1"
        ]  # store the output of NOT
        temp_index_1: int = self.reserved_cells[
            "COMPARISON_TEMP_2"
        ]  # store the output of EQUAL
        z_index: int = output_index
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_3"]
        y_storage_index: int = self.reserved_cells["COMPARISON_TEMP_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [temp_index_0, temp_index_1, x_storage_index, y_storage_index, z_index]
        )
        instruction += self.__copy_cell(x_index, x_storage_index)
        instruction += self.__copy_cell(y_index, y_storage_index)
        instruction += self.__equal_variables(temp_index_1)
        instruction += self.__copy_cell(temp_index_1, x_index)
        instruction += self.__bool_not_variable(temp_index_1)
        instruction += self.__copy_cell(temp_index_1, output_index)
        instruction += self.__copy_cell(x_storage_index, x_index)
        instruction += self.__copy_cell(y_storage_index, y_index)

        return instruction

    def __bool_not_variable(self, output_index: int) -> str:
        left_index: int = self.reserved_cells["COMPARISON_INPUT_1"]
        temp_index_1: int = self.reserved_cells["COMPARISON_TEMP_1"]
        x_storage_index: int = self.reserved_cells["COMPARISON_TEMP_5"]
        instruction: str = ""
        instruction += self.__clear_cells([temp_index_1, x_storage_index, output_index])
        instruction += self.__copy_cell(left_index, x_storage_index)
        instruction += ">" * temp_index_1
        instruction += "+"
        instruction += "<" * temp_index_1
        instruction += ">" * left_index
        instruction += "[[-]"
        instruction += "<" * left_index
        instruction += ">" * temp_index_1
        instruction += "-"
        instruction += "<" * temp_index_1
        instruction += ">" * left_index
        instruction += "]"
        instruction += "<" * left_index
        instruction += ">" * temp_index_1
        instruction += "[-"
        instruction += "<" * temp_index_1
        instruction += ">" * left_index
        instruction += "+"
        instruction += "<" * left_index
        instruction += ">" * temp_index_1
        instruction += "]"
        instruction += "<" * temp_index_1
        instruction += self.__copy_cell(left_index, output_index)
        instruction += self.__copy_cell(x_storage_index, left_index)

        return instruction

    def __convert_cell_to_boolean(self, target_index: int) -> str:
        temp_index_1: int = self.reserved_cells["GENERAL_TEMP_1"]
        instruction: str = ""
        instruction += self.__clear_cell(temp_index_1)
        instruction += ">" * target_index
        instruction += "[[-]"
        instruction += "<" * target_index
        instruction += ">" * temp_index_1
        instruction += "+"
        instruction += "<" * temp_index_1
        instruction += ">" * target_index
        instruction += "]"
        instruction += "<" * target_index
        instruction += self.__copy_cell(temp_index_1, target_index)

        return instruction

    def __run_raw_bf_string(self, args) -> None:
        raw_string: str = args[0]
        instruction: str = ""
        instruction = raw_string[1:-1]

        self.__print(instruction)
        self.instructions.append((instruction, None))

    def __pow_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        temp_index_1: int = self.reserved_cells["MULT_1"]
        temp_index_2: int = self.reserved_cells["MULT_2"]
        temp_index_3: int = self.reserved_cells["POW_1"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                output_index,
                temp_index_1,
                temp_index_2,
                temp_index_3,
            ]
        )
        instruction += self.__copy_cell(index_2, temp_index_3)
        instruction += ">" * temp_index_3
        instruction += "["
        instruction += "<" * temp_index_3
        instruction += self.__copy_cell(index_1, temp_index_1)
        instruction += ">" * temp_index_1
        instruction += "["
        instruction += "<" * temp_index_1
        instruction += self.__copy_cell(index_1, temp_index_2)
        instruction += ">" * temp_index_2
        instruction += "["
        instruction += "<" * temp_index_2
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * temp_index_2
        instruction += "-]"
        instruction += "<" * temp_index_2
        instruction += ">" * temp_index_1
        instruction += "-]"
        instruction += "<" * temp_index_1
        instruction += ">" * temp_index_3
        instruction += "-]"
        instruction += "<" * temp_index_3

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __mod_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        temp_index_1: int = self.reserved_cells["DIVMOD_1"]
        temp_index_2: int = self.reserved_cells["DIVMOD_2"]
        temp_index_3: int = self.reserved_cells["DIVMOD_3"]
        temp_index_4: int = self.reserved_cells["DIVMOD_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                output_index,
                temp_index_1,
                temp_index_2,
                temp_index_3,
                temp_index_4,
            ]
        )
        instruction += self.__copy_cell(index_1, temp_index_1)
        instruction += self.__copy_cell(index_2, temp_index_2)
        instruction += self.__set_cell_to_value(temp_index_3, 1)
        instruction += ">" * temp_index_1
        instruction += "[->-[>+>>]>[+[-<+>]>+>>]<<<<<]"  # ripped from https://esolangs.org/wiki/Brainfuck_algorithms
        instruction += "<" * temp_index_1
        instruction += self.__copy_cell(temp_index_3, output_index)

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __div_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        temp_index_1: int = self.reserved_cells["DIVMOD_1"]
        temp_index_2: int = self.reserved_cells["DIVMOD_2"]
        temp_index_3: int = self.reserved_cells["DIVMOD_3"]
        temp_index_4: int = self.reserved_cells["DIVMOD_4"]
        instruction: str = ""
        instruction += self.__clear_cells(
            [
                output_index,
                temp_index_1,
                temp_index_2,
                temp_index_3,
                temp_index_4,
            ]
        )
        instruction += self.__copy_cell(index_1, temp_index_1)
        instruction += self.__copy_cell(index_2, temp_index_2)
        instruction += self.__set_cell_to_value(temp_index_3, 1)
        instruction += ">" * temp_index_1
        instruction += "[->-[>+>>]>[+[-<+>]>+>>]<<<<<]"  # ripped from https://esolangs.org/wiki/Brainfuck_algorithms
        instruction += "<" * temp_index_1
        instruction += self.__copy_cell(temp_index_4, output_index)

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __mul_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        temp_index_1: int = self.reserved_cells["MULT_1"]
        temp_index_2: int = self.reserved_cells["MULT_2"]
        instruction: str = ""
        instruction += self.__clear_cells([output_index, temp_index_1, temp_index_2])
        instruction += self.__copy_cell(index_1, temp_index_1)
        instruction += ">" * temp_index_1
        instruction += "["
        instruction += "<" * temp_index_1
        instruction += self.__copy_cell(index_2, temp_index_2)
        instruction += ">" * temp_index_2
        instruction += "["
        instruction += "<" * temp_index_2
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * temp_index_2
        instruction += "-]"
        instruction += "<" * temp_index_2
        instruction += ">" * temp_index_1
        instruction += "-]"
        instruction += "<" * temp_index_1

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __set_cell_to_value(self, target_index: int, value: int) -> str:
        instruction: str = ""
        instruction += self.__clear_cell(target_index)
        instruction += ">" * target_index
        instruction += "+" * value
        instruction += "<" * target_index

        return instruction

    def __move_cell(
        self, from_index: int, to_index: int, current_index: int = 0
    ) -> str:
        instruction: str = ""
        instruction += self.__clear_cell(to_index)
        instruction += "<" * current_index
        instruction += ">" * from_index
        instruction += "["
        instruction += "<" * from_index
        instruction += ">" * to_index
        instruction += "+"
        instruction += "<" * to_index
        instruction += ">" * from_index
        instruction += "-]"
        instruction += "<" * from_index
        instruction += ">" * current_index

        return instruction

    def __copy_cell(
        self, from_index: int, to_index: int, current_index: int = 0
    ) -> str:
        temp_index: int = self.reserved_cells["COPYING"]
        instruction: str = ""
        instruction += self.__clear_cells([to_index, temp_index])
        instruction += "<" * current_index
        instruction += ">" * from_index
        instruction += "["
        instruction += "<" * from_index
        instruction += ">" * temp_index
        instruction += "+"
        instruction += "<" * temp_index
        instruction += ">" * to_index
        instruction += "+"
        instruction += "<" * to_index
        instruction += ">" * from_index
        instruction += "-]"
        instruction += "<" * from_index
        instruction += ">" * temp_index
        instruction += "["
        instruction += "<" * temp_index
        instruction += ">" * from_index
        instruction += "+"
        instruction += "<" * from_index
        instruction += ">" * temp_index
        instruction += "-]"
        instruction += "<" * temp_index
        instruction += ">" * current_index

        return instruction

    def __clear_cells(self, target_indexes: List[int]) -> str:
        instruction: str = ""

        for index in target_indexes:
            instruction += self.__clear_cell(index)

        return instruction

    def __clear_cell(self, target_index: int) -> str:
        instruction: str = ""
        instruction += ">" * target_index
        instruction += "[-]"
        instruction += "<" * target_index
        return instruction

    def __sub_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        instruction: str = ""
        instruction += self.__clear_cell(output_index)
        instruction += self.__copy_cell(index_1, output_index)
        instruction += ">" * index_2
        instruction += "["
        instruction += "<" * index_2
        instruction += ">" * output_index
        instruction += "-"
        instruction += "<" * output_index
        instruction += ">" * index_2
        instruction += "-]"
        instruction += "<" * index_2

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __add_variables(self, args) -> None:
        left_input_var: Variable = self.variables[args[0]]
        right_input_var: Variable = self.variables[args[1]]
        output_var: Variable = self.variables[args[2]]
        index_1: int = left_input_var.index
        index_2: int = right_input_var.index
        output_index: int = output_var.index
        instruction: str = ""
        instruction += self.__clear_cell(output_index)
        instruction += self.__copy_cell(index_1, output_index)
        instruction += ">" * index_2
        instruction += "["
        instruction += "<" * index_2
        instruction += ">" * output_index
        instruction += "+"
        instruction += "<" * output_index
        instruction += ">" * index_2
        instruction += "-]"
        instruction += "<" * index_2

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __set_variable(self, args) -> None:
        target_var: Variable = self.variables[args[0]]
        value_type: UserDataType = UserDataType(args[2])
        instruction: str = ""
        match value_type:
            case UserDataType.INT:
                new_value: int = int(args[3])
                instruction += self.__set_cell_to_value(target_var.index, new_value)
            case UserDataType.BOOL:
                bool_value: BooleanValue = BooleanValue(args[3].upper())
                new_value: int = 1 if bool_value == BooleanValue.TRUE else 0
                instruction += self.__set_cell_to_value(target_var.index, new_value)
            case UserDataType.BYTE:
                new_value: int = ord(args[3][1])
                instruction += self.__set_cell_to_value(target_var.index, new_value)
            case UserDataType.BYTE_ARRAY:
                value_args: List[str] = args[3:]
                value_str: str = " ".join(value_args)
                new_values: List[str] = list(
                    value_str[1:-1].encode().decode("unicode_escape")
                )
                instruction += "".join(
                    [
                        self.__set_cell_to_value(
                            target_var.index + index, ord(character)
                        )
                        for index, character in enumerate(new_values)
                    ]
                )
            case UserDataType.VAR:
                providing_var: Variable = self.variables[args[3]]
                target_var_index: int = target_var.index
                providing_var_index: int = providing_var.index
                if target_var.type == VarType.BOOL:
                    temp_index_1: int = self.reserved_cells["GENERAL_TEMP_1"]
                    temp_index_2: int = self.reserved_cells["GENERAL_TEMP_2"]
                    instruction += self.__clear_cells([temp_index_1, temp_index_2])
                    instruction += self.__copy_cell(providing_var_index, temp_index_2)
                    instruction += self.__convert_cell_to_boolean(temp_index_2)
                    instruction += self.__copy_cell(temp_index_2, target_var_index)
                else:
                    instruction += "".join(
                        [
                            self.__copy_cell(
                                providing_var.index + offset, target_var.index + offset
                            )
                            for offset in range(target_var.length)
                        ]
                    )

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __print_variable(self, args) -> None:
        var: Variable = self.variables[args[0]]
        instruction: str = ""
        instruction += ">" * var.index
        for _ in range(var.length):
            instruction += ".>"
        instruction += "<" * (var.index + var.length)

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __read_to_variable(self, args) -> None:
        char_count: int = int(args[0])
        dest_var: Variable = self.variables[args[1]]
        if char_count > dest_var.length:
            raise OverflowError(
                f"{char_count} bytes exceeds the size of variable '{args[1]}' when reading."
            )
        instruction: str = ""
        instruction += ">" * dest_var.index
        for _ in range(char_count):
            instruction += ",>"
        instruction += "<" * (dest_var.index + char_count)

        self.__print(instruction)
        self.instructions.append((instruction, 0))

    def __create_variable(self, args) -> None:
        name: str = args[0]
        var_type: UserDataType = UserDataType(args[1])
        match var_type:
            case UserDataType.INT:
                self.variables[name] = IntegerVariable(index=self.var_offset)
            case UserDataType.BYTE:
                self.variables[name] = ByteVariable(index=self.var_offset)
            case UserDataType.BOOL:
                self.variables[name] = BooleanVariable(index=self.var_offset)
            case _:
                length: int = int(args[1][5:-1])
                self.variables[name] = ByteArrayVariable(
                    index=self.var_offset, length=length
                )
        self.var_offset += self.variables[name].length


if __name__ == "__main__":
    compiler = BrainFogCompiler()
    output: str = compiler.compile_from_file("test.bfg")
