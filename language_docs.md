# Brain Fog Language Docs
```
Reserved cells: 
- As of v0.0.6, reserved cells are now found dynamically based on which ones are actually necessary to run your script

Variable declaration:
var VAR_NAME VAR_TYPE

Variable assignment to integer:
set VAR_NAME = int VALUE

Variable assignment to boolean:
set VAR_NAME = bool TRUE/FALSE

Variable assignment to byte:
set VAR_NAME = byte "C" (any single ASCII character)

Variable assignment to byte[]:
set VAR_NAME = byte[] "VALUE"

Variable assignment to other variable's value:
set VAR_NAME = var TARGET_VAR_NAME

Variable types:
byte, byte[] (hopefully), integer

Arithmetic:
add/sub/mul/div/mod/pow a b c (sets c = a <specified operation> b)

Raw BF string:
raw BF_STRING

Comments:
// this is a comment

Comparison Operators:
> - GT
< - LT
>= - GTE
<= - LTE
== - EQUAL
!= - NOT_EQUAL

If statements:
if (TYPE VALUE COMPARISON_OPERATOR TYPE VALUE) (ex. "var x > int 3")
    OPERATION1
    OPERATION2
endif
```