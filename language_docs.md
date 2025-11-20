# Brain Fog Language Docs
```
Reserved cells:
0-19: Various temporary operations
- 1: used for copying variables non-destructively
- 2-3: used in multiplication/exponentiation
- 4: used in exponentiation
- 5-8: used in division/modulo
- 9: used in if statements
- 10-11: used as input for comparisons
- 12-13: used in comparisons
- 14-19: reserved but currently unused
20-n+20: Variables (n = sum of all variables lengths)
n+21-30k: Unreserved

Variable declaration:
var VAR_NAME VAR_TYPE

Variable assignment to integer:
set VAR_NAME = int VALUE

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


if:
if (TYPE VALUE COMPARISON_OPERATOR TYPE VALUE) (ex. "var x > int 3")
    OPERATION1
    OPERATION2
endif
```