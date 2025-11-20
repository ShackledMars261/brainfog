# Brain Fog Language Docs
```
Reserved cells:
0-16: Various uses:
- 0: used for copying variables non-destructively
- 1-2: used in multiplication/exponentiation
- 3: used in exponentiation
- 4-7: used in division/modulo
- 8-9: used as input for comparisons
- 10-15: used in comparisons
- 16: held as a temporary cell for any operation to use
17-16+n: used as a callback for if statements, if your bfg code contains any (n = max depth of blocks (nested ifs))
17+n-17+n+m: Variables (m = sum of all variables lengths)
18+n+m-30,000: Unreserved

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

If statements:
if (TYPE VALUE COMPARISON_OPERATOR TYPE VALUE) (ex. "var x > int 3")
    OPERATION1
    OPERATION2
endif
```