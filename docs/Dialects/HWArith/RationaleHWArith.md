# HW Arith Dialect Rationale

This document describes the various design points of the HWArith dialect, a
dialect that is used to represent bitwidth extending arithmetic. This follows in the
spirit of other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [HW Arith Dialect Rationale](#hw-arith-dialect-rationale)
  - [Introduction](#introduction)
    - [Q&A](#qa)
  - [Bit Width Rules](#bit-width-rules)
  - [Lowering](#lowering)

## Introduction

The `hwarith` dialect provides a collection of operations that define typical
integer arithmetic operations which are bitwidth extending. These semantics
are expressed through return type inference rules that, based on the types of an
operation and its input operands, infers the type of the result operand.
Importantly, for the initial implementation of this dialect, we do not allow
types with uninferred width in the IR. A user is expected to iteratively build
operations, expect operation return types, and use this to further construct IR.
This dialect is **not** intended to be a "core" RTL dialect which takes part in
RTL codegen, and instead mainly serves as a frontend target. The lifetime of the
operations should be considered short, and lowered to their `comb` counterparts
early in the compilation pipeline.

In general, the bit width inference rules are designed to model SystemVerilog
semantics. However, unlike SystemVerilog, the aim of this dialect is to provide
a strongly-typed hardware arithmetic library with the width inference rules for
every operator explained in a clear and understandable way to avoid ambiguity.

To serve this goal of being strongly typed, the operations of the dialect
explicitly do not accept signless types. As such, only `ui#`, `si#` types are
allowed. A casting operation is provided to serve signedness casting as well as
truncation and padding.

Below we try to capture some common questions on the capabilities of such a
dialect, and how we've addressed/think about the issue.

### Q&A
* **Q:** Does this support n-ary operations?
    * n-ary operations in this dialect would have to solve the issue of how
    width rules in expressions such as the following are handled: `%0 =
    hwarith.add %0, %1, %2 : si6, si5, si4`  
    In the short term, we envision this dialect to be considering strictly
    binary operations. Defining width inference rules for these is simpler and
    fits the immediate usecase. If, in the future, a need for n-ary operations
    comes up and is motivated clearly (e.g. for optimization purposes), the
    dialect should be adapted to support it. In this case, we expect n-ary
    operation rules to be a superset of those defined for binary operations.
* **Q:** Does this support 0-width operands?
    * 0-width values might arise from arithmetic rules which reduce the bit
      width of an expression wrt. the operands to that expression. One case
      where such rules _may_ apply is in the implementation of modulo
      operations.  
    We refrain from adding support at this point in time, since support for
    0-width values in the remainder of the RTL dialects is, at time of writing,
    lacking/undefined. Once support is added in these downstream dialects,
    0-width support in `hwarith` should be reconsidered.
* **Q:** Does this support width inferred types?
    * Relying on width inference is relevant when referencing results of other
      width-inferred value. Without this, a user/frontend must itself know and
      apply width inference rules while generating the IR (either explicitly, or
      implicitly, through op builders). Having width-inferred types will be
      convenient not only for generating IR, but also to leave room for width
      inference rules and optimizations to apply recursively. As an example:
    ```mlir
    %1 = hwarith.add %a, %b : (ui4, ui4) -> (un) // %1 is an unsigned integer of inferred width
    %2 = hwarith.add %c, %1 : (ui3, ui) -> (un)
    ```
    We see merit in having inferred width types, but recognize that the work
    required to get this right is non-trivial and will require significant
    effort. Once need arises, and resources are available, we find width
    inferred types to be a valuable contribution to the dialect, but it will not
    be part of the initial implementation.
* **Q:** Does this support user-provided/alternative rules?
    * Initially, we want to settle on a set of common-case rules which describe
      the semantics required by the immediate users of the dialect, so this will
      not be an immediate goal of the dialect.
* **Q:** What about fixed point operations?
    * Fixed point operations and values may eventually be relevant in an
      `hwarith` context. However, the immediate needs of this dialect is to
      provide bitwidth-extending arithmetic for integer values.
* **Q:** What about floating point operations?
    * As above, with the additional consideration that handling floating point
      operations in hardware usually involves inferring external IP. Considering
      how float operations are added to `hwarith` should only be considered
      after the problem of operator libraries have been solved for CIRCT.

## Bit Width Rules

The core capability of the dialect is the bit width inference rules implemented
for each operation. These define the semantics for how arithmetic overflow is to
be handled both with respect to the width and signedness of operation results,
as well as how this is communicated to a `comb` lowering.

### Preliminaries

The MLIR `ui<w>` types represents values in the interval *[0, 2^w - 1]*. Its
signed counterpart, `si<w>`, represents values in the interval *[-2^(w-1),
2^(w-1) - 1]*. For example, `ui4` covers values between *0* and *15*, whereas
`si4` ranges from *-8* to *7*. For a given bit-width, the unsigned and the
signed type contain the same number of values, but the represented interval is
shifted by half the interval's size.

```
         │                  
         ├──────────────┐   
         │     ui4      │   
 ┌───────┼──────┬───────┘   
 │     si4      │           
 └───────┼──────┘           
─────────┴─────────────────▶
-8       0      7      15   
```

We use the notation `UI_MIN<w>`, `UI_MAX<w>`, `SI_MIN<w>` and `SI_MAX<w>` to
denote the smallest and largest values representable by an unsigned/signed type
of the given bit-width. For example, `UI_MAX<4>` = *15*, and `SI_MIN<4>` = *-8*.

The following relations for all `w > 0` are handy:
```
UI_MIN<w>     = 0
2 * UI_MAX<w> = 2^(w+1) - 2
UI_MAX<w+1>   = 2 * UI_MAX<w> + 1 

SI_MIN<w>     < 0
2 * SI_MIN<w> = -2^w
SI_MIN<w+1>   = 2 * SI_MIN<w>
2 * SI_MAX<w> = 2^w - 2
SI_MAX<w+1>   = 2 * SI_MAX<w> + 1
```

The following result type inference rules are chosen to prevent a loss of
precision and sign information associated with over- and underflows. The
reasoning usually takes the smallest and largest possible result values into
account. For example, if the result can be negative, then the result type must
be signed.

### Addition

#### Overview

|   | LHS type | RHS type | Result type                                |
| - | :------- | :------- | :----------------------------------------- |
|(U)| `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = max(*a*, *b*) + 1           |
|(S)| `si<a>`  | `si<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1           |
|(M)| `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b*    |
|   |          |          | `si<r>`, *r* = *b* + 1 **if** *a* < *b*    |
|(M)| `si<a>`  | `ui<b>`  | Same as `ui<b> + si<a>`                    |

#### Unsigned addition (U)

Result value range: `[0 + 0, UI_MAX<a> + UI_MAX<b>]`

W.l.o.g., let `ui<b>` be the wider type. Then the following inequalities hold:
```
UI_MAX<b> < UI_MAX<a> + UI_MAX<b> ≤ 2 * UI_MAX<b> ≤ UI_MAX<b+1>
```

Example:
```mlir
%0 = hwarith.add %10, %11 : (ui3, ui4) -> ui5
```

#### Signed addition (S)

Result value range: `[SI_MIN<a> + SI_MIN<b>, SI_MAX<a> + SI_MAX<b>]`

We apply the same reasoning as in (U). W.l.o.g., let `si<b>` be the wider type.
Then the following inequalities hold:
```
SI_MIN<b> > SI_MIN<a> + SI_MIN<b> ≥ 2 * SI_MIN<b> ≥ SI_MIN<b+1>
SI_MAX<b> < SI_MAX<a> + SI_MAX<b> ≤ 2 * SI_MAX<b> ≤ SI_MAX<b+1>
```

Example:
```mlir
%1 = hwarith.add %12, %13 : (si3, si3) -> si4
```

#### Mixed addition (M)

The result type must be signed, as the addition of a positive and a negative
value may yield a negative result. Due to commutativity, we only consider
`ui<a> + si<b>`. We distinguish two situations:

* `a ≥ b`, i.e. the unsigned operand's width is greater or equal the signed
operand's width.

    The smallest signed type to represent all values of `ui<a>` is `si<a+1>`.
    We know that `a + 1 > b`, and derive from (S) that the result type must be
    `si<(a+1) + 1>` = `si<a+2>`.

* `a < b`, i.e. the unsigned operand's width is stricly less than the signed
operand's width.

    As `a + 1 ≤ b`, the result via (S) is `si<b+1>`.

Examples:
```mlir
%2 = hwarith.add %14, %15 : (ui3, si4) -> si5
%3 = hwarith.add %16, %17 : (si4, ui6) -> si8
```

### Subtraction

#### Overview

|   | LHS type | RHS type | Result type                              |
| - | :------- | :------- | :--------------------------------------- |
|(U)| `ui<a>`  | `ui<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1         |
|(S)| `si<a>`  | `si<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1         |
|(M)| `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b*  |
|   |          |          | `si<r>`, *r* = *b* + 1 **if** *a* < *b*  |
|(M)| `si<a>`  | `ui<b>`  | Same as `ui<b> - si<a>`                  |

#### Unsigned subtraction (U)

Result value range: `[0 - UI_MAX<b>, UI_MAX<a> - 0]`

As the result can be negative, the result type must be signed. `-UI_MAX<b>` is
covered by `si<b+1>`, and `UI_MAX<a>` is covered by `si<a+1>`, so the narrowest
type capable of representing both extremal values is `si<r>` with
*r* = max(*a*, *b*) + 1.

Example:
```mlir
%0 = hwarith.sub %10, %11 : (ui3, ui4) -> si5
```

#### Signed subtraction (S)

Result value range: `[SI_MIN<a> - SI_MAX<b>, SI_MAX<a> - SI_MIN<b>]`

We rewrite the value range and then appy the same reasoning as for the
[signed addition](#signed-addition-S):
```
  [SI_MIN<a> - SI_MAX<b>, SI_MAX<a> - SI_MIN<b>]
= [SI_MIN<a> + -2^(b-1) + 1, SI_MAX<a> + 2^(b-1)]
= [SI_MIN<a> + SI_MIN<b> + 1, SI_MAX<a> + SI_MAX<b> + 1]
```

W.l.o.g., let `si<b>` be the wider type. Then the following inequalities hold:
```
SI_MIN<b> > SI_MIN<a> + SI_MIN<b> + 1 ≥ 2 * SI_MIN<b> + 1 ≥ SI_MIN<b+1>
SI_MAX<b> < SI_MAX<a> + SI_MAX<b> + 1 ≤ 2 * SI_MAX<b> + 1 ≤ SI_MAX<b+1>
```

Example:
```mlir
%1 = hwarith.sub %12, %13 : (si3, si3) -> si4
```

### Mixed subtraction (M)

See [mixed addition](#mixed-addition-M).

Examples:
```mlir
%2 = hwarith.sub %14, %15 : (ui3, si4) -> si5
%3 = hwarith.sub %16, %17 : (si4, ui6) -> si8
```

### Multiplication

#### Overview

|   | LHS type | RHS type | Result type                              |
| - | :------- | :------- | :--------------------------------------- |
|(U)| `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = *a* + *b*                 |
|(S)| `si<a>`  | `si<b>`  | `si<r>`, *r* = *a* + *b*                 |
|(M)| `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + *b*                 |
|(M)| `si<a>`  | `ui<b>`  | `si<r>`, *r* = *a* + *b*                 |

#### Unsigned multiplication (U)

Result value range: `[0 * 0, UI_MAX<a> * UI_MAX<b>]`

The result type must be at least *a* + *b bits wide:
```
UI_MAX<a> * UI_MAX<b> = (2^a - 1) * (2^b - 1)
                      = 2^(a + b) - 2^a - 2^b + 1 
                      ≤ 2^(a + b) - 1 = UI_MAX<a+b>
```

The result type cannot be smaller than `ui<a+b>`, as the following
counter-example shows:
```
UI_MAX<3> * UI_MAX<4> = 7 * 15 = 105 > UI_MAX<3+4-1>
```

Example:
```mlir
%0 = hwarith.mul %10, %11 : (ui3, ui4) -> ui7
```

#### Signed multiplication (S)

Result value range:
`[min(SI_MIN<a> * SI_MAX<b>, SI_MAX<a> * SI_MIN<b>), SI_MAX<a> * SI_MAX<b>]`

With the same reasoning as in (U), we know that `si<a+b>` is the narrowest type
to accommodate `SI_MAX<a> * SI_MAX<b>`. This type also covers the negative
extremal value (w.l.o.g. let `si<b>` be the wider type here):
```
SI_MIN<a> * SI_MAX<b> = -2^(a-1) * (2^(b-1) - 1)
                      = -2^(a + b - 2) + 2^(a-1)
                      ≥ -2^(a + b - 1) = SI_MIN<a+b>
```

Example:
```mlir
%1 = hwarith.mul %12, %13 : (si3, si3) -> si6
```

#### Mixed multiplication (M)

The result type must be signed. Let us consider `ui<a> * si<b>`; the flipped
case `si<a> * ui<b>` follows analogously.

Result value range: `[UI_MAX<a> * SI_MIN<b>, UI_MAX<a> * SI_MAX<b>]`

Analogously to the previous cases, the result type must be at least *a* + *b*
bits wide:
```
UI_MAX<a> * SI_MIN<b> = (2^a - 1) * -2^(b-1)
                      = -2^(a + b - 1) + 2^(b-1)
                      ≥ 2^(a + b - 1) = SI_MIN<a+b>

UI_MAX<a> * SI_MAX<b> = (2^a - 1) * (2^(b-1) - 1)
                      = 2^(a + b - 1) - 2^a - 2^(b-1) + 1
                      ≤ 2^(a + b - 1) - 1 = SI_MAX<a+b>
```

Again, we cannot choose a narrower type according to the following
counter-example:
```
UI_MAX<3> * SI_MAX<4> = 7 * 7 = 49 ≥ SI_MAX<3+4-1>
```

Example:
```mlir
%2 = hwarith.mul %14, %15 : (si3, ui5) -> si8
```

### Division

#### Overview

|    | LHS type | RHS type | Result type                              |
| -- | :------- | :------- | :--------------------------------------- |
|(U) | `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = *a*                       |
|(S) | `si<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 1                   |
|(M1)| `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 1                   |
|(M2)| `si<a>`  | `ui<b>`  | `si<r>`, *r* = *a*                       |

#### Unsigned division (U)

Result value range: `[0 / 1, UI_MAX<a> / 1]`

The result is also less or equal the dividend, so the result type is `ui<a>`.

Example:
```mlir
%0 = hwarith.div %10, %11 : (ui3, ui4) -> ui3
```

#### Signed division (S)

Result value range: `[SI_MIN<a> / 1, SI_MIN<a> / -1]`

`- SI_MIN<a>` is not convered by `si<a>`, so we have to widen the result type to
`si<a+1>`.

Example:
```mlir
%1 = hwarith.div %12, %13 : (si3, si3) -> si4
```

#### Division with signed divisor (M1)

If at least one operand is signed, then the result type must be signed as well.

Result value range: `[UI_MAX<a> / -1, UI_MAX<a> / 1]`

The result type must be `si<a+1>` in order to accommodate `UI_MAX<a>`.

Example:
```mlir
%2 = hwarith.div %14, %15 : (ui3, si4) -> si4
```

#### Division with signed dividend (M2)

If at least one operand is signed, then the result type must be signed as well.

Result value range: `[SI_MIN<a> / 1, SI_MAX<a> / 1]`]

The type of the dividend, `si<a>`, already covers the entire result value range.

Example:
```mlir
%3 = hwarith.div %16, %17 : (si4, ui6) -> si4
```

### Additional operators
**TODO**: elaborate once operators are provided.
* `hwarith.mod %0, %1 : (#l0, #l1) -> (#l2)`:
* `hwarith.cast %0 : (#l0) -> (#l2)`: 


## Lowering

The general lowering style of the dialect operations consists of padding the
operands of an operation based on an operation rule, and subsequently feeding
these operands to the corresponding `comb` operator.

For instance, given `%res = hwarith.add %lhs, %rhs` and a rule stating `w(res)`
= `max(w(lhs), w(rhs)) + 1` the lowering would be:

```mlir
%res = hwarith.add %lhs, %rhs : (ui3, ui4) -> (ui5)

// lowers to
%z2 = hw.constant 0 : i2
%z1 = hw.constant 0 : i1
%lhs_padded = comb.concat %z2, %lhs : i5
%rhs_padded = comb.concat %z1, %rhs : i5
%res = comb.add %lhs_padded, %rhs_padded : i5
```
