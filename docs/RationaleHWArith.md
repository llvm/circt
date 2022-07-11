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

**TODO**: elaborate once operators are provided.
* `hwarith.add %0, %1 : (#l0, #l1) -> (#l2)`:
* `hwarith.sub %0, %1 : (#l0, #l1) -> (#l2)`:
* `hwarith.mul %0, %1 : (#l0, #l1) -> (#l2)`:
* `hwarith.div %0, %1 : (#l0, #l1) -> (#l2)`:
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
