# HW Arith Dialect Rationale

This document describes the various design points of the HWArith dialect, a dialect that is used to represent bitwidth aware arithmetic. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [HW Arith Dialect Rationale](#hw-arith-dialect-rationale)
  - [Introduction](#introduction)
    - [Q&A](#qa)
  - [Bit Width Rules](#bit-width-rules)
  - [Lowering](#lowering)

## Introduction

The `hwarith` dialect provides a collection of operations that define typical integer arithmetic operations which are bitwidth aware.
This bitwidth awareness is expressed through return type inference rules that, based on the types of an operation and its input operands, infers the type of the result operand.

In general, the bit width inference rules are designed to model SystemVerilog semantics. However, unlike SystemVerilog, the aim of this dialect is to provide a strongly-typed hardware arithmetic library with the width inference rules for every operator explained in a clear and understandable way to avoid ambiguity.

To serve this goal of being strongly typed, the operations of the dialect explicitly does not accept signless types. As such, only `ui#`, `si#` types are allowed. A casting operation is provided to serve signedness casting as well as truncation and padding.

Below we try to capture some common questions on the capabilities of such a dialect, and how we've addressed/think about the issue.

### Q&A
* **Q:** Does this support n-ary operations?
    * n-ary operations in this dialect would have to solve the issue of how width rules in expressions such as the following are handled:
    `%0 = hwarith.add %0, %1, %2 : si6, si5, si4`  
    In the short term, we envision this dialect to be considering strictly binary operations. Defining width inference rules for these is simpler and fits the immediate usecase. If, in the future, a need for n-ary operations comes up and is motivated clearly (e.g. for optimization purposes), the dialect should be adapted to support it. In this case, we expect n-ary operation rules to be a superset of those defined for binary operations.
* **Q:** Does this support 0-width operands?
    * 0-width values might arise from arithmetic rules which reduce the bit width of an expression wrt. the operands to that expression. One case where such rule _may_ apply is in the implementation of modulo operations.  
    We refrain from adding support at this point in time, since support for 0-width values in the remainder of the RTL dialects is, at time of writing, lacking/undefined. Once support is added in these downstream dialects, 0-width support in `hwarith` should be reconsidered.
* **Q:** Does this support width inferred types?
    * Relying on width inference is relevant when referencing results of other width-inferred value. Without this, a user/frontend must itself know and apply width inference rules before generating the IR. Having width-inferred types will be convenient not only for generating IR, but also to leave room for width inference rules and optimizations to apply recursively. As an example:
    ```mlir
    %1 = hwarith.add %a, %b : ui4, ui4
    %2 = hwarith.add %c, %1 : ui3, ui // %1 is an unsigned integer of inferred width
    ```
    @TODO: (When) do we want this?
* **Q:** Does this support user-provided/alternative rules?
    * Initially, we want to settle on a set of common-case rules which describe the semantics required by the immediate users of the dialect. However, seeing as bit width inference rules aren't directly attach to the operations, and with the capability of width-inferred types, this leaves space for interchangeable rules. We are open to discussing this in the future, if the need is there.
* **Q:** What about fixed point operations?
    * Fixed point operations and values may eventually be relevant in an `hwarith` context. However, the immediate needs of this dialect is to provide bitwidth-aware arithmetic for integer values.

## Bit Width Rules

The core capability of the dialect is the bit width inference rules implemented for each operation. These define the semantics for how arithmetic overflow is to be handled both with respect to the width and signedness of operation results, as well as how this is communicated to a `comb` lowering.

**TODO**: elaborate once operators are provided.
* `hwarith.add %0, %1 : #l0, #l1`:
* `hwarith.sub %0, %1 : #l0, #l1`:
* `hwarith.mul %0, %1 : #l0, #l1`:
* `hwarith.div %0, %1 : #l0, #l1`:
* `hwarith.mod %0, %1 : #l0, #l1`:
* `hwarith.cast %0    : #l0, #l1`: 


## Lowering

The general lowering style of the dialect operations consists of padding the operands of an operation based on an operation rule, and subsequently feeding these operands to the corresponding `comb` operator.

For instance, given `%res = hwarith.add %lhs, %rhs` and a rule stating `w(res)` = `max(w(lhs), w(rhs)) + 1` the lowering would be:

```mlir
%res = hwarith.add %lhs, %rhs : ui3, ui4

// lowers to
%z2 = hw.constant 0 : i2
%z1 = hw.constant 0 : i1
%lhs_padded = comb.concat %z2, %lhs : i5
%rhs_padded = comb.concat %z1, %rhs : i5
%res = comb.add %lhs_padded, %rhs_padded : i5
```
