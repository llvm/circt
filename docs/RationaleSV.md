# SV Dialect Rationale

This document describes various design points of the `sv` dialect, a common
dialect that is typically used in conjunction with the `hw` and `comb` dialects.
Please see the [RationaleHW.md](HW Dialect Rationale) for high level insight
on how these work together.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [SV Dialect Rationale](#sv-dialect-rationale)
  - [Introduction to the `sv` dialect](#introduction-to-the-sv-dialect)
  - [`sv` Type System](#sv-type-system)
  - [Overview of `sv` dialect operations](#overview-of-sv-dialect-operations)
  - [Statements](#statements)
  - [Declarations](#declarations)
  - [Expressions](#expressions)
    - [Verbatim op](#verbatim-op)
  - [Cost Model](#cost-model)

## Introduction to the `sv` dialect

The `sv` dialect is one of the dialects that can be mixed into the HW dialect,
providing access to a range of syntactic and behavioral constructs in
SystemVerilog.  The driving focus of this dialect is to provide simple and
predictable access to these features: it is not focused primarily on being easy
to analyze and transform.

The `sv` dialect is designed to build on top of the `hw` dialect and is often
used in conjunction with the `comb` or other dialects, so it does not have its
own operations for combinational logic, modules, or other common functionality.

## `sv` Type System

Like the HW dialect, the SV dialect is designed to tolerate unknown types where
possible, allowing other dialects to mix in with it.  In addition to these
external types, and the types used by the HW dialect, the SV dialect defines
types for SystemVerilog interfaces.

TODO: Describe interface types, modports, etc.

## Overview of `sv` dialect operations

Because the SV dialect aims to align with the textual nature of SystemVerilog,
many of the constructs in the SV dialect have an "AST" style of representation.
The major classes of operations you'll find are:

1) Statements like `sv.if`, `sv.ifdef`, `sv.always` and `sv.initial` that
   expose primary task-like operations and the behavioral model.
1) Procedural assignment operators, including the `sv.bpassign` and `sv.passign`
   operators that expose the blocking (`x = y`) and non-blocking (`x <= y`)
   procedural operators.
1) Directives like `sv.finish` and `sv.alias` and behavioral functions like
   `sv.fwrite`.
1) Access to verification constructs with `sv.assert`, `sv.assume`, and
   `sv.cover`.
1) Escape hatches that allow direct integration of textual expressions
   (`sv.verbatim.expr`) and full statements (`sv.verbatim`).

These operations are designed to directly model the syntax of the SystemVerilog
language and to be easily printable by the ExportVerilog pass.  While there are
still many things in SystemVerilog that we cannot currently express in the SV
dialect, this design makes it easy to incrementally build out new capabilities
over time.

## Statements

TODO.

## Declarations

TODO: Describe `sv.wire`, `sv.reg`, 

## Expressions

TODO: Describe `sv.read_inout` and `sv.array_index_inout`.

### Verbatim op

The verbatim operation produces a typed value expressed by a string of
SystemVerilog.  This can be used to access macros and other values that are
only sensible as Verilog text. There are three kinds of verbatim operations:

 1. VerbatimOp(`sv.verbatim`, the statement form
 2. VerbatimExprOp(`sv.verbatim.expr`), the expression form.
 3. VerbatimExprSEOp(`sv.verbatim.expr.se`), the effectful expression form.

For the verbatim expression form, the text string is assumed to have the
highest precedence - include parentheses in the text if it isn't a single token.
`sv.verbatim.expr` is assumed to not have side effects (is `NoSideEffect` in
MLIR terminology), whereas `sv.verbatim.expr.se` may have side effects.

Verbatim allows operand substitutions with '{{0}}' syntax.
For macro substitution, optional operands and symbols can be added after the 
string. Verbatim ops may also include an array of symbol references.
The indexing begins at 0, and if the index is greater than the
number of operands, then it is used to index into the symbols array.
It is invalid to have macro indices greater than the total number 
of operands and symbols.
Example, 

```
sv.verbatim "MACRO({{0}}, {{1}} reg={{4}}, {{3}})" 
            (%add, %xor) : i8, i8
            {symRefs = [@reg1, @Module1, @instance1]}
```

## Cost Model

The SV dialect is primarily designed for human consumption, not machines.  As
such, transformations should aim to reduce redundancy, eliminate useless
constructs (e.g. eliminate empty ifdef and if blocks), etc.
