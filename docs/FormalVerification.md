# Formal Verification Tooling

Formally verifying hardware designs is a crucial step during the development
process. Various techniques exist, such as logical equivalence checking, model
checking, symbolic execution, etc. The preferred technique depends on the level
of abstraction, the kind of properties to be verified, runtime limitations, etc.
As a hardware compiler collection, CIRCT provides infrastructure to implement
formal verification tooling and already comes with a few tools for common
use-cases. This document provides an introduction to those tools and gives and
overview over the underlying infrastructure for compiler engineers who want to
use CIRCT to implement their custom verification tool.

[TOC]

## Logical Equivalence Checking

The `circt-lec` tool takes one or two MLIR input files with operations of the
HW, Comb, and Seq dialects and the names of two modules to check for
equivalence. To get to know the exact CLI command type `circt-lec --help`.

For example, the below IR contains two modules `@mulByTwo` and `@add`. Calling
`circt-lec input.mlir -c1=mulByTwo -c2=add` will output `c1 == c2` since they
are semantically equivalent.

```mlir
// input.mlir
hw.module @mulByTwo(in %in: i32, out out: i32) {
  dbg.variable "in", %in : i32
  %two = hw.constant 2 : i32
  dbg.variable "two", %two : i32
  %res = comb.mul %in, %two : i32
  hw.output %res : i32
}
hw.module @add(in %in: i32, out out: i32) {
  dbg.variable "in", %in : i32
  %res = comb.add %in, %in : i32
  hw.output %res : i32
}
```

In the above example, the MLIR file is compiled to LLVM IR which is then passed
to the LLVM JIT compiler to directly output the LEC result. Alternatively, there
are command line flags to print the LEC problem in SMT-LIB, LLVM IR, or an
object file that can be linked against the Z3 SMT solver to produce a standalone
binary.

## Bounded Model Checking

TODO

## Infrastructure Overview

This section provides and overview over the relevant dialects and passes for
building formal verification tools based on CIRCT. The
[below diagram](/includes/img/smt_based_formal_verification_infra.svg) provides
a visual overview.

<p align="center"><img src="/includes/img/smt_based_formal_verification_infra.svg"/></p>
