# 'moore' Dialect

This dialect provides operations and types to capture a SystemVerilog design after parsing, type checking, and elaboration.

[TOC]


## Rationale

The main goal of the `moore` dialect is to provide a set of operations and types for the `ImportVerilog` conversion to translate a fully parsed, type-checked, and elaborated Slang AST into MLIR operations. See IEEE 1800-2017 for more details about SystemVerilog. The dialect aims to faithfully capture the full SystemVerilog types and semantics, and provide a platform for transformation passes to resolve language quirks, analyze the design at a high level, and lower it to the core dialects.

In contrast, the `sv` dialect is geared towards emission of SystemVerilog text, and is focused on providing a good lowering target to allow for emission. The `moore` and `sv` dialect may eventually converge into a single dialect. As we are building out the Verilog frontend capabilities of CIRCT it is valuable to have a separate ingestion dialect, such that we do not have to make disruptive changes to the load-bearing `sv` dialect used in production.


## Types

[include "Dialects/MooreTypes.md"]


## Operations

[include "Dialects/MooreOps.md"]
