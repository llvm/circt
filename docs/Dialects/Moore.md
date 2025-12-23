# Moore Dialect

This dialect provides operations and types to capture a SystemVerilog design after parsing, type checking, and elaboration.

[TOC]


## Rationale

The main goal of the `moore` dialect is to provide a set of operations and types for the `ImportVerilog` conversion to translate a fully parsed, type-checked, and elaborated Slang AST into MLIR operations.
See IEEE 1800-2017 for more details about SystemVerilog.
The dialect aims to faithfully capture the full SystemVerilog types and semantics, and provide a platform for transformation passes to resolve language quirks, analyze the design at a high level, and lower it to the core dialects.

In contrast, the `sv` dialect is geared towards emission of SystemVerilog text, and is focused on providing a good lowering target to allow for emission.
The `moore` and `sv` dialect may eventually converge into a single dialect.
As we are building out the Verilog frontend capabilities of CIRCT it is valuable to have a separate ingestion dialect, such that we do not have to make disruptive changes to the load-bearing `sv` dialect used in production.


## Types

### Simple Bit Vector Type

The `moore.iN` and `moore.lN` types represent a two-valued or four-valued simple bit vector of width `N`.

| Verilog    | Moore Dialect |
| ---------- | ------------- |
| `bit`      | `!moore.i1`   |
| `logic`    | `!moore.l1`   |
| `reg`      | `!moore.l1`   |
| `byte`     | `!moore.i8`   |
| `shortint` | `!moore.i16`  |
| `int`      | `!moore.i32`  |
| `integer`  | `!moore.l32`  |
| `longint`  | `!moore.i64`  |
| `time`     | `!moore.l64`  |

### Default Values

Behavior of unconnected ports:

| Port Type        | Unconnected Behavior       |
| ---------------- | -------------------------- |
| Input (Net)      | High-impedance value ('Z)  |
| Input (Variable) | Default initial value      |
| Output           | No effect on Simulation    |
| Inout (Net)      | High-impedance value ('Z)  |
| Inout (Variable) | Default initial value      |
| Ref              | Cannot be left unconnected |
| Interconnect     | Cannot be left unconnected |
| Interface        | Cannot be left unconnected |

Uninitialized variables:

| Type                | Default initial value           |
| ------------------- | ------------------------------- |
| 4-state integral    | 'X                              |
| 2-state integral    | '0                              |
| `real`, `shortreal` | 0.0                             |
| Enumeration         | Base type default initial value |
| `string`            | "" (empty string)               |
| `event`             | New event                       |
| `class`             | `null`                          |
| `interface class`   | `null`                          |
| `chandle`           | `null`                          |
| `virtual interface` | `null`                          |

[include "Dialects/MooreTypes.md"]


## Operations

[include "Dialects/MooreOps.md"]
