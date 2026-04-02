# Simulation Dialect DPI

This document describes the full-semantics-preserving representation for
SystemVerilog DPI-C declarations and calls in the `sim` dialect.

[TOC]

## Motivation

DPI-C functions carry richer semantics than a simple input/output split.
The `sim.func.dpi` declaration needs to faithfully represent:

1. The explicit function return value (the C return, not an output argument).
2. `out` arguments (passed by pointer in the C ABI).
3. `inout` arguments (read-write, passed by pointer in the C ABI).
4. `ref` arguments (open-array / by-reference handles).

These distinctions affect both the call-site shape in MLIR and the lowering to
the foreign C ABI.

## Design

### DPIDirection Enum

Port semantics are encoded as a first-class `DPIDirection` enum:

| Direction | Call operand? | Call result? | C ABI        |
|-----------|--------------|-------------|--------------|
| `in`      | yes          | no          | by value     |
| `out`     | no           | yes         | by pointer   |
| `inout`   | yes          | yes         | by pointer   |
| `return`  | no           | yes (last)  | C return     |
| `ref`     | yes          | no          | pass-through |

### DPIModuleType

All port information is bundled into a single MLIR type,
`!sim.dpi_modty<...>`, which stores an array of `DPIPort` structs (name,
type, direction). This type:

- Is uniqued by the MLIR context (pointer equality for dedup).
- Derives `FunctionType` on demand for `FunctionOpInterface`.
- Constructs `hw::ModuleType` on demand for SimToSV lowering.

Generic form:

```mlir
!sim.dpi_modty<in "a" : i32, out "b" : i32, return "ret" : i8>
```

### Declaration Syntax

```mlir
sim.func.dpi @func(in %a : i32, out b : i32, return ret : i8)
```

- `in`, `inout`, and `ref` ports use SSA-style names (`%name`) because
  they are call operands.
- `out` and `return` ports use bare names because they are not call
  operands.

### Call Semantics

Operands and results are derived from the declaration's port directions:

- **Operands**: `in`, `inout`, and `ref` ports (in declaration order).
- **Results**: `out`, `inout`, and `return` ports (in declaration order).

```mlir
%b, %ret = sim.func.dpi.call @func(%a) : (i32) -> (i32, i8)
```

## Examples

### Void Function with Outputs

```systemverilog
import "DPI-C" function void dpi(input int a, output int b, output int c);
```

```mlir
sim.func.dpi @dpi(in %a : i32, out b : i32, out c : i32)
%b, %c = sim.func.dpi.call @dpi(%a) : (i32) -> (i32, i32)
```

### Function Return Plus Output

```systemverilog
import "DPI-C" function int func(input int size, output int new_size);
```

```mlir
sim.func.dpi @func(in %size : i32, out new_size : i32, return ret : i32)
%new_size, %ret = sim.func.dpi.call @func(%size) : (i32) -> (i32, i32)
```

### Function Return Plus InOut

```systemverilog
import "DPI-C" function int func(input int size, inout int state);
```

```mlir
sim.func.dpi @func(in %size : i32, inout %state : i32, return ret : i32)
%state_after, %ret = sim.func.dpi.call @func(%size, %state) : (i32, i32) -> (i32, i32)
```

### Return, Output, and InOut Together

```systemverilog
import "DPI-C" function int func(input int size, output int new_size,
                                 inout int state);
```

```mlir
sim.func.dpi @func(
  in %size : i32, out new_size : i32,
  inout %state : i32, return ret : i32)
%new_size, %state_after, %ret =
  sim.func.dpi.call @func(%size, %state) : (i32, i32) -> (i32, i32, i32)
```

### By-Reference Open-Array Port

```systemverilog
import "DPI-C" function void write_mem(input int addr, ref int data[]);
```

```mlir
sim.func.dpi @write_mem(in %addr : i32, ref %data : !llvm.ptr)
sim.func.dpi.call @write_mem(%addr, %data) : (i32, !llvm.ptr) -> ()
```

## Lowering to the DPI-C ABI

`LowerDPIFunc` generates a `func.func` wrapper that translates between SSA
call semantics and the pointer-based C ABI:

| Direction | Wrapper behavior                                       |
|-----------|--------------------------------------------------------|
| `input`   | Passed by value                                        |
| `output`  | `alloca` + pass pointer + `load` after call            |
| `inout`   | `alloca` + `store` incoming + pass pointer + `load`    |
| `return`  | Becomes the C return value                             |
| `ref`     | Pointer passed through directly                        |

## SimToSV Lowering

`SimToSV` constructs an `hw::ModuleType` from `DPIModuleType` at the
conversion boundary using `getHWModuleType()`. The `return` direction maps
to an `sv.func` output port annotated with `sv.func.explicitly_returned`.

## Verifier Rules

1. At most one `return` port, which must be the last port.
2. `ref` ports must use `!llvm.ptr` type.
3. `function_type` must match the type derived from `module_type`.

## ImportVerilog Semantics

`ImportVerilog` maps DPI declarations into `moore.func.dpi` using the
source-level semantics. `MooreToCore` then converts to `sim.func.dpi` by
mapping port directions to `DPIDirection` values. Non-input DPI open-array
ports are mapped to `ref` direction with `!llvm.ptr` type.

## Open Questions

1. Whether result ordering should remain declaration order or group explicit
   return last regardless of position.
2. Whether `inout` should eventually admit an alternative lowered form for
   performance-sensitive pipelines.