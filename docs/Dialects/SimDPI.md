# Simulation Dialect DPI Proposal

This document proposes a full-semantics-preserving representation for
SystemVerilog DPI-C declarations and calls in the `sim` and `moore` dialects.

[TOC]

## Motivation

The current `sim.func.dpi` and `sim.func.dpi.call` representation is useful,
but it does not preserve the full distinction between:

1. The explicit function return value.
2. `output` arguments.
3. `inout` arguments.

Today, `sim.func.dpi.call` uses SSA results for DPI outputs, which is a good
high-level abstraction, but `sim.func.dpi` has no way to distinguish a true
function return from an `output` argument. In addition, `inout` ports are
structurally representable in the declaration, but are not supported by the
current lowering path.

This proposal preserves full source semantics while keeping the call surface
amenable to analysis and transformation.

## Goals

1. Preserve the declared distinction between `input`, `output`, and `inout`.
2. Preserve the explicit DPI function return value separately from output
   arguments.
3. Keep call semantics SSA-friendly at the `moore` and `sim` levels.
4. Lower faithfully to the C ABI used by DPI-C.
5. Keep any `llhd.ref`-based optimization as a later lowering choice rather
   than baking it into the high-level semantics.

## Non-Goals

1. This proposal does not attempt to optimize `inout` and `output` handling by
   prematurely introducing `llhd.ref` operands in the canonical `sim` or
   `moore` representation.
2. This proposal does not change the meaning of existing non-DPI `func.func`
   declarations or calls.

## Proposed Declaration Semantics

Both `moore.func.dpi` and `sim.func.dpi` remain module-like declarations backed
by `hw.module_type`.

Directions remain semantically real:

1. `in` means an input-only DPI argument.
2. `out` means an output-only DPI argument.
3. `inout` means a read-write DPI argument.

In addition, exactly zero or one `out` port may be marked as the explicit
function return value.

### Explicit Return Marker

The declaration uses a per-port attribute on one output port to indicate that
this port is the explicit function return. This follows the same conceptual
model already used by `sv.func`.

Constraints:

1. At most one port may be marked explicitly returned.
2. The marked port must have `out` direction.
3. The marked port should be the last output port in declaration order.

If no port is marked explicitly returned, the declaration models a `void`
foreign function.

## Proposed Call Semantics

The operand and result mapping is defined from the declaration, not from the C
ABI directly.

### Operands

Call operands are formed from declaration ports as follows:

1. `in` ports contribute one operand.
2. `inout` ports contribute one operand carrying the incoming value.
3. `out` ports do not contribute an operand.

### Results

Call results are formed from declaration ports as follows:

1. Ordinary `out` ports contribute one result.
2. `inout` ports contribute one result carrying the updated value.
3. The explicitly returned port contributes one result.

Results appear in declaration order among ports that produce values.

## Examples

### Void Function with Outputs

SystemVerilog:

```systemverilog
import "DPI-C" function void dpi(input int a, output int b, output int c);
```

Declaration:

```mlir
sim.func.dpi @dpi(in %a : i32, out %b : i32, out %c : i32)
```

Call:

```mlir
%b, %c = sim.func.dpi.call @dpi(%a) : (i32) -> (i32, i32)
```

### Function Return Plus Output

SystemVerilog:

```systemverilog
import "DPI-C" function int func(input int size, output int new_size);
```

Declaration:

```mlir
sim.func.dpi @func(
  in %size : i32,
  out %new_size : i32,
   out %ret : i32 {sim.func.explicitly_returned}
)
```

Call:

```mlir
%new_size, %ret = sim.func.dpi.call @func(%size) : (i32) -> (i32, i32)
```

### Function Return Plus InOut

SystemVerilog:

```systemverilog
import "DPI-C" function int func(input int size, inout int state);
```

Declaration:

```mlir
sim.func.dpi @func(
  in %size : i32,
  inout %state : i32,
   out %ret : i32 {sim.func.explicitly_returned}
)
```

Call:

```mlir
%state_after, %ret = sim.func.dpi.call @func(%size, %state) : (i32, i32) -> (i32, i32)
```

### Return, Output, and InOut Together

SystemVerilog:

```systemverilog
import "DPI-C" function int func(input int size, output int new_size,
                                 inout int state);
```

Declaration:

```mlir
sim.func.dpi @func(
  in %size : i32,
  out %new_size : i32,
  inout %state : i32,
   out %ret : i32 {sim.func.explicitly_returned}
)
```

Call:

```mlir
%new_size, %state_after, %ret =
  sim.func.dpi.call @func(%size, %state) : (i32, i32) -> (i32, i32, i32)
```

## ImportVerilog Semantics

`ImportVerilog` should lower DPI declarations and calls according to the
declared source-level semantics.

### Declarations

For `moore.func.dpi`:

1. Source `input` becomes `in`.
2. Source `output` becomes `out`.
3. Source `inout` becomes `inout`.
4. The explicit function return is emitted as an `out` port with the explicit
   return marker.
5. Non-input DPI open-array ports remain handle-passed and therefore use
   reference types at the declaration boundary.

### Calls

For `moore.func.dpi.call`:

1. `input` actuals are lowered as ordinary operands.
2. `output` actuals are lowered as lvalues which receive call results.
3. `inout` actuals are lowered by reading the incoming value, passing it as an
   operand, and writing the updated result back to the same lvalue.
4. The explicit function return result becomes the value of the call
   expression.
5. DPI open-array `output` / `inout` / `ref` actuals remain by-reference
   operands instead of becoming SSA call results.

## Lowering to the DPI-C ABI

The foreign C ABI remains pointer-based for outputs and inouts.

For a declaration lowered by a wrapper pass such as `LowerDPIFunc`:

1. `in` ports are passed by value.
2. Ordinary `out` ports are materialized as temporaries, passed by pointer, and
   loaded after the call.
3. `inout` ports are materialized as temporaries, initialized from the incoming
   operand, passed by pointer, and loaded after the call.
4. The explicitly returned port becomes the actual C return type.

Example:

```systemverilog
import "DPI-C" function int func(input int size, output int new_size,
                                 inout int state);
```

Lowers to the foreign ABI shape:

```c
int func(int size, int *new_size, int *state);
```

while preserving SSA call semantics in `sim.func.dpi.call`.

## Verifier Rules

The verifier for `sim.func.dpi` and `moore.func.dpi` should enforce:

1. At most one explicitly returned port.
2. The explicitly returned port must be `out`.
3. The explicitly returned port must be the last output port.
4. The module signature must not contain unsupported port types for the chosen
   lowering path.
5. Non-input DPI open-array ports must remain handle-passed instead of being
   modeled as value results.

The verifier for `sim.func.dpi.call` and `moore.func.dpi.call` should enforce:

1. Operand count matches the number of `in` plus `inout` ports.
2. Result count matches the number of ordinary `out` plus `inout` plus explicit
   return ports.
3. Operand types match declaration-derived input types.
4. Result types match declaration-derived produced-value types.
5. Open-array boundary ports remain operand-only even when their source
   direction is `out` or `inout`.

## Rationale for Deferring `llhd.ref`

`llhd.ref` may still be useful as a lowering detail, especially for `inout`,
but it should not be the canonical high-level representation.

Reasons:

1. Pure `out` arguments do not need storage identity.
2. SSA results provide better analysis and canonicalization properties.
3. Keeping `sim` and `moore` semantics independent of LLHD preserves layering.

An implementation may still choose to lower `inout` to `llhd.ref` internally at
or below the Moore-to-core boundary once the full semantics have already been
modeled explicitly.

## Migration Plan

1. Add the explicit-return marker to `sim.func.dpi`.
2. Extend `sim.func.dpi.call` verification to derive operand/result structure
   from declaration directions.
3. Mirror the same semantics in `moore.func.dpi` and `moore.func.dpi.call`.
4. Update `ImportVerilog` to emit declaration and call semantics faithfully.
5. Update the lowering path to the foreign C ABI.
6. Add end-to-end tests for:
   1. return only
   2. output only
   3. return plus output
   4. inout only
   5. return plus inout
   6. return plus output plus inout

## Open Questions

1. Whether result ordering should remain declaration order or group explicit
   return last regardless of position.
2. Whether `inout` should eventually admit an alternative lowered form for
   performance-sensitive pipelines once the semantic model is established.