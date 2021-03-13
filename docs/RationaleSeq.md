# Seq(uential) Dialect Rationale

This document describes various design points of the `seq` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Digital logic is generally split into two orthogonal categories:
combinational and sequential. CIRCT contains a `comb` dialect to model the
basic combinational operations and the (future) `seq` dialect which is
discussed here. The intention of the `seq` dialect is to provide a set of
stateful constructs which can be used to model sequential logic, independent
of the output method (e.g. SystemVerilog).

We have yet to flesh this dialect out to represent all common stateful
components. We start with a register and will build from there.

## Definitions

For the sake of precision, we use the following definitions:

- Physical devices:
  - **Unclocked Latch:** A memory element which is only sensitive to the
  levels of its inputs. Has no clock. Example: SR Latch.
  - Clocked (gated) **latch:** A latch wherein the inputs are gated by a
  clock. Transparent the entire time the clock is high. Generally referred to
  as a "latch". Examples: "gated SR latch", "D latch".
  - Edge-triggered **flip-flop:** An edge-sensitive memory element. Captures
  the input value on one or both clock edges. Variants: posedge FF, negedge
  FF, "edge-sensitive" FF (captures the input value on both edges),
  resettable FF.
- Abstract models:
  - **Register:** A synchronous, resettable memory element. Can be
  implemented using any of the above "circuit level" elements.

## The register operation

The `seq.reg` op models an abstract notion of a "register", independent of
its implementation (e.g. latch, D flip-flop). It is intended to be 'lowered'
to a specific implementation, which may be an `sv.always_ff` block, a device
primitive instantiation, or even a feedback loop in all `comb` logic. Our
intention is to allow analysis and optimization of sequential logic without
having to reason about timing-dependent behavior. This makes it somewhat
distinct from the SystemVerilog register wherein one has to worry about reset
behavior and implementation (latch vs. flip-flop).

`Reg` has four operands:

- **input**: The value to be captured at 'clock'. Generally called 'q'.
Accepts any type, results in the same type.
- **clock**: Capture 'value' on the positive edge of this signal.
- **reset**: Signal to set the state to 'resetValue'. Optional.
- **resetValue**: A value which the state is set to upon reset. Required iff
'reset' is present.

```mlir
%d = seq.reg %input, %clk [, %reset [, %resetValue] ] : $type(input)
```

### Rationale

Several design decisions were made in defining this op. Mostly, they were
made to simplify it while still providing the common case. Providing support
for all flavors of registers is an anti-goal of this op. If an omitted
feature is needed, it can be added or (if not common or precludes
optimization) another op could be added.

- Logical features:
  - Inclusion of optional 'reset' signal: This operand makes lowering to an
  efficient implementation of reset easier. Omission of it would require some
  (potentially complex) analysis to find the reset mux if required (which it
  usually is to ensure efficient synthesis).
  - Inclusion of 'resetValue': if we have a 'reset' signal, we need to
  include a value.
  - Omission of 'enable': enables are easily modeled via a multiplexer on the
  input with one of the mux inputs as the output of the register. This
  property makes 'enables' easier to detect than reset in the common case.

- Timing / clocking:
  - Omission of 'negedge' event on 'clock': this is easily modeled by
  inverting the clock value.
  - Omission of 'dual edge' event on 'clock': this is not expected to be
  terribly common.
  - Omission of edge conditions on 'reset': The reset style shouldn't affect
  logical correctness. It should, therefore, be determined by a lowering
  pass.

### Future considerations

- Enable signal: if this proves difficult to detect (or non-performant if we
do not detect and generate the SystemVerilog correctly), we can build it into
the reg op.
- Reset style and clock style: how should we model posedge vs negedge clocks?
Async vs sync resets? There are some reasonble options here: attributes on
this op or `clock` and `reset` types which are parameterized with that
information.
- Initial value: this register initializes to `x` (in SystemVerilog
terminology). We will add an `initialValue` attribute if this proves
insufficient.
