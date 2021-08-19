# Seq(uential) Dialect Rationale

This document describes various design points of the `seq` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Digital logic is generally split into two categories: combinational and
sequential. CIRCT contains a `comb` dialect to model the basic combinational
operations and the (future) `seq` dialect which is discussed here. The
intention of the `seq` dialect is to provide a set of stateful constructs
which can be used to model sequential logic, independent of the output method
(e.g. SystemVerilog).

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

## The computational register operation

The `seq.compreg` op models an abstract notion of a "register", independent
of its implementation (e.g. latch, D flip-flop). This specific register op is
intended to support "computation support" or "reset-agnostic code" and thus
it cannot be used to model all the behaviors of a SystemVerilog register.
(E.g. FSM and pipeline registers.) It is intended to be 'lowered' to a
specific implementation, which may be an `sv.always_ff` block, a device
primitive instantiation, or even a feedback loop in all `comb` logic.

Our intention is to allow analysis and optimization of sequential logic
without having to reason about implementation-specific behavior. This makes
it somewhat distinct from the SystemVerilog register wherein one has to
account for details like reset behavior and implementation (latch vs.
flip-flop).

`CompReg` has four operands:

- **input**: The value to be captured at 'clock'. Generally called 'd'.
Accepts any type, results in the same type. Does not support any notion of
addressing, meaning that this operation sets / reads the entire value.
- **clock**: Capture 'value' on the positive edge of this signal.
- **reset**: Signal to set the state to 'resetValue'. Optional.
- **resetValue**: A value which the state is set to upon reset. Required iff
'reset' is present.

```mlir
%q = seq.compreg %input, %clk [, %reset, %resetValue ] : $type(input)
```

Upon initialization, the state is defined to be uninitialized.

### Rationale

Several design decisions were made in defining this op. Mostly, they were
made to simplify it while still providing the common case. Providing support
for all flavors of registers is an anti-goal of this op. If an omitted
feature is needed, it can be added or (if not common or precludes
optimization) another op could be added.

- Logical features:
  - Inclusion of optional 'reset' signal: This operand makes lowering to an
  efficient implementation of reset easier. Omission of it would require some
  (potentially complex) analysis to find the reset mux if required
  - Inclusion of 'resetValue': if we have a 'reset' signal, we need to
  include a value.
  - Omission of 'enable': enables are easily modeled via a multiplexer on the
  input with one of the mux inputs as the output of the register. This -- we
  assume -- property makes 'enables' easier to detect than reset in the
  common case.

- Timing / clocking:
  - Omission of 'negedge' event on 'clock': this is easily modeled by
  inverting the clock value.
  - Omission of 'dual edge' event on 'clock': this is not expected to be
  terribly common.
  - Omission of edge conditions on 'reset': Since this op specifically
  targets "reset-agnostic code", the reset style shouldn't affect logical
  correctness. It should, therefore, be determined by a lowering pass.

### Future considerations

- Enable signal: if this proves difficult to detect (or non-performant if we
do not detect and generate the SystemVerilog correctly), we can build it into
the compreg op.
- Reset style and clock style: how should we model posedge vs negedge clocks?
Async vs sync resets? There are some reasonble options here: attributes on
this op or `clock` and `reset` types which are parameterized with that
information.
- Initial value: this register is uninitialized. Using an uninitialized value
results in undefined behavior. We will add an `initialValue` attribute if
this proves insufficient.

## The Flip Flop Operation

`seq.ff` models a D-Flip Flop operation, with support for various features,
namely:

- clock edge sensitivity: posedge, negedge or edge-triggered
- explicit enable signals
- an optional synchronous & asynchronus reset

This is an abstract general-purpose operation that can be targetted by other
dialects. It should be further lowered down to either vendor-specific primtives
or to the SV dialect.

`seq.ff` has the following MLIR format:

```mlir
%q = seq.ff %input, (posedge|negedge) %clk , %enable, %syncReset, %asyncReset,
%syncResetValue, %asyncResetValue  : $type(input)
```

### Rationale

The ordering of operands are as such, as reset values need not be supplied if
`syncReset` or `asyncReset` is tied to `hw.constant 0`. the `seq.ff` operation
will be verified such that this operand holds.

The common case in this operation would be to set `enable` to `hw.constant 1`.
However, we do not make this operand optional despite the slight benefit of
making the IR smaller, as it will complicate parsing, and make analysis harder
as we would need to deal with an additional null case.

The reset values are not optional in this operation. When as they are not used,
they will be populated with dummy values (eg: zeros). Analysis can statically
determine that they are not used by checking if `%syncReset` is `hw.constant
0`. The presence of optional operands will somewhat complicate parsing. For
example, if `%syncResetValue` and `%asyncResetValue` are optional, when a FF
has both an asynchronous reset, but not a synchronous one, `%syncResetValue`
will need to populated with a dummy value anyway.
