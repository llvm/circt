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
- **name**: A name for the register, defaults to `""`. Inferred from the textual
SSA value name, or passed explicitly in builder APIs. The name will be passed to
the `sv.reg` during lowering.
- **innerSym**: An optional symbol to refer to the register. The symbol will be
passed to the `sv.reg` during lowering if present.

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

## The FIRRTL register operation [Provisional]

The `seq.firreg` carries all the information required to represent a FIRRTL
register and lower it to SystemVerilog.

`FirReg` has the following operands:

- **input**: Value to set the register to on the positive edge of the clock
signal.
- **clk**: Clock signal driving the register.
- **name**: A name for the register, passed directly to the `sv.reg`.
- **inner_sym**: A optional symbol for the register, passed directly to the
`sv.reg`. Is a symbol is not specified and the register is randomised, one is
created during the lowering to SV. Registers without symbols can be removed
from the design.
- **reset**: Signal to trigger the reset.
- **resetValue**: A value which is set upon reset. Must be a constant if the
reset is asynchronous.
- **isAsync**: Optional boolean flag indicating whether the reset is
asynchronous.
- **randomized**: Unit attribute indicating whether the register should be
initialised to a random value.

```mlir
%reg = seq.firreg %input, %clk
    [ sym @sym ]
    [ reset (sync|async) %reset, %value ]
    [ randomized ] : $type(input)
```

Examples of registers:

```mlir
%reg_no_reset = seq.firreg %input, %clk, @sym : i32

%reg_sync_reset_rand  = seq.firreg %input, %clk, @sym
    reset sync %reset, %value randomized : i64

%reg_async_reset = seq.firreg %input, %clk, @sym
    reset async %reset, %value : i1
```

A register without a reset lowers directly to an always block:

```
always @(posedge clk or [posedge reset]) begin
  a <= [%input]
end
```

In the presence of a reset, an if statement and an always block with the
proper triggers are emitted:

```
always @(posedge clk or [posedge reset]) begin
  if ([%reset])
    a <= [%resetValue]
  else
    a <= [%input]
end
```

Additionally, `sv` operations are also included to provide the register with
a randomized preset value.
Since items assigned in an `always_ff` block cannot be initialised in an
`initial` block, this register lowers to `always`.

```
`ifndef SYNTHESIS
  `ifdef RANDOMIZE_REG_INIT
    reg [31:0] _RANDOM;
  `endif
  initial begin
    `INIT_RANDOM_PROLOG_
    `ifdef RANDOMIZE_REG_INIT
      _RANDOM = {`RANDOM};
      a = _RANDOM;
    `endif
  end
`endif
```

Registers expect the logic assignment to them to be in SSA form.
For example, a strict connect to a field of a structure:

```firrtl
%field = firrtl.subfield %a(0)
firrtl.strictconnect %field, %value
```
Is converted into a `hw.struct_inject` operation:
```mlir
%reg = seq.firreg %value, %clk, @sym : i32
%value = hw.struct_inject %reg["x"], %value
```

In order to avoid generating unnecessary assignments, the lowering of the
register to `sv` eliminates the SSA form and emits a single parallel assignment
to the field (`reg.x = value`).

### Rationale

A register specific for FIRRTL is desired as it has a specific lowering while
also requiring a preset value and asynchronous resets.
The lowering must also be compatible with the reference FIRRTL lowering, which
might diverge from the lowering of the computation register.

### Future considerations

- Enable signal: if this proves difficult to detect (or non-performant if we
do not detect and generate the SystemVerilog correctly), we can build it into
the compreg op.
- Reset style and clock style: how should we model posedge vs negedge clocks?
Async vs sync resets? There are some reasonable options here: attributes on
this op or `clock` and `reset` types which are parameterized with that
information.
- Initial value: this register is uninitialized. Using an uninitialized value
results in undefined behavior. We will add an `initialValue` attribute if
this proves insufficient.
