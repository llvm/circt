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

## Definitions

For the sake of precision, we use the following definitions:

- **Latch:** A memory element which is only sensitive to the levels of its
inputs. Has no clock. Example: SR Latch.
- **Clocked (gated) latch:** A latch wherein the inputs are gated by a clock.
Transparent the entire time the clock is high. Generally referred to as a
"latch". Examples: "gated SR latch", "D latch".
- **Flip-flop:** An edge-sensitive memory element. Captures the input value
on one or both clock edges. Variants: posedge FF, negedge FF,
"edge-sensitive" FF (captures the input value on both edges), resettable FF.
- **Register:** A synchronous, resettable memory element. Can be implemented
using any of the above "circuit level" elements.

## The register operation

The `seq.reg` op models an abstract notion of a "register", independent of
its implementation (e.g. latch, D flip-flop). It is intended to be 'lowered'
to a specific implementation, which may be an `sv.always_ff` block, a device
primitive instantiation, or even a feedback loop in all `comb` logic. It has
three operands and one attribute:

- **input**: The value to be captured at 'clock'. Generally called 'q'.
Accepts any type, results in the same type.
- **clock**: Capture 'value' on the positive edge of this signal.
- **reset**: Signal to set the state to 'resetValue'. Optional.
- **resetValue**: A constant which the state is set to upon reset. Optional.

```mlir
%d = seq.reg %input, %clk [, %reset [<resetValue>] ] : $type(input)
```

### Rationale

Several design decisions were made in defining this op:

- Inclusion of 'reset' signal: the most common lowering case is expected to
result in a SystemVerilog `always_ff`, which _should_ include an `if (rst)`
block. This operand makes lowering to that form easier. Omission of if would
require some (potentially complex) analysis to find the reset mux.
- Inclusion of 'resetValue' attribute: if we have a 'reset' signal, we need
to include a value to avoid complex inversion tricks.
- Omission of 'enable': enables are easily modeled via a multiplexer on the
input with one of the muxs inputs as the output of the register. This
property makes 'enables' easier to detect than reset in the common case.
- Omission of 'negedge' event on 'clock': this is easily modeled by inverting the
clock value.
- Omission of 'dual edge' event on 'clock': this is not terribly common. If
we find a compelling use case, we could either create a new op for "dual
edge" register, or create a "clock double" op which would interpose on the
clock value. (And be detected by a lowering step).
- Omission of edge conditions on 'reset': The reset style is generally a
design-wide choice and (if done correctly) shouldn't affect correctness. It
should, therefore, be determined by a lowering pass.
