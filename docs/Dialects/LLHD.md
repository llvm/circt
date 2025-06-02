# LLHD Dialect

This dialect provides operations and types to interact with an event queue in an event-based simulation.
It describes how signals change over time in reaction to changes in other signals and physical time advancing.
Established hardware description languages such as SystemVerilog and VHDL use an event queue as their programming model to describe combinational and sequential logic, as well as test harnesses and test benches.

[TOC]

## Rationale

### Register Reset Values

Resets are problematic since Verilog forces designers to describe them as edge-sensitive triggers.
This does _not_ match the async resets found on almost all standard cell flip-flops, which are level-sensitive.
Therefore a pass lowering from Verilog-style processes to a structural register such as `seq.compreg` would have to verify that the reset value is a constant in order to make the mapping from edge-sensitive Verilog description to level-sensitive standard cell valid.

In practice, designers commonly implement registers encapsulated in a Verilog module, with the reset value being provided as a module input
port.
This makes determining whether the input is a constant much more difficult.
Most commercial tools relax this constraint and simply map the edge-sensitive reset to a level-sensitive one.
This does not preserve the semantics of the input, which is bad.
Most synthesis tools will then go ahead and fail during synthesis if a register's reset value does not end up being a constant value.

Therefore the Deseq pass does not verify that a register's reset value is a constant.
Instead, it applies the same transform from edge-sensitive to level-sensitive reset as most other tools.

## Types

[include "Dialects/LLHDTypes.md"]

## Attributes

[include "Dialects/LLHDAttributes.md"]

## Operations

[include "Dialects/LLHDOps.md"]

## Passes

[include "LLHDPasses.md"]
