# Arc Dialect

This dialect provides operations and types to represent state transfer functions in a circuit, enabling efficient scheduling of operations for simulation.

[TOC]


## Rationale

The main goal of the Arc dialect is to provide an intermediate representation of hardware designs that is optimized for simulation.
It transforms hardware descriptions from the HW, Seq, and Comb dialects into a form where all module hierarchies have been flattened, combinational logic is represented as callable "arcs" (state transfer functions), and sequential elements are modeled explicitly.

The Arc dialect is used by the *arcilator* simulation tool, which compiles Arc IR to a binary object via LLVM for fast simulation.


## Types

[include "Dialects/ArcTypes.md"]


## Operations

[include "Dialects/ArcOps.md"]


## Passes

[include "ArcPasses.md"]
