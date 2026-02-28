# Simulation Dialect

This dialect provides a high-level representation for simulator-specific
operations. The purpose of the dialect is to provide a high-level representation
for constructs which interact with simulators (Verilator, VCS, Arc, ...) that
are easy to analyze and transform in the compiler.

[TOC]

## Rationale

### Plusargs

The `sim.plusarg_test` and `sim.plusarg_value` operations are wrappers around
the SystemVerilog built-ins which access command-line arguments.
They are cleaner from a data-flow perspective, as they package the wires
and if-statements involved into compact operations that can be trivially
handled by analyses.

## Types

[include "Dialects/SimTypes.md"]

## Operations

[include "Dialects/SimOps.md"]
