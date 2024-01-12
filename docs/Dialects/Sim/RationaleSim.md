# Simulation (Sim) Dialect Rationale

This document describes various design points of the `sim` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

The `sim` dialect provides a high-level representation for simulator-specific
operations. The purpose of the dialect is to provide a high-level representation
for constructs which interact with simulators (Verilator, VCS, Arc, ...) that
are easy to analyze and transform in the compiler.

## Operations

### Plusargs

The `sim.plusarg_test` and `sim.plusarg_value` operations are wrappers around
the SystemVerilog built-ins which access command-line arguments.
They are cleaner from a data-flow perspective, as they package the wires
and if-statements involved into compact operations that can be trivially
handled by analyses.
