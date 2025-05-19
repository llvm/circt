# AIG Longest Path Analysis

This document describes the longest path analysis for the AIG dialect, which calculates the maximum delay through combinational paths in a circuit where each AIG and-inverter operation is considered to have a unit delay.

[TOC]

## Overview

The longest path analysis is a static analysis that computes the maximum delay from inputs to outputs in a circuit. It is particularly useful for:

1. Identifying critical paths in a design
2. Estimating circuit performance
3. Guiding optimization passes to focus on high-delay paths

The analysis treats each AIG and-inverter operation as having a unit delay (1), and calculates the maximum delay through all possible paths in the circuit.

## Implementation Details

### Data Structures

The analysis uses several key data structures:

- **Object**: Represents a specific bit of a value at a specific instance path
- **OpenPath**: Represents an open path from a fan-in with an associated delay
- **DataflowPath**: Represents a complete path from a source to a sink
- **DebugPoint**: Provides debug information for a point in the path

### Algorithm

The analysis works by:

1. Traversing the circuit from registers and module inputs (sources)
2. Following combinational paths through AIG operations
3. Computing the maximum delay to each point in the circuit
4. Recording the paths with their associated delays

For each operation, the analysis:
- Computes the delay as the maximum of input delays plus the operation's delay (1 for AIG operations)
- Records the path information including the source, sink, and intermediate points
- Handles multi-bit values by analyzing each bit position separately

### Handling Module Hierarchies

The analysis handles module hierarchies by:

1. Building an instance graph of the design
2. Analyzing modules in post-order (bottom-up)
3. Propagating delay information across module boundaries
4. Maintaining instance paths to track the full hierarchical path to each value

These steps are executed in parallel across modules.

## Usage

### Analysis Interface

The `LongestPathAnalysis` class provides the following key methods:

- `getResults(Value, size_t, SmallVectorImpl<DataflowPath>&)`: Gets all paths to the given value and bit position.
- `getAverageMaxDelay(Value)`: Returns the average of maximum delays across all bits of a value
- `getClosedPaths(StringAttr, SmallVectorImpl<DataflowPath>&)`: Gets all closed paths (register-to-register) in a module
- `getOpenPaths(StringAttr, ...)`: Gets all open paths (input-to-register and register-to-output) in a module
- `isAnalysisAvailable(StringAttr)`: Checks if analysis is available for a module
