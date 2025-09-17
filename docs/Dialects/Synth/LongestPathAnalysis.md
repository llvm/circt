# Synth Longest Path Analysis

This document describes the longest path analysis for the Synth dialect, which calculates the maximum delay through combinational paths in a circuit where Synth primitive operations are considered to have a unit delay.

[TOC]

## Overview

The longest path analysis is a static analysis that computes the maximum delay from inputs to outputs in a circuit. It is particularly useful for:

1. Identifying critical paths in a design
2. Estimating circuit performance
3. Guiding optimization passes to focus on high-delay paths

The analysis treats each primitive logic operation as having a unit delay (1), and calculates the maximum delay through all possible paths in the circuit.


## Design

The longest path analysis is by nature very expensive analysis:
* Dataflow must be tracked in a bit-sensitive manner
* The analysis must handle module instances and hierarchies
* The dataflow graph can be very large, especially for multi core designs with deep hierarchies and wide datapaths
* The analysis must provide debuggable information if requested.

To make the analysis tractable, we use several techniques:
* Redundant Path Elimination: If a path is closed under specific hierarchical context, it's not necessary to propagate it its parents.
* Graph Parallelism: The analysis is designed to be executed in parallel across the instance graph. Each module is analyzed independently, and the results are combined across the hierarchy.
* Early pruning of paths: Paths that are known to be not part of the longest paths are pruned early.

## Implementation Details

### Data Structures

The analysis uses several key data structures:

- **Object**: Represents a specific bit of a value at a specific instance path
- **OpenPath**: Represents an open path from a fan-in with an associated delay
- **DataflowPath**: Represents a complete path from a fanout to a fanin
- **DebugPoint**: Provides debug information for a point in the path

### Algorithm

The analysis works by:

1. Traverse instance graph in post-order
2. Walk operations, and start tracing from fanOuts (FFs and module outputs)
3. Following combinational paths through logic operations
4. Computing the maximum delay to each point in the circuit
5. Recording the paths with their associated delays
6. 1-5 are parallelly executed across all modules in the design

For each operation, the analysis:
- Computes the delay as the maximum of input delays plus the operation's delay
- Records the path information including the source, sink, and intermediate points
- Handles multi-bit values by analyzing each bit position separately

## Usage

### Analysis Interface

The `LongestPathAnalysis` class provides the following key methods:

#### Core Query Methods

- `getResults(Value, size_t, SmallVectorImpl<DataflowPath>&)`: Gets all paths to the given value and bit position.
- `getAverageMaxDelay(Value)`: Returns the average of maximum delays across all bits of a value
- `isAnalysisAvailable(StringAttr)`: Checks if analysis is available for a module

#### Path Query Methods

- `getClosedPaths(StringAttr, SmallVectorImpl<DataflowPath>&, bool elaboratePaths = false)`: Gets all closed paths (register-to-register) in a module
- `getOpenPathsFromInputPortsToInternal(StringAttr, SmallVectorImpl<DataflowPath>&)`: Gets open paths from module input ports to internal sequential elements
- `getOpenPathsFromInternalToOutputPorts(StringAttr, SmallVectorImpl<DataflowPath>&)`: Gets open paths from internal sequential elements to module output ports
- `getAllPaths(StringAttr, SmallVectorImpl<DataflowPath>&, bool elaboratePaths = false)`: Convenience method that combines all path types

#### Hierarchical Path Support

The analysis now supports hierarchical path elaboration through the `elaboratePaths` parameter. When enabled, paths include full hierarchical instance information across module boundaries, providing complete visibility into timing paths in hierarchical designs.
