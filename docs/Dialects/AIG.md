# 'aig' Dialect

This document outlines the rationale of the AIG dialect, a dialect used for representing and transforming And-Inverter Graphs.

[TOC]

## Rationale

# Why use the AIG dialect instead of the `comb` dialect?

And-Inverter Graphs have proven to be a scalable approach for logic synthesis, serving as the underlying data structure for ABC, one of the most performant open-source logic synthesis tools.

While it's technically possible to represent `aig.and_inv` using a combination of `comb.and`, `comb.xor`, and `hw.constant`, the ability to represent everything with `aig.and_inv` offers significant advantages. This unified representation simplifies complex analyses such as path retiming and area analysis, as well as logic mappings. Moreover, it allows for direct application of existing AIG research results and tools, further enhancing its utility in the synthesis process.

# AIG delay analysis

AIG delay analysis provides a way to determine the maximum delay through an AIG.

## Graph rewrite as part of analysis
Unlike the ordinary anaysis in MLIR, this analysis currently mutates the IR as part of the analysis process.

For scalability, this analysis uses lazy-bit-blasting technique.
This is due to the fact that we still need to construct some graph data structures for the analysis even when using lazy-bit-blasting. 

We could implement a similar analysis purely in-memory without mutating the IR, but for now this approach is chosen for its relative simplicity.

## Operations

[include "Dialects/AIGOps.md"]
