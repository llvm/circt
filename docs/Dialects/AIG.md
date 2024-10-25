# 'aig' Dialect

This document outlines the rationale of the AIG dialect, a dialect used for representing and transforming And-Inverter Graphs.

[TOC]

## Rationale

# Why use the AIG dialect instead of the `comb` dialect?

And-Inverter Graphs have proven to be a scalable approach for logic synthesis, serving as the underlying data structure for ABC, one of the most performant open-source logic synthesis tools.

While it's technically possible to represent `aig.and_inv` using a combination of `comb.and`, `comb.xor`, and `hw.constant`, the ability to represent everything with `aig.and_inv` offers significant advantages. This unified representation simplifies complex analyses such as path retiming and area analysis, as well as logic mappings. Moreover, it allows for direct application of existing AIG research results and tools, further enhancing its utility in the synthesis process.

# Operations
## aig.and_inv

The `aig.and_inv` operation directly represents an And-Node in the AIG. Traditionally, an And-Node in AIG has two operands. However, `aig.and_inv` extends this concept by allowing variadic operands and non-i1 integer types. Although the final stage of the pipeline requires lowering everything to i1-binary operands, it's more efficient to progressively lower the variadic multibit operations.

Variadic operands have demonstrated their utility in low-level optimizations within the `comb` dialect. Furthermore, in synthesis, it's common practice to re-balance the logic path. Variadic operands enable the compiler to select more efficient solutions without the need to traverse binary trees multiple times.

The ability to represent multibit operations during synthesis is crucial for scalable logic optimization. This approach enables a form of vectorization, allowing for batch processing of logic synthesis when multibit operations are constructed in a similar manner. Such vectorization can significantly improve the efficiency and performance of the synthesis process.

## aig.cut

The `aig.cut` operation represents a "cut" in the logic tree. This operation possesses the `IsolatedAbove` trait and contains a single block. Its input operands represent the input edges, while the returned value represents the output edges.

This operation proves particularly useful for progressive LUT mapping. For instance, a k-input cut can be readily mapped to a k-input LUT. Consequently, the subsequent stages of the pipeline can concentrate on replacing combinational logic with k-input Cut operations, simplifying the overall process.


## Operations

[include "Dialects/AIGOps.md"]
