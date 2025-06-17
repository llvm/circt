# 'aig' Dialect

This document outlines the rationale of the AIG dialect, a dialect used for representing and transforming And-Inverter Graphs.

[TOC]

## Rationale

# Why use the AIG dialect instead of the `comb` dialect?

And-Inverter Graphs have proven to be a scalable approach for logic synthesis, serving as the underlying data structure for ABC, one of the most performant open-source logic synthesis tools.

While it's technically possible to represent `aig.and_inv` using a combination of `comb.and`, `comb.xor`, and `hw.constant`, the ability to represent everything with `aig.and_inv` offers significant advantages. This unified representation simplifies complex analyses such as path retiming and area analysis, as well as logic mappings. Moreover, it allows for direct application of existing AIG research results and tools, further enhancing its utility in the synthesis process.
