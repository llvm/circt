# 'synth' Dialect

This document outlines the rationale of the Synth dialect, a dialect for logic synthesis which includes boolean operations specifically designed for logic synthesis such as and-inverter and majority-inverter.

[TOC]

## Rationale

### Logic Representations: AIG and MIG

AIG (And-Inverter Graph) and MIG (Majority-Inverter Graph) have emerged as fundamental representations in logic synthesis. AIG represents Boolean functions using only AND gates with optional input inversions, providing a canonical and efficient representation that has been the foundation of synthesis tools like ABC for decades. Recent research trends have explored MIG as an alternative representation using majority gates (3-input gates that output the majority value) with optional input inversions, which can represent some functions more compactly than AIG and enables novel optimization strategies, particularly for arithmetic circuits and emerging computing paradigms.

CIRCT synthesis pipeline is designed to support both AIG and MIG, and can be extended to support other representations in the future.

### Why use the Synth dialect instead of the `comb` dialect?

And-Inverter Graphs have proven to be a scalable approach for logic synthesis, serving as the underlying data structure for ABC, one of the most performant open-source logic synthesis tools.

While it's technically possible to represent `synth.aig.and_inv` using a combination of `comb.and`, `comb.xor`, and `hw.constant`, the ability to represent everything with `synth.aig.and_inv` offers significant advantages. This unified representation simplifies complex analyses such as path retiming and area analysis, as well as logic mappings. Moreover, it allows for direct application of existing AIG research results and tools, further enhancing its utility in the synthesis process.
