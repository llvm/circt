//===-- Passes.td - Comb pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_PASSES_TD
#define CIRCT_DIALECT_COMB_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerComb : Pass<"lower-comb"> {
  let summary = "Lowers the some of the comb ops";
  let description = [{
    Some operations in the comb dialect (e.g. `comb.truth_table`) are not
    directly supported by ExportVerilog. They need to be lowered into ops which
    are supported. There are many ways to lower these ops so we do this in a
    separate pass. This also allows the lowered form to participate in
    optimizations like the comb canonicalizers.
  }];
}

def CombIntRangeNarrowing : Pass<"comb-int-range-narrowing"> {
  let summary = "Reduce comb op bitwidth based on integer range analysis.";
  let description = [{
    Compute a basic value range analysis, by propagating integer intervals 
    through the domain. The analysis is limited by a lack of sign-extension 
    operator in the comb dialect, leading to an over-approximation. 
    Particularly for signed arithmetic, a single interval is often an 
    over-approximation, a more precise analysis would require a union of 
    intervals.
  }];
}

#endif // CIRCT_DIALECT_COMB_PASSES_TD
