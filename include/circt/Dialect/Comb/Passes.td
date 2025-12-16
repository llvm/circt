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

def CombOverflowAnnotating : Pass<"comb-overflow-annotating"> {
  let summary = "Annotate comb ops with no overflow signal (nuw).";
  let description = [{
    Using integer range analysis, annotate comb ops which cannot overflow with
    LLVMs no unsigned wrap (nuw) signal. This is useful for downstream passes 
    which need to handle overflow conditions.
  }];
}

def BalanceMux : Pass<"comb-balance-mux"> {
  let summary = "Balance and optimize mux chains";
  let description = [{
    Optimizes mux chains through enhanced folding and priority mux
    rebalancing. Converts index comparisons to arrays and rebalances
    linear chains into balanced trees, reducing depth from O(n) to O(log n).
  }];
  let options = [Option<
      "muxChainThreshold", "mux-chain-threshold", "unsigned", "16",
      "Minimum number of linear mux chains to trigger rebalancing">];
}

def AssumeTwoValued : Pass<"comb-assume-two-valued"> {
  let summary = "Simplify under assumption of two-valued logic";
  let description = [{
    Performs simplifying transformations that are valid if all integer values
    are always two-valued (i.e., all bits are either 1 or 0).
  }];
}


def SimplifyTruthTable : Pass<"comb-simplify-tt"> {
  let summary = "Simplify truth tables that depend on 1 or less inputs";
  let description = [{
    Simplifies truth tables that are constant or depend only on single input.
    For truth tables that depend on no input, reduce them to hw.constant. For
    truth tables that depend on single input, reduce to identity of input or
    negation of input.
  }];
}

#endif // CIRCT_DIALECT_COMB_PASSES_TD
