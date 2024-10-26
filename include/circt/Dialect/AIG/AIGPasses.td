//===- AIGPasses.td - AIG dialect passes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_AIGPASSES_TD
#define CIRCT_DIALECT_AIG_AIGPASSES_TD

include "mlir/Pass/PassBase.td"

def LowerCutToLUT : Pass<"aig-lower-cut-to-lut", "hw::HWModuleOp"> {
  let summary = "Lower a cut to a LUT";
  let dependentDialects = ["comb::CombDialect"];
}

def LowerVariadic : Pass<"aig-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic AndInverter operations to binary AndInverter";
}

def LowerWordToBits : Pass<"aig-lower-word-to-bits", "hw::HWModuleOp"> {
  let summary = "Lower multi-bit AIG operations to single-bit ones";
  let dependentDialects = ["comb::CombDialect"];
}

def GreedyCutDecomp : Pass<"aig-greedy-cut-decomp", "hw::HWModuleOp"> {
  let summary = "Decompose AIGs into k-feasible Cuts using a greedy algorithm";
  let dependentDialects = ["comb::CombDialect"];
  let options = [
    Option<"cutSizes", "cut-sizes", "uint32_t", "6",
           "The sizes of the cuts to decompose">,
  ];
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
