//===- ArcPasses.td - Arc dialect passes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCPASSES_TD
#define CIRCT_DIALECT_ARC_ARCPASSES_TD

include "mlir/Pass/PassBase.td"

def LowerCutToLUT : Pass<"aig-lower-cut-to-lut", "hw::HWModuleOp"> {
  let summary = "Lower a cut to a LUT";
  let dependentDialects = ["comb::CombDialect"];
}

#endif // CIRCT_DIALECT_ARC_ARCPASSES_TD
