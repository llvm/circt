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

def LowerVariadic : Pass<"aig-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic AndInverter operations to binary AndInverter";
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
