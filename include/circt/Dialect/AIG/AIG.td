//===- AIG.td - AIG Definitions ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_AIG_DIALECT_TD
#define CIRCT_AIG_DIALECT_TD

include "mlir/IR/DialectBase.td"

def AIG_Dialect : Dialect {
  let name = "aig";
  let cppNamespace = "::circt::aig";
  let summary = "Representation of AIGs";
  let dependentDialects = ["::circt::comb::CombDialect"];
}

include "circt/Dialect/AIG/AIGOps.td"

#endif // CIRCT_AIG_DIALECT_TD
