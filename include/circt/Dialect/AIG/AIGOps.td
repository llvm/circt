//===- AIGOps.td - AIG Op Definitions -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for AIG.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_OPS_TD
#define CIRCT_DIALECT_AIG_OPS_TD

include "circt/Dialect/AIG/AIG.td"
include "mlir/IR/OpBase.td"

class AIGOp<string mnemonic, list<Trait> traits = []> :
  Op<AIG_Dialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_AIG_OPS_TD
