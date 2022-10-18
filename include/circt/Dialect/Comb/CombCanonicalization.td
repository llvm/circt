//===- CombCanonicalization.td - Comb canonicalization     -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The canonicalization patterns for the Commb dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD
#define CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD

include "circt/Dialect/Comb/Comb.td"
include "mlir/IR/PatternBase.td"

//def ShlConst : Pat<
//  (ShlOp $x, (isConstant I32Attr:$c), $bin),
//  (ConcatOp (ExtractOp $x, $c), createConstant($c, 0) ...),
//  []>;

def ExtractChain : Pat <
  (ExtractOp (ExtractOp $x, I32Attr:$ilo), I32Attr:$olo),
  (ExtractOp $x, (NativeCodeCall<"$_builder.getIntegerAttr($0.getType(), $0.getValue().getZExtValue() + $1.getValue().getZExtValue())"> $olo, $ilo)),
  []>;

#endif // CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD
