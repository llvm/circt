//===- CombCanonicalization.td - Comb canonicalization -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The canonicalization patterns for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD
#define CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD

include "circt/Dialect/Comb/Comb.td"
include "circt/Dialect/HW/HW.td"

def isAllOnes : Constraint<CPred<"$0.isAllOnesValue()">>;
def isZero : Constraint<CPred<"$0.isNullValue()">>;

// extract(mux(c,a,b)) -> mux(c,extract(a),extract(b))
def NarrowMux : Pat <
  (ExtractOp (MuxOp $c, $a, $b), $lb),
  (MuxOp $c, (ExtractOp $a, $lb), (ExtractOp $b, $lb)),
  []>;

// mux(not(c),a,b)) -> mux(c,a,b)
def NarrowMux : Pat <
  (ExtractOp (MuxOp $c, $a, $b), $lb),
  (MuxOp $c, (ExtractOp $a, $lb), (ExtractOp $b, $lb)),
  []>;


#endif // CIRCT_DIALECT_COMB_COMBCANONICALIZATION_TD
