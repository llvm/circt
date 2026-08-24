//===- PIROps.cpp ==-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/PIR/PIROps.h"
#include "circt/Dialect/PIR/PIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace pir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion Ops
//===----------------------------------------------------------------------===//

// (clk-seq-to-clk-prop (bool-to-clk-seq b)) == (bool-to-clk-prop b)
LogicalResult
ClockedSeqToClockedPropOp::canonicalize(ClockedSeqToClockedPropOp op,
                                        PatternRewriter &rewriter) {
  if (auto bToClkSeq = op.getInput().getDefiningOp<BoolToClockedSeqOp>()) {
    rewriter.replaceOpWithNewOp<BoolToClockedPropOp>(op, bToClkSeq.getInput());
    return success();
  }
  return failure();
}

#define GET_OP_CLASSES
#include "circt/Dialect/PIR/PIR.cpp.inc"
