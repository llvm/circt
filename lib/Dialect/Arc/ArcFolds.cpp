//===- ArcFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  // When there are no names attached, the state is not externaly observable.
  // When there are also no internal users, we can remove it.
  if (op->use_empty() && !op->hasAttr("name") && !op->hasAttr("names")) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}
