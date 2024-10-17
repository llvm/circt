//===- LoopScheduleOps.cpp - LoopSchedule CIRCT Operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the AIG ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace circt::aig;

#define GET_OP_CLASSES
#include "circt/Dialect/AIG/AIG.cpp.inc"

void AIGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/AIG/AIG.cpp.inc"
      >();
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs()) {
    if (hasNoInvertedInputs())
      return getLhs();

    if (getInvertLhs() != getInvertRhs())
      return IntegerAttr::get(
          getType(), APInt::getZero(getType().getIntOrFloatBitWidth()));
  }

  return {};
}

LogicalResult AndOp::canonicalize(AndOp op, PatternRewriter &rewriter) {
  if (!op.getInvertLhs() && op.getInvertRhs()) {
    // Always invert the lhs.
    rewriter.replaceOpWithNewOp<AndOp>(op, op.getRhs(), op.getLhs(), true,
                                       false);
    return success();
  }
  return failure();
}

APInt AndOp::evaluate(const APInt &a, const APInt &b) {
  if (getInvertLhs() && getInvertRhs())
    return ~a & ~b;
  if (getInvertLhs())
    return ~a & b;
  if (getInvertRhs())
    return a & ~b;
  return a & b;
}

#include "circt/Dialect/AIG/AIGDialect.cpp.inc"
