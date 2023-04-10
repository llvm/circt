//===- ArcFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isAlways(Attribute attr, bool expected) {
  if (auto enable = dyn_cast_or_null<IntegerAttr>(attr))
    return enable.getValue().getBoolValue() == expected;
  return false;
}

static bool isAlways(Value value, bool expected) {
  if (!value)
    return false;

  if (auto constOp = value.getDefiningOp<hw::ConstantOp>())
    return constOp.getValue().getBoolValue() == expected;

  return false;
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  if ((isAlways(adaptor.getEnable(), false) ||
       isAlways(adaptor.getReset(), true)) &&
      !getOperation()->hasAttr("name") && !getOperation()->hasAttr("names")) {
    // We can fold to zero here because the states are zero-initialized and
    // don't ever change.
    for (auto resTy : getResultTypes())
      results.push_back(IntegerAttr::get(resTy, 0));
    return success();
  }

  // Remove operand when input is default value.
  if (isAlways(adaptor.getReset(), false))
    return getResetMutable().clear(), success();

  // Remove operand when input is default value.
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();

  return failure();
}

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  // When there are no names attached, the state is not externaly observable.
  // When there are also no internal users, we can remove it.
  if (op->use_empty() && !op->hasAttr("name") && !op->hasAttr("names")) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MemoryReadPortOp
//===----------------------------------------------------------------------===//

OpFoldResult MemoryReadPortOp::fold(FoldAdaptor adaptor) {
  // The result is undefined in this case, but we just return 0.
  if (isAlways(adaptor.getEnable(), false))
    return IntegerAttr::get(getType(), 0);

  if (isAlways(adaptor.getEnable(), true))
    return this->getEnableMutable().clear(), this->getResult();

  return {};
}

//===----------------------------------------------------------------------===//
// MemoryWritePortOp
//===----------------------------------------------------------------------===//

LogicalResult MemoryWritePortOp::fold(FoldAdaptor adaptor,
                                      SmallVectorImpl<OpFoldResult> &results) {
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();
  return failure();
}

LogicalResult MemoryWritePortOp::canonicalize(MemoryWritePortOp op,
                                              PatternRewriter &rewriter) {
  if (isAlways(op.getEnable(), false))
    return rewriter.eraseOp(op), success();
  return failure();
}

//===----------------------------------------------------------------------===//
// MemoryWriteOp
//===----------------------------------------------------------------------===//

LogicalResult MemoryWriteOp::fold(FoldAdaptor adaptor,
                                  SmallVectorImpl<OpFoldResult> &results) {
  if (isAlways(adaptor.getEnable(), true))
    return getEnableMutable().clear(), success();
  return failure();
}

LogicalResult MemoryWriteOp::canonicalize(MemoryWriteOp op,
                                          PatternRewriter &rewriter) {
  if (isAlways(op.getEnable(), false))
    return rewriter.eraseOp(op), success();
  return failure();
}
