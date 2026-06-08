//===- ArcFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

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

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {

  if (getNumResults() > 0 && !getOperation()->hasAttr("name") &&
      !getOperation()->hasAttr("names")) {
    bool hasExplicitInitials = !getInitials().empty();
    bool allInitialsConstant =
        !hasExplicitInitials ||
        llvm::all_of(adaptor.getInitials(),
                     [&](Attribute attr) { return !!attr; });
    if (isAlways(adaptor.getEnable(), false) && allInitialsConstant) {
      // Fold to the explicit or implicit initial value if
      // the state is never enabled and the initial values
      // are compile-time constants.
      if (hasExplicitInitials)
        results.append(adaptor.getInitials().begin(),
                       adaptor.getInitials().end());
      else
        for (auto resTy : getResultTypes())
          results.push_back(IntegerAttr::get(resTy, 0));
      return success();
    }
    if (!hasExplicitInitials && isAlways(adaptor.getReset(), true)) {
      // We assume both the implicit initial value and the
      // implicit (synchronous) reset value to be zero.
      for (auto resTy : getResultTypes())
        results.push_back(IntegerAttr::get(resTy, 0));
      return success();
    }
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
// StorageGetOp
//===----------------------------------------------------------------------===//

LogicalResult StorageGetOp::canonicalize(StorageGetOp op,
                                         PatternRewriter &rewriter) {
  if (auto pred = op.getStorage().getDefiningOp<StorageGetOp>()) {
    rewriter.modifyOpInPlace(op, [&] {
      op.getStorageMutable().assign(pred.getStorage());
      op.setOffset(op.getOffset() + pred.getOffset());
    });
    return success();
  }
  return failure();
}
