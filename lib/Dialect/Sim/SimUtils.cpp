//===- SimUtils.cpp - Sim utility entry points ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for the Sim dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimUtils.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

namespace circt {
namespace sim {

// Returns true for producer ops that are safe to remove during cascading erase:
// they are side-effect-free formatting/file-handle construction nodes whose
// only relevant liveness is whether their results are still used.
static bool isDeleteCascadable(Operation *op) {
  return isa<FormatLiteralOp, FormatHexOp, FormatOctOp, FormatBinOp,
             FormatScientificOp, FormatFloatOp, FormatGeneralOp, FormatDecOp,
             FormatCharOp, FormatHierPathOp, FormatStringConcatOp, GetFileOp>(
      op);
}

template <typename PrintOpTy>
static void cascadeErasePrintImpl(PrintOpTy op, mlir::RewriterBase &rewriter) {
  auto *root = op.getOperation();
  llvm::DenseSet<Operation *> scheduled;
  std::queue<Operation *> toErase;
  toErase.push(root);
  scheduled.insert(root);
  while (!toErase.empty()) {
    auto *currentOp = toErase.front();
    toErase.pop();

    llvm::DenseSet<Operation *> seenProducers;
    for (auto operand : currentOp->getOperands()) {
      if (auto *definingOp = operand.getDefiningOp();
          definingOp && isDeleteCascadable(definingOp) &&
          seenProducers.insert(definingOp).second) {
        bool allUsesFromCurrent =
            llvm::all_of(definingOp->getResults(), [&](Value result) {
              return llvm::all_of(result.getUsers(), [&](Operation *user) {
                return user == currentOp;
              });
            });
        if (allUsesFromCurrent && scheduled.insert(definingOp).second)
          toErase.push(definingOp);
      }
    }
    rewriter.eraseOp(currentOp);
  }
}

void cascadeErasePrint(PrintFormattedOp op, mlir::RewriterBase &rewriter) {
  cascadeErasePrintImpl(op, rewriter);
}

void cascadeErasePrint(PrintFormattedProcOp op, mlir::RewriterBase &rewriter) {
  cascadeErasePrintImpl(op, rewriter);
}

} // namespace sim
} // namespace circt
