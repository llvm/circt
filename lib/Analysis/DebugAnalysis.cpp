//===- DebugAnalysis.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace debug;
using namespace mlir;

namespace {
struct DebugAnalysisBuilder {
  DebugAnalysisBuilder(Operation *rootOp) : rootOp(rootOp) {}
  void run();
  void addDebugOp(Operation *op);
  void addDebugValue(Value value);
  void addDebugOperand(OpOperand *operand);
  void maybeDebugOp(Operation *op);

  Operation *rootOp;
  SetVector<Operation *> worklist;

  DenseSet<Operation *> debugOps;
  DenseSet<Value> debugValues;
  DenseSet<OpOperand *> debugOperands;
};
} // namespace

void DebugAnalysisBuilder::run() {
  // Find all debug ops nested under the root op and mark them as debug-only
  // to kickstart the analysis.
  rootOp->walk([&](Operation *op) {
    if (isa_and_nonnull<debug::DebugDialect>(op->getDialect())) {
      addDebugOp(op);
      return;
    }
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto arg : block.getArguments())
          if (isa<debug::DebugDialect>(arg.getType().getDialect()))
            addDebugValue(arg);
    for (auto result : op->getResults())
      if (isa<debug::DebugDialect>(result.getType().getDialect()))
        addDebugValue(result);
  });

  // Visit operations and check if all their operands or all their uses are
  // marked as debug-only. If they are, mark the op itself as debug-only.
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    if (debugOps.contains(op))
      continue;

    // Do not propagate through stateful elements. This should probably be
    // configurable, since certain forms of debug info extraction would be able
    // to pull entire state machines out of the design. For now this just
    // represents the common denominator across all debug infos.
    if (!isa_and_nonnull<hw::HWDialect, comb::CombDialect>(op->getDialect()))
      continue;
    if (op->hasAttr("name"))
      continue;

    if (op->getNumResults() > 0) {
      auto allUsesDebug = llvm::all_of(op->getUses(), [&](auto &use) {
        return debugOperands.contains(&use);
      });
      if (allUsesDebug) {
        addDebugOp(op);
        continue;
      }
    }

    if (op->getNumOperands() > 0) {
      auto allOperandsDebug =
          llvm::all_of(op->getOperands(), [&](auto operand) {
            return debugValues.contains(operand);
          });
      if (allOperandsDebug) {
        addDebugOp(op);
        continue;
      }
    }
  }
}

void DebugAnalysisBuilder::addDebugOp(Operation *op) {
  if (debugOps.insert(op).second) {
    for (auto &operand : op->getOpOperands())
      addDebugOperand(&operand);
    for (auto result : op->getResults())
      addDebugValue(result);
  }
}

void DebugAnalysisBuilder::addDebugValue(Value value) {
  if (debugValues.insert(value).second) {
    for (auto *user : value.getUsers())
      maybeDebugOp(user);
  }
}

void DebugAnalysisBuilder::addDebugOperand(OpOperand *operand) {
  if (debugOperands.insert(operand).second)
    maybeDebugOp(operand->get().getDefiningOp());
}

void DebugAnalysisBuilder::maybeDebugOp(Operation *op) {
  if (!op || debugOps.contains(op))
    return;
  worklist.insert(op);
}

DebugAnalysis::DebugAnalysis(Operation *op) {
  DebugAnalysisBuilder builder(op);
  builder.run();
  debugOps = std::move(builder.debugOps);
  debugValues = std::move(builder.debugValues);
  debugOperands = std::move(builder.debugOperands);
}
