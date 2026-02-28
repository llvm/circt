//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_UNUSEDOPPRUNER_H
#define CIRCT_SUPPORT_UNUSEDOPPRUNER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {

/// Utility that tracks operations that have potentially become unused and
/// allows them to be cleaned up at a later point. Useful to make passes clean
/// up dead code.
struct UnusedOpPruner {
  /// Mark an op the be erased later if it is unused at that point.
  void eraseLaterIfUnused(Operation *op) {
    assert(op);
    opsToEraseIfUnused.insert(op);
  }

  /// Mark the defining op of a value to be erased later if the op is unused at
  /// that point.
  void eraseLaterIfUnused(Value value) {
    if (!value)
      return;
    if (auto *defOp = value.getDefiningOp())
      eraseLaterIfUnused(defOp);
    else
      blockArgsToEraseIfUnused.insert(cast<BlockArgument>(value));
  }

  /// Mark the defining ops of a range of values to be erased later if the ops
  /// are unused at that point.
  void eraseLaterIfUnused(ValueRange values) {
    for (auto value : values)
      eraseLaterIfUnused(value);
  }

  /// Erase an operation immediately, and remove it from the set of ops to be
  /// removed later. The op is erase regardless of whether it has any side
  /// effects or not.
  void eraseNow(Operation *op) {
    eraseLaterIfUnused(op->getOperands());
    opsToEraseIfUnused.erase(op);
    op->erase();
  }

  // Erase tracked operations that are side-effect free and have become unused.
  void eraseNow() {
    using mlir::BranchOpInterface;
    while (!opsToEraseIfUnused.empty() || !blockArgsToEraseIfUnused.empty()) {
      while (!opsToEraseIfUnused.empty()) {
        auto it = opsToEraseIfUnused.begin();
        auto *op = *it;
        opsToEraseIfUnused.erase(it);
        if (!isOpTriviallyDead(op))
          continue;
        eraseLaterIfUnused(op->getOperands());
        op->erase();
      }
      while (!blockArgsToEraseIfUnused.empty()) {
        auto it = blockArgsToEraseIfUnused.begin();
        auto arg = *it;
        blockArgsToEraseIfUnused.erase(it);
        if (!arg.use_empty())
          continue;
        if (!llvm::all_of(arg.getOwner()->getUses(), [](auto &blockOperand) {
              return isa<BranchOpInterface>(blockOperand.getOwner());
            }))
          continue;
        unsigned argIdx = arg.getArgNumber();
        for (auto &blockOperand : arg.getOwner()->getUses()) {
          auto branchOp = cast<BranchOpInterface>(blockOperand.getOwner());
          auto operands =
              branchOp.getSuccessorOperands(blockOperand.getOperandNumber());
          eraseLaterIfUnused(operands[argIdx]);
          operands.erase(argIdx);
        }
        arg.getOwner()->eraseArgument(argIdx);
      }
    }
  }

private:
  /// The set of operations that may have become unused.
  llvm::SmallDenseSet<Operation *> opsToEraseIfUnused;
  /// The set of block arguments that may have become unused.
  llvm::SmallDenseSet<BlockArgument> blockArgsToEraseIfUnused;
};

} // namespace circt

#endif // CIRCT_SUPPORT_UNUSEDOPPRUNER_H
