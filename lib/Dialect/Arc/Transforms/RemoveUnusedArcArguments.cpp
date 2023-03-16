//===- RemoveUnusedArcArguments.cpp - Implement RemoveUnusedArcArgs Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to remove unused arguments of arc::DefineOps. Also adjusts the
// arc::StateOps referencing the arc.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#define DEBUG_TYPE "arc-remove-unused-arc-arguments"

using namespace mlir;
using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArguments pass
//===----------------------------------------------------------------------===//

namespace {
struct RemoveUnusedArcArgumentsPass
    : public RemoveUnusedArcArgumentsBase<RemoveUnusedArcArgumentsPass> {
  void runOnOperation() override;
  void runOnStateOp(arc::StateOp stateOp, arc::DefineOp arc,
                    DenseMap<arc::DefineOp, BitVector> &removedArgsMap);
};
} // namespace

void RemoveUnusedArcArgumentsPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  DenseMap<arc::DefineOp, BitVector> removedArgsMap;

  getOperation()->walk([&](arc::StateOp stateOp) {
    auto arc = cast<arc::DefineOp>(cast<CallOpInterface>(stateOp.getOperation())
                                       .resolveCallable(&symbolTable));
    runOnStateOp(stateOp, arc, removedArgsMap);
  });
}

void RemoveUnusedArcArgumentsPass::runOnStateOp(
    arc::StateOp stateOp, arc::DefineOp arc,
    DenseMap<arc::DefineOp, BitVector> &removedArgsMap) {
  if (!removedArgsMap.count(arc)) {
    BitVector toErase(arc.getNumArguments());
    for (auto [i, arg] : llvm::enumerate(arc.getArguments())) {
      if (arg.use_empty())
        toErase.set(i);
    }
    arc.eraseArguments(toErase);
    removedArgsMap[arc] = toErase;
  }

  BitVector toErase = removedArgsMap[arc];
  for (int i = toErase.size() - 1; i >= 0; --i) {
    if (toErase[i])
      stateOp.getInputsMutable().erase(i);
  }
}

std::unique_ptr<Pass> arc::createRemoveUnusedArcArgumentsPass() {
  return std::make_unique<RemoveUnusedArcArgumentsPass>();
}
