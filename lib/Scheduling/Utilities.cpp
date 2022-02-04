//===- Utilities.cpp - Library of scheduling utilities --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains useful helpers for scheduler implementations.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;

LogicalResult scheduling::handleOperationsInTopologicalOrder(Problem &prob,
                                                             HandleOpFn fun) {
  auto &allOps = prob.getOperations();
  SmallVector<Operation *> unhandledOps;
  unhandledOps.insert(unhandledOps.begin(), allOps.begin(), allOps.end());

  while (!unhandledOps.empty()) {
    // Remember how many unhandled operations we have at the beginning of this
    // attempt. This is a fail-safe for cyclic dependence graphs: If we do not
    // successfully handle at least one operation per attempt, we have
    // encountered a cycle.
    unsigned numUnhandledBefore = unhandledOps.size();

    // Set up the worklist for this attempt, and initialize it in reverse order
    // so that we can pop off its back later.
    SmallVector<Operation *> worklist;
    worklist.insert(worklist.begin(), unhandledOps.rbegin(),
                    unhandledOps.rend());
    unhandledOps.clear();

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      auto res = fun(op);
      if (failed(res))
        unhandledOps.push_back(op);
    }

    if (numUnhandledBefore == unhandledOps.size())
      return prob.getContainingOp()->emitError() << "dependence cycle detected";
  }

  return success();
}
