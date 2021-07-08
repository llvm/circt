//===- ASAPScheduler.cpp - As-soon-as-possible list scheduler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of an as-soon-as-possible list scheduler for acyclic problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/ASAPScheduler.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;

LogicalResult ASAPScheduler::schedule() {
  if (failed(check()))
    return failure();

  // Keep track of ops that don't have a start time yet
  llvm::SmallVector<Operation *> unscheduledOps;
  unscheduledOps.insert(unscheduledOps.begin(), getOperations().begin(),
                        getOperations().end());

  // We may need multiple attempts to schedule all operations in case the
  // problem's operation list is not in a topological order w.r.t. the
  // dependence graph.
  while (!unscheduledOps.empty()) {
    // Remember how many unscheduled operations we have at the beginning of this
    // attempt. This is a fail-safe for cyclic dependence graphs: If we do not
    // schedule at least one operation per attempt, we have encountered a cycle.
    unsigned numUnscheduledBefore = unscheduledOps.size();

    // Set up the worklist for this attempt, and initialize it in reverse order
    // so that we can pop off its back later.
    llvm::SmallVector<Operation *> worklist;
    worklist.insert(worklist.begin(), unscheduledOps.rbegin(),
                    unscheduledOps.rend());
    unscheduledOps.clear();

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();

      // Operations with no predecessors are scheduled at time step 0
      if (getDependences(op).empty()) {
        setStartTime(op, 0);
        continue;
      }

      // op has at least one predecessor. Compute start time as:
      //   max_{p : preds} startTime[p] + latency[linkedOpr[p]]
      unsigned startTime = 0;
      bool startTimeIsValid = true;
      for (auto &dep : getDependences(op)) {
        Operation *pred = dep.getSource();
        if (auto predStart = getStartTime(pred)) {
          // pred is already scheduled
          unsigned predLatency = *getLatency(*getLinkedOperatorType(pred));
          startTime = std::max(startTime, *predStart + predLatency);
        } else {
          // pred is not yet scheduled, give up and try again later
          startTimeIsValid = false;
          break;
        }
      }

      if (startTimeIsValid)
        setStartTime(op, startTime);
      else
        unscheduledOps.push_back(op);
    }

    // Fail if no progress was made during this attempt
    if (numUnscheduledBefore == unscheduledOps.size())
      return getContainingOp()->emitError() << "dependence cycle detected";
  }

  return success();
}
