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

#include <deque>

using namespace circt;
using namespace circt::scheduling;

LogicalResult ASAPScheduler::schedule() {
  if (failed(check()))
    return failure();

  // initialize a worklist with the block's operations
  std::deque<Operation *> worklist;
  worklist.insert(worklist.begin(), getOperations().begin(),
                  getOperations().end());

  // iterate until all operations are scheduled
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (getDependences(op).empty()) {
      // operations with no predecessors are scheduled at time step 0
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
      worklist.push_back(op);
  }

  return success();
}
