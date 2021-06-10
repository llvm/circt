//===- ASAPScheduler.cpp - As-soon-as-possible list scheduler ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an acyclic, as-soon-as-possible list scheduler.
//
//===----------------------------------------------------------------------===//
#include "circt/Support/Scheduling/ASAPScheduler.h"

#include <deque>

using namespace circt;
using namespace circt::sched;

LogicalResult ASAPScheduler::schedule() {
  if (failed(check())) // TODO: should check for cycles
    return failure();

  // determine each op's predecessors
  OpProp<SmallVector<OperationHandle, 4>> preds;
  for (auto dep : getDependences()) {
    OperationHandle i, j;
    std::tie(i, std::ignore, j, std::ignore) = dep;
    preds[j].push_back(i);
  }

  // initialize a worklist with the block's operations
  std::deque<OperationHandle> worklist;
  worklist.insert(worklist.begin(), getOperations().begin(),
                  getOperations().end());

  // iterate until all operations are scheduled
  while (!worklist.empty()) {
    auto op = worklist.front();
    worklist.pop_front();
    if (preds[op].empty()) {
      // operations with no predecessors are scheduled at time step 0
      setStartTime(op, 0);
      continue;
    }

    // op has at least one predecessor. Compute start time as:
    //   max_{p : preds} startTime[p] + latency[p]
    unsigned startTime = 0;
    bool startTimeIsValid = true;
    for (auto pred : preds[op]) {
      if (hasStartTime(pred)) {
        // pred is already scheduled
        unsigned predStart = getStartTime(pred);
        unsigned predLatency = getLatency(getAssociatedOperatorType(pred));
        startTime = std::max(startTime, predStart + predLatency);
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

  return LogicalResult::success();
}
