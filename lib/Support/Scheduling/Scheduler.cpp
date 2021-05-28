//===- Scheduler.cpp - Common interface for scheduling algorithms -===========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements base classes and utilities for scheduling algorithms.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Scheduling/Scheduler.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "scheduler"

using namespace circt;
using namespace circt::sched;

LogicalResult Scheduler::checkOp(Scheduler::OperationHandle op) {
  if (!hasAssociatedOperatorType(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Operation '" << op
                            << "' is not associated with an operator type\n");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkDep(Scheduler::Dependence dep) {
  OperationHandle i, j;
  unsigned fri, toi;
  std::tie(i, fri, j, toi) = dep;
  if (!(hasOperation(i) && hasOperation(j))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Dependence '" << i << "':" << fri << " --> '" << j
               << "':" << toi << " references unregistered operations\n");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkOpr(Scheduler::OperatorTypeHandle opr) {
  if (!hasLatency(opr)) {
    LLVM_DEBUG(llvm::dbgs() << "Operator '" << opr << "' has no latency\n");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkProb() { return success(); }

/// Check overall problem by delegating to the component-specific checkers.
LogicalResult Scheduler::check() {
  for (auto op : getOperations()) {
    if (failed(checkOp(op)))
      return failure();
  }

  for (auto dep : getDependences()) {
    if (failed(checkDep(dep)))
      return failure();
  }

  for (auto opr : getOperatorTypes()) {
    if (failed(checkOpr(opr)))
      return failure();
  }

  return checkProb();
}

LogicalResult Scheduler::verifyOp(Scheduler::OperationHandle op) {
  if (!hasStartTime(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Operation '" << op << "' has no start time\n");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::verifyDep(Scheduler::Dependence dep) {
  OperationHandle i, j;
  unsigned fri, toi;
  std::tie(i, fri, j, toi) = dep;

  unsigned stI, latI, stJ;
  stI = getStartTime(i);
  latI = getLatency(getAssociatedOperatorType(i));
  stJ = getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Precedence violated for dependence '" << i << "':" << fri
               << " --> '" << j << "':" << toi << ", " << stI << '+' << latI
               << '>' << stJ << '\n');
    return failure();
  }
  return success();
}

LogicalResult Scheduler::verifyOpr(Scheduler::OperatorTypeHandle opr) {
  return success();
}

LogicalResult Scheduler::verifyProb() { return success(); }

/// Verify overall solution by delegating to the component-specific verifiers.
LogicalResult Scheduler::verify() {
  for (auto op : getOperations()) {
    if (failed(verifyOp(op)))
      return failure();
  }

  for (auto dep : getDependences()) {
    if (failed(verifyDep(dep)))
      return failure();
  }

  for (auto opr : getOperatorTypes()) {
    if (failed(verifyOpr(opr)))
      return failure();
  }

  return verifyProb();
}
