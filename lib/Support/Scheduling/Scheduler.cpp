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

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::sched;

LogicalResult Scheduler::checkOp(Operation *op) {
  if (!hasAssociatedOperatorType(op)) {
    op->emitError("Operation is not associated with an operator type");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkDep(Dependence dep) {
  Operation *i = dep.getSrc();
  Operation *j = dep.getDst();
  if (!(hasOperation(i) && hasOperation(j))) {
    containingOp->emitError()
        << "Scheduling problem contains dependence with unregistered endpoints."
        << "\n  from: " << *i << (hasOperation(i) ? "" : " (unregistered)")
        << "\n  to:   " << *j << (hasOperation(j) ? "" : " (unregistered)");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkOpr(OperatorTypeId opr) {
  if (!hasLatency(opr)) {
    containingOp->emitError() << "Operator type '" << opr << "' has no latency";
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

LogicalResult Scheduler::verifyOp(Operation *op) {
  if (!hasStartTime(op)) {
    op->emitError("Operation has no start time");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::verifyDep(Dependence dep) {
  Operation *i = dep.getSrc();
  Operation *j = dep.getDst();

  unsigned stI, latI, stJ;
  stI = getStartTime(i);
  latI = getLatency(getAssociatedOperatorType(i));
  stJ = getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ)) {
    containingOp->emitError()
        << "Precedence violated for dependence."
        << "\n  from: " << *i << ", result available in t=" << (stI + latI)
        << "\n  to:   " << *j << ", starts in t=" << stJ;
    return failure();
  }
  return success();
}

LogicalResult Scheduler::verifyOpr(OperatorTypeId opr) { return success(); }

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
