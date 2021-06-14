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

#include "circt/Scheduling/Scheduler.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::sched;

Scheduler::Scheduler(Operation *containingOp) : containingOp(containingOp) {
  uniquer.registerParametricStorageType<Dependence>();
  uniquer.registerParametricStorageType<OperatorType>();
}

Dependence *Scheduler::getOrInsertDependence(Operation *src, unsigned srcIdx,
                                             Operation *dst, unsigned dstIdx) {
  auto *dep = uniquer.get<Dependence>({}, src, dst, 0, 0);
  deps.insert(dep);
  return dep;
}

OperatorType *Scheduler::getOrInsertOperatorType(StringRef name) {
  auto *opr = uniquer.get<OperatorType>({}, name);
  oprs.insert(opr);
  return opr;
}

LogicalResult Scheduler::checkOp(Operation *op) {
  if (!hasAssociatedOperatorType(op)) {
    op->emitError("Operation is not associated with an operator type");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkDep(Dependence *dep) {
  Operation *i = dep->getSource();
  Operation *j = dep->getDestination();
  if (!(hasOperation(i) && hasOperation(j))) {
    containingOp->emitError()
        << "Scheduling problem contains dependence with unregistered endpoints."
        << "\n  from: " << *i << (hasOperation(i) ? "" : " (unregistered)")
        << "\n  to:   " << *j << (hasOperation(j) ? "" : " (unregistered)");
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkOpr(OperatorType *opr) {
  if (!hasLatency(opr)) {
    containingOp->emitError()
        << "Operator type '" << opr->getName() << "' has no latency";
    return failure();
  }
  return success();
}

LogicalResult Scheduler::checkProb() { return success(); }

/// Check overall problem by delegating to the component-specific checkers.
LogicalResult Scheduler::check() {
  for (auto *op : getOperations()) {
    if (failed(checkOp(op)))
      return failure();
  }

  for (auto *dep : getDependences()) {
    if (failed(checkDep(dep)))
      return failure();
  }

  for (auto *opr : getOperatorTypes()) {
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

LogicalResult Scheduler::verifyDep(Dependence *dep) {
  Operation *i = dep->getSource();
  Operation *j = dep->getDestination();

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

LogicalResult Scheduler::verifyOpr(OperatorType *opr) { return success(); }

LogicalResult Scheduler::verifyProb() { return success(); }

/// Verify overall solution by delegating to the component-specific verifiers.
LogicalResult Scheduler::verify() {
  for (auto *op : getOperations()) {
    if (failed(verifyOp(op)))
      return failure();
  }

  for (auto *dep : getDependences()) {
    if (failed(verifyDep(dep)))
      return failure();
  }

  for (auto *opr : getOperatorTypes()) {
    if (failed(verifyOpr(opr)))
      return failure();
  }

  return verifyProb();
}

Dependence *
Dependence::construct(mlir::StorageUniquer::StorageAllocator &allocator,
                      const KeyTy &key) {
  auto *result = allocator.allocate<Dependence>();
  std::tie(result->src, result->dst, result->srcIdx, result->dstIdx) = key;
  return result;
}

OperatorType *
OperatorType::construct(mlir::StorageUniquer::StorageAllocator &allocator,
                        const KeyTy &key) {
  auto *result = allocator.allocate<OperatorType>();
  result->name = allocator.copyInto(key);
  return result;
}
