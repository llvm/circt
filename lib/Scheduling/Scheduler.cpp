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
using namespace circt::scheduling;

Scheduler::Scheduler(Operation *containingOp) : containingOp(containingOp) {
  uniquer.registerParametricStorageType<Dependence>();
  uniquer.registerParametricStorageType<OperatorType>();
}

Dependence *Scheduler::getOrInsertDependence(Operation *src, unsigned srcIdx,
                                             Operation *dst, unsigned dstIdx) {
  auto *dep = uniquer.get<Dependence>({}, src, dst, 0, 0);
  dependences.insert(dep);
  return dep;
}

OperatorType *Scheduler::getOrInsertOperatorType(StringRef name) {
  auto *opr = uniquer.get<OperatorType>({}, name);
  operatorTypes.insert(opr);
  return opr;
}

LogicalResult Scheduler::checkOperation(Operation *op) {
  if (!hasLinkedOperatorType(op))
    return op->emitError("Operation is not linked to an operator type");
  return success();
}

LogicalResult Scheduler::checkDependence(Dependence *dep) {
  Operation *i = dep->getSource();
  Operation *j = dep->getDestination();

  if (!(hasOperation(i) && hasOperation(j)))
    return containingOp->emitError()
           << "Scheduling problem contains dependence with unregistered "
              "endpoints."
           << "\n  from: " << *i << (hasOperation(i) ? "" : " (unregistered)")
           << "\n  to:   " << *j << (hasOperation(j) ? "" : " (unregistered)");

  return success();
}

LogicalResult Scheduler::checkOperatorType(OperatorType *opr) {
  if (!hasLatency(opr))
    return containingOp->emitError()
           << "Operator type '" << opr->getName() << "' has no latency";

  return success();
}

LogicalResult Scheduler::checkProblem() { return success(); }

/// Check overall problem by delegating to the component-specific checkers.
LogicalResult Scheduler::check() {
  for (auto *op : getOperations())
    if (failed(checkOperation(op)))
      return failure();

  for (auto *dep : getDependences())
    if (failed(checkDependence(dep)))
      return failure();

  for (auto *opr : getOperatorTypes())
    if (failed(checkOperatorType(opr)))
      return failure();

  return checkProblem();
}

LogicalResult Scheduler::verifyOperation(Operation *op) {
  if (!hasStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult Scheduler::verifyDependence(Dependence *dep) {
  Operation *i = dep->getSource();
  Operation *j = dep->getDestination();

  unsigned stI, latI, stJ;
  stI = getStartTime(i);
  latI = getLatency(getLinkedOperatorType(i));
  stJ = getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ))
    return containingOp->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ;

  return success();
}

LogicalResult Scheduler::verifyOperatorType(OperatorType *opr) {
  return success();
}

LogicalResult Scheduler::verifyProblem() { return success(); }

/// Verify overall solution by delegating to the component-specific verifiers.
LogicalResult Scheduler::verify() {
  for (auto *op : getOperations())
    if (failed(verifyOperation(op)))
      return failure();

  for (auto *dep : getDependences())
    if (failed(verifyDependence(dep)))
      return failure();

  for (auto *opr : getOperatorTypes())
    if (failed(verifyOperatorType(opr)))
      return failure();

  return verifyProblem();
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
