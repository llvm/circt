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
#include "circt/Scheduling/DependenceIterator.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;
using namespace circt::scheduling::detail;

//===----------------------------------------------------------------------===//
// Scheduler
//===----------------------------------------------------------------------===//

Scheduler::DependenceId
Scheduler::getOrAssignDependenceId(DefUseDependenceHandle dep) {
  auto it = defUseDependenceIds.find(dep);
  if (it != defUseDependenceIds.end())
    return it->second;

  return defUseDependenceIds[dep] = nextDependenceId++;
}

Scheduler::DependenceId
Scheduler::getOrAssignDependenceId(AuxDependenceHandle dep) {
  auto it = auxDependenceIds.find(dep);
  if (it != auxDependenceIds.end())
    return it->second;

  return auxDependenceIds[dep] = nextDependenceId++;
}

Scheduler::OperatorType Scheduler::getOrInsertOperatorType(StringRef name) {
  auto opr = OperatorType::get(name, containingOp->getContext());
  operatorTypes.insert(opr);
  return opr;
}

Scheduler::DependenceRange Scheduler::getDependences(Operation *op) {
  return DependenceRange(DependenceIterator(*this, op),
                         DependenceIterator(*this, op, /*end=*/true));
}

LogicalResult Scheduler::checkOperation(Operation *op) {
  if (!getLinkedOperatorType(op))
    return op->emitError("Operation is not linked to an operator type");
  if (!hasOperatorType(*getLinkedOperatorType(op)))
    return op->emitError("Operation uses an unregistered operator type");
  return success();
}

LogicalResult Scheduler::checkDependence(const Dependence &dep) {
  return success();
}

LogicalResult Scheduler::checkOperatorType(OperatorType opr) {
  if (!getLatency(opr))
    return containingOp->emitError()
           << "Operator type '" << opr << "' has no latency";

  return success();
}

LogicalResult Scheduler::checkProblem() {
  // Check the auxiliary dependences for validity, as these have been provided
  // by the client. The def-use-dependences (from the SSA subgraph) only include
  // registered operations by construction.
  for (auto kv : auxDependences) {
    for (auto &dep : getDependences(kv.first)) {
      Operation *i = dep.src;
      Operation *j = dep.dst;

      if (!(hasOperation(i) && hasOperation(j)))
        return containingOp->emitError()
               << "Scheduling problem contains dependence with unregistered "
                  "endpoints."
               << "\n  from: " << *i << (hasOperation(i) ? "" : " (unreg)")
               << "\n  to:   " << *j << (hasOperation(j) ? "" : " (unreg)");
    }
  }

  return success();
}

/// Check overall problem by delegating to the component-specific checkers.
LogicalResult Scheduler::check() {
  for (auto *op : getOperations())
    if (failed(checkOperation(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(checkDependence(dep)))
        return failure();

  for (auto opr : getOperatorTypes())
    if (failed(checkOperatorType(opr)))
      return failure();

  return checkProblem();
}

LogicalResult Scheduler::verifyOperation(Operation *op) {
  if (!getStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult Scheduler::verifyDependence(const Dependence &dep) {
  Operation *i = dep.src;
  Operation *j = dep.dst;

  unsigned stI, latI, stJ;
  stI = *getStartTime(i);
  latI = *getLatency(*getLinkedOperatorType(i));
  stJ = *getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ))
    return containingOp->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ;

  return success();
}

LogicalResult Scheduler::verifyOperatorType(OperatorType opr) {
  return success();
}

LogicalResult Scheduler::verifyProblem() { return success(); }

/// Verify overall solution by delegating to the component-specific verifiers.
LogicalResult Scheduler::verify() {
  for (auto *op : getOperations())
    if (failed(verifyOperation(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(verifyDependence(dep)))
        return failure();

  for (auto opr : getOperatorTypes())
    if (failed(verifyOperatorType(opr)))
      return failure();

  return verifyProblem();
}

//===----------------------------------------------------------------------===//
// Dependence and DependenceIterator
//===----------------------------------------------------------------------===//

bool Dependence::operator==(const Dependence &other) const {
  return src == other.src && dst == other.dst && srcIdx == other.srcIdx &&
         dstIdx == other.dstIdx && id == other.id;
};

constexpr Dependence DependenceIterator::invalid;

DependenceIterator::DependenceIterator(Scheduler &scheduler, Operation *op,
                                       bool end)
    : scheduler(scheduler), op(op), operandIdx(0), auxPredIdx(0),
      auxPreds(nullptr), dependence(invalid) {
  if (!end) {
    if (scheduler.auxDependences.count(op))
      auxPreds = &scheduler.auxDependences[op];

    findNextDependence();
  }
}

void DependenceIterator::findNextDependence() {
  while (operandIdx < op->getNumOperands()) {
    OpOperand &operand = op->getOpOperand(operandIdx++);
    Operation *src = operand.get().getDefiningOp();

    // Skip definitions that are not an operation, or unregistered in the
    // scheduling problem.
    if (!src || !scheduler.hasOperation(src))
      continue;

    OpResult result = operand.get().dyn_cast<OpResult>();
    Optional<DependenceId> depId = None;
    auto mapIt = scheduler.defUseDependenceIds.find(&operand);
    if (mapIt != scheduler.defUseDependenceIds.end())
      depId = mapIt->second;

    dependence = {src, op, result.getResultNumber(), operandIdx, depId};
    return;
  }

  if (auxPreds && auxPredIdx < auxPreds->size()) {
    Operation *src = (*auxPreds)[auxPredIdx++];

    Optional<DependenceId> depId = None;
    auto mapIt = scheduler.auxDependenceIds.find(
        Scheduler::AuxDependenceHandle(src, op));
    if (mapIt != scheduler.auxDependenceIds.end())
      depId = mapIt->second;

    dependence = {src, op, None, None, depId};
    return;
  }

  dependence = invalid;
}
