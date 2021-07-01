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

LogicalResult Scheduler::insertDependence(Dependence dep) {
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();

  // Fail early on invalid dependences (src == dst == null), and def-use
  // dependences that cannot be added because the source value is not the result
  // of an operation (e.g., a BlockArgument).
  if (!src || !dst)
    return failure();

  // record auxiliary dependences explicitly
  if (dep.isAuxiliary())
    auxDependences[dst].insert(src);

  // auto-register the endpoints
  operations.insert(src);
  operations.insert(dst);

  return success();
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

LogicalResult Scheduler::checkDependence(Dependence dep) { return success(); }

LogicalResult Scheduler::checkOperatorType(OperatorType opr) {
  if (!getLatency(opr))
    return containingOp->emitError()
           << "Operator type '" << opr << "' has no latency";

  return success();
}

LogicalResult Scheduler::checkProblem() { return success(); }

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

LogicalResult Scheduler::verifyDependence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

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
// Dependence
//===----------------------------------------------------------------------===//

Operation *Dependence::getSource() const {
  return isDefUse() ? defUse->get().getDefiningOp() : auxSrc;
}

Operation *Dependence::getDestination() const {
  return isDefUse() ? defUse->getOwner() : auxDst;
}

Optional<unsigned> Dependence::getSourceIndex() const {
  if (!isDefUse())
    return None;

  assert(defUse->get().isa<OpResult>() && "source is not an operation");
  return defUse->get().dyn_cast<OpResult>().getResultNumber();
}

Optional<unsigned> Dependence::getDestinationIndex() const {
  if (!isDefUse())
    return None;
  return defUse->getOperandNumber();
}

Dependence::TupleRepr Dependence::getAsTuple() const {
  return TupleRepr(getSource(), getDestination(), getSourceIndex(),
                   getDestinationIndex());
}

bool Dependence::operator==(const Dependence &other) const {
  return getAsTuple() == other.getAsTuple();
};

//===----------------------------------------------------------------------===//
// DependenceIterator
//===----------------------------------------------------------------------===//

DependenceIterator::DependenceIterator(Scheduler &scheduler, Operation *op,
                                       bool end)
    : scheduler(scheduler), op(op), operandIdx(0), auxPredIdx(0),
      auxPreds(nullptr), dep() {
  if (!end) {
    if (scheduler.auxDependences.count(op))
      auxPreds = &scheduler.auxDependences[op];

    findNextDependence();
  }
}

void DependenceIterator::findNextDependence() {
  // Yield dependences corresponding to values used by `op`'s operands...
  while (operandIdx < op->getNumOperands()) {
    dep = Dependence(&op->getOpOperand(operandIdx++));
    Operation *src = dep.getSource();

    // ... but only if they are outgoing from operations that are registered in
    // the scheduling problem.
    if (src && scheduler.hasOperation(src))
      return;
  }

  // Then, yield auxiliary dependences, if present.
  if (auxPreds && auxPredIdx < auxPreds->size()) {
    dep = Dependence((*auxPreds)[auxPredIdx++], op);
    return;
  }

  // An invalid dependence signals the end of iteration.
  dep = Dependence();
}
