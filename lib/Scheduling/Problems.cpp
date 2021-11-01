//===- Problems.h - Modeling of scheduling problems -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements base classes for scheduling problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/DependenceIterator.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;
using namespace circt::scheduling::detail;

//===----------------------------------------------------------------------===//
// Problem
//===----------------------------------------------------------------------===//

LogicalResult Problem::insertDependence(Dependence dep) {
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

Problem::OperatorType Problem::getOrInsertOperatorType(StringRef name) {
  auto opr = OperatorType::get(name, containingOp->getContext());
  operatorTypes.insert(opr);
  return opr;
}

Problem::DependenceRange Problem::getDependences(Operation *op) {
  return DependenceRange(DependenceIterator(*this, op),
                         DependenceIterator(*this, op, /*end=*/true));
}

LogicalResult Problem::checkLinkedOperatorType(Operation *op) {
  if (!getLinkedOperatorType(op))
    return op->emitError("Operation is not linked to an operator type");
  if (!hasOperatorType(*getLinkedOperatorType(op)))
    return op->emitError("Operation uses an unregistered operator type");
  return success();
}

LogicalResult Problem::checkLatency(OperatorType opr) {
  if (!getLatency(opr))
    return getContainingOp()->emitError()
           << "Operator type '" << opr << "' has no latency";

  return success();
}

LogicalResult Problem::check() {
  for (auto *op : getOperations())
    if (failed(checkLinkedOperatorType(op)))
      return failure();

  for (auto opr : getOperatorTypes())
    if (failed(checkLatency(opr)))
      return failure();

  return success();
}

LogicalResult Problem::verifyStartTime(Operation *op) {
  if (!getStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult Problem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ;

  return success();
}

LogicalResult Problem::verify() {
  for (auto *op : getOperations())
    if (failed(verifyStartTime(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(verifyPrecedence(dep)))
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CyclicProblem
//===----------------------------------------------------------------------===//

LogicalResult CyclicProblem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);
  unsigned dist = getDistance(dep).getValueOr(0); // optional property
  unsigned ii = *getInitiationInterval();

  // check if i's result is available before j starts (dist iterations later)
  if (!(stI + latI <= stJ + dist * ii))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ
           << "\n  dist: " << dist << ", II=" << ii;

  return success();
}

LogicalResult CyclicProblem::verifyInitiationInterval() {
  if (!getInitiationInterval() || *getInitiationInterval() == 0)
    return getContainingOp()->emitError("Invalid initiation interval");
  return success();
}

LogicalResult CyclicProblem::verify() {
  if (failed(verifyInitiationInterval()) || failed(Problem::verify()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// SharedPipelinedOperatorsProblem
//===----------------------------------------------------------------------===//

LogicalResult SharedPipelinedOperatorsProblem::checkLatency(OperatorType opr) {
  if (failed(Problem::checkLatency(opr)))
    return failure();

  auto limit = getLimit(opr);
  if (limit && *limit > 0 && *getLatency(opr) == 0)
    return getContainingOp()->emitError()
           << "Limited operator type '" << opr << "' has zero latency.";
  return success();
}

LogicalResult
SharedPipelinedOperatorsProblem::verifyUtilization(OperatorType opr) {
  auto limit = getLimit(opr);
  if (!limit)
    return success();

  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerTimeStep;
  for (auto *op : getOperations())
    if (opr == *getLinkedOperatorType(op))
      ++nOpsPerTimeStep[*getStartTime(op)];

  for (auto &kv : nOpsPerTimeStep)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Operator type '" << opr << "' is oversubscribed."
             << "\n  time step: " << kv.first
             << "\n  #operations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult SharedPipelinedOperatorsProblem::verify() {
  if (failed(Problem::verify()))
    return failure();

  for (auto opr : getOperatorTypes())
    if (failed(verifyUtilization(opr)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloProblem
//===----------------------------------------------------------------------===//

LogicalResult ModuloProblem::verifyUtilization(OperatorType opr) {
  auto limit = getLimit(opr);
  if (!limit)
    return success();

  unsigned ii = *getInitiationInterval();
  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerCongruenceClass;
  for (auto *op : getOperations())
    if (opr == *getLinkedOperatorType(op))
      ++nOpsPerCongruenceClass[*getStartTime(op) % ii];

  for (auto &kv : nOpsPerCongruenceClass)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Operator type '" << opr << "' is oversubscribed."
             << "\n  congruence class: " << kv.first
             << "\n  #operations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult ModuloProblem::verify() {
  if (failed(CyclicProblem::verify()))
    return failure();

  // Don't call SharedPipelinedOperatorsProblem::verify() here to prevent
  // redundant verification of the base problem.
  for (auto opr : getOperatorTypes())
    if (failed(verifyUtilization(opr)))
      return failure();

  return success();
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
}

//===----------------------------------------------------------------------===//
// DependenceIterator
//===----------------------------------------------------------------------===//

DependenceIterator::DependenceIterator(Problem &problem, Operation *op,
                                       bool end)
    : problem(problem), op(op), operandIdx(0), auxPredIdx(0), auxPreds(nullptr),
      dep() {
  if (!end) {
    if (problem.auxDependences.count(op))
      auxPreds = &problem.auxDependences[op];

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
    if (src && problem.hasOperation(src))
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
