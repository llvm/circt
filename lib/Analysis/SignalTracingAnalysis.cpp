//===- SignalTracingAnalysis.cpp - signal tracing dataflow analysis -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SignalTracingAnalysis using the dataflow analysis
// framework.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SignalTracingAnalysis.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace mlir;
using namespace circt::analysis;
using namespace circt::hw;
using namespace circt::sv;

ChangeResult circt::analysis::SignalState::reset() {
  if (!traced)
    return ChangeResult::NoChange;
  traced = false;
  return ChangeResult::Change;
}

bool circt::analysis::SignalState::operator==(const SignalState &rhs) const {
  return traced == rhs.traced;
}

SignalState circt::analysis::SignalState::join(const SignalState &lhs,
                                               const SignalState &rhs) {
  SignalState state;
  state.setTraced(lhs.getTraced() || rhs.getTraced());
  return state;
}

ChangeResult circt::analysis::SignalState::setTraced(bool isTraced) {
  bool before = traced;
  traced = isTraced;
  if (before != traced)
    return ChangeResult::Change;
  return ChangeResult::NoChange;
}

bool circt::analysis::SignalState::getTraced() const { return traced; }

void circt::analysis::SignalState::print(llvm::raw_ostream &os) const {
  os << "SignalState(" << traced << ")\n";
}

void circt::analysis::SignalTracingAnalysis::visitOperation(
    Operation *op, ArrayRef<const Lattice<SignalState> *> operands,
    ArrayRef<Lattice<SignalState> *> results) {
  // Start propagation from sources.
  if (sources.contains(op)) {
    for (Lattice<SignalState> *lattice : results)
      propagateIfChanged(lattice, lattice->join(SignalState(true)));
    return;
  }

  // End propagation at sinks.
  if (sinks.contains(op))
    return;

  // Propagate lattice states from instance operands to module block arguments.
  if (isa<InstanceOp>(op)) {
    HWModuleOp mod = cast<HWModuleOp>(instanceGraph.getReferencedModule(op));
    assert(operands.size() == mod.getBodyBlock()->getNumArguments());
    for (size_t i = 0, e = operands.size(); i < e; ++i) {
      Lattice<SignalState> *argLattice =
          getLatticeElement(mod.getBodyBlock()->getArgument(i));
      propagateIfChanged(argLattice, argLattice->join(*operands[i]));
    }
    return;
  }

  // Propagate lattice states from module outputs to instance results.
  if (isa<OutputOp>(op) && op->getNumOperands() > 0) {
    HWModuleOp mod = op->getParentOfType<HWModuleOp>();
    for (InstanceRecord *use : instanceGraph.lookup(mod)->uses()) {
      assert(operands.size() == use->getInstance()->getNumResults());
      for (size_t i = 0, e = operands.size(); i < e; ++i) {
        Lattice<SignalState> *outputLattice =
            getLatticeElement(use->getInstance()->getResult(i));
        propagateIfChanged(outputLattice, outputLattice->join(*operands[i]));
      }
    }
    return;
  }

  // Propagate lattice states from assigns to the destination wire.
  if (auto assign = dyn_cast<AssignOp>(op)) {
    Lattice<SignalState> *lattice = getLatticeElement(assign.getDest());
    assert(operands.size() == 2);
    propagateIfChanged(lattice, lattice->join(*operands[1]));
    return;
  }

  // All other ops propagate any operands to all results.
  SignalState allOperandState;
  for (const Lattice<SignalState> *operandState : operands)
    allOperandState =
        SignalState::join(allOperandState, operandState->getValue());

  for (Lattice<SignalState> *resultState : results)
    propagateIfChanged(resultState, resultState->join(allOperandState));
}

void circt::analysis::SignalTracingAnalysis::setToEntryState(
    Lattice<SignalState> *lattice) {
  propagateIfChanged(lattice, lattice->getValue().reset());
}
