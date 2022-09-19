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
#include "mlir/Analysis/DataFlowFramework.h"

using namespace mlir;
using namespace circt::analysis;
using namespace circt::hw;

ChangeResult circt::analysis::SignalState::reset() {
  if (!traced)
    return ChangeResult::NoChange;
  traced = false;
  return ChangeResult::Change;
}

ChangeResult
circt::analysis::SignalState::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const SignalState &>(lattice);
  bool before = traced;
  traced |= rhs.traced;
  if (before != traced)
    return ChangeResult::Change;
  return ChangeResult::NoChange;
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
    Operation *op, const SignalState &before, SignalState *after) {
  ChangeResult result = ChangeResult::NoChange;

  llvm::outs() << *op << ", before: " << before << ", after: " << *after;

  if (sources.contains(op))
    result |= after->setTraced(true);

  if (before.getTraced())
    result |= after->setTraced(true);

  if (sinks.contains(op))
    result = ChangeResult::NoChange;

  if (result == ChangeResult::Change) {
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      Operation *mod = instance.getReferencedModule();
      SignalState *modState = getLattice(mod);
      modState->join(*after);
      addDependency(after, mod);
    }
  }

  propagateIfChanged(after, result);
}

void circt::analysis::SignalTracingAnalysis::setToEntryState(
    SignalState *lattice) {
  propagateIfChanged(lattice, lattice->reset());
}
