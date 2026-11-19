//===- RemoveUnreachableStates.cpp - Remove unreachable FSM states --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Removes the states of a machine that cannot be reached from its initial
// state.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace fsm {
#define GEN_PASS_DEF_REMOVEUNREACHABLESTATES
#include "circt/Dialect/FSM/Passes.h.inc"
} // namespace fsm
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

namespace {
/// Erases the states of `machine` that cannot be reached from its initial
/// state. Returns true if any state was erased.
bool removeUnreachableStates(MachineOp machine) {
  StateOp initialState = machine.getInitialStateOp();
  // The verifier reports a missing initial state.
  if (!initialState)
    return false;

  llvm::SetVector<StateOp> reachable;
  SmallVector<StateOp> worklist = {initialState};
  while (!worklist.empty()) {
    StateOp state = worklist.pop_back_val();
    if (!reachable.insert(state))
      continue;
    llvm::append_range(worklist, state.getNextStates());
  }

  // Collect in IR order before erasing. A reachable state never transitions to
  // an unreachable one, so the transitions that name an unreachable state sit
  // inside unreachable states and are erased with them.
  SmallVector<StateOp> unreachable;
  for (StateOp state : machine.getOps<StateOp>())
    if (!reachable.contains(state))
      unreachable.push_back(state);

  for (StateOp state : unreachable)
    state.erase();
  return !unreachable.empty();
}

struct RemoveUnreachableStatesPass
    : public circt::fsm::impl::RemoveUnreachableStatesBase<
          RemoveUnreachableStatesPass> {
  void runOnOperation() override {
    bool changed = false;
    getOperation().walk(
        [&](MachineOp machine) { changed |= removeUnreachableStates(machine); });
    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::fsm::createRemoveUnreachableStatesPass() {
  return std::make_unique<RemoveUnreachableStatesPass>();
}
