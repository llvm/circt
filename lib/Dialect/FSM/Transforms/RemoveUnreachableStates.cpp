//===- FSMRemoveUnreachableStates.cpp - Print the instance graph *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Removes unreachable states in the FSM.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace fsm;

namespace {

// Returns all states that are unreachable from the initial state.
static SmallVector<StateOp> unreachableStates(MachineOp machine) {
  llvm::SetVector<StateOp> reachableStates;
  SmallVector<StateOp, 4> queue;
  queue.push_back(machine.getInitialStateOp());
  while (!queue.empty()) {
    auto *state = queue.begin();
    queue.erase(state);
    if (reachableStates.contains(*state))
      continue;
    reachableStates.insert(*state);
    llvm::copy(state->getNextStates(), std::back_inserter(queue));
  }

  // Get the difference between reachable states and all states in the machine.
  auto allStates = machine.getOps<StateOp>();
  SmallVector<StateOp> unreachableStates;
  std::set_difference(allStates.begin(), allStates.end(),
                      reachableStates.begin(), reachableStates.end(),
                      std::back_inserter(unreachableStates));
  return unreachableStates;
}

struct RemoveUnreachableStatesPass
    : public RemoveUnreachableStatesBase<RemoveUnreachableStatesPass> {
  void runOnOperation() override {
    // Erase any unreachable state ops.
    for (auto unreachable : unreachableStates(getOperation()))
      unreachable.erase();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::fsm::createRemoveUnreachableStatesPass() {
  return std::make_unique<RemoveUnreachableStatesPass>();
}
