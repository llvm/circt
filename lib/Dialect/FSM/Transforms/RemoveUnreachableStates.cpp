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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
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

// Removes any transition to the provided 'unreachableState' within the
// 'machine'. It is expected that any transition with references the
// 'unreachableState' is similarly unreachable.
static LogicalResult
removeTransitionsToState(MachineOp machine, ArrayRef<StateOp> unreachableStates,
                         StateOp unreachableState) {
  auto stateUses = SymbolTable::getSymbolUses(unreachableState, machine);
  if (stateUses.hasValue()) {
    for (auto use : stateUses.getValue()) {
      Operation *user = use.getUser();
      auto transition = dyn_cast<TransitionOp>(user);
      if (!transition)
        return user->emitOpError()
               << "unreachable state '" << unreachableState.getName()
               << "' referenced by something else than a "
                  "transition op. "
               << "This is unhandled behaviour - erroring out";

      assert(llvm::find(unreachableStates,
                        transition->getParentOfType<StateOp>()) !=
                 unreachableStates.end() &&
             "unreachable state referenced by some state which is reachable?");
      transition.erase();
    }
  }
  return success();
}

struct RemoveUnreachableStatesPass
    : public RemoveUnreachableStatesBase<RemoveUnreachableStatesPass> {
  void runOnOperation() override {
    auto machine = getOperation();
    // Erase any unreachable state ops.
    auto unreachables = unreachableStates(machine);
    for (auto unreachable : unreachables) {
      // To safely delete the unreachable states, we first need to delete any
      // transitions which reference them. This occurs when there is some
      // strongly connected set of states which are all unreachable from the
      // entry state.
      if (failed(
              removeTransitionsToState(machine, unreachables, unreachable))) {
        signalPassFailure();
        return;
      }
      unreachable.erase();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::fsm::createRemoveUnreachableStatesPass() {
  return std::make_unique<RemoveUnreachableStatesPass>();
}
