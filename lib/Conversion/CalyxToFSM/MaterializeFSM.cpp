//===- MaterializeCalyxToFSM.cpp - FSM Materialization Pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the FSM materialization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace calyx;
using namespace mlir;
using namespace fsm;

namespace {

// A set of group names. Use SetVector to ensure deterministic ordering.
using GroupSet = SetVector<StringAttr>;

struct MaterializeCalyxToFSMPass
    : public MaterializeCalyxToFSMBase<MaterializeCalyxToFSMPass> {
  void runOnOperation() override;

  struct StateEnableInfo {
    llvm::SmallVector<Value> outputOperands;
    llvm::SmallVector<Value> doneGuards;
  };

  /// Returns an operand vector for the states which should be enabled for a
  /// given state, as well as the state done signals that should be considered
  /// for the state transition guard. We do this here to avoid
  /// assignStateOutputOperands/assignStateTransitionGuard having to each walk
  /// the set of output operations.
  StateEnableInfo gatherStateEnableInfo(StateOp stateOp) {
    StateEnableInfo info;
    size_t portIndex = 0;
    auto &enabledGroups = stateEnables[stateOp];
    for (auto group : referencedGroups) {
      if (enabledGroups.contains(group)) {
        info.outputOperands.push_back(c1);
        info.doneGuards.push_back(machineOp.getArgument(portIndex));
      } else
        info.outputOperands.push_back(c0);

      ++portIndex;
    }
    return info;
  }

  /// Assigns the 'fsm.output' operation of the provided 'state' to enabled the
  /// set of provided groups. If 'topLevelDone' is set, also asserts the
  /// top-level done signal.
  void assignStateOutputOperands(StateOp stateOp,
                                 llvm::SmallVector<Value> outputOperands,
                                 bool topLevelDone = false) {
    assert(outputOperands.size() == machineOp.getNumArguments() - 1 &&
           "Expected exactly one value for each uniquely referenced group in "
           "this machine");
    // outputOperands is expected to only have
    outputOperands.push_back(topLevelDone ? c1 : c0);
    auto outputOp = stateOp.output().getOps<fsm::OutputOp>();
    assert(!outputOp.empty() &&
           "Expected an fsm.output op inside the state output region");
    (*outputOp.begin()).getOperation()->setOperands(outputOperands);
  }

  /// Extends every `fsm.return` guard in the transitions of this state to also
  /// include the provided set of 'doneGuards'.
  void assignStateTransitionGuard(OpBuilder &b, StateOp stateOp,
                                  llvm::ArrayRef<Value> doneGuards) {
    for (auto transition : stateOp.transitions().getOps<fsm::TransitionOp>()) {
      auto guardOp = transition.getGuardReturn();
      llvm::SmallVector<Value> guards;
      llvm::append_range(guards, doneGuards);
      if (guardOp.operand())
        guards.push_back(guardOp.operand());

      if (guards.empty())
        continue;

      b.setInsertionPoint(guardOp);
      Value guardConjunction;
      if (guards.size() == 1)
        guardConjunction = guards.front();
      else
        guardConjunction = b.create<comb::AndOp>(transition.getLoc(), guards);
      guardOp.setOperand(guardConjunction);
    }
  }

  /// Maintain a set of all groups referenced within this fsm.machine.
  GroupSet referencedGroups;

  /// Maintain a relation between states and the groups which they enable.
  DenseMap<fsm::StateOp, GroupSet> stateEnables;

  /// A handle to the machine under transformation.
  MachineOp machineOp;

  /// Commonly used constants.
  Value c1, c0;
};

} // end anonymous namespace

void MaterializeCalyxToFSMPass::runOnOperation() {
  ComponentOp component = getOperation();
  auto *ctx = &getContext();
  auto b = OpBuilder(ctx);
  auto controlOp = component.getControlOp();
  machineOp = dyn_cast_or_null<fsm::MachineOp>(controlOp.getBody()->front());
  if (!machineOp) {
    controlOp.emitOpError()
        << "expected an 'fsm.machine' operation as the top-level operation "
           "within the control region of this component";
    signalPassFailure();
    return;
  }

  // Ensure a well-formed FSM.
  auto graph = FSMGraph(machineOp);
  auto *entryState = graph.lookup(b.getStringAttr(calyxToFSM::sEntryStateName));
  auto *exitState = graph.lookup(b.getStringAttr(calyxToFSM::sExitStateName));

  if (!(entryState && exitState)) {
    machineOp.emitOpError()
        << "Expected an '" << calyxToFSM::sEntryStateName << "' and '"
        << calyxToFSM::sExitStateName << "' to be present in the FSM";
    signalPassFailure();
    return;
  }

  // Generate commonly used constants.
  b.setInsertionPointToStart(&machineOp.getBody().front());
  c1 = b.create<hw::ConstantOp>(machineOp.getLoc(), b.getI1Type(), 1);
  c0 = b.create<hw::ConstantOp>(machineOp.getLoc(), b.getI1Type(), 0);

  // Walk the states of the machine and gather the relation between states and
  // the groups which they enable as well as the set of all enabled states.
  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    for (auto enableOp : llvm::make_early_inc_range(
             stateOp.output().getOps<calyx::EnableOp>())) {
      auto groupName = enableOp.groupNameAttr().getAttr();
      stateEnables[stateOp].insert(groupName);
      referencedGroups.insert(groupName);
      // Erase the enable op now that we've recorded the information.
      enableOp.erase();
    }
  }

  // Materialize the top-level I/O ports of the fsm.machine. We add an in- and
  // output for every unique group referenced within the machine, as well as an
  // additional in- and output to represent the top-level "go" input and "done"
  // output ports.
  SmallVector<Type> ioTypes = SmallVector<Type>(
      referencedGroups.size() + /*top-level go/done*/ 1, b.getI1Type());
  size_t nGroups = ioTypes.size() - 1;
  machineOp.setType(b.getFunctionType(ioTypes, ioTypes));
  assert(machineOp.getBody().getNumArguments() == 0 &&
         "expected no inputs to the FSM");
  machineOp.getBody().addArguments(
      ioTypes, SmallVector<Location, 4>(ioTypes.size(), b.getUnknownLoc()));

  // Build output assignments and transition guards in every state. We here
  // assume that the ordering of states in referencedGroups is fixed and
  // deterministic, since it is used as an analogue for port I/O ordering.
  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    StateEnableInfo info = gatherStateEnableInfo(stateOp);
    assignStateOutputOperands(stateOp, info.outputOperands,
                              /*topLevelDone=*/false);
    assignStateTransitionGuard(b, stateOp, info.doneGuards);
  }

  // Assign top-level go guard in the transition state.
  size_t topLevelGoIdx = nGroups;
  assignStateTransitionGuard(b, entryState->getState(),
                             machineOp.getArgument(topLevelGoIdx));

  // Assign top-level done in the exit state.
  assignStateOutputOperands(exitState->getState(),
                            SmallVector<Value>(nGroups, c0),
                            /*topLevelDone=*/true);
}

std::unique_ptr<mlir::Pass> circt::createMaterializeCalyxToFSMPass() {
  return std::make_unique<MaterializeCalyxToFSMPass>();
}
