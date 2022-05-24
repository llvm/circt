//===- MaterializeFSM.cpp - FSM Outlining Pass ------------------*- C++ -*-===//
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

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

struct MaterializeFSMPass : public MaterializeFSMBase<MaterializeFSMPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void MaterializeFSMPass::runOnOperation() {
  ComponentOp component = getOperation();
  auto *ctx = &getContext();
  auto controlOp = component.getControlOp();
  auto machineOp =
      dyn_cast_or_null<fsm::MachineOp>(controlOp.getBody()->front());
  if (!machineOp) {
    controlOp.emitOpError()
        << "expected an `fsm.machine` operation as the top-level operation "
           "within the control region of this component";
    signalPassFailure();
    return;
  }

  OpBuilder builder(ctx);

  // Commonly used constants.
  builder.setInsertionPointToStart(&machineOp.getBody().front());
  auto c1 = builder.create<hw::ConstantOp>(machineOp.getLoc(),
                                           builder.getI1Type(), 1);
  auto c0 = builder.create<hw::ConstantOp>(machineOp.getLoc(),
                                           builder.getI1Type(), 0);

  // Walk the states of the machine and gather the relation between states and
  // the groups which they enable as well as the set of all enabled states.
  DenseMap<fsm::StateOp, DenseSet<Attribute>> stateEnables;
  DenseSet<StringAttr> allEnabledGroups;

  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    for (auto enableOp : llvm::make_early_inc_range(
             stateOp.output().getOps<calyx::EnableOp>())) {
      stateEnables[stateOp].insert(enableOp.groupNameAttr().getAttr());
      allEnabledGroups.insert(enableOp.groupNameAttr().getAttr());
      // Erase the enable op now that we've recorded the information.
      enableOp.erase();
    }
  }

  // Create a new fsm.machine to include the materialized I/O ports. We must
  // create a new operation, since an fsm.machine with type () -> () does _not_
  // have an operand storage, and an operand storage can only be allocated on
  // construction.
  SmallVector<Type> IOTypes =
      SmallVector<Type>(allEnabledGroups.size(), builder.getI1Type());
  machineOp.setType(builder.getFunctionType(IOTypes, IOTypes));
  assert(machineOp.getBody().getNumArguments() == 0 &&
         "expected no inputs to the FSM");
  machineOp.getBody().addArguments(
      IOTypes,
      SmallVector<Location, 4>(IOTypes.size(), builder.getUnknownLoc()));

  // Build output assignments and transition guards in every state. We here
  // assume that the ordering of states in allEnabledGroups is fixed, since it
  // is used as an analogue for port I/O ordering.
  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    llvm::SmallVector<Value> operands;
    llvm::SmallVector<Value> doneGuards;
    auto &enabledGroups = stateEnables[stateOp];
    // Note: llvm::enumerate does not play nicely with DenseSet
    size_t portIndex = 0;
    for (auto group : allEnabledGroups) {
      if (enabledGroups.contains(group)) {
        operands.push_back(c1);
        doneGuards.push_back(machineOp.getArgument(portIndex));
      } else {
        operands.push_back(c0);
      }
      ++portIndex;
    }

    // Assign the output op.
    auto outputOp = stateOp.output().getOps<fsm::OutputOp>();
    assert(!outputOp.empty() &&
           "Expected an fsm.output op inside the state output region");
    (*outputOp.begin()).getOperation()->setOperands(operands);

    // Assign the transition guards.
    for (auto transition : stateOp.transitions().getOps<fsm::TransitionOp>()) {
      auto guardOp = transition.getGuardReturn();
      llvm::SmallVector<Value> guards;
      llvm::append_range(guards, doneGuards);
      if (guardOp.operand())
        guards.push_back(guardOp.operand());

      if (guards.empty())
        continue;

      builder.setInsertionPoint(guardOp);
      Value guardConjunction =
          builder.create<comb::AndOp>(transition.getLoc(), guards);
      guardOp.setOperand(guardConjunction);
    }
  }
}

std::unique_ptr<mlir::Pass> circt::calyx::createMaterializeFSMPass() {
  return std::make_unique<MaterializeFSMPass>();
}
