//===- FSMToStandard.cpp - Convert FSM to HW and SV Dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToStandard/FSMToStandard.h"
#include "../PassDetail.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace fsm;

namespace {
struct FSMToStandardPass : public ConvertFSMToStandardBase<FSMToStandardPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createConvertFSMToStandardPass() {
  return std::make_unique<FSMToStandardPass>();
}

static std::pair<scf::IfOp, bool> convertGuardRegion(TransitionOp transition,
                                                     OpBuilder &b) {
  // If the transition is always taken, we don't need if statement.
  if (transition.isAlwaysTaken())
    return {nullptr, true};

  scf::IfOp guardIfOp = nullptr;
  for (auto &op : transition.guard().front()) {
    if (op.getDialect()->getNamespace() == "std") {
      auto cloneStdOp = b.clone(op);
      cloneStdOp->remove();
      op.replaceAllUsesWith(b.insert(cloneStdOp));

    } else if (auto returnOp = dyn_cast<fsm::ReturnOp>(op)) {
      guardIfOp = b.create<scf::IfOp>(returnOp.getLoc(), returnOp.getOperand(0),
                                      /*withElseRegion=*/true);
    } else {
      op.emitOpError("found unsupported op in the guard region");
      return {guardIfOp, false};
    }
  }
  return {guardIfOp, true};
}

static bool convertActionRegion(Region &region, OpBuilder &b) {
  for (auto &op : region.front()) {
    if (op.getDialect()->getNamespace() == "std") {
      auto cloneStdOp = b.clone(op);
      cloneStdOp->remove();
      op.replaceAllUsesWith(b.insert(cloneStdOp));

    } else if (auto update = dyn_cast<UpdateOp>(op))
      b.create<memref::StoreOp>(update.getLoc(), update.src(), update.dst());
    else if (!isa<fsm::ReturnOp>(op))
      return op.emitOpError("found unsupported op in the action region"), false;
  }
  return true;
}

/// This is the main entrypoint for the lowering pass.
void FSMToStandardPass::runOnOperation() {
  auto b = OpBuilder(getOperation());
  SmallVector<Operation *, 16> opToErase;
  DenseMap<Value, SmallVector<Value>> memRefMap;

  // Traverse all functions.
  for (auto func : getOperation().getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() == "fsm") {
        b.setInsertionPoint(op);

        if (auto instance = dyn_cast<fsm::InstanceOp>(op)) {
          auto machine = instance.getReferencedMachine();
          auto &memRefs = memRefMap[instance];

          // Alloca memrefs for the state and all variables.
          for (auto variable : machine.getOps<VariableOp>()) {
            auto varMemRef = b.create<memref::AllocaOp>(
                variable.getLoc(), MemRefType::get({}, variable.getType()));
            memRefs.push_back(varMemRef);
          }
          auto stateMemRef = b.create<memref::AllocaOp>(
              machine.getLoc(), MemRefType::get({}, machine.stateType()));
          memRefs.push_back(stateMemRef);

          opToErase.push_back(instance);

        } else if (auto trigger = dyn_cast<fsm::TriggerOp>(op)) {
          auto instance = trigger.instance().getDefiningOp<fsm::InstanceOp>();

          // Add the original inputs and also the memrefs associated to the
          // machine instance to the operand list.
          SmallVector<Value, 16> operands(trigger.inputs());
          operands.append(memRefMap[trigger.instance()]);
          auto callOp =
              b.create<CallOp>(trigger.getLoc(), instance.machineAttr(),
                               trigger.getResultTypes(), operands);

          // Replace all uses. Also drop all references.
          trigger.replaceAllUsesWith(callOp);
          trigger->dropAllReferences();
          opToErase.push_back(trigger);

        } else {
          op->emitOpError("found unsupported FSM op in a function");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }

  // Traverse all machines.
  for (auto machine : getOperation().getOps<MachineOp>()) {
    b.setInsertionPoint(machine);
    auto machineLoc = machine.getLoc();
    auto stateType = machine.stateType().cast<IntegerType>();

    SmallVector<Type, 16> argTypes(machine.getArgumentTypes());
    // Convert the types of state and internal variables into memrefs and add to
    // the argument list.
    for (auto variable : machine.getOps<VariableOp>())
      argTypes.push_back(MemRefType::get({}, variable.getType()));
    argTypes.push_back(MemRefType::get({}, stateType));

    // Create a new function for the machine.
    auto func = b.create<FuncOp>(
        machineLoc, machine.getName(),
        b.getFunctionType(argTypes, machine.getType().getResults()));
    auto entryBlock = func.addEntryBlock();
    b.setInsertionPointToStart(entryBlock);

    // Replace all uses of the machine arguments with the arguments of the new
    // created function.
    unsigned argIndex = 0;
    for (auto input : machine.getArguments()) {
      input.replaceAllUsesWith(func.getArgument(argIndex));
      ++argIndex;
    }

    // We encode each machine state with binary encoding and use `constant` to
    // store the encoded value of each state. This is used to map each state op
    // to its corresponding `constant`.
    SmallDenseMap<Operation *, Value> stateValMap;

    unsigned stateIndex = 0;
    // Traverse all operations in the machine.
    for (auto &op : machine.front()) {
      if (op.getDialect()->getNamespace() == "std") {
        auto cloneStdOp = b.clone(op);
        cloneStdOp->remove();
        op.replaceAllUsesWith(b.insert(cloneStdOp));

      } else if (auto variable = dyn_cast<fsm::VariableOp>(op)) {
        auto varMemRef = func.getArgument(argIndex);
        auto varValue = b.create<memref::LoadOp>(variable.getLoc(), varMemRef);

        // `varMemRef` is lvalue and `varValue` is rvalue. Only when the memref
        // is used as the `dst` of `fsm.update`, replace uses with `varMemRef`.
        variable.getResult().replaceUsesWithIf(varMemRef, [&](OpOperand &use) {
          return isa<UpdateOp>(use.getOwner()) && use.getOperandNumber() == 0;
        });
        // Replace other uses with `varValue`, which is the rvalue.
        variable.getResult().replaceAllUsesWith(varValue);
        ++argIndex;

      } else if (auto state = dyn_cast<fsm::StateOp>(op)) {
        // Create an `constant` to store the encoded value and insert it into
        // the map.
        auto encode = APInt(stateType.getWidth(), stateIndex);
        stateValMap[state] = b.create<ConstantOp>(
            state.getLoc(), b.getIntegerAttr(stateType, encode));
        ++stateIndex;

      } else if (auto output = dyn_cast<fsm::OutputOp>(op)) {
        auto returnOp =
            b.create<mlir::ReturnOp>(output.getLoc(), output.getOperands());
        b.setInsertionPoint(returnOp);
      } else {
        op.emitOpError("found unsupported op in the state machine");
        return;
      }
    }

    // Get the state memref and its rvalue `stateValue`.
    auto stateMemRef = func.getArguments().back();
    auto stateValue = b.create<memref::LoadOp>(machineLoc, stateMemRef);

    // Build an `if-else` chain for all `state` ops.
    stateIndex = 0;
    auto stateRange = machine.getOps<StateOp>();
    for (auto stateIt = stateRange.begin(), stateEnd = stateRange.end();
         stateIt != stateEnd; ++stateIt) {
      auto state = *stateIt;
      // Create `scf.if` op for the current `state` op.
      auto stateCond = b.create<CmpIOp>(state.getLoc(), CmpIPredicate::eq,
                                        stateValue, stateValMap[state]);
      auto stateIfOp = b.create<scf::IfOp>(state.getLoc(), stateCond,
                                           std::next(stateIt) != stateEnd);
      b.setInsertionPointToStart(stateIfOp.thenBlock());

      // Build an `if-else` chain for `transition` ops.
      auto transRange = state.transitions().getOps<TransitionOp>();
      for (auto transIt = transRange.begin(), transEnd = transRange.end();
           transIt != transEnd; ++transIt) {
        auto transition = *transIt;
        // Convert the guard region and create `if` op if applicable.
        auto ifOpAndSucceeded = convertGuardRegion(transition, b);
        if (!ifOpAndSucceeded.second) {
          transition.emitOpError("failed to convert the guard region");
          return;
        }

        // Set insertion point to the then block of `if` op if applicable.
        auto guardIfOp = ifOpAndSucceeded.first;
        if (guardIfOp)
          b.setInsertionPointToStart(guardIfOp.thenBlock());

        // Procedural assign state register to the next state.
        auto nextState = transition.getReferencedNextState();
        b.create<memref::StoreOp>(transition.getLoc(), stateValMap[nextState],
                                  stateMemRef);

        // Convert action regions accordingly.
        if (!convertActionRegion(state.exit(), b)) {
          state.emitOpError("failed to convert the exit region");
          return;
        }
        if (!convertActionRegion(transition.action(), b)) {
          transition.emitOpError("failed to convert the action region");
          return;
        }
        if (!convertActionRegion(nextState.entry(), b)) {
          nextState.emitOpError("failed to convert the entry region");
          return;
        }

        // If the current transition is not always taken and is not the last
        // one, switch to the next transition.
        if (guardIfOp && std::next(transIt) != transEnd)
          b.setInsertionPointToStart(guardIfOp.elseBlock());
        else
          break;
      }

      // Switch to the next state.
      if (std::next(stateIt) != stateEnd)
        b.setInsertionPointToStart(stateIfOp.elseBlock());
    }
    // Erase the original machine op.
    opToErase.push_back(machine);
  }

  // Finish the conversion.
  for (auto op : opToErase)
    op->erase();
}
