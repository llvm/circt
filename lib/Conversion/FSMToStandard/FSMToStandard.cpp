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

static bool dispatchStdOps(Operation &op, OpBuilder &b) {
  if (op.getDialect()->getNamespace() == "std") {
    auto cloneStdOp = b.clone(op);
    cloneStdOp->remove();
    op.replaceAllUsesWith(b.insert(cloneStdOp));
    return true;
  }
  return false;
}

static scf::IfOp convertGuardRegion(Region &region, OpBuilder &b) {
  for (auto &op : region.front()) {
    if (dispatchStdOps(op, b))
      continue;

    if (auto returnOp = dyn_cast<fsm::ReturnOp>(op))
      return b.create<scf::IfOp>(returnOp.getLoc(), returnOp.getOperand(0),
                                 true);
    else
      return op.emitOpError("unsupported op in the guard region"), nullptr;
  }
  return nullptr;
}

static bool convertActionRegion(Region &region, OpBuilder &b) {
  for (auto &op : region.front()) {
    if (dispatchStdOps(op, b))
      continue;

    if (auto update = dyn_cast<UpdateOp>(op))
      b.create<memref::StoreOp>(update.getLoc(), update.src(), update.dst());
    else if (!isa<fsm::ReturnOp>(op))
      return op.emitOpError("unsupported op in the action region"), false;
  }
  return true;
}

static bool convertOutputRegion(Region &region, OpBuilder &b) {
  for (auto &op : region.front()) {
    if (dispatchStdOps(op, b))
      continue;

    if (auto output = dyn_cast<fsm::OutputOp>(op))
      b.create<scf::YieldOp>(output.getLoc(), output.getOperands());
    else
      return op.emitOpError("unsupported op in the output region"), false;
  }
  return true;
}

static bool
convertTransitionsRegion(Region &region, OpBuilder &b, Value stateMemRef,
                         SmallDenseMap<Operation *, Value> &stateEncodeMap) {
  // Build an `if-else` chain for `transition` ops.
  auto transRange = region.getOps<TransitionOp>();
  for (auto transIt = transRange.begin(), transEnd = transRange.end();
       transIt != transEnd; ++transIt) {
    auto trans = *transIt;

    // Convert the guard region and create `if` op if transition is not always
    // taken. Then, set insertion point to the then block of `if` op.
    scf::IfOp guardIfOp = nullptr;
    if (!trans.isAlwaysTaken()) {
      guardIfOp = convertGuardRegion(trans.guard(), b);
      if (!guardIfOp)
        return trans.emitOpError("failed to convert the guard region"), false;
      b.setInsertionPointToStart(guardIfOp.thenBlock());
    }

    // Store the next state to the state memref.
    auto nextState = trans.getReferencedNextState();
    b.create<memref::StoreOp>(trans.getLoc(), stateEncodeMap[nextState],
                              stateMemRef);

    // Convert the action region.
    if (!convertActionRegion(trans.action(), b))
      return trans.emitOpError("failed to convert the action region"), false;

    // If the current transition is not always taken and is not the last
    // one, switch to the next transition.
    if (guardIfOp && std::next(transIt) != transEnd)
      b.setInsertionPointToStart(guardIfOp.elseBlock());
    else
      break;
  }
  return true;
}

/// This is the main entrypoint for the lowering pass.
void FSMToStandardPass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

  // Store the memref type values of all `fsm.variable`s of each `fsm.machine`.
  DenseMap<Value, SmallVector<Value>> memRefMap;
  // Traverse all instances and triggers.
  module.walk([&](Operation *op) {
    if (auto instance = dyn_cast<fsm::InstanceOp>(op)) {
      b.setInsertionPoint(instance);
      auto machine = instance.getReferencedMachine();
      auto stateType = machine.stateType();
      auto &memRefs = memRefMap[instance];

      // Alloca memrefs for the state and all variables. Then, store the initial
      // value into them.
      for (auto variable : machine.getOps<VariableOp>()) {
        auto varMemRef = b.create<memref::AllocaOp>(
            variable.getLoc(), MemRefType::get({}, variable.getType()));
        auto initValue =
            b.create<mlir::ConstantOp>(variable.getLoc(), variable.initValue());
        b.create<memref::StoreOp>(machine.getLoc(), initValue, varMemRef);
        memRefs.push_back(varMemRef);
      }

      auto stateMemRef = b.create<memref::AllocaOp>(
          machine.getLoc(), MemRefType::get({}, stateType));
      // The encoded value of the default state is always zero.
      auto constZero = b.create<mlir::ConstantOp>(machine.getLoc(),
                                                  b.getZeroAttr(stateType));
      b.create<memref::StoreOp>(machine.getLoc(), constZero, stateMemRef);
      memRefs.push_back(stateMemRef);

      opToErase.push_back(instance);

    } else if (auto trigger = dyn_cast<fsm::TriggerOp>(op)) {
      b.setInsertionPoint(trigger);
      auto instance = trigger.instance().getDefiningOp<fsm::InstanceOp>();

      // Add the original inputs and also the memrefs associated to the
      // machine instance to the operand list.
      SmallVector<Value, 16> operands(trigger.inputs());
      operands.append(memRefMap[trigger.instance()]);
      auto callOp = b.create<CallOp>(trigger.getLoc(), instance.machineAttr(),
                                     trigger.getResultTypes(), operands);

      // Replace all uses. Also drop all references.
      trigger.replaceAllUsesWith(callOp);
      trigger->dropAllReferences();
      opToErase.push_back(trigger);
    }
  });

  // Traverse all machines.
  for (auto machine : module.getOps<MachineOp>()) {
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
    SmallDenseMap<Operation *, Value> stateEncodeMap;

    unsigned stateIndex = 0;
    // Traverse all operations in the machine.
    for (auto &op : machine.front()) {
      if (dispatchStdOps(op, b))
        continue;

      if (auto variable = dyn_cast<fsm::VariableOp>(op)) {
        auto varMemRef = func.getArgument(argIndex);
        auto varValue = b.create<memref::LoadOp>(variable.getLoc(), varMemRef);

        // `varMemRef` is lvalue and `varValue` is rvalue. Only when the memref
        // is used as the `dst` of `fsm.update`, replace uses with `varMemRef`.
        variable.getResult().replaceUsesWithIf(varMemRef, [&](OpOperand &use) {
          if (auto update = dyn_cast<UpdateOp>(use.getOwner()))
            if (use.get() == update.dst())
              return true;
          if (auto output = dyn_cast<OutputOp>(use.getOwner()))
            return true;
          return false;
        });
        // Replace other uses with `varValue`, which is the rvalue.
        variable.getResult().replaceAllUsesWith(varValue);
        ++argIndex;

      } else if (auto state = dyn_cast<fsm::StateOp>(op)) {
        // Create an `constant` to store the encoded value and insert it into
        // the map.
        auto encode = APInt(stateType.getWidth(), stateIndex);
        stateEncodeMap[state] = b.create<ConstantOp>(
            state.getLoc(), b.getIntegerAttr(stateType, encode));
        ++stateIndex;

      } else {
        op.emitOpError("found unsupported op in the state machine");
        signalPassFailure();
        return;
      }
    }

    // Get the state memref and its rvalue `stateValue`.
    auto stateMemRef = func.getArguments().back();
    auto stateValue = b.create<memref::LoadOp>(machineLoc, stateMemRef);

    // Build an `if-else` chain for all `state`s to perform the transition.
    for (auto state : machine.getOps<StateOp>()) {
      // Create `scf.if` op for the current `state` op.
      auto stateCond = b.create<CmpIOp>(state.getLoc(), CmpIPredicate::eq,
                                        stateValue, stateEncodeMap[state]);
      auto stateIfOp = b.create<scf::IfOp>(state.getLoc(), stateCond, true);

      // Convert the `transtions` region.
      b.setInsertionPointToStart(stateIfOp.thenBlock());
      if (!convertTransitionsRegion(state.transitions(), b, stateMemRef,
                                    stateEncodeMap)) {
        state.emitOpError("failed to convert the transition region");
        signalPassFailure();
        return;
      }
      // Switch to the next state.
      b.setInsertionPointToStart(stateIfOp.elseBlock());
    }

    auto transTrueOp =
        b.create<mlir::ConstantOp>(machineLoc, b.getBoolAttr(true));
    b.create<AssertOp>(machineLoc, transTrueOp,
                       b.getStringAttr("invalid state"));

    // Get the new state value after transition.
    b.setInsertionPointToEnd(&func.front());
    auto newStateValue = b.create<memref::LoadOp>(machineLoc, stateMemRef);

    // Build an `if-else` chain for all `state`s to generate outputs.
    stateIndex = 0;
    for (auto state : machine.getOps<StateOp>()) {
      // Create `scf.if` op for the current `state` op.
      auto stateCond = b.create<CmpIOp>(state.getLoc(), CmpIPredicate::eq,
                                        newStateValue, stateEncodeMap[state]);
      auto stateIfOp = b.create<scf::IfOp>(
          state.getLoc(), machine.getType().getResults(), stateCond, true);

      // Return or yield for the results of `scf.if` op.
      if (stateIndex++ == 0)
        b.create<mlir::ReturnOp>(state.getLoc(), stateIfOp.getResults());
      else
        b.create<scf::YieldOp>(state.getLoc(), stateIfOp.getResults());

      // Convert the `output` region.
      b.setInsertionPointToStart(stateIfOp.thenBlock());
      if (!convertOutputRegion(state.output(), b)) {
        state.emitOpError("failed to convert the output region");
        signalPassFailure();
        return;
      }
      // Switch to the next state.
      b.setInsertionPointToStart(stateIfOp.elseBlock());
    }

    // Default case should always assert.
    auto outputTrueOp =
        b.create<mlir::ConstantOp>(machineLoc, b.getBoolAttr(true));
    b.create<AssertOp>(machineLoc, outputTrueOp,
                       b.getStringAttr("invalid state"));

    SmallVector<Value, 8> zeroValues;
    for (auto outputType : machine.getType().getResults())
      zeroValues.push_back(
          b.create<mlir::ConstantOp>(machineLoc, b.getZeroAttr(outputType)));
    b.create<scf::YieldOp>(machineLoc, zeroValues);

    // Erase the original machine op.
    opToErase.push_back(machine);
  }

  // Finish the conversion.
  for (auto op : opToErase)
    op->erase();
}
