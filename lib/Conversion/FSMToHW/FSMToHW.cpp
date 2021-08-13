//===- FSMToHW.cpp - Convert FSM to HW and SV Dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToHW/FSMToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace fsm;

namespace {
struct FSMToHWPass : public ConvertFSMToHWBase<FSMToHWPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createConvertFSMToHWPass() {
  return std::make_unique<FSMToHWPass>();
}

/// Get the port info of a FSM machine. Clock and reset port are also added.
static void getMachinePortInfo(SmallVectorImpl<hw::ModulePortInfo> &ports,
                               MachineOp machine, OpBuilder &b) {
  // Get the port info of the machine inputs and outputs.
  machine.getHWPortInfo(ports);

  // Add clock port.
  hw::ModulePortInfo clock;
  clock.name = b.getStringAttr("clk");
  clock.direction = hw::PortDirection::INPUT;
  clock.type = b.getI1Type();
  clock.argNum = machine.getNumArguments();
  ports.push_back(clock);

  // Add reset port.
  hw::ModulePortInfo reset;
  reset.name = b.getStringAttr("rst_n");
  reset.direction = hw::PortDirection::INPUT;
  reset.type = b.getI1Type();
  reset.argNum = machine.getNumArguments() + 1;
  ports.push_back(reset);
}

template <typename CombType, typename StdType>
static void convertStdOp(StdType stdOp, OpBuilder &b) {
  auto combOp =
      b.create<CombType>(stdOp.getLoc(), stdOp.getType(), stdOp.getOperands());
  stdOp.getResult().replaceAllUsesWith(combOp);
}

static bool dispatchStdOps(Operation *op, OpBuilder &b) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<mlir::AddIOp>(
          [&](auto op) { return convertStdOp<comb::AddOp>(op, b), true; })
      .Case<mlir::SubIOp>(
          [&](auto op) { return convertStdOp<comb::SubOp>(op, b), true; })
      .Case<mlir::MulIOp>(
          [&](auto op) { return convertStdOp<comb::MulOp>(op, b), true; })
      .Case<mlir::AndOp>(
          [&](auto op) { return convertStdOp<comb::AndOp>(op, b), true; })
      .Case<mlir::OrOp>(
          [&](auto op) { return convertStdOp<comb::OrOp>(op, b), true; })
      .Case<mlir::XOrOp>(
          [&](auto op) { return convertStdOp<comb::XorOp>(op, b), true; })
      .Case<mlir::CmpIOp>([&](mlir::CmpIOp op) {
        auto predicate = comb::symbolizeICmpPredicate((uint64_t)op.predicate());
        auto icmpOp = b.create<comb::ICmpOp>(op.getLoc(), predicate.getValue(),
                                             op.rhs(), op.lhs());
        return op.getResult().replaceAllUsesWith(icmpOp), true;
      })
      .Case<mlir::ConstantOp>([&](mlir::ConstantOp op) {
        auto valAttr = op.value().dyn_cast<IntegerAttr>();
        if (!valAttr)
          return op.emitOpError("constant op must be integer type"), false;

        auto constantOp = b.create<hw::ConstantOp>(op.getLoc(), valAttr);
        return op.getResult().replaceAllUsesWith(constantOp), true;
      })
      .Default([&](auto) { return false; });
}

static std::pair<sv::IfOp, bool> convertGuardRegion(TransitionOp transition,
                                                    OpBuilder &b) {
  // If the transition is always taken, we don't need if statement.
  if (transition.isAlwaysTaken())
    return {nullptr, true};

  sv::IfOp guardIfOp = nullptr;
  for (auto &op : transition.guard().front()) {
    if (dispatchStdOps(&op, b))
      continue;

    if (auto returnOp = dyn_cast<fsm::ReturnOp>(op))
      guardIfOp = b.create<sv::IfOp>(returnOp.getLoc(), returnOp.getOperand(0));
    else {
      op.emitOpError("found unsupported op in the guard region");
      return {guardIfOp, false};
    }
  }
  return {guardIfOp, true};
}

// FIXME: As the entry and exit region of state can be converted multiple times,
// the current implementation will cause some bugs.
static bool convertActionRegion(Region &region, OpBuilder &b) {
  for (auto &op : region.front()) {
    if (dispatchStdOps(&op, b))
      continue;

    if (auto update = dyn_cast<UpdateOp>(op))
      b.create<sv::PAssignOp>(update.getLoc(), update.dst(), update.src());
    else if (!isa<fsm::ReturnOp>(op))
      return op.emitOpError("found unsupported op in the action region"), false;
  }
  return true;
}

/// This is the main entrypoint for the lowering pass.
void FSMToHWPass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

  // Traverse all machine instances.
  auto walkResult = module.walk([&](fsm::HWInstanceOp instance) {
    if (!instance.clock() || !instance.reset()) {
      instance.emitOpError("must have clock and reset operand");
      return WalkResult::interrupt();
    }

    b.setInsertionPoint(instance);
    auto hwInstance = b.create<hw::InstanceOp>(
        instance.getLoc(), instance->getResultTypes(), instance.sym_nameAttr(),
        instance.machineAttr(), instance->getOperands(), nullptr,
        instance.sym_nameAttr());
    instance.replaceAllUsesWith(hwInstance);

    opToErase.push_back(instance);
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Traverse all machines.
  for (auto machine : module.getOps<MachineOp>()) {
    b.setInsertionPoint(machine);
    auto machineLoc = machine.getLoc();
    auto stateType = machine.stateType().cast<IntegerType>();

    // Get the port info of the machine and create a new HW module for it.
    SmallVector<hw::ModulePortInfo, 16> ports;
    getMachinePortInfo(ports, machine, b);
    auto hwModule =
        b.create<hw::HWModuleOp>(machineLoc, machine.sym_nameAttr(), ports);
    b.setInsertionPointToStart(&hwModule.front());

    // Replace all uses of the machine arguments with the arguments of the new
    // created HW module.
    for (auto args :
         llvm::zip(machine.getArguments(), hwModule.getArguments())) {
      auto machineArg = std::get<0>(args);
      auto hwModuleArg = std::get<1>(args);
      machineArg.replaceAllUsesWith(hwModuleArg);
    }

    // Create the `state` register of the machine and get its rvalue.
    auto stateReg =
        b.create<sv::RegOp>(machineLoc, stateType, b.getStringAttr("state"));
    auto stateRegElem = b.create<sv::ReadInOutOp>(machineLoc, stateReg);

    // Used to store all the internal registers and their rvalues.
    SmallVector<sv::RegOp, 8> regs;
    SmallVector<Value, 8> regElems;

    // We encode each machine state with binary encoding and use `hw.constant`
    // to store the encoded value of each state. This is used to map each state
    // op to its corresponding `hw.constant`.
    SmallDenseMap<Operation *, Value> stateValMap;
    // Store the encoded pattern as an attribute for each state. This will be
    // used to construct the `sv.casez` statement later.
    SmallVector<Attribute, 16> stateCases;

    unsigned stateIndex = 0;
    // Traverse all operations in the machine.
    for (auto &op : machine.front()) {
      if (dispatchStdOps(&op, b))
        continue;

      if (auto variable = dyn_cast<fsm::VariableOp>(op)) {
        // Convert `variable` op to register.
        auto reg = b.create<sv::RegOp>(variable.getLoc(), variable.getType(),
                                       variable.nameAttr());
        auto regElem = b.create<sv::ReadInOutOp>(op.getLoc(), reg);
        regs.push_back(reg);
        regElems.push_back(regElem);

        // `reg` is lvalue and `regElem` is rvalue. Only when the register is
        // used as the `dst` of `fsm.update` op, replace the use with `reg`.
        variable.getResult().replaceUsesWithIf(reg, [&](OpOperand &use) {
          if (auto update = dyn_cast<UpdateOp>(use.getOwner()))
            if (use.get() == update.dst())
              return true;
          return false;
        });
        // Replace other register uses with `regElem`, which is the rvalue.
        variable.getResult().replaceAllUsesWith(regElem);

      } else if (auto state = dyn_cast<fsm::StateOp>(op)) {
        // Create an `hw.constant` to store the encoded value and insert it into
        // the map.
        auto width = stateType.getWidth();
        stateValMap[state] =
            b.create<hw::ConstantOp>(state.getLoc(), APInt(width, stateIndex));

        // Generate a case pattern (see the definition of `sv.casez` op) as an
        // integer attribute.
        auto pattern = APInt(width * 2, 0);
        for (int64_t i = width - 1; i >= 0; --i) {
          pattern <<= 2;
          pattern |= (stateIndex >> i) % 2;
        }
        stateCases.push_back(
            b.getIntegerAttr(b.getIntegerType(width * 2), pattern));
        ++stateIndex;

      } else if (auto output = dyn_cast<fsm::OutputOp>(op)) {
        hwModule.front().getTerminator()->setOperands(output.getOperands());
      } else {
        op.emitOpError("found unsupported op in the state machine");
        signalPassFailure();
        return;
      }
    }

    // We adopt a 1-always FSM coding style. Create the `always_ff` statement.
    auto clock = hwModule.getArgument(hwModule.getNumArguments() - 2);
    auto reset = hwModule.getArgument(hwModule.getNumArguments() - 1);
    auto alwaysff = b.create<sv::AlwaysFFOp>(
        machineLoc, sv::EventControl::AtPosEdge, clock, ResetType::AsyncReset,
        sv::EventControl::AtNegEdge, reset);

    // Generate the reset block of `always_ff`. Reset internal registers and the
    // state register.
    auto defaultState = machine.getDefaultState();
    b.setInsertionPointToStart(alwaysff.getResetBlock());
    b.create<sv::PAssignOp>(machineLoc, stateReg, stateValMap[defaultState]);
    // FIXME: Use initValue to initalize registers.
    for (auto reg : regs) {
      auto constZero =
          b.create<hw::ConstantOp>(reg.getLoc(), reg.getElementType(), 0);
      b.create<sv::PAssignOp>(reg.getLoc(), reg, constZero);
    }
    if (!convertActionRegion(defaultState.entry(), b)) {
      defaultState.emitOpError("failed to convert the entry region");
      signalPassFailure();
      return;
    }

    // Begin to generate the body block of `always_ff`. By default, we assign
    // internal registers and state to themselves.
    b.setInsertionPointToStart(alwaysff.getBodyBlock());
    b.create<sv::PAssignOp>(machineLoc, stateReg, stateRegElem);
    for (auto regAndElem : llvm::zip(regs, regElems)) {
      auto reg = std::get<0>(regAndElem);
      auto elem = std::get<1>(regAndElem);
      b.create<sv::PAssignOp>(reg.getLoc(), reg, elem);
    }

    // Create a `casez` statement with the state register as condition.
    auto casez =
        b.create<sv::CaseZOp>(machineLoc, stateRegElem,
                              b.getArrayAttr(stateCases), stateCases.size());

    // In the `casez` statement, we build one block for each `state` op.
    stateIndex = 0;
    for (auto state : machine.getOps<StateOp>()) {
      b.setInsertionPointToStart(&casez.getRegion(stateIndex++).emplaceBlock());

      // Build an `if-else` chain for `transition` ops.
      auto transRange = state.transitions().getOps<TransitionOp>();
      for (auto it = transRange.begin(), e = transRange.end(); it != e; ++it) {
        auto transition = *it;
        // Convert the guard region and create `if` op if applicable.
        auto ifOpAndSucceeded = convertGuardRegion(transition, b);
        if (!ifOpAndSucceeded.second) {
          transition.emitOpError("failed to convert the guard region");
          signalPassFailure();
          return;
        }

        // Set insertion point to the then block of `if` op if applicable.
        auto guardIfOp = ifOpAndSucceeded.first;
        if (guardIfOp)
          b.setInsertionPointToStart(guardIfOp.getThenBlock());

        // Procedural assign state register to the next state.
        auto nextState = transition.getReferencedNextState();
        b.create<sv::PAssignOp>(transition.getLoc(), stateReg,
                                stateValMap[nextState]);

        // Convert action regions accordingly.
        if (!convertActionRegion(state.exit(), b)) {
          state.emitOpError("failed to convert the exit region");
          signalPassFailure();
          return;
        }
        if (!convertActionRegion(transition.action(), b)) {
          transition.emitOpError("failed to convert the action region");
          signalPassFailure();
          return;
        }
        if (!convertActionRegion(nextState.entry(), b)) {
          nextState.emitOpError("failed to convert the entry region");
          signalPassFailure();
          return;
        }

        // If the current transition is not the last one, create `else` block.
        if (guardIfOp && std::next(it) != e)
          b.setInsertionPointToStart(&guardIfOp.elseRegion().emplaceBlock());
        else
          break;
      }
    }
    // Erase the original machine op.
    opToErase.push_back(machine);
  }

  // Finish the conversion.
  for (auto op : opToErase)
    op->erase();
}
