//===- FSMToHW.cpp - Convert FSM to HW and SV Dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace fsm;

/// Get the port info of a FSM machine. Clock and reset port are also added.
static void getMachinePortInfo(SmallVectorImpl<hw::PortInfo> &ports,
                               MachineOp machine, OpBuilder &b) {
  // Get the port info of the machine inputs and outputs.
  machine.getHWPortInfo(ports);

  // Add clock port.
  hw::PortInfo clock;
  clock.name = b.getStringAttr("clk");
  clock.direction = hw::PortDirection::INPUT;
  clock.type = b.getI1Type();
  clock.argNum = machine.getNumArguments();
  ports.push_back(clock);

  // Add reset port.
  hw::PortInfo reset;
  reset.name = b.getStringAttr("rst");
  reset.direction = hw::PortDirection::INPUT;
  reset.type = b.getI1Type();
  reset.argNum = machine.getNumArguments() + 1;
  ports.push_back(reset);
}

class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp)
      : machineOp(machineOp), b(builder) {}

  LogicalResult dispatch();

private:
  struct StateConversionResult {
    // Value of the next state output signal of the converted state.
    Value nextState;
    // Value of the output signals of the converted state.
    llvm::SmallVector<Value> outputs;
  };

  // Converts a StateOp within this machine, and returns the value corresponding
  // to the next-state output of the op.
  FailureOr<StateConversionResult> convertState(StateOp state);

  // Converts the outgoing transitions of a state and returns the value
  // corresponding to the next-state output of the op.
  // Transitions are priority encoded in the order which they appear in the
  // state transition region.
  FailureOr<Value> convertTransitions(StateOp currentState,
                                      ArrayRef<TransitionOp> transitions);

  // Moves operations from 'block' into module scope, failing if any op were
  // deemed illegal. Returns the final op in the block if the op was a
  // terminator. An optional 'exclude' filer can be provided to dynamically
  // exclude some ops from being moved.
  FailureOr<Operation *>
  moveOps(Block *block,
          llvm::function_ref<bool(Operation *)> exclude = nullptr) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (!isa<comb::CombDialect, hw::HWDialect, fsm::FSMDialect>(
              op.getDialect()))
        return op.emitOpError()
               << "is unsupported (op from the "
               << op.getDialect()->getNamespace() << " dialect).";

      if (exclude && exclude(&op))
        continue;

      if (op.hasTrait<OpTrait::IsTerminator>())
        return &op;

      op.moveBefore(&hwModuleOp.front(), b.getInsertionPoint());
    }
    return nullptr;
  }
  // A mapping between a StateOp and its corresponding encoded value.
  SmallDenseMap<StateOp, Value> stateEncodeMap;

  // A handle to the MachineOp being converted.
  MachineOp machineOp;

  // A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  OpBuilder &b;
};

LogicalResult MachineOpConverter::dispatch() {
  if (auto varOps = machineOp.front().getOps<VariableOp>(); !varOps.empty())
    return (*varOps.begin())->emitOpError()
           << "FSM variables not yet supported for HW "
              "lowering.";

  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  if (machineOp.getNumStates() < 2)
    return machineOp.emitOpError() << "expected at least 2 states.";
  auto stateType =
      b.getIntegerType(llvm::Log2_64_Ceil(machineOp.getNumStates()));

  // Get the port info of the machine and create a new HW module for it.
  SmallVector<hw::PortInfo, 16> ports;
  getMachinePortInfo(ports, machineOp, b);
  hwModuleOp = b.create<hw::HWModuleOp>(loc, machineOp.sym_nameAttr(), ports);
  b.setInsertionPointToStart(&hwModuleOp.front());

  // Replace all uses of the machine arguments with the arguments of the
  // new created HW module.
  for (auto args :
       llvm::zip(machineOp.getArguments(), hwModuleOp.front().getArguments())) {
    auto machineArg = std::get<0>(args);
    auto hwModuleArg = std::get<1>(args);
    machineArg.replaceAllUsesWith(hwModuleArg);
  }

  auto clock =
      hwModuleOp.front().getArgument(hwModuleOp.front().getNumArguments() - 2);
  auto reset =
      hwModuleOp.front().getArgument(hwModuleOp.front().getNumArguments() - 1);

  // Create the state register of the machine.
  BackedgeBuilder bb(b, loc);
  auto nextStateBackedge = bb.get(stateType);
  auto defaultStateBackedge = bb.get(stateType);
  auto stateReg = b.create<seq::CompRegOp>(
      loc, stateType, nextStateBackedge, clock, "state_reg", reset,
      defaultStateBackedge, nullptr, nullptr);

  // Build state encoding. We assign these to named SV wires to allow for
  // propagating state names into the generated SV. An inner symbol is
  // attached to the wire to avoid it being optimized away.
  for (auto state : machineOp.front().getOps<fsm::StateOp>()) {
    auto stateEncodingWire = b.create<sv::WireOp>(
        loc, stateType, state.getNameAttr(), /*inner_sym=*/state.getNameAttr());
    b.create<sv::AssignOp>(
        loc, stateEncodingWire,
        b.create<hw::ConstantOp>(
            loc, APInt(stateType.getWidth(), stateEncodeMap.size())));
    stateEncodeMap[state] = b.create<sv::ReadInOutOp>(loc, stateEncodingWire);

    if (machineOp.getInitialStateOp() == state)
      defaultStateBackedge.setValue(stateEncodeMap[state]);
  }

  // Move any operations at the machine-level scope, excluding state ops, which
  // are handled separately.
  if (failed(moveOps(&machineOp.front(),
                     [](Operation *op) { return isa<fsm::StateOp>(op); }))) {
    bb.abandon();
    return failure();
  }

  // Gather the states in a deterministic datastructure which will be used for
  // any subsequent iteration over states.
  llvm::SmallVector<StateOp> orderedStates;
  llvm::SmallVector<Value> nextStateValues;

  // Convert states
  DenseMap<StateOp, StateConversionResult> stateConvResults;
  for (auto state : machineOp.front().getOps<fsm::StateOp>()) {
    auto stateConvRes = convertState(state);
    if (failed(stateConvRes)) {
      bb.abandon();
      return failure();
    }
    stateConvResults[state] = stateConvRes.getValue();
    orderedStates.push_back(state);
    nextStateValues.insert(nextStateValues.begin(),
                           stateConvRes.getValue().nextState);
  }

  // Create next-state mux.
  auto nextStateMux = b.create<hw::ArrayCreateOp>(loc, nextStateValues);
  nextStateMux->setAttr("sv.namehint", b.getStringAttr("next_state_mux"));
  auto nextState = b.create<hw::ArrayGetOp>(loc, nextStateMux, stateReg);
  nextState->setAttr("sv.namehint", b.getStringAttr("state_next"));
  nextStateBackedge.setValue(nextState);

  // Create output muxes.
  llvm::SmallVector<Value> outputMuxes;
  llvm::SmallVector<llvm::SmallVector<Value, 4>, 4> outputPortValues(
      machineOp.getNumResults());
  for (auto &state : orderedStates) {
    for (auto it : llvm::enumerate(stateConvResults[state].outputs))
      outputPortValues[it.index()].insert(outputPortValues[it.index()].begin(),
                                          it.value());
  }

  for (auto outputPortValueIt : llvm::enumerate(outputPortValues)) {
    auto outputMuxValues =
        b.create<hw::ArrayCreateOp>(loc, outputPortValueIt.value());
    outputMuxValues->setAttr(
        "sv.namehint",
        b.getStringAttr("output_" + std::to_string(outputPortValueIt.index()) +
                        "_mux"));
    auto outputMux = b.create<hw::ArrayGetOp>(loc, outputMuxValues, stateReg);
    outputMuxes.push_back(outputMux);
  }

  // Delete the default created output op and replace it with the output muxes.
  auto *oldOutputOp = hwModuleOp.front().getTerminator();
  b.create<hw::OutputOp>(loc, outputMuxes);
  oldOutputOp->erase();

  // Erase the original machine op.
  machineOp.erase();

  return success();
}

FailureOr<Value>
MachineOpConverter::convertTransitions( // NOLINT(misc-no-recursion)
    StateOp currentState, ArrayRef<TransitionOp> transitions) {
  Value nextState;
  if (transitions.empty())
    // Base case - transition to the current state.
    nextState = stateEncodeMap[currentState];
  else {
    // Recursive case - transition to a named state.
    auto transition = cast<fsm::TransitionOp>(transitions.front());
    nextState = stateEncodeMap[transition.getNextState()];
    if (transition.hasGuard()) {
      // Not always taken; recurse and mux between the targeted next state and
      // the recursion result, selecting based on the provided guard.
      auto guardOpRes = moveOps(&transition.guard().front());
      if (failed(guardOpRes))
        return failure();

      auto guard = cast<ReturnOp>(*guardOpRes).getOperand(0);
      auto otherNextState =
          convertTransitions(currentState, transitions.drop_front());
      if (failed(otherNextState))
        return failure();
      comb::MuxOp nextStateMux = b.create<comb::MuxOp>(
          transition.getLoc(), guard, nextState, *otherNextState);
      nextState = nextStateMux;
    }
  }
  return nextState;
}

FailureOr<MachineOpConverter::StateConversionResult>
MachineOpConverter::convertState(StateOp state) {
  MachineOpConverter::StateConversionResult res;

  // Convert the output region by moving the operations into the module scope
  // and gathering the operands of the output op.
  auto outputOpRes = moveOps(&state.output().front());
  if (failed(outputOpRes))
    return failure();

  OutputOp outputOp = cast<fsm::OutputOp>(outputOpRes.getValue());
  res.outputs = outputOp.getOperands();

  auto transitions = llvm::SmallVector<TransitionOp>(
      state.transitions().getOps<TransitionOp>());
  auto nextStateRes = convertTransitions(state, transitions);
  if (failed(nextStateRes))
    return failure();
  res.nextState = nextStateRes.getValue();
  return res;
}

namespace {
struct FSMToHWPass : public ConvertFSMToHWBase<FSMToHWPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto b = OpBuilder(module);
    SmallVector<Operation *, 16> opToErase;

    // Traverse all machines and convert.
    for (auto machine :
         llvm::make_early_inc_range(module.getOps<MachineOp>())) {
      MachineOpConverter converter(b, machine);

      if (failed(converter.dispatch())) {
        signalPassFailure();
        return;
      }
    }

    // Traverse all machine instances and convert to hw instances.
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      for (auto instance :
           llvm::make_early_inc_range(hwModule.getOps<fsm::HWInstanceOp>())) {
        auto fsmHWModule =
            module.lookupSymbol<hw::HWModuleOp>(instance.machine());
        assert(fsmHWModule &&
               "FSM machine should have been converted to a hw.module");

        b.setInsertionPoint(instance);
        llvm::SmallVector<Value, 4> operands;
        llvm::transform(instance.getOperands(), std::back_inserter(operands),
                        [&](auto operand) { return operand; });
        auto hwInstance = b.create<hw::InstanceOp>(
            instance.getLoc(), fsmHWModule, b.getStringAttr(instance.getName()),
            operands, nullptr);
        instance.replaceAllUsesWith(hwInstance);
        instance.erase();
      }
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToHWPass() {
  return std::make_unique<FSMToHWPass>();
}
