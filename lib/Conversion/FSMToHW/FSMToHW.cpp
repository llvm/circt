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

#include <memory>

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

namespace {

class StateEncoding {
  // An interface for handling state encoding.
public:
  StateEncoding(OpBuilder &b, MachineOp machine, hw::HWModuleOp hwModule)
      : b(b), machine(machine), hwModule(hwModule) {}
  virtual ~StateEncoding() {}

  Value encode(StateOp state) {
    auto it = stateToValue.find(state);
    assert(it != stateToValue.end() && "state not found");
    return it->second;
  }
  StateOp decode(Value value) {
    auto it = valueToState.find(value);
    assert(it != valueToState.end() && "encoded state not found");
    return it->second;
  }

  // Returns the type which encodes the state values.
  virtual IntegerType getStateType() = 0;

protected:
  // Creates a wired constant value in the module for the given encoded state
  // and records the state value in the mappings. An inner symbol is
  // attached to the wire to avoid it being optimized away.
  void setEncoding(StateOp state, APInt v) {
    assert(stateToValue.find(state) == stateToValue.end() &&
           "state already encoded");

    auto loc = machine.getLoc();
    auto stateType = getStateType();
    auto stateEncodingWire =
        b.create<sv::WireOp>(loc, stateType, state.getNameAttr(),
                             /*inner_sym=*/state.getNameAttr());
    b.create<sv::AssignOp>(loc, stateEncodingWire,
                           b.create<hw::ConstantOp>(loc, v));
    Value encodedValue = b.create<sv::ReadInOutOp>(loc, stateEncodingWire);
    stateToValue[state] = encodedValue;
    valueToState[encodedValue] = state;
  }

  // A mapping between a StateOp and its corresponding encoded value.
  SmallDenseMap<StateOp, Value> stateToValue;
  // A mapping between an encoded value and its corresponding StateOp.
  SmallDenseMap<Value, StateOp> valueToState;

  OpBuilder &b;
  MachineOp machine;
  hw::HWModuleOp hwModule;
};

class BinaryStateEncoding : public StateEncoding {
public:
  BinaryStateEncoding(OpBuilder &b, MachineOp machine, hw::HWModuleOp hwModule)
      : StateEncoding(b, machine, hwModule) {

    // Dead simple integer encoding.
    auto stateType = getStateType();
    for (auto stateIt : llvm::enumerate(machine.getBody().getOps<StateOp>()))
      setEncoding(stateIt.value(),
                  APInt(stateType.getWidth(), stateIt.index()));
  }

  IntegerType getStateType() override {
    return b.getIntegerType(llvm::Log2_64_Ceil(machine.getNumStates()));
  }
};

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

  using StateConversionResults = DenseMap<StateOp, StateConversionResult>;

  // Converts a StateOp within this machine, and returns the value corresponding
  // to the next-state output of the op.
  FailureOr<StateConversionResult> convertState(StateOp state);

  // Converts the outgoing transitions of a state and returns the value
  // corresponding to the next-state output of the op.
  // Transitions are priority encoded in the order which they appear in the
  // state transition region.
  FailureOr<Value> convertTransitions(StateOp currentState,
                                      ArrayRef<TransitionOp> transitions);

  // Returns the value that must be assigned to the hw output port of this
  // machine for a given output port index.
  Value getOutputAssignment(Location loc,
                            StateConversionResults &stateConvResults,
                            size_t portIndex);

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

  // A handle to the state encoder for this machine.
  std::unique_ptr<StateEncoding> encoding;

  // A deterministic ordering of the states in this machine.
  llvm::SmallVector<StateOp> orderedStates;

  // A handle to the MachineOp being converted.
  MachineOp machineOp;

  // A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  // A handle to the state register of the machine.
  seq::CompRegOp stateReg;

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

  // Build state register.
  encoding = std::make_unique<BinaryStateEncoding>(b, machineOp, hwModuleOp);
  auto stateType = encoding->getStateType();

  BackedgeBuilder bb(b, loc);
  auto nextStateBackedge = bb.get(stateType);
  stateReg = b.create<seq::CompRegOp>(
      loc, stateType, nextStateBackedge, clock, "state_reg", reset,
      /*reset value=*/encoding->encode(machineOp.getInitialStateOp()), nullptr,
      nullptr);

  // Move any operations at the machine-level scope, excluding state ops, which
  // are handled separately.
  if (failed(moveOps(&machineOp.front(),
                     [](Operation *op) { return isa<fsm::StateOp>(op); }))) {
    bb.abandon();
    return failure();
  }

  // Gather the states in a deterministic datastructure which will be used for
  // any subsequent iteration over states.
  llvm::SmallVector<Value> nextStateValues;

  // Convert states
  StateConversionResults stateConvResults;
  for (auto state : machineOp.getBody().getOps<StateOp>()) {
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
  // @todo: this assumes that stateReg is binary encoded. Encoding should
  // give an option to mux between values based on a state, and then it is
  // implementation defined how that muxing occurs.
  auto nextStateMux = b.create<hw::ArrayCreateOp>(loc, nextStateValues);
  nextStateMux->setAttr("sv.namehint", b.getStringAttr("next_state_mux"));
  auto nextState = b.create<hw::ArrayGetOp>(loc, nextStateMux, stateReg);
  nextState->setAttr("sv.namehint", b.getStringAttr("state_next"));
  nextStateBackedge.setValue(nextState);

  // Create output port assignments.
  llvm::SmallVector<Value> outputValues;
  for (size_t i = 0; i < machineOp.getNumResults(); i++)
    outputValues.push_back(getOutputAssignment(loc, stateConvResults, i));

  // Delete the default created output op and replace it with the output muxes.
  auto *oldOutputOp = hwModuleOp.front().getTerminator();
  b.create<hw::OutputOp>(loc, outputValues);
  oldOutputOp->erase();

  // Erase the original machine op.
  machineOp.erase();

  return success();
}

FailureOr<Value>
MachineOpConverter::convertTransitions( // NOLINT(misc-no-recursion)
    StateOp currentState, ArrayRef<TransitionOp> transitions) {
  Value nextState;
  if (transitions.empty()) {
    // Base case - transition to the current state.
    nextState = encoding->encode(currentState);
  } else {
    // Recursive case - transition to a named state.
    auto transition = cast<fsm::TransitionOp>(transitions.front());
    nextState = encoding->encode(transition.getNextState());
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

Value MachineOpConverter::getOutputAssignment(
    Location loc, StateConversionResults &convResults, size_t portIndex) {
  llvm::SmallVector<Value, 4> outputPortValues;
  for (auto &state : orderedStates)
    outputPortValues.insert(outputPortValues.begin(),
                            convResults[state].outputs[portIndex]);

  auto outputMuxValues = b.create<hw::ArrayCreateOp>(loc, outputPortValues);
  outputMuxValues->setAttr(
      "sv.namehint",
      b.getStringAttr("output_" + std::to_string(portIndex) + "_mux"));
  auto outputMux = b.create<hw::ArrayGetOp>(loc, outputMuxValues, stateReg);
  return outputMux;
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
