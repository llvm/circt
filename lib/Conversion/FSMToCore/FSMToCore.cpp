//===- FSMToCore.cpp - Convert FSM to HW Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToCore.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTFSMTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

namespace {
struct ClkRstIdxs {
  size_t clockIdx;
  size_t resetIdx;
};

/// Clones constants implicitly captured by the region, into the region.
static void cloneConstantsIntoRegion(Region &region, OpBuilder &builder) {
  // Values implicitly captured by the region.
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, region, captures);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&region.front());

  // Clone ConstantLike operations into the region.
  for (auto &capture : captures) {
    Operation *op = capture.getDefiningOp();
    if (!op || !op->hasTrait<OpTrait::ConstantLike>())
      continue;

    Operation *cloned = builder.clone(*op);
    for (auto [orig, replacement] :
         llvm::zip(op->getResults(), cloned->getResults()))
      replaceAllUsesInRegionWith(orig, replacement, region);
  }
}

static ClkRstIdxs getMachinePortInfo(SmallVectorImpl<hw::PortInfo> &ports,
                                     MachineOp machine, OpBuilder &b) {
  // Get the port info of the machine inputs and outputs.
  machine.getHWPortInfo(ports);
  ClkRstIdxs specialPorts;

  // Add clock port.
  hw::PortInfo clock;
  clock.name = b.getStringAttr("clk");
  clock.dir = hw::ModulePort::Direction::Input;
  clock.type = seq::ClockType::get(b.getContext());
  clock.argNum = machine.getNumArguments();
  ports.push_back(clock);
  specialPorts.clockIdx = clock.argNum;

  // Add reset port.
  hw::PortInfo reset;
  reset.name = b.getStringAttr("rst");
  reset.dir = hw::ModulePort::Direction::Input;
  reset.type = b.getI1Type();
  reset.argNum = machine.getNumArguments() + 1;
  ports.push_back(reset);
  specialPorts.resetIdx = reset.argNum;

  return specialPorts;
}
} // namespace

namespace {

class StateEncoding {

public:
  StateEncoding(OpBuilder &b, MachineOp machine, hw::HWModuleOp hwModule);

  /// Get the encoded value for a state.
  Value encode(StateOp state);
  /// Get the state corresponding to an encoded value.
  StateOp decode(Value value);

  /// Returns the type which encodes the state values.
  Type getStateType() { return stateType; }

protected:
  /// Creates a constant value in the module for the given encoded state
  /// and records the state value in the mappings.
  void setEncoding(StateOp state, Value v);

  /// A mapping between a StateOp and its corresponding encoded value.
  SmallDenseMap<StateOp, Value> stateToValue;

  /// A mapping between an encoded value and its corresponding StateOp.
  SmallDenseMap<Value, StateOp> valueToState;

  Type stateType;

  OpBuilder &b;
  MachineOp machine;
  hw::HWModuleOp hwModule;
};

StateEncoding::StateEncoding(OpBuilder &b, MachineOp machine,
                             hw::HWModuleOp hwModule)
    : b(b), machine(machine), hwModule(hwModule) {
  Location loc = machine.getLoc();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&hwModule.getBodyRegion().front());
  // If stateType is explicitly provided, use this - otherwise, calculate the
  // minimum int size that can represent all states
  if (machine->getAttr("stateType")) {
    // We already checked that a static cast is valid
    stateType = cast<TypeAttr>(machine->getAttr("stateType")).getValue();
  } else {
    int numStates = std::distance(machine.getBody().getOps<StateOp>().begin(),
                                  machine.getBody().getOps<StateOp>().end());
    stateType =
        IntegerType::get(machine.getContext(), llvm::Log2_64_Ceil(numStates));
  }
  int stateValue = 0;
  // And create values for the states
  b.setInsertionPointToStart(&hwModule.getBody().front());
  for (auto state : machine.getBody().getOps<StateOp>()) {
    auto constantOp = hw::ConstantOp::create(b, loc, stateType, stateValue++);
    setEncoding(state, constantOp);
  }
}

// Get the encoded value for a state.
Value StateEncoding::encode(StateOp state) {
  auto it = stateToValue.find(state);
  assert(it != stateToValue.end() && "state not found");
  return it->second;
}

void StateEncoding::setEncoding(StateOp state, Value v) {
  assert(stateToValue.find(state) == stateToValue.end() &&
         "state already encoded");
  stateToValue[state] = v;
  valueToState[v] = state;
}
} // namespace

namespace {
class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp)
      : machineOp(machineOp), b(builder),
        bb(BackedgeBuilder(builder, machineOp->getLoc())) {}

  /// Converts the machine op to a hardware module.
  /// 1. Creates a HWModuleOp for the machine op, with the same I/O as the FSM +
  /// clk/reset ports.
  /// 2. Creates a state register + encodings for the states visible in the
  /// machine.
  /// 3. Iterates over all states in the machine
  ///  3.1. Moves all `comb` logic into the body of the HW module
  ///  3.2. Extend the output logic mux chains with cases for this state
  ///  3.3. Iterates over the transitions of the state
  ///    3.3.1. Moves all `comb` logic in the transition guard/action regions to
  ///            the body of the HW module.
  ///    3.3.2. Extends the next state mux chain with an optionally guarded case
  ///    for this transition.
  /// 4. Connect the state and variable mux chain outputs to the corresponding
  /// register inputs.
  LogicalResult dispatch();

private:
  /// Converts a StateOp within this machine, and returns the value
  /// corresponding to the next-state output of the op.
  LogicalResult convertState(StateOp state);

  /// Converts the outgoing transitions of a state and returns the value
  /// corresponding to the next-state output of the op.
  /// Transitions are priority encoded in the order which they appear in the
  /// state transition region.
  FailureOr<Value> convertTransitions(StateOp currentState,
                                      ArrayRef<TransitionOp> transitions);

  /// Moves operations from 'block' into module scope, failing if any op were
  /// deemed illegal. Returns the final op in the block if the op was a
  /// terminator. An optional 'exclude' filer can be provided to dynamically
  /// exclude some ops from being moved.
  FailureOr<Operation *>
  moveOps(Block *block,
          llvm::function_ref<bool(Operation *)> exclude = nullptr);

  DenseMap<Value, std::string> backedgeMap;

  /// A handle to the state encoder for this machine.
  std::unique_ptr<StateEncoding> encoding;

  /// A mapping from a fsm.variable op to its register.
  llvm::MapVector<VariableOp, seq::CompRegOp> variableToRegister;

  /// A mapping from a fsm.variable op to the output of the mux chain that
  /// calculates its next value.
  llvm::MapVector<VariableOp, mlir::Value> variableToMuxChainOut;

  /// Mapping from a hw port to
  llvm::SmallVector<mlir::Value> outputMuxChainOuts;

  /// A handle to the MachineOp being converted.
  MachineOp machineOp;

  /// A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  /// A handle to the state register of the machine.
  seq::CompRegOp stateReg;

  OpBuilder &b;

  mlir::Value stateMuxChainOut;

  BackedgeBuilder bb;
};
} // namespace

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();

  // Clone all referenced constants into the machine body - constants may have
  // been moved to the machine parent due to the lack of IsolationFromAbove.
  cloneConstantsIntoRegion(machineOp.getBody(), b);

  // 1) Get the port info of the machine and create a new HW module for it.
  SmallVector<hw::PortInfo, 16> ports;
  auto clkRstIdxs = getMachinePortInfo(ports, machineOp, b);
  hwModuleOp =
      hw::HWModuleOp::create(b, loc, machineOp.getSymNameAttr(), ports);
  b.setInsertionPointToStart(hwModuleOp.getBodyBlock());

  // Replace all uses of the machine arguments with the arguments of the
  // newly created HW module.
  for (auto [machineArg, hwModuleArg] :
       llvm::zip(machineOp.getArguments(),
                 hwModuleOp.getBodyBlock()->getArguments())) {
    machineArg.replaceAllUsesWith(hwModuleArg);
  }

  auto clock = hwModuleOp.getBodyBlock()->getArgument(clkRstIdxs.clockIdx);
  auto reset = hwModuleOp.getBodyBlock()->getArgument(clkRstIdxs.resetIdx);

  // 2) Build state and variable registers.

  encoding = std::make_unique<StateEncoding>(b, machineOp, hwModuleOp);
  auto stateType = encoding->getStateType();

  Backedge nextStateWire = bb.get(stateType);

  auto initialStateOp = machineOp.getInitialStateOp();
  stateReg = seq::CompRegOp::create(
      b, loc, nextStateWire, clock, reset,
      /*reset value=*/encoding->encode(initialStateOp), "state_reg",
      /*powerOn value=*/
      seq::createConstantInitialValue(
          b, encoding->encode(initialStateOp).getDefiningOp()));
  stateMuxChainOut = stateReg;

  llvm::DenseMap<VariableOp, Backedge> variableNextStateWires;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto initValueAttr = cast<IntegerAttr>(variableOp.getInitValueAttr());
    Type varType = variableOp.getType();
    auto varLoc = variableOp.getLoc();
    auto nextVariableStateWire = bb.get(varType);
    auto varResetVal = hw::ConstantOp::create(b, varLoc, initValueAttr);
    auto variableReg = seq::CompRegOp::create(
        b, varLoc, nextVariableStateWire, clock, reset, varResetVal,
        b.getStringAttr(variableOp.getName()),
        seq::createConstantInitialValue(b, varResetVal));
    variableToRegister[variableOp] = variableReg;
    variableNextStateWires[variableOp] = nextVariableStateWire;
    variableToMuxChainOut[variableOp] = variableReg;
    // Postpone value replacement until all logic has been created.
    // fsm::UpdateOp's require their target variables to refer to a
    // fsm::VariableOp - if this is not the case, they'll throw an assert.
  }

  // Move any operations at the machine-level scope, excluding state ops,
  // which are handled separately.
  if (failed(moveOps(&machineOp.front(), [](Operation *op) {
        return isa<fsm::StateOp, fsm::VariableOp>(op);
      })))
    return failure();

  // Begin mux chains for outputs
  auto hwPortList = hwModuleOp.getPortList();
  llvm::SmallVector<Backedge> outputBackedges;
  for (auto &port : hwPortList)
    if (port.isOutput())
      outputMuxChainOuts.push_back(Value());

  // 3) Convert states and record their next-state value assignments.
  for (auto state : machineOp.getBody().getOps<StateOp>()) {
    auto stateConvRes = convertState(state);
    if (failed(stateConvRes))
      return failure();
  }

  // 4) Set the input of the state and variable registers to the output of their
  // mux chains.
  nextStateWire.setValue(stateMuxChainOut);
  for (auto [variable, muxChainOut] : variableToMuxChainOut) {
    variableNextStateWires[variable].setValue(muxChainOut);
  }

  // Replace variable values with their register counterparts.
  for (auto [variableOp, variableReg] : variableToRegister)
    variableOp.getResult().replaceAllUsesWith(variableReg);

  // Cast to values to appease builder
  llvm::SmallVector<Value> outputValues;
  for (auto backedge : outputMuxChainOuts) {
    outputValues.push_back(backedge);
  }
  auto *oldOutputOp = hwModuleOp.getBodyBlock()->getTerminator();
  b.setInsertionPointToEnd(oldOutputOp->getBlock());
  oldOutputOp->erase();
  hw::OutputOp::create(b, loc, outputValues);
  machineOp.erase();
  return success();
}

FailureOr<Value>
MachineOpConverter::convertTransitions( // NOLINT(misc-no-recursion)
    StateOp currentState, ArrayRef<TransitionOp> transitions) {
  Value nextState;
  llvm::MapVector<fsm::VariableOp, Value> variableUpdates;
  auto stateCmp =
      comb::ICmpOp::create(b, machineOp.getLoc(), comb::ICmpPredicate::eq,
                           stateReg, encoding->encode(currentState));
  if (transitions.empty()) {
    // Base case
    // State: transition to the current state.
    nextState = encoding->encode(currentState);
  } else {
    // Recursive case - transition to a named state.
    auto transition = cast<fsm::TransitionOp>(transitions.front());
    nextState = encoding->encode(transition.getNextStateOp());
    mlir::Value varUpdateCondition;
    // Action conversion
    if (transition.hasAction()) {
      // Move any ops from the action region to the general scope, excluding
      // variable update ops.
      auto actionMoveOpsRes =
          moveOps(&transition.getAction().front(),
                  [](Operation *op) { return isa<fsm::UpdateOp>(op); });
      if (failed(actionMoveOpsRes))
        return failure();

      // Gather variable updates during the action.
      for (auto updateOp : transition.getAction().getOps<fsm::UpdateOp>()) {
        VariableOp variableOp = updateOp.getVariableOp();
        variableUpdates[variableOp] = updateOp.getValue();
      }
    }

    // Guard conversion
    if (transition.hasGuard()) {
      // Not always taken; recurse and mux between the targeted next state and
      // the recursion result, selecting based on the provided guard.
      auto guardOpRes = moveOps(&transition.getGuard().front());
      if (failed(guardOpRes))
        return failure();

      auto guardOp = cast<ReturnOp>(*guardOpRes);
      assert(guardOp && "guard should be defined");
      auto guard = guardOp.getOperand();
      auto otherNextState =
          convertTransitions(currentState, transitions.drop_front());
      if (failed(otherNextState))
        return failure();
      comb::MuxOp nextStateMux = comb::MuxOp::create(
          b, transition.getLoc(), guard, nextState, *otherNextState, false);
      nextState = nextStateMux;
      varUpdateCondition =
          comb::AndOp::create(b, machineOp.getLoc(), guard, stateCmp);
    } else
      varUpdateCondition = stateCmp;
    // Handle variable updates
    for (auto variableUpdate : variableUpdates) {
      auto muxChainOut = variableToMuxChainOut[variableUpdate.first];
      auto newMuxChainOut =
          comb::MuxOp::create(b, machineOp.getLoc(), varUpdateCondition,
                              variableUpdate.second, muxChainOut, false);
      variableToMuxChainOut[variableUpdate.first] = newMuxChainOut;
    }
  }

  stateMuxChainOut = comb::MuxOp::create(b, machineOp.getLoc(), stateCmp,
                                         nextState, stateMuxChainOut);
  assert(nextState && "next state should be defined");
  return nextState;
}

FailureOr<Operation *>
MachineOpConverter::moveOps(Block *block,
                            llvm::function_ref<bool(Operation *)> exclude) {
  for (auto &op : llvm::make_early_inc_range(*block)) {
    if (!isa<comb::CombDialect, hw::HWDialect, fsm::FSMDialect>(
            op.getDialect())) {
      // Avoid giving unrelated errors about unbound backedges.
      bb.abandon();
      return op.emitOpError()
             << "is unsupported (op from the "
             << op.getDialect()->getNamespace() << " dialect).";
    }
    if (exclude && exclude(&op))
      continue;

    if (op.hasTrait<OpTrait::IsTerminator>())
      return &op;

    op.moveBefore(hwModuleOp.getBodyBlock(), b.getInsertionPoint());
  }
  return nullptr;
}

LogicalResult MachineOpConverter::convertState(StateOp state) {
  // 3.1) Convert the output region by moving the operations into the module
  // scope and gathering the operands of the output op.
  if (!state.getOutput().empty()) {
    auto outputOpRes = moveOps(&state.getOutput().front());
    if (failed(outputOpRes))
      return failure();

    // 3.2) Extend the output mux chains with a comparison on this state
    OutputOp outputOp = cast<fsm::OutputOp>(*outputOpRes);
    auto stateCmp =
        comb::ICmpOp::create(b, machineOp.getLoc(), comb::ICmpPredicate::eq,
                             stateReg, encoding->encode(state));

    for (auto [i, operand] : llvm::enumerate(outputOp.getOperands())) {
      auto muxChainOut = outputMuxChainOuts[i];
      // If this is the first node in the mux chain, just use this value
      // directly as the default
      if (!muxChainOut) {
        outputMuxChainOuts[i] = operand;
        continue;
      }
      auto muxOp = comb::MuxOp::create(b, machineOp.getLoc(), stateCmp, operand,
                                       muxChainOut);
      outputMuxChainOuts[i] = muxOp;
    }
  }

  auto transitions = llvm::SmallVector<TransitionOp>(
      state.getTransitions().getOps<TransitionOp>());
  // 3.3) Convert the transitions and add a case to the next-state mux
  // chain for each
  auto nextStateRes = convertTransitions(state, transitions);
  if (failed(nextStateRes))
    return failure();
  return success();
}
namespace {
struct FSMToCorePass : public circt::impl::ConvertFSMToCoreBase<FSMToCorePass> {
  void runOnOperation() override;
};

void FSMToCorePass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

  b.setInsertionPointToStart(module.getBody());
  // Traverse all machines and convert.
  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {

    // Check validity of the FSM while we can still easily error out
    if (machine->getAttr("stateType")) {
      auto stateType = dyn_cast<TypeAttr>(machine->getAttr("stateType"));
      if (!stateType) {
        machine->emitError("stateType attribute does not name a type");
        signalPassFailure();
        return;
      }
      if (!isa<IntegerType>(stateType.getValue())) {
        machine->emitError("stateType attribute must name an integer type");
        signalPassFailure();
        return;
      }
    }
    for (auto variableOp : machine.front().getOps<fsm::VariableOp>()) {
      if (!isa<IntegerType>(variableOp.getType())) {
        variableOp.emitOpError(
            "only integer variables are currently supported");
        signalPassFailure();
        return;
      }
    }

    MachineOpConverter converter(b, machine);

    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
  }

  // Traverse all machine instances and convert to hw instances.
  llvm::SmallVector<HWInstanceOp> instances;
  module.walk([&](HWInstanceOp instance) { instances.push_back(instance); });
  for (auto instance : instances) {
    auto fsmHWModule =
        module.lookupSymbol<hw::HWModuleOp>(instance.getMachine());
    assert(fsmHWModule &&
           "FSM machine should have been converted to a hw.module");

    b.setInsertionPoint(instance);
    llvm::SmallVector<Value, 4> operands;
    llvm::transform(instance.getOperands(), std::back_inserter(operands),
                    [&](auto operand) { return operand; });
    auto hwInstance = hw::InstanceOp::create(
        b, instance.getLoc(), fsmHWModule, b.getStringAttr(instance.getName()),
        operands, nullptr);
    instance.replaceAllUsesWith(hwInstance);
    instance.erase();
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToCorePass() {
  return std::make_unique<FSMToCorePass>();
}
