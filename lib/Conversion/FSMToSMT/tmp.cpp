//===- FSMToCore.cpp - Convert FSM to HW Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToCore.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

using namespace mlir;
using namespace circt;
using namespace fsm;

/// Get the port info of a FSM machine. Clock and reset port are also added.
namespace {
struct ClkRstIdxs {
  size_t clockIdx;
  size_t resetIdx;
};

// Clones constants implicitly captured by the region, into the region.
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
  // An class for handling state encoding. The class is designed to
  // abstract away how states are selected in case patterns, referred to as
  // values, and used as selection signals for muxes.

public:
  StateEncoding(OpBuilder &b, MachineOp machine, hw::HWModuleOp hwModule);

  // Get the encoded value for a state.
  Value encode(StateOp state);
  // Get the state corresponding to an encoded value.
  StateOp decode(Value value);

  // Returns the type which encodes the state values.
  Type getStateType() { return stateType; }

  // Returns a case pattern which matches the provided state.
  std::unique_ptr<sv::CasePattern> getCasePattern(StateOp state);

protected:
  // Creates a constant value in the module for the given encoded state
  // and records the state value in the mappings. An inner symbol is
  // attached to the wire to avoid it being optimized away.
  // The constant can optionally be assigned behind a sv wire - doing so at this
  // point ensures that constants don't end up behind "_GEN#" wires in the
  // module.
  void setEncoding(StateOp state, Value v, bool wire = false);

  // A mapping between a StateOp and its corresponding encoded value.
  SmallDenseMap<StateOp, Value> stateToValue;

  // A mapping between an encoded value and its corresponding StateOp.
  SmallDenseMap<Value, StateOp> valueToState;

  // A mapping between an encoded value and the source value in the IR.
  SmallDenseMap<Value, Value> valueToSrcValue;

  // The enum type for the states.
  Type stateType;

  OpBuilder &b;
  MachineOp machine;
  hw::HWModuleOp hwModule;
};

StateEncoding::StateEncoding(OpBuilder &b, MachineOp machine,
                             hw::HWModuleOp hwModule)
    : b(b), machine(machine), hwModule(hwModule) {
  Location loc = machine.getLoc();
  llvm::SmallVector<Attribute> stateNames;

  for (auto state : machine.getBody().getOps<StateOp>())
    stateNames.push_back(b.getStringAttr(state.getName()));

  // Create an enum typedef for the states.
  Type rawEnumType =
      hw::EnumType::get(b.getContext(), b.getArrayAttr(stateNames));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&hwModule.getBodyRegion().front());
  // If stateType is explicitly provided, use this - otherwise, calculate the
  // minimum int size that can represent all states
  if (machine->getAttr("stateType"))
    stateType = machine->getAttr("stateType").cast<TypeAttr>().getValue();
  else {
    int numOps = std::distance(machine.getBody().getOps<StateOp>().begin(),
                               machine.getBody().getOps<StateOp>().end());
    // Manual log2
    int width = 1;
    int maxVal = 2;
    while (maxVal < numOps) {
      width++;
      maxVal *= 2;
    }
    stateType = IntegerType::get(machine.getContext(), width);
  }
  int stateValue = 0;
  // And create values for the states
  b.setInsertionPointToStart(&hwModule.getBody().front());
  for (auto state : machine.getBody().getOps<StateOp>()) {
    auto constantOp = b.create<hw::ConstantOp>(loc, stateType, stateValue++);
    setEncoding(state, constantOp,
                /*wire=*/true);
  }
}

// Get the encoded value for a state.
Value StateEncoding::encode(StateOp state) {
  auto it = stateToValue.find(state);
  assert(it != stateToValue.end() && "state not found");
  return it->second;
}
// Get the state corresponding to an encoded value.
StateOp StateEncoding::decode(Value value) {
  auto it = valueToState.find(value);
  assert(it != valueToState.end() && "encoded state not found");
  return it->second;
}

// Returns a case pattern which matches the provided state.
std::unique_ptr<sv::CasePattern> StateEncoding::getCasePattern(StateOp state) {
  // Get the field attribute for the state - fetch it through the encoding.
  auto fieldAttr =
      cast<hw::EnumConstantOp>(valueToSrcValue[encode(state)].getDefiningOp())
          .getFieldAttr();
  return std::make_unique<sv::CaseEnumPattern>(fieldAttr);
}

void StateEncoding::setEncoding(StateOp state, Value v, bool wire) {
  assert(stateToValue.find(state) == stateToValue.end() &&
         "state already encoded");

  Value encodedValue;
  if (wire) {
    auto loc = machine.getLoc();
    auto stateType = getStateType();
    // auto stateEncodingWire = b.create<sv: :RegOp>(
    //     loc, stateType, b.getStringAttr("to_" + state.getName()),
    //     hw::InnerSymAttr::get(state.getNameAttr()));
    encodedValue =
        v; //= b.create<comb::ReplicateOp>(loc, v.getType(), v)->getResult(0);
    // encodedValue = b.create<sv: :ReadInOutOp>(loc, stateEncodingWire);
  } else
    encodedValue = v;
  stateToValue[state] = encodedValue;
  valueToState[encodedValue] = state;
  valueToSrcValue[encodedValue] = v;
}
} // namespace

namespace {
class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp)
      : machineOp(machineOp), b(builder) {}

  // Converts the machine op to a hardware module.
  // 1. Creates a HWModuleOp for the machine op, with the same I/O as the FSM +
  // clk/reset ports.
  // 2. Creates a state register + encodings for the states visible in the
  // machine.
  // 3. Iterates over all states in the machine
  //  3.1. Moves all `comb` logic into the body of the HW module
  //  3.2. Records the SSA value(s) associated to the output ports in the state
  //  3.3. iterates of the transitions of the state
  //    3.3.1. Moves all `comb` logic in the transition guard/action regions to
  //            the body of the HW module.
  //    3.3.2. Creates a case pattern for the transition guard
  //  3.4. Creates a next-state value for the state based on the transition
  //  guards.
  // 4. Assigns next-state values for the states in a case statement on the
  // state reg.
  // 5. Assigns the current-state outputs for the states in a case statement
  // on the state reg.
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

  // Moves operations from 'block' into module scope, failing if any op were
  // deemed illegal. Returns the final op in the block if the op was a
  // terminator. An optional 'exclude' filer can be provided to dynamically
  // exclude some ops from being moved.
  FailureOr<Operation *>
  moveOps(Block *block,
          llvm::function_ref<bool(Operation *)> exclude = nullptr);

  struct CaseMuxItem;
  using StateCaseMapping =
      llvm::SmallDenseMap<StateOp,
                          std::variant<Value, std::shared_ptr<CaseMuxItem>>>;
  struct CaseMuxItem {
    // The target wire to be assigned.
    Backedge wire;

    // The case select signal to be used.
    Value select;

    // A mapping between a state and an assignment within that state.
    // An assignment can either be a value or a nested CaseMuxItem. The latter
    // case will create nested case statements.
    StateCaseMapping assignmentInState;

    // An optional default value to be assigned before the case statement, if
    // the case is not fully specified for all states.
    std::optional<Value> defaultValue = {};
  };

  // Build an SV-based case mux for the given assignments. Assignments are
  // merged into the same case statement. Caller is expected to ensure that the
  // insertion point is within an `always_...` block.
  void
  buildStateCaseMux(llvm::MutableArrayRef<CaseMuxItem> assignments,
                    std::optional<mlir::Value> outerCondition = std::nullopt);

  void printStateCaseMux(CaseMuxItem assignment, int ws);

  DenseMap<Value, std::string> backedgeMap;

  // A handle to the state encoder for this machine.
  std::unique_ptr<StateEncoding> encoding;

  // A deterministic ordering of the states in this machine.
  llvm::SmallVector<StateOp> orderedStates;

  // A mapping from a fsm.variable op to its register.
  llvm::SmallDenseMap<VariableOp, seq::CompRegOp> variableToRegister;

  // A mapping from a fsm.variable op to the output of the mux chain that
  // calculates its next value.
  llvm::SmallDenseMap<VariableOp, mlir::Value> variableToMuxChainOut;

  // Mapping from a hw port to
  llvm::SmallVector<mlir::Value> outputMuxChainOuts;

  // A mapping from a state to variable updates performed during outgoing state
  // transitions.
  llvm::SmallDenseMap<
      /*currentState*/ StateOp,
      llvm::SmallDenseMap<
          /*targetState*/ StateOp,
          llvm::DenseMap</*targetVariable*/ VariableOp, /*targetValue*/ Value>>>
      stateToVariableUpdates;

  // A handle to the MachineOp being converted.
  MachineOp machineOp;

  // A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  // A handle to the state register of the machine.
  seq::CompRegOp stateReg;

  OpBuilder &b;

  mlir::Value stateMuxChainOut;
};
} // namespace

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  if (machineOp.getNumStates() < 2)
    return machineOp.emitOpError() << "expected at least 2 states.";

  // Clone all referenced constants into the machine body - constants may have
  // been moved to the machine parent due to the lack of IsolationFromAbove.
  cloneConstantsIntoRegion(machineOp.getBody(), b);

  // 1) Get the port info of the machine and create a new HW module for it.
  SmallVector<hw::PortInfo, 16> ports;
  auto clkRstIdxs = getMachinePortInfo(ports, machineOp, b);
  hwModuleOp = b.create<hw::HWModuleOp>(loc, machineOp.getSymNameAttr(), ports);
  b.setInsertionPointToStart(hwModuleOp.getBodyBlock());

  // Replace all uses of the machine arguments with the arguments of the
  // new created HW module.
  for (auto args : llvm::zip(machineOp.getArguments(),
                             hwModuleOp.getBodyBlock()->getArguments())) {
    auto machineArg = std::get<0>(args);
    auto hwModuleArg = std::get<1>(args);
    machineArg.replaceAllUsesWith(hwModuleArg);
  }

  auto clock = hwModuleOp.getBodyBlock()->getArgument(clkRstIdxs.clockIdx);
  auto reset = hwModuleOp.getBodyBlock()->getArgument(clkRstIdxs.resetIdx);

  // 2) Build state and variable registers.

  encoding = std::make_unique<StateEncoding>(b, machineOp, hwModuleOp);
  auto stateType = encoding->getStateType();

  BackedgeBuilder bb(b, loc);

  Backedge nextStateWire = bb.get(stateType);

  stateReg = b.create<seq::CompRegOp>(
      loc, nextStateWire, clock, reset,
      /*reset value=*/encoding->encode(machineOp.getInitialStateOp()),
      "state_reg",
      /*powerOn value=*/encoding->encode(machineOp.getInitialStateOp()));

  stateMuxChainOut = stateReg;

  llvm::DenseMap<VariableOp, Backedge> variableNextStateWires;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto initValueAttr = variableOp.getInitValueAttr().dyn_cast<IntegerAttr>();
    if (!initValueAttr)
      return variableOp.emitOpError() << "expected an integer attribute "
                                         "for the initial value.";
    Type varType = variableOp.getType();
    auto varLoc = variableOp.getLoc();
    auto nextVariableStateWire = bb.get(varType);
    backedgeMap.insert(
        std::pair(nextVariableStateWire, "nextVariableStateWire"));
    auto varResetVal = b.create<hw::ConstantOp>(varLoc, initValueAttr);
    auto variableReg = b.create<seq::CompRegOp>(
        varLoc, nextVariableStateWire, clock, reset, varResetVal,
        b.getStringAttr(variableOp.getName()), varResetVal);
    auto varNextState = variableReg;
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
  size_t portIndex = 0;
  llvm::SmallVector<Backedge> outputBackedges;
  for (auto &port : hwPortList) {
    if (!port.isOutput())
      continue;
    auto outputPortType = port.type;
    auto nextOutputStateWire = bb.get(outputPortType);
    outputMuxChainOuts.push_back(nextOutputStateWire);
    outputBackedges.push_back(nextOutputStateWire);
  }

  // 3) Convert states and record their next-state value assignments.
  StateCaseMapping nextStateFromState;
  StateConversionResults stateConvResults;
  for (auto state : machineOp.getBody().getOps<StateOp>()) {
    auto stateConvRes = convertState(state);
    if (failed(stateConvRes))
      return failure();
  }

  nextStateWire.setValue(stateMuxChainOut);
  for (auto varPair : variableToMuxChainOut) {
    variableNextStateWires[varPair.first].setValue(varPair.second);
  }
  for (int i = 0; i < outputBackedges.size(); i++) {
    outputBackedges[i].setValue(outputMuxChainOuts[i]);
  }

  // Replace variable values with their register counterparts.
  for (auto &[variableOp, variableReg] : variableToRegister)
    variableOp.getResult().replaceAllUsesWith(variableReg);

  // Cast to values to appease builder
  llvm::SmallVector<Value> outputValues;
  for (auto backedge : outputBackedges) {
    outputValues.push_back(backedge);
  }
  auto *oldOutputOp = hwModuleOp.getBodyBlock()->getTerminator();
  // b.setInsertionPointToEnd();
  b.setInsertionPointToEnd(oldOutputOp->getBlock());
  oldOutputOp->erase();
  auto op = b.create<hw::OutputOp>(loc, outputValues);
  machineOp.erase();
  return success();
}

FailureOr<Value>
MachineOpConverter::convertTransitions( // NOLINT(misc-no-recursion)
    StateOp currentState, ArrayRef<TransitionOp> transitions) {
  Value nextState;
  DenseMap<fsm::VariableOp, Value> variableUpdates;
  auto stateCmp =
      b.create<comb::ICmpOp>(machineOp.getLoc(), comb::ICmpPredicate::eq,
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

      stateToVariableUpdates[currentState][transition.getNextStateOp()] =
          variableUpdates;
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
      comb::MuxOp nextStateMux = b.create<comb::MuxOp>(
          transition.getLoc(), guard, nextState, *otherNextState, false);
      nextState = nextStateMux;
      varUpdateCondition =
          b.create<comb::AndOp>(machineOp.getLoc(), guard, stateCmp);
    } else
      varUpdateCondition = stateCmp;
    // Handle variable updates
    for (auto variableUpdate : variableUpdates) {
      auto muxChainOut = variableToMuxChainOut[variableUpdate.first];
      auto newMuxChainOut =
          b.create<comb::MuxOp>(machineOp.getLoc(), varUpdateCondition,
                                variableUpdate.second, muxChainOut, false);
      variableToMuxChainOut[variableUpdate.first] = newMuxChainOut;
    }
  }

  stateMuxChainOut = b.create<comb::MuxOp>(machineOp.getLoc(), stateCmp,
                                           nextState, stateMuxChainOut);
  assert(nextState && "next state should be defined");
  return nextState;
}

FailureOr<Operation *>
MachineOpConverter::moveOps(Block *block,
                            llvm::function_ref<bool(Operation *)> exclude) {
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

    op.moveBefore(hwModuleOp.getBodyBlock(), b.getInsertionPoint());
  }
  return nullptr;
}

FailureOr<MachineOpConverter::StateConversionResult>
MachineOpConverter::convertState(StateOp state) {
  MachineOpConverter::StateConversionResult res;

  // 3.1) Convert the output region by moving the operations into the module
  // scope and gathering the operands of the output op.
  if (!state.getOutput().empty()) {
    auto outputOpRes = moveOps(&state.getOutput().front());
    if (failed(outputOpRes))
      return failure();

    OutputOp outputOp = cast<fsm::OutputOp>(*outputOpRes);
    // TODO: two of these, dedup - one in convertTransitions too
    auto stateCmp =
        b.create<comb::ICmpOp>(machineOp.getLoc(), comb::ICmpPredicate::eq,
                               stateReg, encoding->encode(state));

    for (int i = 0; i < outputOp->getNumOperands(); i++) {
      auto muxChainOut = outputMuxChainOuts[i];
      auto muxOp = b.create<comb::MuxOp>(machineOp->getLoc(), stateCmp,
                                         outputOp->getOperand(i), muxChainOut);
      outputMuxChainOuts[i] = muxOp;
    }
    res.outputs = outputOp.getOperands(); // 3.2
  }

  auto transitions = llvm::SmallVector<TransitionOp>(
      state.getTransitions().getOps<TransitionOp>());
  // 3.3, 3.4) Convert the transitions and record the next-state value
  // derived from the transitions being selected in a priority-encoded manner.
  auto nextStateRes = convertTransitions(state, transitions);
  if (failed(nextStateRes))
    return failure();
  res.nextState = *nextStateRes;
  return res;
}
namespace {
struct FSMToCorePass : public ConvertFSMToCoreBase<FSMToCorePass> {
  void runOnOperation() override;
};

void FSMToCorePass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

  b.setInsertionPointToStart(module.getBody());
  // Traverse all machines and convert.
  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
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
    auto hwInstance = b.create<hw::InstanceOp>(
        instance.getLoc(), fsmHWModule, b.getStringAttr(instance.getName()),
        operands, nullptr);
    instance.replaceAllUsesWith(hwInstance);
    instance.erase();
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToCorePass() {
  return std::make_unique<FSMToCorePass>();
}