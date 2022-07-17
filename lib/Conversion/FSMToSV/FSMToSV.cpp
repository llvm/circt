//===- FSMToSV.cpp - Convert FSM to HW and SV Dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToSV.h"
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
namespace {
struct ClkRstIdxs {
  size_t clockIdx;
  size_t resetIdx;
};
} // namespace
static ClkRstIdxs getMachinePortInfo(SmallVectorImpl<hw::PortInfo> &ports,
                                     MachineOp machine, OpBuilder &b) {
  // Get the port info of the machine inputs and outputs.
  machine.getHWPortInfo(ports);
  ClkRstIdxs specialPorts;

  // Add clock port.
  hw::PortInfo clock;
  clock.name = b.getStringAttr("clk");
  clock.direction = hw::PortDirection::INPUT;
  clock.type = b.getI1Type();
  clock.argNum = machine.getNumArguments();
  ports.push_back(clock);
  specialPorts.clockIdx = clock.argNum;

  // Add reset port.
  hw::PortInfo reset;
  reset.name = b.getStringAttr("rst");
  reset.direction = hw::PortDirection::INPUT;
  reset.type = b.getI1Type();
  reset.argNum = machine.getNumArguments() + 1;
  ports.push_back(reset);
  specialPorts.resetIdx = reset.argNum;

  return specialPorts;
}

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
  b.setInsertionPoint(hwModule);
  auto typeScope = b.create<hw::TypeScopeOp>(
      loc, b.getStringAttr(hwModule.getName() + "_enum_typedecls"));
  typeScope.getBodyRegion().push_back(new Block());

  b.setInsertionPointToStart(&typeScope.getBodyRegion().front());
  auto typedeclEnumType = b.create<hw::TypedeclOp>(
      loc, b.getStringAttr(hwModule.getName() + "_state_t"),
      TypeAttr::get(rawEnumType), nullptr);

  stateType = hw::TypeAliasType::get(
      SymbolRefAttr::get(typeScope.getSymNameAttr(),
                         {FlatSymbolRefAttr::get(typedeclEnumType)}),
      rawEnumType);

  // And create enum values for the states
  b.setInsertionPointToStart(&hwModule.getBody().front());
  for (auto state : machine.getBody().getOps<StateOp>()) {
    auto fieldAttr = hw::EnumFieldAttr::get(
        loc, b.getStringAttr(state.getName()), stateType);
    auto enumConstantOp = b.create<hw::EnumConstantOp>(
        loc, fieldAttr.getType().getValue(), fieldAttr);
    setEncoding(state, enumConstantOp,
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
    auto stateEncodingWire = b.create<sv::WireOp>(
        loc, stateType, b.getStringAttr("to_" + state.getName()),
        /*inner_sym=*/state.getNameAttr());
    b.create<sv::AssignOp>(loc, stateEncodingWire, v);
    encodedValue = b.create<sv::ReadInOutOp>(loc, stateEncodingWire);
  } else
    encodedValue = v;
  stateToValue[state] = encodedValue;
  valueToState[encodedValue] = state;
  valueToSrcValue[encodedValue] = v;
}

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

  // Returns the value that must be assigned to the hw output ports of this
  // machine.
  llvm::SmallVector<Value>
  getOutputAssignments(StateConversionResults &stateConvResults);

  // Moves operations from 'block' into module scope, failing if any op were
  // deemed illegal. Returns the final op in the block if the op was a
  // terminator. An optional 'exclude' filer can be provided to dynamically
  // exclude some ops from being moved.
  FailureOr<Operation *>
  moveOps(Block *block,
          llvm::function_ref<bool(Operation *)> exclude = nullptr);

  // Build a SV case-based combinational mux the values provided in
  // 'stateToValue' to a retured wire.
  // 'stateToValue' being a list implies that multiple muxes can be emitted at
  // once, avoiding bloating the IR with a case statement for every muxed value.
  // A wire is returned for each srcMap provided.
  // 'nameF' can be provided to specify the name of the output wire created for
  // each source map.
  llvm::SmallVector<Value>
  buildStateCaseMux(Location loc, Value sel,
                    llvm::ArrayRef<llvm::SmallDenseMap<StateOp, Value>> srcMaps,
                    llvm::function_ref<StringAttr(size_t)> nameF = {});

  // A handle to the state encoder for this machine.
  std::unique_ptr<StateEncoding> encoding;

  // A deterministic ordering of the states in this machine.
  llvm::SmallVector<StateOp> orderedStates;

  // A mapping from a state op to its next-state value.
  llvm::SmallDenseMap<StateOp, Value> nextStateFromState;

  // A handle to the MachineOp being converted.
  MachineOp machineOp;

  // A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  // A handle to the state register of the machine.
  seq::CompRegOp stateReg;

  OpBuilder &b;
};

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

    op.moveBefore(&hwModuleOp.front(), b.getInsertionPoint());
  }
  return nullptr;
}

llvm::SmallVector<Value> MachineOpConverter::buildStateCaseMux(
    Location loc, Value sel,
    llvm::ArrayRef<llvm::SmallDenseMap<StateOp, Value>> srcMaps,
    llvm::function_ref<StringAttr(size_t)> nameF) {
  sv::CaseOp caseMux;
  auto caseMuxCtor = [&]() {
    caseMux = b.create<sv::CaseOp>(loc, CaseStmtType::CaseStmt, sel,
                                   /*numCases=*/machineOp.getNumStates(),
                                   [&](size_t caseIdx) {
                                     StateOp state = orderedStates[caseIdx];
                                     return encoding->getCasePattern(state);
                                   });
  };
  b.create<sv::AlwaysCombOp>(loc, caseMuxCtor);

  llvm::SmallVector<Value> dsts;
  // note: cannot use llvm::enumerate, makes the underlying iterator const.
  size_t idx = 0;
  for (auto srcMap : srcMaps) {
    auto valueType = srcMap.begin()->second.getType();
    StringAttr name;
    if (nameF)
      name = nameF(idx);

    auto dst = b.create<sv::RegOp>(loc, valueType, name);
    OpBuilder::InsertionGuard g(b);
    for (auto [caseInfo, stateOp] :
         llvm::zip(caseMux.getCases(), orderedStates)) {
      b.setInsertionPointToEnd(caseInfo.block);
      b.create<sv::BPAssignOp>(loc, dst, srcMap[stateOp]);
    }
    dsts.push_back(dst);
    idx++;
  }
  return dsts;
}

LogicalResult MachineOpConverter::dispatch() {
  if (auto varOps = machineOp.front().getOps<VariableOp>(); !varOps.empty())
    return (*varOps.begin())->emitOpError()
           << "FSM variables not yet supported for SV "
              "lowering.";

  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  if (machineOp.getNumStates() < 2)
    return machineOp.emitOpError() << "expected at least 2 states.";

  // 1) Get the port info of the machine and create a new HW module for it.
  SmallVector<hw::PortInfo, 16> ports;
  auto clkRstIdxs = getMachinePortInfo(ports, machineOp, b);
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

  auto clock = hwModuleOp.front().getArgument(clkRstIdxs.clockIdx);
  auto reset = hwModuleOp.front().getArgument(clkRstIdxs.resetIdx);

  // 2) Build state register.
  encoding = std::make_unique<StateEncoding>(b, machineOp, hwModuleOp);
  auto stateType = encoding->getStateType();

  BackedgeBuilder bb(b, loc);
  auto nextStateBackedge = bb.get(stateType);
  stateReg = b.create<seq::CompRegOp>(
      loc, stateType, nextStateBackedge, clock, "state_reg", reset,
      /*reset value=*/encoding->encode(machineOp.getInitialStateOp()), nullptr);

  // Move any operations at the machine-level scope, excluding state ops, which
  // are handled separately.
  if (failed(moveOps(&machineOp.front(),
                     [](Operation *op) { return isa<fsm::StateOp>(op); }))) {
    bb.abandon();
    return failure();
  }

  // 3) Convert states and record their next-state value.
  StateConversionResults stateConvResults;
  for (auto state : machineOp.getBody().getOps<StateOp>()) {
    auto stateConvRes = convertState(state);
    if (failed(stateConvRes)) {
      bb.abandon();
      return failure();
    }
    stateConvResults[state] = stateConvRes.getValue();
    orderedStates.push_back(state);
    nextStateFromState[state] = stateConvRes.getValue().nextState;
  }

  // 4/5) Create next-state maps for each output and the next-state signal in a
  // format suitable for creating a case mux.
  llvm::SmallVector<llvm::SmallDenseMap<StateOp, Value>, 4> nextStateMaps;
  nextStateMaps.push_back(nextStateFromState);
  for (size_t portIndex = 0; portIndex < machineOp.getNumResults();
       portIndex++) {
    auto &nsmap = nextStateMaps.emplace_back();
    for (auto &state : orderedStates)
      nsmap[state] = stateConvResults[state].outputs[portIndex];
  }

  // Materialize the case mux. We do this in a single call to have a single
  // always_comb block.
  auto stateCaseMuxes = buildStateCaseMux(
      machineOp.getLoc(), stateReg, nextStateMaps, [&](size_t idx) {
        if (idx == 0)
          return b.getStringAttr("next_state");

        return b.getStringAttr("output_" + std::to_string(idx - 1));
      });

  nextStateBackedge.setValue(b.create<sv::ReadInOutOp>(loc, stateCaseMuxes[0]));

  llvm::SmallVector<Value> outputPortAssignments;
  for (auto outputMux : llvm::makeArrayRef(stateCaseMuxes).drop_front())
    outputPortAssignments.push_back(
        b.create<sv::ReadInOutOp>(machineOp.getLoc(), outputMux));

  // Delete the default created output op and replace it with the output muxes.
  auto *oldOutputOp = hwModuleOp.front().getTerminator();
  b.create<hw::OutputOp>(loc, outputPortAssignments);
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

  assert(nextState && "next state should be defined");
  return nextState;
}

llvm::SmallVector<Value>
MachineOpConverter::getOutputAssignments(StateConversionResults &convResults) {

  // One for each output port.
  llvm::SmallVector<llvm::SmallDenseMap<StateOp, Value>> outputPortValues(
      machineOp.getNumResults());
  for (auto &state : orderedStates) {
    for (size_t portIndex = 0; portIndex < machineOp.getNumResults();
         portIndex++)
      outputPortValues[portIndex][state] =
          convResults[state].outputs[portIndex];
  }

  llvm::SmallVector<Value> outputPortAssignments;

  auto outputMuxes = buildStateCaseMux(
      machineOp.getLoc(), stateReg, outputPortValues, [&](size_t idx) {
        return b.getStringAttr("output_" + std::to_string(idx));
      });

  for (auto outputMux : outputMuxes)
    outputPortAssignments.push_back(
        b.create<sv::ReadInOutOp>(machineOp.getLoc(), outputMux));

  return outputPortAssignments;
}

FailureOr<MachineOpConverter::StateConversionResult>
MachineOpConverter::convertState(StateOp state) {
  MachineOpConverter::StateConversionResult res;

  // 3.1) Convert the output region by moving the operations into the module
  // scope and gathering the operands of the output op.
  auto outputOpRes = moveOps(&state.output().front());
  if (failed(outputOpRes))
    return failure();

  OutputOp outputOp = cast<fsm::OutputOp>(outputOpRes.getValue());
  res.outputs = outputOp.getOperands(); // 3.2

  auto transitions = llvm::SmallVector<TransitionOp>(
      state.transitions().getOps<TransitionOp>());
  // 3.3, 3.4) Convert the transitions and record the next-state value
  // derived from the transitions being selected in a priority-encoded manner.
  auto nextStateRes = convertTransitions(state, transitions);
  if (failed(nextStateRes))
    return failure();
  res.nextState = nextStateRes.getValue();
  return res;
}

struct FSMToSVPass : public ConvertFSMToSVBase<FSMToSVPass> {
  void runOnOperation() override;
};

void FSMToSVPass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

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
    auto fsmHWModule = module.lookupSymbol<hw::HWModuleOp>(instance.machine());
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

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSVPass() {
  return std::make_unique<FSMToSVPass>();
}
