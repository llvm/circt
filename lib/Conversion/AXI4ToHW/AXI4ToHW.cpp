//===- AXI4ToHW.cpp - AXI4 to HW conversion pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers an AXI4 network specification to a concrete RTL description.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AXI4ToHW.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <array>

namespace circt {
#define GEN_PASS_DEF_AXI4TOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

// axi4.ports split into fifteen signals - a valid, a ready and a payload for
// each channel

namespace {
/// An AXI4 channel's name, and whether a manager drives its payload.
struct ChannelInfo {
  AXI4Channel channel;
  StringLiteral name;
  bool isRequest;
};
} // namespace

static constexpr ChannelInfo kChannels[] = {
    {AXI4Channel::AW, StringLiteral("aw"), true},
    {AXI4Channel::W, StringLiteral("w"), true},
    {AXI4Channel::B, StringLiteral("b"), false},
    {AXI4Channel::AR, StringLiteral("ar"), true},
    {AXI4Channel::R, StringLiteral("r"), false}};

namespace {
/// One of the signals an `!axi4.port` explodes into.
struct SignalInfo {
  std::string suffix;
  Type type;
  bool managerDrives;
};
} // namespace

/// The signals `port` explodes into. A channel's ready travels against its
/// payload, so the end that drives one receives the other.
static SmallVector<SignalInfo, 15> portSignals(PortType port) {
  Type i1 = IntegerType::get(port.getContext(), 1);
  SmallVector<SignalInfo, 15> signals;
  for (const ChannelInfo &info : kChannels) {
    Type payload = getChannelPayloadType(port, info.channel);
    signals.push_back(
        {(Twine("_") + info.name).str(), payload, info.isRequest});
    signals.push_back(
        {(Twine("_") + info.name + "valid").str(), i1, info.isRequest});
    signals.push_back(
        {(Twine("_") + info.name + "ready").str(), i1, !info.isRequest});
  }
  return signals;
}

/// The types of `ports`, in order.
static SmallVector<Type> portTypes(ArrayRef<hw::PortInfo> ports) {
  return llvm::map_to_vector(
      ports, [](const hw::PortInfo &port) { return port.type; });
}

//===----------------------------------------------------------------------===//
// Bridges
//===----------------------------------------------------------------------===//

namespace {
/// Placeholder clock and reset for struct<->port ops that are temporarily
/// materialized between PortConverter invocations
class FillerDomain {
public:
  /// The filler for `block`, created on first use.
  std::pair<Value, Value> get(Block *block);
  /// Whether `value` is a placeholder, and so carries no domain of its own.
  bool isFiller(Value value) const;
  /// Erase the fillers that are no longer used (should erase all fillers at the
  /// end of the run).
  void eraseUnused();

private:
  DenseMap<Block *, std::pair<Value, Value>> fillers;
};
} // namespace

std::pair<Value, Value> FillerDomain::get(Block *block) {
  auto it = fillers.find(block);
  if (it != fillers.end())
    return it->second;

  ImplicitLocOpBuilder b(block->getParentOp()->getLoc(), block, block->begin());
  std::pair<Value, Value> filler{
      seq::ConstClockOp::create(b, seq::ClockConst::Low),
      hw::ConstantOp::create(b, APInt(1, 0))};
  fillers.insert({block, filler});
  return filler;
}

bool FillerDomain::isFiller(Value value) const {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  auto it = fillers.find(op->getBlock());
  if (it == fillers.end())
    return false;
  return value == it->second.first || value == it->second.second;
}

void FillerDomain::eraseUnused() {
  for (auto &[clock, reset] : llvm::make_second_range(fillers)) {
    if (clock.use_empty())
      clock.getDefiningOp()->erase();
    if (reset.use_empty())
      reset.getDefiningOp()->erase();
  }
}

/// Report two conversion ops connected by a port but in different domains.
static LogicalResult emitDomainCrossing(Operation *op, Operation *other,
                                        StringRef domain) {
  auto diag = op->emitOpError()
              << "is in a different " << domain << " domain to the '"
              << other->getName().getStringRef() << "' connected to it";
  diag.attachNote(other->getLoc()) << "connected operation here";
  return failure();
}

/// Wire through and erase every back-to-back bridge pair - errors if there's a
/// domain crossing.
static LogicalResult annihilateBridges(ModuleOp module, FillerDomain &filler) {
  SmallVector<ChannelStructsToPortOp> toPortOps;
  module.walk([&](ChannelStructsToPortOp op) { toPortOps.push_back(op); });

  for (ChannelStructsToPortOp toPort : toPortOps) {
    if (!toPort.getPort().hasOneUse())
      continue;
    auto fromPort =
        dyn_cast<PortToChannelStructsOp>(*toPort.getPort().user_begin());
    if (!fromPort)
      continue;

    // Complain if we have two non-filler converter back to back in different
    // domains
    if (!filler.isFiller(toPort.getClock()) &&
        !filler.isFiller(fromPort.getClock())) {
      if (toPort.getClock() != fromPort.getClock())
        return emitDomainCrossing(fromPort, toPort, "clock");
      if (toPort.getReset() != fromPort.getReset())
        return emitDomainCrossing(fromPort, toPort, "reset");
    }

    // Each op returns the signals the other was given: the manager-driven ones
    // one way and the subordinate-driven ones the other. Both operand lists
    // drop the clock and the reset, and `fromPort` also drops its port operand
    for (auto [result, operand] : llvm::zip_equal(
             fromPort.getResults(), toPort.getOperands().drop_front(2)))
      result.replaceAllUsesWith(operand);
    for (auto [result, operand] :
         llvm::zip_equal(toPort.getResults().drop_front(),
                         fromPort.getOperands().drop_front(3)))
      result.replaceAllUsesWith(operand);

    fromPort.erase();
    toPort.erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Port conversion
//===----------------------------------------------------------------------===//

namespace {
/// Lowers an `!axi4.port` module port to a payload, valid and ready port per
/// channel, bridged to the original port value by a channel struct op.
class AXI4PortConversion : public hw::PortConversion {
public:
  AXI4PortConversion(hw::PortConverterImpl &converter, hw::PortInfo origPort,
                     FillerDomain &filler)
      : PortConversion(converter, origPort), filler(filler),
        portType(cast<PortType>(origPort.type)) {}

protected:
  void buildInputSignals() override;
  void buildOutputSignals() override;
  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  /// Whether this module is the manager on the port, and so drives the signals
  /// a manager drives.
  bool isManager() const {
    return origPort.dir == hw::ModulePort::Direction::Output;
  }
  /// Create the exploded ports this module takes, and return their values.
  SmallVector<Value> createInputPorts();
  /// Create an output port for each value in `values` (and connect
  /// accordingly).
  void createOutputPorts(ValueRange values);
  /// The types of the exploded signals this module drives.
  SmallVector<Type> drivenTypes();
  /// The signals the new instance drives for this port, in signal order. They
  /// come back as backedges because the instance is not built yet.
  SmallVector<Value> instanceDriven(ArrayRef<Backedge> newResults);
  /// The number of outstanding writes/reads that can concurrently be in flight
  /// down this port
  std::array<NamedAttribute, 2> concurrency(Builder &b);

  FillerDomain &filler;
  PortType portType;
  /// The generated ports, split by this module's own direction, in AXI signal
  /// order
  SmallVector<hw::PortInfo> inputPorts, outputPorts;
};

class AXI4PortConversionBuilder : public hw::PortConversionBuilder {
public:
  AXI4PortConversionBuilder(hw::PortConverterImpl &converter,
                            FillerDomain &filler)
      : PortConversionBuilder(converter), filler(filler) {}

  FailureOr<std::unique_ptr<hw::PortConversion>>
  build(hw::PortInfo port) override {
    if (isa<PortType>(port.type))
      return {std::make_unique<AXI4PortConversion>(converter, port, filler)};
    return PortConversionBuilder::build(port);
  }

private:
  FillerDomain &filler;
};
} // namespace

SmallVector<Value> AXI4PortConversion::createInputPorts() {
  SmallVector<Value> values;
  for (const SignalInfo &signal : portSignals(portType)) {
    if (signal.managerDrives == isManager())
      continue;
    hw::PortInfo &port = inputPorts.emplace_back();
    values.push_back(
        converter.createNewInput(origPort, signal.suffix, signal.type, port));
  }
  return values;
}

void AXI4PortConversion::createOutputPorts(ValueRange values) {
  for (const SignalInfo &signal : portSignals(portType)) {
    if (signal.managerDrives != isManager())
      continue;
    hw::PortInfo &port = outputPorts.emplace_back();
    Value value = values.empty() ? Value{} : values[outputPorts.size() - 1];
    converter.createNewOutput(origPort, signal.suffix, signal.type, value,
                              port);
  }
}

SmallVector<Type> AXI4PortConversion::drivenTypes() {
  SmallVector<Type> types;
  for (const SignalInfo &signal : portSignals(portType))
    if (signal.managerDrives == isManager())
      types.push_back(signal.type);
  return types;
}

SmallVector<Value>
AXI4PortConversion::instanceDriven(ArrayRef<Backedge> newResults) {
  return llvm::map_to_vector(outputPorts,
                             [&](const hw::PortInfo &port) -> Value {
                               return newResults[port.argNum];
                             });
}

/// Build the corresponding ports for an !axi4.port input
void AXI4PortConversion::buildInputSignals() {
  // This hook is called when an !axi.port is an input to a module, so we're in
  // a subordinate
  SmallVector<Value> inputs = createInputPorts();

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body) {
    createOutputPorts({});
    return;
  }

  // Turn the !axi.port value that was already being digested into a set of
  // signals
  auto [clock, reset] = filler.get(body);
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands, inputs);
  SmallVector<Type> resultTypes{portType};
  llvm::append_range(resultTypes, drivenTypes());

  ImplicitLocOpBuilder b(origPort.loc, body->getTerminator());
  auto toPort = ChannelStructsToPortOp::create(b, resultTypes, operands);
  body->getArgument(origPort.argNum).replaceAllUsesWith(toPort.getPort());
  createOutputPorts(toPort.getResults().drop_front());
}

/// Build the corresponding ports for an !axi4.port output
void AXI4PortConversion::buildOutputSignals() {
  // This hook is called when an !axi.port is an output of a module, so we're in
  // a manager
  SmallVector<Value> inputs = createInputPorts();

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body) {
    createOutputPorts({});
    return;
  }

  // Turn the !axi.port value that was already being driven into a set of
  // signals
  auto [clock, reset] = filler.get(body);
  Operation *terminator = body->getTerminator();
  SmallVector<Value> operands{clock, reset,
                              terminator->getOperand(origPort.argNum)};
  llvm::append_range(operands, inputs);

  ImplicitLocOpBuilder b(origPort.loc, terminator);
  auto fromPort = PortToChannelStructsOp::create(b, drivenTypes(), operands,
                                                 concurrency(b));
  createOutputPorts(fromPort.getResults());
}

std::array<NamedAttribute, 2> AXI4PortConversion::concurrency(Builder &b) {
  return {b.getNamedAttr("concurrent_writes",
                         b.getI32IntegerAttr(portType.getOutstandingWrites())),
          b.getNamedAttr("concurrent_reads",
                         b.getI32IntegerAttr(portType.getOutstandingReads()))};
}

// Map to the newly created ports on the instances of the modified subordinate
void AXI4PortConversion::mapInputSignals(OpBuilder &b, Operation *inst,
                                         Value instValue,
                                         SmallVectorImpl<Value> &newOperands,
                                         ArrayRef<Backedge> newResults) {
  // The instance takes a port: convert it into the signals the new instance
  // takes, driven by the ones it produces.
  auto [clock, reset] = filler.get(inst->getBlock());
  SmallVector<Value> operands{clock, reset, instValue};
  llvm::append_range(operands, instanceDriven(newResults));

  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  auto fromPort = PortToChannelStructsOp::create(
      builder, portTypes(inputPorts), operands, concurrency(builder));

  for (auto [port, value] : llvm::zip_equal(inputPorts, fromPort.getResults()))
    newOperands[port.argNum] = value;
}

// Map to the newly created ports on the instances of the modified manager
void AXI4PortConversion::mapOutputSignals(OpBuilder &b, Operation *inst,
                                          Value instValue,
                                          SmallVectorImpl<Value> &newOperands,
                                          ArrayRef<Backedge> newResults) {
  // The instance produces a port: convert the signals the new instance
  // produces into one, and feed it back the ones it takes.
  auto [clock, reset] = filler.get(inst->getBlock());
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands, instanceDriven(newResults));
  SmallVector<Type> resultTypes{portType};
  llvm::append_range(resultTypes, portTypes(inputPorts));

  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  auto toPort = ChannelStructsToPortOp::create(builder, resultTypes, operands);

  instValue.replaceAllUsesWith(toPort.getPort());
  for (auto [port, value] :
       llvm::zip_equal(inputPorts, toPort.getResults().drop_front()))
    newOperands[port.argNum] = value;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Report a port this pass has no signals to wire to.
static LogicalResult checkLowerable(Value port) {
  if (!isa<PortType>(port.getType()) || !port.use_empty())
    return success();
  return mlir::emitError(port.getLoc())
         << "AXI4 port has no uses, so cannot be lowered";
}

/// Warn about each bodyless module we are about to change the ports of - its
/// implementation is not ours to rewrite, so changing them silently would
/// leave it mismatched against whatever supplies it.
static void warnPortsChanging(ModuleOp module) {
  for (auto mod : module.getOps<hw::HWMutableModuleLike>()) {
    if (mod.getBodyBlock())
      continue;
    SmallVector<StringRef> names;
    for (const hw::PortInfo &port : mod.getPortList())
      if (isa<PortType>(port.type))
        names.push_back(port.getName());
    if (names.empty())
      continue;

    auto diag = mod->emitWarning()
                << "lowering AXI4 port" << (names.size() == 1 ? " " : "s ");
    llvm::interleaveComma(names, diag,
                          [&](StringRef name) { diag << "'" << name << "'"; });
    diag << " changes the ports of this module; its implementation must match "
            "the new port list";
  }
}

namespace {
struct AXI4ToHWPass : public circt::impl::AXI4ToHWBase<AXI4ToHWPass> {
  void runOnOperation() override;
};
} // namespace

void AXI4ToHWPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool anyFailed = false;

  // Reject what cannot be lowered before mutating anything.
  module.walk([&](Operation *op) {
    if (isa<AbstractManagerOp, AbstractSubordinateOp>(op)) {
      op->emitOpError("models an endpoint with no RTL, so cannot be lowered");
      anyFailed = true;
    } else if (isa<XbarOp>(op)) {
      op->emitOpError("lowering is not yet implemented");
      anyFailed = true;
    }

    for (Value result : op->getResults())
      anyFailed |= failed(checkLowerable(result));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          anyFailed |= failed(checkLowerable(arg));
  });
  if (anyFailed)
    return signalPassFailure();

  warnPortsChanging(module);

  FillerDomain filler;
  hw::InstanceGraph &instanceGraph = getAnalysis<hw::InstanceGraph>();
  for (auto mod : module.getOps<hw::HWMutableModuleLike>())
    if (failed(hw::PortConverter<AXI4PortConversionBuilder>(instanceGraph, mod,
                                                            filler)
                   .run()))
      return signalPassFailure();

  if (failed(annihilateBridges(module, filler)))
    return signalPassFailure();
  filler.eraseUnused();

  // A bridge that failed to pair up would carry a filler clock into the output.
  module.walk([&](Operation *op) {
    if (isa_and_nonnull<AXI4Dialect>(op->getDialect())) {
      op->emitOpError("could not be lowered to HW");
      anyFailed = true;
    }
  });
  if (anyFailed)
    signalPassFailure();
}
