//===- ESILowerPorts.cpp - Lower ESI ports pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

/// Return a attribute with the specified suffix appended.
static StringAttr appendToRtlName(StringAttr base, const Twine &suffix) {
  auto *context = base.getContext();
  return StringAttr::get(context, base.getValue() + suffix);
}

// Returns either the string dialect attr stored in 'op' going by the name
// 'attrName' or 'def' if the attribute doesn't exist in 'op'.
inline static StringRef getStringAttributeOr(Operation *op, StringRef attrName,
                                             StringRef def) {
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (attr)
    return attr.getValue();
  return def;
}

namespace {
class SignalingStandard;

/// Responsible for: lowering module ports, updating the module body, and
/// updating said modules instances.
class ChannelRewriter {
public:
  ChannelRewriter(hw::HWMutableModuleLike mod)
      : inSuffix(getStringAttributeOr(mod, extModPortInSuffix, "")),
        outSuffix(getStringAttributeOr(mod, extModPortOutSuffix, "")),
        validSuffix(getStringAttributeOr(mod, extModPortValidSuffix, "_valid")),
        readySuffix(getStringAttributeOr(mod, extModPortReadySuffix, "_ready")),
        rdenSuffix(getStringAttributeOr(mod, extModPortRdenSuffix, "_rden")),
        emptySuffix(getStringAttributeOr(mod, extModPortEmptySuffix, "_empty")),
        mod(mod),
        flattenStructs(mod->hasAttr(extModPortFlattenStructsAttrName)),
        body(nullptr), foundEsiPorts(false) {

    if (mod->getNumRegions() == 1 && mod->getRegion(0).hasOneBlock())
      body = &mod->getRegion(0).front();
  }

  /// Convert all input and output ChannelTypes into the specified wire-level
  /// signaling standard. Try not to change the order and materialize ops in
  /// reasonably intuitive locations. Will modify the module and body only if
  /// one exists.
  LogicalResult rewriteChannelsOnModule();

  /// Update an instance pointing to this module. Uses the bookkeeping
  /// information stored in this class to ease the update instead of recreating
  /// the algorithm which did the mapping initially.
  void updateInstance(InstanceOp inst);

  hw::HWMutableModuleLike getModule() const { return mod; }
  Block *getBody() const { return body; }

  /// These two methods take care of allocating new ports in the correct place
  /// based on the position of 'origPort'. The new port is based on the original
  /// name and suffix. The specification for the new port is given by `newPort`
  /// and is recorded internally. Any changes to 'newPort' after calling this
  /// will not be reflected in the modules new port list.
  Value createNewInput(PortInfo origPort, const Twine &suffix, Type type,
                       PortInfo &newPort);
  /// Same as above. 'output' is the value fed into the new port and is required
  /// if 'body' is non-null. Important note: cannot be a backedge which gets
  /// replaced since this isn't attached to an op until later in the pass.
  void createNewOutput(PortInfo origPort, const Twine &suffix, Type type,
                       Value output, PortInfo &newPort);

  bool shouldFlattenStructs() const { return flattenStructs; }

  /// Some external modules use unusual port naming conventions. Since we want
  /// to avoid needing to write wrappers, provide some flexibility in the naming
  /// convention.
  const StringRef inSuffix, outSuffix;
  const StringRef validSuffix, readySuffix, rdenSuffix, emptySuffix;

private:
  hw::HWMutableModuleLike mod;
  // Does the module demand that we break out all the struct fields into
  // individual fields?
  bool flattenStructs;
  // If the module has a block and it wants to be modified, this'll be non-null.
  Block *body;
  // Did we find an ESI port?
  bool foundEsiPorts;
  // Keep around a reference to the specific signaling standard classes to
  // facilitate updating the instance ops. Indexed by the original port
  // location.
  SmallVector<std::unique_ptr<SignalingStandard>> loweredInputs;
  SmallVector<std::unique_ptr<SignalingStandard>> loweredOutputs;

  // Tracking information to modify the module. Populated by the
  // 'createNew(Input|Output)' methods. Not needed by `updateInstance`, so we
  // can clear them once the module ports have been modified. Default length is
  // 0 to save memory since we'll be keeping this around for later use.
  SmallVector<std::pair<unsigned, PortInfo>, 0> newInputs;
  SmallVector<std::pair<unsigned, PortInfo>, 0> newOutputs;
  SmallVector<Value, 0> newOutputValues;
};

/// Base class for the signaling standard of a particular port. Abstracts the
/// details of a particular signaling standard from the port layout. Subclasses
/// keep around port mapping information to use when updating instances.
class SignalingStandard {
public:
  SignalingStandard(ChannelRewriter &rewriter, PortInfo origPort)
      : rewriter(rewriter), body(rewriter.getBody()), origPort(origPort) {}
  virtual ~SignalingStandard() = default;

  // Lower the specified (possibly high-level ESI) port into a wire-level
  // signaling protocol. The two virtual methods 'build*Signals' should be
  // overridden by subclasses. They should use the 'create*' methods in
  // 'ChannelRewriter' to create the necessary ports.
  void lowerPort() {
    if (origPort.direction == PortDirection::OUTPUT)
      buildOutputSignals();
    else
      buildInputSignals();
  }

  /// Update an instance port to the new port information. Also adds the proper
  /// ESI ops to map the channel to the wire signaling standard. These get
  /// lowered away in a later pass.
  virtual void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                               SmallVectorImpl<Value> &newOperands,
                               ArrayRef<Backedge> newResults) = 0;
  virtual void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                SmallVectorImpl<Value> &newOperands,
                                ArrayRef<Backedge> newResults) = 0;

protected:
  virtual void buildInputSignals() = 0;
  virtual void buildOutputSignals() = 0;

  ChannelRewriter &rewriter;
  Block *body;
  PortInfo origPort;

  hw::HWMutableModuleLike getModule() { return rewriter.getModule(); }
  MLIRContext *getContext() { return getModule()->getContext(); }

  SmallVector<PortInfo> dataPorts;

  /// Data ports aren't (generally) treated differently by different signaling
  /// standards. Since we want to support "flattened" structs as well as not
  /// (which is orthogonal to the control signals), put all of the data port
  /// transform logic in these general methods.
  Value buildInputDataPorts();
  void buildOutputDataPorts(Value data);
  void mapInputDataPorts(OpBuilder &b, Value unwrappedData,
                         SmallVectorImpl<Value> &newOperands);
  Value mapOutputDataPorts(OpBuilder &b, ArrayRef<Backedge> newResults);
};

// Build the input data ports. If it's a channel of a struct (and we've been
// asked to flatten it), flatten it into a port for each wire. If there's a
// body, re-construct the struct.
Value SignalingStandard::buildInputDataPorts() {
  auto chanTy = origPort.type.dyn_cast<ChannelType>();
  Type dataPortType = chanTy ? chanTy.getInner() : origPort.type;
  hw::StructType origStruct =
      chanTy ? dataPortType.dyn_cast<hw::StructType>() : hw::StructType();

  if (!rewriter.shouldFlattenStructs() || !origStruct) {
    dataPorts.push_back({});
    return rewriter.createNewInput(origPort, "", dataPortType, dataPorts[0]);
  }

  ArrayRef<hw::StructType::FieldInfo> elements = origStruct.getElements();
  dataPorts.append(elements.size(), {});
  SmallVector<Value, 16> elementValues;
  for (auto [idx, element] : llvm::enumerate(elements))
    elementValues.push_back(rewriter.createNewInput(
        origPort, "_" + element.name.getValue(), element.type, dataPorts[idx]));
  if (!body)
    return {};
  return OpBuilder::atBlockBegin(body).create<hw::StructCreateOp>(
      origPort.loc, origStruct, elementValues);
}

// Map a data value into the new operands for an instance. If the original type
// was a channel of a struct (and we've been asked to flatten it), break it up
// into per-field values.
void SignalingStandard::mapInputDataPorts(OpBuilder &b, Value unwrappedData,
                                          SmallVectorImpl<Value> &newOperands) {
  auto chanTy = origPort.type.dyn_cast<ChannelType>();
  Type dataPortType = chanTy ? chanTy.getInner() : origPort.type;
  hw::StructType origStruct =
      chanTy ? dataPortType.dyn_cast<hw::StructType>() : hw::StructType();

  if (!rewriter.shouldFlattenStructs() || !origStruct) {
    newOperands[dataPorts[0].argNum] = unwrappedData;
  } else {
    auto explode = b.create<hw::StructExplodeOp>(origPort.loc, unwrappedData);
    assert(explode->getNumResults() == dataPorts.size());
    for (auto [dataPort, fieldValue] :
         llvm::zip(dataPorts, explode.getResults()))
      newOperands[dataPort.argNum] = fieldValue;
  }
}

// Build the data ports for outputs. If the original type was a channel of a
// struct (and we've been asked to flatten it), explode the struct to create
// individual ports.
void SignalingStandard::buildOutputDataPorts(Value data) {
  auto chanTy = origPort.type.dyn_cast<ChannelType>();
  Type dataPortType = chanTy ? chanTy.getInner() : origPort.type;
  hw::StructType origStruct =
      chanTy ? dataPortType.dyn_cast<hw::StructType>() : hw::StructType();

  if (!rewriter.shouldFlattenStructs() || !origStruct) {
    dataPorts.push_back({});
    rewriter.createNewOutput(origPort, "", dataPortType, data, dataPorts[0]);
  } else {
    ArrayRef<hw::StructType::FieldInfo> elements = origStruct.getElements();
    dataPorts.append(elements.size(), {});

    Operation *explode = nullptr;
    if (body)
      explode = OpBuilder::atBlockTerminator(body).create<hw::StructExplodeOp>(
          origPort.loc, data);

    for (size_t idx = 0, e = elements.size(); idx < e; ++idx) {
      auto field = elements[idx];
      Value fieldValue = explode ? explode->getResult(idx) : Value();
      rewriter.createNewOutput(origPort, "_" + field.name.getValue(),
                               field.type, fieldValue, dataPorts[idx]);
    }
  }
}

// Map the data ports coming off an instance back into the original ports. If
// the original type was a channel of a struct (and we've been asked to flatten
// it), construct the original struct from the new ports.
Value SignalingStandard::mapOutputDataPorts(OpBuilder &b,
                                            ArrayRef<Backedge> newResults) {
  auto chanTy = origPort.type.dyn_cast<ChannelType>();
  Type dataPortType = chanTy ? chanTy.getInner() : origPort.type;
  hw::StructType origStruct =
      chanTy ? dataPortType.dyn_cast<hw::StructType>() : hw::StructType();

  if (!rewriter.shouldFlattenStructs() || !origStruct)
    return newResults[dataPorts[0].argNum];

  SmallVector<Value, 16> fieldValues;
  for (auto portInfo : dataPorts)
    fieldValues.push_back(newResults[portInfo.argNum]);
  return b.create<hw::StructCreateOp>(origPort.loc, origStruct, fieldValues);
}

/// We consider non-ESI ports to be ad-hoc signaling or 'raw wires'. (Which
/// counts as a signaling protocol if one squints pretty hard). We mostly do
/// this since it allows us a more consistent internal API.
class RawWires : public SignalingStandard {
public:
  using SignalingStandard::SignalingStandard;

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override {
    mapInputDataPorts(b, instValue, newOperands);
  }
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override {
    instValue.replaceAllUsesWith(mapOutputDataPorts(b, newResults));
  }

private:
  void buildInputSignals() override {
    Value newValue = buildInputDataPorts();
    if (body)
      body->getArgument(origPort.argNum).replaceAllUsesWith(newValue);
  }

  void buildOutputSignals() override {
    Value output;
    if (body)
      output = body->getTerminator()->getOperand(origPort.argNum);
    buildOutputDataPorts(output);
  }
};

/// Implement the Valid/Ready signaling standard.
class ValidReady : public SignalingStandard {
public:
  using SignalingStandard::SignalingStandard;

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Keep around information about the port numbers of the relevant ports and
  // use that later to update the instances.
  PortInfo validPort;
  PortInfo readyPort;
};

/// Implement the FIFO signaling standard.
class FIFO : public SignalingStandard {
public:
  using SignalingStandard::SignalingStandard;

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Keep around information about the port numbers of the relevant ports and
  // use that later to update the instances.
  PortInfo rdenPort;
  PortInfo emptyPort;
};
} // namespace

Value ChannelRewriter::createNewInput(PortInfo origPort, const Twine &suffix,
                                      Type type, PortInfo &newPort) {
  newPort = PortInfo{appendToRtlName(origPort.name, suffix.isTriviallyEmpty()
                                                        ? ""
                                                        : suffix + inSuffix),
                     PortDirection::INPUT,
                     type,
                     newInputs.size(),
                     {},
                     origPort.loc};
  newInputs.emplace_back(0, newPort);

  if (!body)
    return {};
  return body->addArgument(type, origPort.loc);
}

void ChannelRewriter::createNewOutput(PortInfo origPort, const Twine &suffix,
                                      Type type, Value output,
                                      PortInfo &newPort) {
  newPort = PortInfo{appendToRtlName(origPort.name, suffix.isTriviallyEmpty()
                                                        ? ""
                                                        : suffix + outSuffix),
                     PortDirection::OUTPUT,
                     type,
                     newOutputs.size(),
                     {},
                     origPort.loc};
  newOutputs.emplace_back(0, newPort);

  if (!body)
    return;
  newOutputValues.push_back(output);
}

LogicalResult ChannelRewriter::rewriteChannelsOnModule() {
  // Build ops in the module.
  ModulePortInfo ports = mod.getPorts();

  // Determine and create a `SignalingStandard` for said port.
  auto createLowering = [&](PortInfo port) -> LogicalResult {
    auto &loweredPorts = port.direction == PortDirection::OUTPUT
                             ? loweredOutputs
                             : loweredInputs;

    auto chanTy = port.type.dyn_cast<ChannelType>();
    if (!chanTy) {
      loweredPorts.emplace_back(new RawWires(*this, port));
    } else {
      // Mark this as a module which needs port lowering.
      foundEsiPorts = true;

      // Determine which ESI signaling standard is specified.
      ChannelSignaling signaling = chanTy.getSignaling();
      if (signaling == ChannelSignaling::ValidReady) {
        loweredPorts.emplace_back(new ValidReady(*this, port));
      } else if (signaling == ChannelSignaling::FIFO0) {
        loweredPorts.emplace_back(new FIFO(*this, port));
      } else {
        auto error =
            mod.emitOpError("encountered unknown signaling standard on port '")
            << stringifyEnum(signaling) << "'";
        error.attachNote(port.loc);
        return error;
      }
    }
    return success();
  };

  // Find the ESI ports and decide the signaling standard.
  for (PortInfo port : ports.inputs)
    if (failed(createLowering(port)))
      return failure();
  for (PortInfo port : ports.outputs)
    if (failed(createLowering(port)))
      return failure();

  // Bail early if we didn't find any.
  if (!foundEsiPorts) {
    // Memory optimization.
    loweredInputs.clear();
    loweredOutputs.clear();
    return success();
  }

  // Lower the ESI ports -- this mutates the body directly and builds the port
  // lists.
  for (auto &lowering : loweredInputs)
    lowering->lowerPort();
  for (auto &lowering : loweredOutputs)
    lowering->lowerPort();

  // Set up vectors to erase _all_ the ports. It's easier to rebuild everything
  // (including the non-ESI ports) than reason about interleaving the newly
  // lowered ESI ports with the non-ESI ports. Also, the 'modifyPorts' method
  // ends up rebuilding the port lists anyway, so this isn't nearly as expensive
  // as it may seem.
  SmallVector<unsigned> inputsToErase;
  for (size_t i = 0, e = mod.getNumInputs(); i < e; ++i)
    inputsToErase.push_back(i);
  SmallVector<unsigned> outputsToErase;
  for (size_t i = 0, e = mod.getNumOutputs(); i < e; ++i)
    outputsToErase.push_back(i);

  mod.modifyPorts(newInputs, newOutputs, inputsToErase, outputsToErase);

  if (!body)
    return success();

  // We should only erase the original arguments. New ones were appended with
  // the `createInput` method call.
  body->eraseArguments([&ports](BlockArgument arg) {
    return arg.getArgNumber() < ports.inputs.size();
  });
  // Set the new operands, overwriting the old ones.
  body->getTerminator()->setOperands(newOutputValues);

  // Memory optimization -- we don't need these anymore.
  newInputs.clear();
  newOutputs.clear();
  newOutputValues.clear();
  return success();
}

void ValidReady::buildInputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  // When we find one, add a data and valid signal to the new args.
  Value data = buildInputDataPorts();
  Value valid =
      rewriter.createNewInput(origPort, rewriter.validSuffix, i1, validPort);

  Value ready;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Build the ESI wrap operation to translate the lowered signals to what
    // they were. (A later pass takes care of eliminating the ESI ops.)
    auto wrap = b.create<WrapValidReadyOp>(data, valid);
    ready = wrap.getReady();
    // Replace uses of the old ESI port argument with the new one from the
    // wrap.
    body->getArgument(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  }

  rewriter.createNewOutput(origPort, rewriter.readySuffix, i1, ready,
                           readyPort);
}

void ValidReady::mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                 SmallVectorImpl<Value> &newOperands,
                                 ArrayRef<Backedge> newResults) {
  auto unwrap = b.create<UnwrapValidReadyOp>(inst->getLoc(),
                                             inst->getOperand(origPort.argNum),
                                             newResults[readyPort.argNum]);
  mapInputDataPorts(b, unwrap.getRawOutput(), newOperands);
  newOperands[validPort.argNum] = unwrap.getValid();
}

void ValidReady::buildOutputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  Value ready =
      rewriter.createNewInput(origPort, rewriter.readySuffix, i1, readyPort);
  Value data, valid;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);

    auto unwrap = b.create<UnwrapValidReadyOp>(
        terminator->getOperand(origPort.argNum), ready);
    data = unwrap.getRawOutput();
    valid = unwrap.getValid();
  }

  // New outputs.
  buildOutputDataPorts(data);
  rewriter.createNewOutput(origPort, rewriter.validSuffix, i1, valid,
                           validPort);
}

void ValidReady::mapOutputSignals(OpBuilder &b, Operation *inst,
                                  Value instValue,
                                  SmallVectorImpl<Value> &newOperands,
                                  ArrayRef<Backedge> newResults) {
  Value data = mapOutputDataPorts(b, newResults);
  auto wrap = b.create<WrapValidReadyOp>(inst->getLoc(), data,
                                         newResults[validPort.argNum]);
  inst->getResult(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  newOperands[readyPort.argNum] = wrap.getReady();
}

void FIFO::buildInputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto chanTy = origPort.type.cast<ChannelType>();

  // When we find one, add a data and valid signal to the new args.
  Value data = buildInputDataPorts();
  Value empty =
      rewriter.createNewInput(origPort, rewriter.emptySuffix, i1, emptyPort);

  Value rden;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Build the ESI wrap operation to translate the lowered signals to what
    // they were. (A later pass takes care of eliminating the ESI ops.)
    auto wrap = b.create<WrapFIFOOp>(ArrayRef<Type>({chanTy, b.getI1Type()}),
                                     data, empty);
    rden = wrap.getRden();
    // Replace uses of the old ESI port argument with the new one from the
    // wrap.
    body->getArgument(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  }

  rewriter.createNewOutput(origPort, rewriter.rdenSuffix, i1, rden, rdenPort);
}

void FIFO::mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                           SmallVectorImpl<Value> &newOperands,
                           ArrayRef<Backedge> newResults) {
  auto unwrap =
      b.create<UnwrapFIFOOp>(inst->getLoc(), inst->getOperand(origPort.argNum),
                             newResults[rdenPort.argNum]);
  mapInputDataPorts(b, unwrap.getData(), newOperands);
  newOperands[emptyPort.argNum] = unwrap.getEmpty();
}

void FIFO::buildOutputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  Value rden =
      rewriter.createNewInput(origPort, rewriter.rdenSuffix, i1, rdenPort);
  Value data, empty;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);

    auto unwrap =
        b.create<UnwrapFIFOOp>(terminator->getOperand(origPort.argNum), rden);
    data = unwrap.getData();
    empty = unwrap.getEmpty();
  }

  // New outputs.
  buildOutputDataPorts(data);
  rewriter.createNewOutput(origPort, rewriter.emptySuffix, i1, empty,
                           emptyPort);
}

void FIFO::mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                            SmallVectorImpl<Value> &newOperands,
                            ArrayRef<Backedge> newResults) {
  Value data = mapOutputDataPorts(b, newResults);
  auto wrap = b.create<WrapFIFOOp>(
      inst->getLoc(), ArrayRef<Type>({origPort.type, b.getI1Type()}), data,
      newResults[emptyPort.argNum]);
  inst->getResult(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  newOperands[rdenPort.argNum] = wrap.getRden();
}

/// Update an instance of an updated module by adding `esi.[un]wrap.vr`
/// ops around the instance. Lowering or folding away `[un]wrap` ops is
/// another pass.
void ChannelRewriter::updateInstance(InstanceOp inst) {
  if (!foundEsiPorts)
    return;

  ImplicitLocOpBuilder b(inst.getLoc(), inst);
  BackedgeBuilder beb(b, inst.getLoc());
  ModulePortInfo ports = mod.getPorts();

  // Create backedges for the future instance results so the signal mappers can
  // use the future results as values.
  SmallVector<Backedge> newResults;
  for (PortInfo outputPort : ports.outputs)
    newResults.push_back(beb.get(outputPort.type));

  // Map the operands.
  SmallVector<Value> newOperands(ports.inputs.size(), {});
  for (size_t oldOpIdx = 0, e = inst.getNumOperands(); oldOpIdx < e; ++oldOpIdx)
    loweredInputs[oldOpIdx]->mapInputSignals(
        b, inst, inst->getOperand(oldOpIdx), newOperands, newResults);

  // Map the results.
  for (size_t oldResIdx = 0, e = inst.getNumResults(); oldResIdx < e;
       ++oldResIdx)
    loweredOutputs[oldResIdx]->mapOutputSignals(
        b, inst, inst->getResult(oldResIdx), newOperands, newResults);

  // Clone the instance. We cannot just modifiy the existing one since the
  // result types might have changed types and number of them.
  assert(llvm::none_of(newOperands, [](Value v) { return !v; }));
  b.setInsertionPointAfter(inst);
  auto newInst =
      b.create<InstanceOp>(mod, inst.getInstanceNameAttr(), newOperands,
                           inst.getParameters(), inst.getInnerSymAttr());
  newInst->setDialectAttrs(inst->getDialectAttrs());

  // Assign the backedges to the new results.
  for (auto [idx, be] : llvm::enumerate(newResults))
    be.setValue(newInst.getResult(idx));

  // Erase the old instance.
  inst.erase();
}

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV
/// interfaces for now on external modules, ready/valid to modules defined
/// internally. In the future, it may be possible to select a different
/// format.
struct ESIPortsPass : public LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation() override;

private:
  bool updateFunc(HWModuleExternOp mod);
  void updateInstance(HWModuleExternOp mod, InstanceOp inst);
  ESIHWBuilder *build;
};
} // anonymous namespace

/// Iterate through the `hw.module[.extern]`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIHWBuilder b(top);
  build = &b;

  // Find all externmodules and try to modify them. Remember the modified
  // ones.
  DenseMap<SymbolRefAttr, HWModuleExternOp> externModsMutated;
  for (auto mod : top.getOps<HWModuleExternOp>())
    if (mod->hasAttrOfType<UnitAttr>(extModBundleSignalsAttrName) &&
        updateFunc(mod))
      externModsMutated[FlatSymbolRefAttr::get(mod)] = mod;

  // Find all instances and update them.
  top.walk([&externModsMutated, this](InstanceOp inst) {
    auto mapIter = externModsMutated.find(inst.getModuleNameAttr());
    if (mapIter != externModsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  // Find all modules and try to modify them to have wires with valid/ready
  // semantics. Remember the modified ones.
  DenseMap<SymbolRefAttr, ChannelRewriter> modsMutated;
  for (auto mod : top.getOps<HWMutableModuleLike>()) {
    auto modSym = FlatSymbolRefAttr::get(mod);
    if (externModsMutated.find(modSym) != externModsMutated.end())
      continue;
    auto [entry, emplaced] = modsMutated.try_emplace(modSym, mod);
    if (!emplaced) {
      auto error = mod.emitOpError("Detected duplicate symbol on module: ")
                   << modSym;
      error.attachNote(entry->second.getModule().getLoc());
      signalPassFailure();
      continue;
    }
    if (failed(entry->second.rewriteChannelsOnModule()))
      signalPassFailure();
  }

  // Find all instances and update them.
  top.walk([&modsMutated](InstanceOp inst) {
    auto mapIter = modsMutated.find(inst.getModuleNameAttr());
    if (mapIter != modsMutated.end())
      mapIter->second.updateInstance(inst);
  });

  build = nullptr;
}

/// Convert all input and output ChannelTypes into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Returns true
/// if 'mod' was updated. Delay updating the instances to amortize the IR walk
/// over all the module updates.
bool ESIPortsPass::updateFunc(HWModuleExternOp mod) {
  auto *ctxt = &getContext();

  bool updated = false;

  SmallVector<Attribute> newArgNames, newArgLocs, newResultNames, newResultLocs;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found.
  SmallVector<Type, 16> newArgTypes;
  size_t nextArgNo = 0;
  for (auto argTy : mod.getArgumentTypes()) {
    auto chanTy = argTy.dyn_cast<ChannelType>();
    newArgNames.push_back(getModuleArgumentNameAttr(mod, nextArgNo));
    newArgLocs.push_back(getModuleArgumentLocAttr(mod, nextArgNo));
    nextArgNo++;

    if (!chanTy) {
      newArgTypes.push_back(argTy);
      continue;
    }

    // When we find one, construct an interface, and add the 'source' modport
    // to the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(ESIHWBuilder::sourceStr));
    updated = true;
  }

  // Iterate through the results and append to one of the two below lists. The
  // first for non-ESI-ports. The second, ports which have been re-located to
  // an operand.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  auto funcType = mod.getFunctionType();
  for (size_t resNum = 0, numRes = mod.getNumResults(); resNum < numRes;
       ++resNum) {
    Type resTy = funcType.getResult(resNum);
    auto chanTy = resTy.dyn_cast<ChannelType>();
    auto resNameAttr = getModuleResultNameAttr(mod, resNum);
    auto resLocAttr = getModuleResultLocAttr(mod, resNum);
    if (!chanTy) {
      newResultTypes.push_back(resTy);
      newResultNames.push_back(resNameAttr);
      newResultLocs.push_back(resLocAttr);
      continue;
    }

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    sv::InterfaceOp iface = build->getOrConstructInterface(chanTy);
    sv::ModportType sinkPort = iface.getModportType(ESIHWBuilder::sinkStr);
    newArgTypes.push_back(sinkPort);
    newArgNames.push_back(resNameAttr);
    newArgLocs.push_back(resLocAttr);
    updated = true;
  }

  mod->removeAttr(extModBundleSignalsAttrName);
  if (!updated)
    return false;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  setModuleArgumentNames(mod, newArgNames);
  setModuleArgumentLocs(mod, newArgLocs);
  setModuleResultNames(mod, newResultNames);
  setModuleResultLocs(mod, newResultLocs);
  return true;
}

static StringRef getOperandName(Value operand) {
  if (BlockArgument arg = operand.dyn_cast<BlockArgument>()) {
    auto *op = arg.getParentBlock()->getParentOp();
    if (op && hw::isAnyModule(op))
      return hw::getModuleArgumentName(op, arg.getArgNumber());
  } else {
    auto *srcOp = operand.getDefiningOp();
    if (auto instOp = dyn_cast<InstanceOp>(srcOp))
      return instOp.getInstanceName();

    if (auto srcName = srcOp->getAttrOfType<StringAttr>("name"))
      return srcName.getValue();
  }
  return "";
}

/// Create a reasonable name for a SV interface instance.
static std::string &constructInstanceName(Value operand, sv::InterfaceOp iface,
                                          std::string &name) {
  llvm::raw_string_ostream s(name);
  // Drop the "IValidReady_" part of the interface name.
  s << llvm::toLower(iface.getSymName()[12]) << iface.getSymName().substr(13);

  // Indicate to where the source is connected.
  if (operand.hasOneUse()) {
    Operation *dstOp = *operand.getUsers().begin();
    if (auto instOp = dyn_cast<InstanceOp>(dstOp))
      s << "To" << llvm::toUpper(instOp.getInstanceName()[0])
        << instOp.getInstanceName().substr(1);
    else if (auto dstName = dstOp->getAttrOfType<StringAttr>("name"))
      s << "To" << dstName.getValue();
  }

  // Indicate to where the sink is connected.
  StringRef operName = getOperandName(operand);
  if (!operName.empty())
    s << "From" << llvm::toUpper(operName[0]) << operName.substr(1);
  return s.str();
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(HWModuleExternOp mod, InstanceOp inst) {
  using namespace circt::sv;
  circt::ImplicitLocOpBuilder instBuilder(inst.getLoc(), inst);
  FunctionType funcTy = mod.getFunctionType();

  // op counter for error reporting purposes.
  size_t opNum = 0;
  // List of new operands.
  SmallVector<Value, 16> newOperands;

  // Fill the new operand list with old plain operands and mutated ones.
  std::string nameStringBuffer; // raw_string_ostream uses std::string.
  for (auto op : inst.getOperands()) {
    auto instChanTy = op.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newOperands.push_back(op);
      ++opNum;
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sourceStr) !=
        funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (operand #")
          << opNum << ") doesn't match module!";
      ++opNum;
      newOperands.push_back(op);
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.unwrap.iface` and the other end to the instance.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(op, iface, nameStringBuffer)));
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    instBuilder.create<UnwrapSVInterfaceOp>(op, sinkModport);
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    // Finally, add the correct modport to the list of operands.
    newOperands.push_back(sourceModport);
  }

  // Go through the results and get both a list of the plain old values being
  // produced and their types.
  SmallVector<Value, 8> newResults;
  SmallVector<Type, 8> newResultTypes;
  for (size_t resNum = 0, numRes = inst.getNumResults(); resNum < numRes;
       ++resNum) {
    Value res = inst.getResult(resNum);
    auto instChanTy = res.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sinkStr) != funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (result #")
          << resNum << ", operand #" << opNum << ") doesn't match module!";
      ++opNum;
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.wrap.iface` and the other end to the instance. Append it to the
    // operand list.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(res, iface, nameStringBuffer)));
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    auto newChannel =
        instBuilder.create<WrapSVInterfaceOp>(res.getType(), sourceModport);
    // Connect all the old users of the output channel with the newly
    // wrapped replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sinkModport);
  }

  // Create the new instance!
  InstanceOp newInst = instBuilder.create<InstanceOp>(
      mod, inst.getInstanceNameAttr(), newOperands, inst.getParameters(),
      inst.getInnerSymAttr());

  // Go through the old list of non-ESI result values, and replace them with
  // the new non-ESI results.
  for (size_t resNum = 0, numRes = newResults.size(); resNum < numRes;
       ++resNum) {
    newResults[resNum].replaceAllUsesWith(newInst.getResult(resNum));
  }
  // Erase the old instance!
  inst.erase();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
