//===- ESIPasses.cpp - ESI to HW/SV conversion passes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower ESI to HW and SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#include <memory>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#endif

using namespace circt;
using namespace circt::comb;
using namespace circt::esi;
using namespace circt::hw;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

namespace {
/// Assist the lowering steps for conversions which need to create auxiliary IR.
class ESIHWBuilder : public circt::ImplicitLocOpBuilder {
public:
  ESIHWBuilder(Operation *top);

  ArrayAttr getStageParameterList(Attribute value);

  HWModuleExternOp declareStage(Operation *symTable, PipelineStageOp);
  // Will be unused when CAPNP is undefined
  HWModuleExternOp declareCosimEndpointOp(Operation *symTable, Type sendType,
                                          Type recvType) LLVM_ATTRIBUTE_UNUSED;

  InterfaceOp getOrConstructInterface(ChannelType);
  InterfaceOp constructInterface(ChannelType);

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr dataOutValid, dataOutReady, dataOut, dataInValid,
      dataInReady, dataIn;
  const StringAttr clk, rst;
  const StringAttr width;

  // Various identifier strings. Keep them all here in case we rename them.
  static constexpr char dataStr[] = "data", validStr[] = "valid",
                        readyStr[] = "ready", sourceStr[] = "source",
                        sinkStr[] = "sink";

private:
  /// Construct a type-appropriate name for the interface, making sure it's not
  /// taken in the symbol table.
  StringAttr constructInterfaceName(ChannelType);

  llvm::DenseMap<Type, HWModuleExternOp> declaredStage;
  llvm::DenseMap<std::pair<Type, Type>, HWModuleExternOp>
      declaredCosimEndpointOp;
  llvm::DenseMap<Type, InterfaceOp> portTypeLookup;
};
} // anonymous namespace

// C++ requires this for showing it what object file it should store these
// symbols in. They should be inline but that feature wasn't added until C++17.
constexpr char ESIHWBuilder::dataStr[], ESIHWBuilder::validStr[],
    ESIHWBuilder::readyStr[], ESIHWBuilder::sourceStr[],
    ESIHWBuilder::sinkStr[];

ESIHWBuilder::ESIHWBuilder(Operation *top)
    : ImplicitLocOpBuilder(UnknownLoc::get(top->getContext()), top),
      a(StringAttr::get(getContext(), "a")),
      aValid(StringAttr::get(getContext(), "a_valid")),
      aReady(StringAttr::get(getContext(), "a_ready")),
      x(StringAttr::get(getContext(), "x")),
      xValid(StringAttr::get(getContext(), "x_valid")),
      xReady(StringAttr::get(getContext(), "x_ready")),
      dataOutValid(StringAttr::get(getContext(), "DataOutValid")),
      dataOutReady(StringAttr::get(getContext(), "DataOutReady")),
      dataOut(StringAttr::get(getContext(), "DataOut")),
      dataInValid(StringAttr::get(getContext(), "DataInValid")),
      dataInReady(StringAttr::get(getContext(), "DataInReady")),
      dataIn(StringAttr::get(getContext(), "DataIn")),
      clk(StringAttr::get(getContext(), "clk")),
      rst(StringAttr::get(getContext(), "rst")),
      width(StringAttr::get(getContext(), "WIDTH")) {

  auto regions = top->getRegions();
  if (regions.size() == 0) {
    top->emitError("ESI HW Builder needs a region to insert HW.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

static StringAttr constructUniqueSymbol(Operation *tableOp,
                                        StringRef proposedNameRef) {
  SmallString<64> proposedName = proposedNameRef;

  // Normalize the type name.
  for (char &ch : proposedName) {
    if (isalpha(ch) || isdigit(ch) || ch == '_')
      continue;
    ch = '_';
  }

  // Make sure that this symbol isn't taken. If it is, append a number and try
  // again.
  size_t baseLength = proposedName.size();
  size_t tries = 0;
  while (SymbolTable::lookupSymbolIn(tableOp, proposedName)) {
    proposedName.resize(baseLength);
    proposedName.append(llvm::utostr(++tries));
  }

  return StringAttr::get(tableOp->getContext(), proposedName);
}

StringAttr ESIHWBuilder::constructInterfaceName(ChannelType port) {
  Operation *tableOp =
      getInsertionPoint()->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  // Get a name based on the type.
  std::string portTypeName;
  llvm::raw_string_ostream nameOS(portTypeName);
  TypeSwitch<Type>(port.getInner())
      .Case([&](hw::ArrayType arr) {
        nameOS << "ArrayOf" << arr.getSize() << 'x' << arr.getElementType();
      })
      .Case([&](hw::StructType t) { nameOS << "Struct"; })
      .Default([&](Type t) { nameOS << port.getInner(); });

  // Don't allow the name to end with '_'.
  ssize_t i = portTypeName.size() - 1;
  while (i >= 0 && portTypeName[i] == '_') {
    --i;
  }
  portTypeName = portTypeName.substr(0, i + 1);

  // All stage names start with this.
  SmallString<64> proposedName("IValidReady_");
  proposedName.append(portTypeName);
  return constructUniqueSymbol(tableOp, proposedName);
}

/// Return a parameter list for the stage module with the specified value.
ArrayAttr ESIHWBuilder::getStageParameterList(Attribute value) {
  auto type = IntegerType::get(width.getContext(), 32, IntegerType::Unsigned);
  auto widthParam = ParamDeclAttr::get(width.getContext(), width, type, value);
  return ArrayAttr::get(width.getContext(), widthParam);
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
HWModuleExternOp ESIHWBuilder::declareStage(Operation *symTable,
                                            PipelineStageOp stage) {
  Type dataType = stage.innerType();
  HWModuleExternOp &stageMod = declaredStage[dataType];
  if (stageMod)
    return stageMod;

  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  size_t argn = 0;
  size_t resn = 0;
  llvm::SmallVector<PortInfo> ports = {
      {clk, PortDirection::INPUT, getI1Type(), argn++},
      {rst, PortDirection::INPUT, getI1Type(), argn++}};

  ports.push_back({a, PortDirection::INPUT, dataType, argn++});
  ports.push_back({aValid, PortDirection::INPUT, getI1Type(), argn++});
  ports.push_back({aReady, PortDirection::OUTPUT, getI1Type(), resn++});
  ports.push_back({x, PortDirection::OUTPUT, dataType, resn++});

  ports.push_back({xValid, PortDirection::OUTPUT, getI1Type(), resn++});
  ports.push_back({xReady, PortDirection::INPUT, getI1Type(), argn++});

  stageMod = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "ESI_PipelineStage"), ports,
      "ESI_PipelineStage", getStageParameterList({}));
  return stageMod;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
HWModuleExternOp ESIHWBuilder::declareCosimEndpointOp(Operation *symTable,
                                                      Type sendType,
                                                      Type recvType) {
  HWModuleExternOp &endpoint =
      declaredCosimEndpointOp[std::make_pair(sendType, recvType)];
  if (endpoint)
    return endpoint;
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  PortInfo ports[] = {{clk, PortDirection::INPUT, getI1Type(), 0},
                      {rst, PortDirection::INPUT, getI1Type(), 1},
                      {dataOutValid, PortDirection::OUTPUT, getI1Type(), 0},
                      {dataOutReady, PortDirection::INPUT, getI1Type(), 2},
                      {dataOut, PortDirection::OUTPUT, recvType, 1},
                      {dataInValid, PortDirection::INPUT, getI1Type(), 3},
                      {dataInReady, PortDirection::OUTPUT, getI1Type(), 2},
                      {dataIn, PortDirection::INPUT, sendType, 4}};
  SmallVector<Attribute, 8> params;
  params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", getStringAttr("")));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("SEND_TYPE_SIZE_BITS", getI32Type()));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("RECV_TYPE_SIZE_BITS", getI32Type()));
  endpoint = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "Cosim_Endpoint"), ports,
      "Cosim_Endpoint", ArrayAttr::get(getContext(), params));
  return endpoint;
}

/// Return the InterfaceType which corresponds to an ESI port type. If it
/// doesn't exist in the cache, build the InterfaceOp and the corresponding
/// type.
InterfaceOp ESIHWBuilder::getOrConstructInterface(ChannelType t) {
  auto ifaceIter = portTypeLookup.find(t);
  if (ifaceIter != portTypeLookup.end())
    return ifaceIter->second;
  auto iface = constructInterface(t);
  portTypeLookup[t] = iface;
  return iface;
}

InterfaceOp ESIHWBuilder::constructInterface(ChannelType chan) {
  return create<InterfaceOp>(constructInterfaceName(chan).getValue(), [&]() {
    create<InterfaceSignalOp>(validStr, getI1Type());
    create<InterfaceSignalOp>(readyStr, getI1Type());
    create<InterfaceSignalOp>(dataStr, chan.getInner());
    llvm::SmallVector<StringRef> validDataStrs;
    validDataStrs.push_back(validStr);
    validDataStrs.push_back(dataStr);
    create<InterfaceModportOp>(sinkStr,
                               /*inputs=*/ArrayRef<StringRef>{readyStr},
                               /*outputs=*/validDataStrs);
    create<InterfaceModportOp>(sourceStr,
                               /*inputs=*/validDataStrs,
                               /*outputs=*/ArrayRef<StringRef>{readyStr});
  });
}

//===----------------------------------------------------------------------===//
// Lower to physical conversions and pass.
//===----------------------------------------------------------------------===//

namespace {
/// Lower `ChannelBufferOp`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct ChannelBufferLowering : public OpConversionPattern<ChannelBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ChannelBufferOp buffer, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult ChannelBufferLowering::matchAndRewrite(
    ChannelBufferOp buffer, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = buffer.getLoc();

  auto type = buffer.getType();

  // Expand 'abstract' buffer into 'physical' stages.
  auto stages = buffer.getStagesAttr();
  uint64_t numStages = 1;
  if (stages) {
    // Guaranteed positive by the parser.
    numStages = stages.getValue().getLimitedValue();
  }
  Value input = buffer.getInput();
  StringAttr bufferName = buffer.getNameAttr();
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStageOp>(loc, type, buffer.getClk(),
                                                  buffer.getRst(), input);
    if (bufferName) {
      SmallString<64> stageName(
          {bufferName.getValue(), "_stage", std::to_string(i)});
      stage->setAttr("name", StringAttr::get(rewriter.getContext(), stageName));
    }
    input = stage;
  }

  // Replace the buffer.
  rewriter.replaceOp(buffer, input);
  return success();
}

namespace {
/// Lower pure modules into hw.modules.
struct PureModuleLowering : public OpConversionPattern<ESIPureModuleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
PureModuleLowering::matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = pureMod.getLoc();
  Block *body = &pureMod.getBody().front();

  // Track existing names (so we can de-dup) and get op result when we want to
  // replace it with the block args.
  DenseMap<StringAttr, ESIPureModuleInputOp> inputPortNames;
  // Build the port list for `hw.module` construction.
  SmallVector<hw::PortInfo> ports;
  // List the input and output ops.
  SmallVector<ESIPureModuleInputOp> inputs;
  SmallVector<ESIPureModuleOutputOp> outputs;
  SmallVector<Attribute> params;

  for (Operation &op : llvm::make_early_inc_range(body->getOperations())) {
    if (auto port = dyn_cast<ESIPureModuleInputOp>(op)) {
      // If we already have an input port of the same name, replace the result
      // value with the previous one. Checking that the types match is done in
      // the pure module verifier.
      auto existingPort = inputPortNames.find(port.getNameAttr());
      if (existingPort != inputPortNames.end()) {
        rewriter.replaceAllUsesWith(port.getResult(),
                                    existingPort->getSecond().getResult());
        rewriter.eraseOp(port);
        continue;
      }
      // Normal port construction.
      ports.push_back(hw::PortInfo{port.getNameAttr(),
                                   hw::PortDirection::INPUT,
                                   port.getResult().getType(),
                                   inputs.size(),
                                   {},
                                   port.getLoc()});
      inputs.push_back(port);
    } else if (auto port = dyn_cast<ESIPureModuleOutputOp>(op)) {
      ports.push_back(hw::PortInfo{port.getNameAttr(),
                                   hw::PortDirection::OUTPUT,
                                   port.getValue().getType(),
                                   outputs.size(),
                                   {},
                                   port.getLoc()});
      outputs.push_back(port);
    } else if (auto param = dyn_cast<ESIPureModuleParamOp>(op)) {
      params.push_back(
          ParamDeclAttr::get(param.getNameAttr(), param.getType()));
      rewriter.eraseOp(param);
    }
  }

  // Create the replacement `hw.module`.
  auto hwMod = rewriter.create<hw::HWModuleOp>(
      loc, pureMod.getNameAttr(), ports, ArrayAttr::get(getContext(), params));
  hwMod->setDialectAttrs(pureMod->getDialectAttrs());
  rewriter.eraseBlock(hwMod.getBodyBlock());
  rewriter.inlineRegionBefore(*body->getParent(), hwMod.getBodyRegion(),
                              hwMod.getBodyRegion().end());
  body = hwMod.getBodyBlock();

  // Re-wire the inputs and erase them.
  for (auto input : inputs) {
    BlockArgument newArg;
    rewriter.updateRootInPlace(hwMod, [&]() {
      newArg = body->addArgument(input.getResult().getType(), input.getLoc());
    });
    rewriter.replaceAllUsesWith(input.getResult(), newArg);
    rewriter.eraseOp(input);
  }

  // Assemble the output values.
  SmallVector<Value> hwOutputOperands;
  for (auto output : outputs) {
    hwOutputOperands.push_back(output.getValue());
    rewriter.eraseOp(output);
  }
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<hw::OutputOp>(pureMod.getLoc(), hwOutputOperands);

  // Erase the original op.
  rewriter.eraseOp(pureMod);
  return success();
}

namespace {
/// Run all the physical lowerings.
struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIToPhysicalPass::runOnOperation() {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<ChannelBufferOp>();
  target.addIllegalOp<ESIPureModuleOp>();

  // Add all the conversion patterns.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ChannelBufferLowering>(&getContext());
  patterns.insert<PureModuleLowering>(&getContext());

  // Run the conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Lower ESI ports pass.
//===----------------------------------------------------------------------===//

/// Return a attribute with the specified suffix appended.
static StringAttr appendToRtlName(StringAttr base, Twine suffix) {
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
  Value createNewInput(PortInfo origPort, Twine suffix, Type type,
                       PortInfo &newPort);
  /// Same as above. 'output' is the value fed into the new port and is required
  /// if 'body' is non-null. Important note: cannot be a backedge which gets
  /// replaced since this isn't attached to an op until later in the pass.
  void createNewOutput(PortInfo origPort, Twine suffix, Type type, Value output,
                       PortInfo &newPort);

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

Value ChannelRewriter::createNewInput(PortInfo origPort, Twine suffix,
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

void ChannelRewriter::createNewOutput(PortInfo origPort, Twine suffix,
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
    InterfaceOp iface = build->getOrConstructInterface(chanTy);
    ModportType sinkPort = iface.getModportType(ESIHWBuilder::sinkStr);
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
    auto op = arg.getParentBlock()->getParentOp();
    if (op && hw::isAnyModule(op))
      return hw::getModuleArgumentName(op, arg.getArgNumber());
  } else {
    auto srcOp = operand.getDefiningOp();
    if (auto instOp = dyn_cast<InstanceOp>(srcOp))
      return instOp.instanceName();

    if (auto srcName = srcOp->getAttrOfType<StringAttr>("name"))
      return srcName.getValue();
  }
  return "";
}

/// Create a reasonable name for a SV interface instance.
static std::string &constructInstanceName(Value operand, InterfaceOp iface,
                                          std::string &name) {
  llvm::raw_string_ostream s(name);
  // Drop the "IValidReady_" part of the interface name.
  s << llvm::toLower(iface.getSymName()[12]) << iface.getSymName().substr(13);

  // Indicate to where the source is connected.
  if (operand.hasOneUse()) {
    Operation *dstOp = *operand.getUsers().begin();
    if (auto instOp = dyn_cast<InstanceOp>(dstOp))
      s << "To" << llvm::toUpper(instOp.instanceName()[0])
        << instOp.instanceName().substr(1);
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

//===----------------------------------------------------------------------===//
// Lower to HW/SV conversions and pass.
//===----------------------------------------------------------------------===//

namespace {
/// Lower PipelineStageOp ops to an HW implementation. Unwrap and re-wrap
/// appropriately. Another conversion will take care merging the resulting
/// adjacent wrap/unwrap ops.
struct PipelineStageLowering : public OpConversionPattern<PipelineStageOp> {
public:
  PipelineStageLowering(ESIHWBuilder &builder, MLIRContext *ctxt)
      : OpConversionPattern(ctxt), builder(builder) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PipelineStageOp stage, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult PipelineStageLowering::matchAndRewrite(
    PipelineStageOp stage, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = stage.getLoc();
  auto chPort = stage.getInput().getType().dyn_cast<ChannelType>();
  if (!chPort)
    return rewriter.notifyMatchFailure(stage, "stage had wrong type");
  Operation *symTable = stage->getParentWithTrait<OpTrait::SymbolTable>();
  auto stageModule = builder.declareStage(symTable, stage);

  size_t width = circt::hw::getBitWidth(chPort.getInner());

  ArrayAttr stageParams =
      builder.getStageParameterList(rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet,
  // so create a temp value and replace it later. Give this constant an
  // odd-looking type to make debugging easier.
  circt::BackedgeBuilder back(rewriter, loc);
  circt::Backedge wrapReady = back.get(rewriter.getI1Type());
  auto unwrap =
      rewriter.create<UnwrapValidReadyOp>(loc, stage.getInput(), wrapReady);

  StringRef pipeStageName = "pipelineStage";
  if (auto name = stage->getAttrOfType<StringAttr>("name"))
    pipeStageName = name.getValue();

  // Instantiate the "ESI_PipelineStage" external module.
  circt::Backedge stageReady = back.get(rewriter.getI1Type());
  llvm::SmallVector<Value> operands = {stage.getClk(), stage.getRst()};
  operands.push_back(unwrap.getRawOutput());
  operands.push_back(unwrap.getValid());
  operands.push_back(stageReady);
  auto stageInst = rewriter.create<InstanceOp>(loc, stageModule, pipeStageName,
                                               operands, stageParams);
  auto stageInstResults = stageInst.getResults();

  // Set a_ready (from the unwrap) back edge correctly to its output from
  // stage.
  wrapReady.setValue(stageInstResults[0]);
  Value x, xValid;
  x = stageInstResults[1];
  xValid = stageInstResults[2];

  // Wrap up the output of the HW stage module.
  auto wrap = rewriter.create<WrapValidReadyOp>(
      loc, chPort, rewriter.getI1Type(), x, xValid);
  // Set the stages x_ready backedge correctly.
  stageReady.setValue(wrap.getReady());

  rewriter.replaceOp(stage, wrap.getChanOutput());
  return success();
}

namespace {
struct NullSourceOpLowering : public OpConversionPattern<NullSourceOp> {
public:
  NullSourceOpLowering(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NullSourceOp nullop, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult NullSourceOpLowering::matchAndRewrite(
    NullSourceOp nullop, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto innerType = nullop.getOut().getType().cast<ChannelType>().getInner();
  Location loc = nullop.getLoc();
  int64_t width = hw::getBitWidth(innerType);
  if (width == -1)
    return rewriter.notifyMatchFailure(
        nullop, "NullOp lowering only supports hw types");
  auto valid =
      rewriter.create<hw::ConstantOp>(nullop.getLoc(), rewriter.getI1Type(), 0);
  auto zero =
      rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(width), 0);
  auto typedZero = rewriter.create<hw::BitcastOp>(loc, innerType, zero);
  auto wrap = rewriter.create<WrapValidReadyOp>(loc, typedZero, valid);
  wrap->setAttr("name", rewriter.getStringAttr("nullsource"));
  rewriter.replaceOp(nullop, {wrap.getChanOutput()});
  return success();
}

namespace {
/// Eliminate back-to-back wrap-unwraps to reduce the number of ESI channels.
struct RemoveWrapUnwrap : public ConversionPattern {
public:
  RemoveWrapUnwrap(MLIRContext *context)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value valid, ready, data;
    WrapValidReadyOp wrap = dyn_cast<WrapValidReadyOp>(op);
    UnwrapValidReadyOp unwrap = dyn_cast<UnwrapValidReadyOp>(op);
    if (wrap) {
      if (!wrap.getChanOutput().hasOneUse() ||
          !(unwrap = dyn_cast<UnwrapValidReadyOp>(
                wrap.getChanOutput().use_begin()->getOwner())))
        return rewriter.notifyMatchFailure(
            wrap, "This conversion only supports wrap-unwrap back-to-back. "
                  "Could not find 'unwrap'.");

      data = operands[0];
      valid = operands[1];
      ready = unwrap.getReady();
    } else if (unwrap) {
      wrap = dyn_cast<WrapValidReadyOp>(operands[0].getDefiningOp());
      if (!wrap)
        return rewriter.notifyMatchFailure(
            operands[0].getDefiningOp(),
            "This conversion only supports wrap-unwrap back-to-back. "
            "Could not find 'wrap'.");
      valid = wrap.getValid();
      data = wrap.getRawInput();
      ready = operands[1];
    } else {
      return failure();
    }

    if (!wrap.getChanOutput().hasOneUse())
      return rewriter.notifyMatchFailure(wrap, [](Diagnostic &d) {
        d << "This conversion only supports wrap-unwrap back-to-back. "
             "Wrap didn't have exactly one use.";
      });
    rewriter.replaceOp(wrap, {nullptr, ready});
    rewriter.replaceOp(unwrap, {data, valid});
    return success();
  }
};
} // anonymous namespace

namespace {
/// Use the op canonicalizer to lower away the op. Assumes the canonicalizer
/// deletes the op.
template <typename Op>
struct CanonicalizerOpLowering : public OpConversionPattern<Op> {
public:
  CanonicalizerOpLowering(MLIRContext *ctxt) : OpConversionPattern<Op>(ctxt) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(Op::canonicalize(op, rewriter)))
      return rewriter.notifyMatchFailure(op->getLoc(), "canonicalizer failed");
    return success();
  }
};
} // anonymous namespace

namespace {
struct ESItoHWPass : public LowerESItoHWBase<ESItoHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower a `wrap.iface` to `wrap.vr` by extracting the wires then feeding the
/// new `wrap.vr`.
struct WrapInterfaceLower : public OpConversionPattern<WrapSVInterfaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
WrapInterfaceLower::matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(wrap, [&operands](Diagnostic &d) {
      d << "wrap.iface has 1 argument. Got " << operands.size() << "operands";
    });
  auto sinkModport = dyn_cast<GetModportOp>(operands[0].getDefiningOp());
  if (!sinkModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sinkModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = wrap.getLoc();
  auto validSignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr);
  Value dataSignal;
  dataSignal = rewriter.create<ReadInterfaceSignalOp>(loc, ifaceInstance,
                                                      ESIHWBuilder::dataStr);
  auto wrapVR = rewriter.create<WrapValidReadyOp>(loc, dataSignal, validSignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr, wrapVR.getReady());
  rewriter.replaceOp(wrap, {wrapVR.getChanOutput()});
  return success();
}

namespace {
/// Lower an unwrap interface to just extract the wires and feed them into an
/// `unwrap.vr`.
struct UnwrapInterfaceLower : public OpConversionPattern<UnwrapSVInterfaceOp> {
public:
  UnwrapInterfaceLower(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnwrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult UnwrapInterfaceLower::matchAndRewrite(
    UnwrapSVInterfaceOp unwrap, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 2)
    return rewriter.notifyMatchFailure(unwrap, [&operands](Diagnostic &d) {
      d << "Unwrap.iface has 2 arguments. Got " << operands.size()
        << "operands";
    });

  auto sourceModport = dyn_cast<GetModportOp>(operands[1].getDefiningOp());
  if (!sourceModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sourceModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = unwrap.getLoc();
  auto readySignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr);
  auto unwrapVR =
      rewriter.create<UnwrapValidReadyOp>(loc, operands[0], readySignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr, unwrapVR.getValid());

  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::dataStr, unwrapVR.getRawOutput());
  rewriter.eraseOp(unwrap);
  return success();
}

namespace {
/// Lower `CosimEndpointOp` ops to a SystemVerilog extern module and a Capnp
/// gasket op.
struct CosimLowering : public OpConversionPattern<CosimEndpointOp> {
public:
  CosimLowering(ESIHWBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimEndpointOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult
CosimLowering::matchAndRewrite(CosimEndpointOp ep, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
#ifndef CAPNP
  (void)builder;
  return rewriter.notifyMatchFailure(
      ep, "Cosim lowering requires the ESI capnp plugin, which was disabled.");
#else
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  auto operands = adaptor.getOperands();
  Value clk = operands[0];
  Value rst = operands[1];
  Value send = operands[2];

  circt::BackedgeBuilder bb(rewriter, loc);
  Type ui64Type =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);
  capnp::TypeSchema sendTypeSchema(send.getType());
  if (!sendTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Send type not supported yet");
  capnp::TypeSchema recvTypeSchema(ep.getRecv().getType());
  if (!recvTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Recv type not supported yet");

  // Set all the parameters.
  SmallVector<Attribute, 8> params;
  if (auto ext = ep->getAttrOfType<StringAttr>("name_ext"))
    params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", ext));
  else
    params.push_back(
        ParamDeclAttr::get("ENDPOINT_ID_EXT", StringAttr::get(ctxt, "")));
  params.push_back(ParamDeclAttr::get(
      "SEND_TYPE_ID",
      IntegerAttr::get(ui64Type, sendTypeSchema.capnpTypeID())));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(sendTypeSchema.size())));
  params.push_back(ParamDeclAttr::get(
      "RECV_TYPE_ID",
      IntegerAttr::get(ui64Type, recvTypeSchema.capnpTypeID())));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(recvTypeSchema.size())));

  // Set up the egest route to drive the EP's send ports.
  ArrayType egestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), sendTypeSchema.size());
  auto sendReady = bb.get(rewriter.getI1Type());
  UnwrapValidReadyOp unwrapSend =
      rewriter.create<UnwrapValidReadyOp>(loc, send, sendReady);
  auto encodeData = rewriter.create<CapnpEncodeOp>(loc, egestBitArrayType, clk,
                                                   unwrapSend.getValid(),
                                                   unwrapSend.getRawOutput());

  // Get information necessary for injest path.
  auto recvReady = bb.get(rewriter.getI1Type());
  ArrayType ingestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), recvTypeSchema.size());

  // Build or get the cached Cosim Endpoint module parameterization.
  Operation *symTable = ep->getParentWithTrait<OpTrait::SymbolTable>();
  HWModuleExternOp endpoint = builder.declareCosimEndpointOp(
      symTable, egestBitArrayType, ingestBitArrayType);

  // Create replacement Cosim_Endpoint instance.
  StringAttr nameAttr = ep->getAttr("name").dyn_cast_or_null<StringAttr>();
  StringRef name = nameAttr ? nameAttr.getValue() : "CosimEndpointOp";
  Value epInstInputs[] = {
      clk, rst, recvReady, unwrapSend.getValid(), encodeData.getCapnpBits(),
  };

  auto cosimEpModule =
      rewriter.create<InstanceOp>(loc, endpoint, name, epInstInputs,
                                  ArrayAttr::get(ctxt, params), StringAttr());
  sendReady.setValue(cosimEpModule.getResult(2));

  // Set up the injest path.
  Value recvDataFromCosim = cosimEpModule.getResult(1);
  Value recvValidFromCosim = cosimEpModule.getResult(0);
  auto decodeData =
      rewriter.create<CapnpDecodeOp>(loc, recvTypeSchema.getType(), clk,
                                     recvValidFromCosim, recvDataFromCosim);
  WrapValidReadyOp wrapRecv = rewriter.create<WrapValidReadyOp>(
      loc, decodeData.getDecodedData(), recvValidFromCosim);
  recvReady.setValue(wrapRecv.getReady());

  // Replace the CosimEndpointOp op.
  rewriter.replaceOp(ep, wrapRecv.getChanOutput());

  return success();
#endif // CAPNP
}

namespace {
/// Lower the encode gasket to SV/HW.
struct EncoderLowering : public OpConversionPattern<CapnpEncodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpEncodeOp enc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(enc,
                                       "encode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema encodeType(enc.getDataToEncode().getType());
    if (!encodeType.isSupported())
      return rewriter.notifyMatchFailure(enc, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value encoderOutput = encodeType.buildEncoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(encoderOutput && "Error in TypeSchema.buildEncoder()");
    rewriter.replaceOp(enc, encoderOutput);
    return success();
#endif
  }
};
} // anonymous namespace

namespace {
/// Lower the decode gasket to SV/HW.
struct DecoderLowering : public OpConversionPattern<CapnpDecodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpDecodeOp dec, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(dec,
                                       "decode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema decodeType(dec.getDecodedData().getType());
    if (!decodeType.isSupported())
      return rewriter.notifyMatchFailure(dec, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value decoderOutput = decodeType.buildDecoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(decoderOutput && "Error in TypeSchema.buildDecoder()");
    rewriter.replaceOp(dec, decoderOutput);
    return success();
#endif
  }
};
} // namespace

void ESItoHWPass::runOnOperation() {
  auto top = getOperation();
  auto ctxt = &getContext();

  // Set up a conversion and give it a set of laws.
  ConversionTarget pass1Target(*ctxt);
  pass1Target.addLegalDialect<CombDialect>();
  pass1Target.addLegalDialect<HWDialect>();
  pass1Target.addLegalDialect<SVDialect>();
  pass1Target.addLegalOp<WrapValidReadyOp, UnwrapValidReadyOp>();
  pass1Target.addLegalOp<CapnpDecodeOp, CapnpEncodeOp>();

  pass1Target.addIllegalOp<WrapSVInterfaceOp, UnwrapSVInterfaceOp>();
  pass1Target.addIllegalOp<PipelineStageOp>();

  // Add all the conversion patterns.
  ESIHWBuilder esiBuilder(top);
  RewritePatternSet pass1Patterns(ctxt);
  pass1Patterns.insert<PipelineStageLowering>(esiBuilder, ctxt);
  pass1Patterns.insert<WrapInterfaceLower>(ctxt);
  pass1Patterns.insert<UnwrapInterfaceLower>(ctxt);
  pass1Patterns.insert<CosimLowering>(esiBuilder);
  pass1Patterns.insert<NullSourceOpLowering>(ctxt);

  // Run the conversion.
  if (failed(
          applyPartialConversion(top, pass1Target, std::move(pass1Patterns))))
    signalPassFailure();

  ConversionTarget pass2Target(*ctxt);
  pass2Target.addLegalDialect<CombDialect>();
  pass2Target.addLegalDialect<HWDialect>();
  pass2Target.addLegalDialect<SVDialect>();
  pass2Target.addIllegalDialect<ESIDialect>();

  RewritePatternSet pass2Patterns(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<UnwrapFIFOOp>>(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<WrapFIFOOp>>(ctxt);
  pass2Patterns.insert<RemoveWrapUnwrap>(ctxt);
  pass2Patterns.insert<EncoderLowering>(ctxt);
  pass2Patterns.insert<DecoderLowering>(ctxt);
  if (failed(
          applyPartialConversion(top, pass2Target, std::move(pass2Patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Emit ESI collateral pass. Collateral includes the capnp schema and a JSON
// descriptor of the service hierarchy.
//===----------------------------------------------------------------------===//

static llvm::json::Value toJSON(Type type) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  StringRef dialect = type.getDialect().getNamespace();
  std::string m;
  Object o = TypeSwitch<Type, Object>(type)
                 .Case([&](ChannelType t) {
                   m = "channel";
                   return Object({{"inner", toJSON(t.getInner())}});
                 })
                 .Case([&](AnyType t) {
                   m = "any";
                   return Object();
                 })
                 .Case([&](StructType t) {
                   m = "struct";
                   Array fields;
                   for (auto field : t.getElements())
                     fields.push_back(Object({{"name", field.name.getValue()},
                                              {"type", toJSON(field.type)}}));
                   return Object({{"fields", Value(std::move(fields))}});
                 })
                 .Default([&](Type t) {
                   llvm::raw_string_ostream(m) << t;
                   return Object();
                 });
  o["dialect"] = dialect;
  if (m.length())
    o["mnemonic"] = m;
  return o;
}

// Serialize an attribute to a JSON value.
static llvm::json::Value toJSON(Attribute attr) {
  // TODO: This is far from complete. Build out as necessary.
  using llvm::json::Value;
  return TypeSwitch<Attribute, Value>(attr)
      .Case([&](StringAttr a) { return a.getValue(); })
      .Case([&](IntegerAttr a) { return a.getValue().getLimitedValue(); })
      .Case([&](TypeAttr a) {
        Type t = a.getValue();
        llvm::json::Object typeMD;
        typeMD["type_desc"] = toJSON(t);

        std::string buf;
        llvm::raw_string_ostream(buf) << t;
        typeMD["mlir_name"] = buf;

        if (auto chanType = t.dyn_cast<ChannelType>()) {
          Type inner = chanType.getInner();
          typeMD["hw_bitwidth"] = hw::getBitWidth(inner);
#ifdef CAPNP
          capnp::TypeSchema schema(inner);
          typeMD["capnp_type_id"] = schema.capnpTypeID();
          typeMD["capnp_name"] = schema.name().str();
#endif
        } else {
          typeMD["hw_bitwidth"] = hw::getBitWidth(t);
        }
        return typeMD;
      })
      .Case([&](ArrayAttr a) {
        return llvm::json::Array(
            llvm::map_range(a, [](Attribute a) { return toJSON(a); }));
      })
      .Case([&](DictionaryAttr a) {
        llvm::json::Object dict;
        for (auto &entry : a.getValue())
          dict[entry.getName().getValue()] = toJSON(entry.getValue());
        return dict;
      })
      .Case([&](InnerRefAttr ref) {
        llvm::json::Object dict;
        dict["outer_sym"] = ref.getModule().getValue();
        dict["inner"] = ref.getName().getValue();
        return dict;
      })
      .Default([&](Attribute a) {
        std::string buff;
        llvm::raw_string_ostream(buff) << a;
        return buff;
      });
}

namespace {
/// Run all the physical lowerings.
struct ESIEmitCollateralPass
    : public ESIEmitCollateralBase<ESIEmitCollateralPass> {
  void runOnOperation() override;

  /// Emit service hierarchy info in JSON format.
  void emitServiceJSON();
};
} // anonymous namespace

void ESIEmitCollateralPass::emitServiceJSON() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();
  SymbolCache topSyms;
  topSyms.addDefinitions(mod);

  // Check for invalid top names.
  for (StringRef topModName : tops)
    if (topSyms.getDefinition(FlatSymbolRefAttr::get(ctxt, topModName)) ==
        nullptr) {
      mod.emitError("Could not find module named '") << topModName << "'\n";
      signalPassFailure();
      return;
    }

  std::string jsonStrBuffer;
  llvm::raw_string_ostream os(jsonStrBuffer);
  llvm::json::OStream j(os, 2);

  // Emit the list of ports of a service declaration.
  auto emitPorts = [&](ServiceDeclOpInterface decl) {
    SmallVector<ServicePortInfo> ports;
    decl.getPortList(ports);
    for (ServicePortInfo port : ports) {
      j.object([&] {
        j.attribute("name", port.name.getValue());
        if (port.toClientType)
          j.attribute("to-client-type", toJSON(port.toClientType));
        if (port.toServerType)
          j.attribute("to-server-type", toJSON(port.toServerType));
      });
    }
  };

  j.object([&] {
    // Emit a list of the service declarations in a design.
    j.attributeArray("declarations", [&] {
      for (auto *op : llvm::make_pointer_range(mod.getOps())) {
        if (auto decl = dyn_cast<ServiceDeclOpInterface>(op)) {
          j.object([&] {
            j.attribute("name", SymbolTable::getSymbolName(op).getValue());
            j.attributeArray("ports", [&] { emitPorts(decl); });
          });
        }
      }
    });

    j.attributeArray("top_levels", [&] {
      for (auto topModName : tops) {
        j.object([&] {
          auto sym = FlatSymbolRefAttr::get(ctxt, topModName);
          Operation *hwMod = topSyms.getDefinition(sym);
          j.attribute("module", toJSON(sym));
          j.attributeArray("services", [&] {
            hwMod->walk([&](ServiceHierarchyMetadataOp md) {
              j.object([&] {
                j.attribute("service", md.getServiceSymbol());
                j.attribute("instance_path",
                            toJSON(md.getServerNamePathAttr()));
              });
            });
          });
        });
      }
    });

    // Get a list of metadata ops which originated in modules (path is empty).
    SmallVector<
        std::pair<hw::HWModuleLike, SmallVector<ServiceHierarchyMetadataOp, 0>>>
        modsWithLocalServices;
    for (auto hwmod : mod.getOps<hw::HWModuleLike>()) {
      SmallVector<ServiceHierarchyMetadataOp, 0> metadataOps;
      hwmod.walk([&metadataOps](ServiceHierarchyMetadataOp md) {
        if (md.getServerNamePath().empty())
          metadataOps.push_back(md);
      });
      if (!metadataOps.empty())
        modsWithLocalServices.push_back(std::make_pair(hwmod, metadataOps));
    }

    // Then output metadata for those modules exclusively.
    j.attributeArray("modules", [&] {
      for (auto &nameMdPair : modsWithLocalServices) {
        hw::HWModuleLike hwmod = nameMdPair.first;
        auto &mdOps = nameMdPair.second;
        j.object([&] {
          j.attribute("symbol", hwmod.moduleName());
          j.attributeArray("services", [&] {
            for (ServiceHierarchyMetadataOp metadata : mdOps) {
              j.object([&] {
                j.attribute("service", metadata.getServiceSymbol());
                j.attribute("impl_type", metadata.getImplType());
                if (metadata.getImplDetailsAttr())
                  j.attribute("impl_details",
                              toJSON(metadata.getImplDetailsAttr()));
                j.attributeArray("clients", [&] {
                  for (auto client : metadata.getClients())
                    j.value(toJSON(client));
                });
              });
            }
          });
        });
      }
    });
  });

  j.flush();
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, "services.json");
  verbatim->setAttr("output_file", outputFileAttr);

  // By now, we should be done with all of the service declarations and
  // metadata ops so we should delete them.
  mod.walk([&](ServiceHierarchyMetadataOp op) { op.erase(); });
  // Track declarations which are still used so that the service impl reqs are
  // still valid.
  DenseSet<StringAttr> stillUsed;
  mod.walk([&](ServiceImplementReqOp req) {
    auto sym = req.getServiceSymbol();
    if (sym.has_value())
      stillUsed.insert(StringAttr::get(req.getContext(), *sym));
  });
  mod.walk([&](ServiceDeclOpInterface decl) {
    if (!stillUsed.contains(SymbolTable::getSymbolName(decl)))
      decl.getOperation()->erase();
  });
}

void ESIEmitCollateralPass::runOnOperation() {
  ModuleOp mod = getOperation();
  auto *ctxt = &getContext();

  emitServiceJSON();

  // Check for cosim endpoints in the design. If the design doesn't have any
  // we don't need a schema.
  WalkResult cosimWalk =
      mod.walk([](CosimEndpointOp _) { return WalkResult::interrupt(); });
  if (!cosimWalk.wasInterrupted())
    return;

  // Generate the schema
  std::string schemaStrBuffer;
  llvm::raw_string_ostream os(schemaStrBuffer);
  if (failed(exportCosimSchema(mod, os))) {
    signalPassFailure();
    return;
  }

  // And stuff if in a verbatim op with a filename, optionally.
  OpBuilder b = OpBuilder::atBlockEnd(mod.getBody());
  auto verbatim = b.create<sv::VerbatimOp>(b.getUnknownLoc(),
                                           StringAttr::get(ctxt, os.str()));
  if (!schemaFile.empty()) {
    auto outputFileAttr = OutputFileAttr::getFromFilename(ctxt, schemaFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIEmitCollateralPass() {
  return std::make_unique<ESIEmitCollateralPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPhysicalLoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> circt::esi::createESItoHWPass() {
  return std::make_unique<ESItoHWPass>();
}

void circt::esi::registerESIPasses() { registerPasses(); }
