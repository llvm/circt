//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ImplicitLocOpBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#endif

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::comb;
using namespace circt::esi;
using namespace circt::rtl;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

namespace {
/// Assist the lowering steps for conversions which need to create auxiliary IR.
class ESIRTLBuilder : public circt::ImplicitLocOpBuilder {
public:
  ESIRTLBuilder(Operation *top);

  RTLModuleExternOp declareStage();
  RTLModuleExternOp declareCosimEndpoint();

  InterfaceOp getOrConstructInterface(ChannelPort);
  InterfaceOp constructInterface(ChannelPort);

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr dataOutValid, dataOutReady, dataOut, dataInValid,
      dataInReady, dataIn;
  const StringAttr clk, rstn;
  const Identifier width;

  // Various identifier strings. Keep them all here in case we rename them.
  static constexpr char dataStr[] = "data", validStr[] = "valid",
                        readyStr[] = "ready", sourceStr[] = "source",
                        sinkStr[] = "sink";

private:
  /// Construct a type-appropriate name for the interface, making sure it's not
  /// taken in the symbol table.
  StringAttr constructInterfaceName(ChannelPort);

  RTLModuleExternOp declaredStage;
  RTLModuleExternOp declaredCosimEndpoint;
  llvm::DenseMap<Type, InterfaceOp> portTypeLookup;
};
} // anonymous namespace

// C++ requires this for showing it what object file it should store these
// symbols in. They should be inline but that feature wasn't added until C++17.
constexpr char ESIRTLBuilder::dataStr[], ESIRTLBuilder::validStr[],
    ESIRTLBuilder::readyStr[], ESIRTLBuilder::sourceStr[],
    ESIRTLBuilder::sinkStr[];

ESIRTLBuilder::ESIRTLBuilder(Operation *top)
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
      rstn(StringAttr::get(getContext(), "rstn")),
      width(Identifier::get("WIDTH", getContext())), declaredStage(nullptr) {

  auto regions = top->getRegions();
  if (regions.size() == 0) {
    top->emitError("ESI RTL Builder needs a region to insert RTL.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

StringAttr ESIRTLBuilder::constructInterfaceName(ChannelPort port) {
  Operation *tableOp =
      getInsertionPoint()->getParentWithTrait<OpTrait::SymbolTable>();

  // Get a name based on the type.
  std::string portTypeName;
  llvm::raw_string_ostream nameOS(portTypeName);
  TypeSwitch<Type>(port.getInner())
      .Case([&](ArrayType arr) {
        nameOS << "ArrayOf" << arr.getSize() << 'x' << arr.getElementType();
      })
      .Default([&](Type t) { nameOS << port.getInner(); });

  // Normalize the type name.
  for (char &ch : portTypeName) {
    if (isalpha(ch) || isdigit(ch) || ch == '_')
      continue;
    ch = '_';
  }

  // Don't allow the name to end with '_'.
  ssize_t i = portTypeName.size() - 1;
  while (i >= 0 && portTypeName[i] == '_') {
    --i;
  }
  portTypeName = portTypeName.substr(0, i + 1);

  // All stage names start with this.
  SmallString<64> proposedName("IValidReady_");
  proposedName.append(portTypeName);

  // Make sure that this symbol isn't taken. If it is, append a number and try
  // again.
  size_t baseLength = proposedName.size();
  size_t tries = 0;
  while (SymbolTable::lookupSymbolIn(tableOp, proposedName)) {
    proposedName.resize(baseLength);
    proposedName.append(llvm::utostr(++tries));
  }

  return StringAttr::get(getContext(), proposedName);
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
RTLModuleExternOp ESIRTLBuilder::declareStage() {
  if (declaredStage)
    return declaredStage;

  auto name = StringAttr::get(getContext(), "ESI_PipelineStage");
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  ModulePortInfo ports[] = {{clk, PortDirection::INPUT, getI1Type(), 0},
                            {rstn, PortDirection::INPUT, getI1Type(), 1},
                            {a, PortDirection::INPUT, getNoneType(), 2},
                            {aValid, PortDirection::INPUT, getI1Type(), 3},
                            {aReady, PortDirection::OUTPUT, getI1Type(), 0},
                            {x, PortDirection::OUTPUT, getNoneType(), 1},
                            {xValid, PortDirection::OUTPUT, getI1Type(), 2},
                            {xReady, PortDirection::INPUT, getI1Type(), 4}};
  declaredStage = create<RTLModuleExternOp>(name, ports);
  return declaredStage;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
RTLModuleExternOp ESIRTLBuilder::declareCosimEndpoint() {
  if (declaredCosimEndpoint)
    return declaredCosimEndpoint;
  auto name = StringAttr::get(getContext(), "Cosim_Endpoint");
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  ModulePortInfo ports[] = {
      {clk, PortDirection::INPUT, getI1Type(), 0},
      {rstn, PortDirection::INPUT, getI1Type(), 1},
      {dataOutValid, PortDirection::OUTPUT, getI1Type(), 0},
      {dataOutReady, PortDirection::INPUT, getI1Type(), 2},
      {dataOut, PortDirection::OUTPUT, getNoneType(), 1},
      {dataInValid, PortDirection::INPUT, getI1Type(), 3},
      {dataInReady, PortDirection::OUTPUT, getI1Type(), 2},
      {dataIn, PortDirection::INPUT, getNoneType(), 4}};
  declaredCosimEndpoint = create<RTLModuleExternOp>(name, ports);
  return declaredCosimEndpoint;
}

/// Return the InterfaceType which corresponds to an ESI port type. If it
/// doesn't exist in the cache, build the InterfaceOp and the corresponding
/// type.
InterfaceOp ESIRTLBuilder::getOrConstructInterface(ChannelPort t) {
  auto ifaceIter = portTypeLookup.find(t);
  if (ifaceIter != portTypeLookup.end())
    return ifaceIter->second;
  auto iface = constructInterface(t);
  portTypeLookup[t] = iface;
  return iface;
}

InterfaceOp ESIRTLBuilder::constructInterface(ChannelPort chan) {
  InterfaceOp iface = create<InterfaceOp>(constructInterfaceName(chan));
  ImplicitLocOpBuilder ib(getLoc(), iface.getRegion());
  ib.createBlock(&iface.getRegion());

  InterfaceSignalOp s;
  ib.create<InterfaceSignalOp>(validStr, getI1Type());
  ib.create<InterfaceSignalOp>(readyStr, getI1Type());
  ib.create<InterfaceSignalOp>(dataStr, chan.getInner());
  ib.create<InterfaceModportOp>(
      sinkStr, /*inputs=*/ArrayRef<StringRef>{readyStr},
      /*outputs=*/ArrayRef<StringRef>{validStr, dataStr});
  ib.create<InterfaceModportOp>(
      sourceStr,
      /*inputs=*/ArrayRef<StringRef>{validStr, dataStr},
      /*outputs=*/ArrayRef<StringRef>{readyStr});
  ib.create<TypeDeclTerminatorOp>();
  return iface;
}

//===----------------------------------------------------------------------===//
// Lower to physical conversions and pass.
//===----------------------------------------------------------------------===//

namespace {
/// Lower `ChannelBuffer`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct ChannelBufferLowering : public OpConversionPattern<ChannelBuffer> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ChannelBuffer buffer, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult ChannelBufferLowering::matchAndRewrite(
    ChannelBuffer buffer, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto loc = buffer.getLoc();

  ChannelBufferOptions opts = buffer.options();
  auto type = buffer.getType();

  // Expand 'abstract' buffer into 'physical' stages.
  auto stages = opts.stages();
  uint64_t numStages = 1;
  if (stages) {
    // Guaranteed positive by the parser.
    numStages = stages.getValue().getLimitedValue();
  }
  Value input = buffer.input();
  StringAttr bufferName = buffer.options().name();
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStage>(loc, type, buffer.clk(),
                                                buffer.rstn(), input);
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
/// Run all the physical lowerings.
struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIToPhysicalPass::runOnOperation() {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.addLegalDialect<ESIDialect>();
  target.addIllegalOp<ChannelBuffer>();

  // Add all the conversion patterns.
  OwningRewritePatternList patterns;
  patterns.insert<ChannelBufferLowering>(&getContext());

  // Run the conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
};

//===----------------------------------------------------------------------===//
// Lower ESI ports pass.
//===----------------------------------------------------------------------===//

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV interfaces
/// for now. In the future, it may be possible to select a different format.
struct ESIPortsPass : public LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation() override;

private:
  bool updateFunc(RTLModuleExternOp mod);
  void updateInstance(RTLModuleExternOp mod, InstanceOp inst);
  ESIRTLBuilder *build;
};
} // anonymous namespace

/// Iterate through the `rtl.module.extern`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIRTLBuilder b(top);
  build = &b;

  // Find all externmodules and try to modify them. Remember the modified ones.
  DenseMap<StringRef, RTLModuleExternOp> modsMutated;
  for (auto mod : top.getOps<RTLModuleExternOp>())
    if (updateFunc(mod))
      modsMutated[mod.getName()] = mod;

  // Find all instances and update them.
  top.walk([&modsMutated, this](InstanceOp inst) {
    auto mapIter = modsMutated.find(inst.moduleName());
    if (mapIter != modsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  build = nullptr;
}

/// Convert all input and output ChannelPorts into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Returns true
/// if 'mod' was updated. Delay updating the instances to amortize the IR walk
/// over all the module updates.
bool ESIPortsPass::updateFunc(RTLModuleExternOp mod) {
  auto *ctxt = &getContext();
  auto funcType = mod.getType();

  bool updated = false;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found.
  SmallVector<Type, 16> newArgTypes;
  for (auto argTy : funcType.getInputs()) {
    auto chanTy = argTy.dyn_cast<ChannelPort>();
    if (!chanTy) {
      newArgTypes.push_back(argTy);
      continue;
    }

    // When we find one, construct an interface, and add the 'source' modport to
    // the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(ESIRTLBuilder::sourceStr));
    updated = true;
  }

  SmallVector<DictionaryAttr, 16> argAttrs;
  mod.getAllArgAttrs(argAttrs);
  SmallVector<DictionaryAttr, 16> resAttrs;
  mod.getAllResultAttrs(resAttrs);

  // Iterate through the results and append to one of the two below lists. The
  // first for non-ESI-ports. The second, ports which have been re-located to an
  // operand.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  for (size_t resNum = 0, numRes = funcType.getNumResults(); resNum < numRes;
       ++resNum) {
    Type resTy = funcType.getResult(resNum);
    auto chanTy = resTy.dyn_cast<ChannelPort>();
    if (!chanTy) {
      newResultTypes.push_back(resTy);
      newResultAttrs.push_back(resAttrs[resNum]);
      continue;
    }

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    InterfaceOp iface = build->getOrConstructInterface(chanTy);
    ModportType sinkPort = iface.getModportType(ESIRTLBuilder::sinkStr);
    newArgTypes.push_back(sinkPort);
    argAttrs.push_back(resAttrs[resNum]);
    updated = true;
  }

  if (!updated)
    return false;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  mod.setAllArgAttrs(argAttrs);
  mod.setAllResultAttrs(newResultAttrs);
  return true;
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(RTLModuleExternOp mod, InstanceOp inst) {
  using namespace circt::sv;
  circt::ImplicitLocOpBuilder instBuilder(inst);
  FunctionType funcTy = mod.getType();

  // op counter for error reporting purposes.
  size_t opNum = 0;
  // List of new operands.
  SmallVector<Value, 16> newOperands;

  // Fill the new operand list with old plain operands and mutated ones.
  for (auto op : inst.getOperands()) {
    auto instChanTy = op.getType().dyn_cast<ChannelPort>();
    if (!instChanTy) {
      newOperands.push_back(op);
      ++opNum;
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIRTLBuilder::sourceStr) !=
        funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelPort (operand #")
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
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIRTLBuilder::sinkStr);
    instBuilder.create<UnwrapSVInterface>(op, sinkModport);
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIRTLBuilder::sourceStr);
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
    auto instChanTy = res.getType().dyn_cast<ChannelPort>();
    if (!instChanTy) {
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIRTLBuilder::sinkStr) !=
        funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelPort (result #")
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
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIRTLBuilder::sourceStr);
    auto newChannel =
        instBuilder.create<WrapSVInterface>(res.getType(), sourceModport);
    // Connect all the old users of the output channel with the newly wrapped
    // replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIRTLBuilder::sinkStr);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sinkModport);
  }

  // Create the new instance!
  InstanceOp newInst = instBuilder.create<InstanceOp>(
      newResultTypes, newOperands, inst.getAttrs());
  // Go through the old list of non-ESI result values, and replace them with the
  // new non-ESI results.
  for (size_t resNum = 0, numRes = newResults.size(); resNum < numRes;
       ++resNum) {
    newResults[resNum].replaceAllUsesWith(newInst.getResult(resNum));
  }
  // Erase the old instance!
  inst.erase();
}

//===----------------------------------------------------------------------===//
// Lower to RTL/SV conversions and pass.
//===----------------------------------------------------------------------===//

namespace {
/// Lower PipelineStage ops to an RTL implementation. Unwrap and re-wrap
/// appropriately. Another conversion will take care merging the resulting
/// adjacent wrap/unwrap ops.
struct PipelineStageLowering : public OpConversionPattern<PipelineStage> {
public:
  PipelineStageLowering(ESIRTLBuilder &builder, MLIRContext *ctxt)
      : OpConversionPattern(ctxt), builder(builder) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PipelineStage stage, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIRTLBuilder &builder;
};
} // anonymous namespace

LogicalResult PipelineStageLowering::matchAndRewrite(
    PipelineStage stage, ArrayRef<Value> stageOperands,
    ConversionPatternRewriter &rewriter) const {
  auto loc = stage.getLoc();
  auto chPort = stage.input().getType().dyn_cast<ChannelPort>();
  if (!chPort)
    return failure();
  auto stageModule = builder.declareStage();

  NamedAttrList stageParams;
  size_t width = circt::rtl::getBitWidth(chPort.getInner());
  stageParams.set(builder.width, rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet, so
  // create a temp value and replace it later. Give this constant an odd-looking
  // type to make debugging easier.
  circt::BackedgeBuilder back(rewriter, loc);
  circt::Backedge wrapReady = back.get(rewriter.getI1Type());
  auto unwrap =
      rewriter.create<UnwrapValidReady>(loc, stage.input(), wrapReady);

  StringRef pipeStageName = "pipelineStage";
  if (auto name = stage->getAttrOfType<StringAttr>("name"))
    pipeStageName = name.getValue();

  // Instantiate the "ESI_PipelineStage" external module.
  circt::Backedge stageReady = back.get(rewriter.getI1Type());
  Value operands[] = {stage.clk(), stage.rstn(), unwrap.rawOutput(),
                      unwrap.valid(), stageReady};
  Type resultTypes[] = {rewriter.getI1Type(), unwrap.rawOutput().getType(),
                        rewriter.getI1Type()};
  auto stageInst = rewriter.create<InstanceOp>(
      loc, resultTypes, pipeStageName, stageModule.getName(), operands,
      stageParams.getDictionary(rewriter.getContext()));
  auto stageInstResults = stageInst.getResults();

  // Set a_ready (from the unwrap) back edge correctly to its output from stage.
  wrapReady.setValue(stageInstResults[0]);

  Value x = stageInstResults[1];
  Value xValid = stageInstResults[2];

  // Wrap up the output of the RTL stage module.
  auto wrap = rewriter.create<WrapValidReady>(loc, chPort, rewriter.getI1Type(),
                                              x, xValid);
  // Set the stages x_ready backedge correctly.
  stageReady.setValue(wrap.ready());

  rewriter.replaceOp(stage, wrap.chanOutput());
  return success();
}
namespace {
/// Eliminate back-to-back wrap-unwraps to reduce the number of ESI channels.
struct RemoveWrapUnwrap : public ConversionPattern {
public:
  RemoveWrapUnwrap() : ConversionPattern(/*benefit=*/1, MatchAnyOpTypeTag()) {}

  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value valid, ready, data;
    WrapValidReady wrap = dyn_cast<WrapValidReady>(op);
    UnwrapValidReady unwrap = dyn_cast<UnwrapValidReady>(op);
    if (wrap) {
      unwrap =
          dyn_cast<UnwrapValidReady>(wrap.chanOutput().use_begin()->getOwner());
      if (!unwrap)
        return rewriter.notifyMatchFailure(wrap, [](Diagnostic &d) {
          d << "This conversion only supports wrap-unwrap back-to-back. "
               "Could not find 'unwrap'.";
        });
      data = operands[0];
      valid = operands[1];
      ready = unwrap.ready();
    } else if (unwrap) {
      wrap = dyn_cast<WrapValidReady>(operands[0].getDefiningOp());
      if (!wrap)
        return rewriter.notifyMatchFailure(wrap, [](Diagnostic &d) {
          d << "This conversion only supports wrap-unwrap back-to-back. "
               "Could not find 'wrap'.";
        });
      valid = wrap.valid();
      data = wrap.rawInput();
      ready = operands[1];
    } else {
      return failure();
    }

    if (!wrap.chanOutput().hasOneUse())
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
struct ESItoRTLPass : public LowerESItoRTLBase<ESItoRTLPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower a `wrap.iface` to `wrap.vr` by extracting the wires then feeding the
/// new `wrap.vr`.
struct WrapInterfaceLower : public OpConversionPattern<WrapSVInterface> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WrapSVInterface wrap, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
WrapInterfaceLower::matchAndRewrite(WrapSVInterface wrap,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(wrap, [&operands](Diagnostic &d) {
      d << "wrap.iface has 1 argument. Got " << operands.size() << "operands";
    });
  auto sinkModport = dyn_cast<GetModportOp>(operands[0].getDefiningOp());
  if (!sinkModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sinkModport.iface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = wrap.getLoc();
  auto validSignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::validStr);
  auto dataSignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::dataStr);
  auto wrapVR = rewriter.create<WrapValidReady>(loc, dataSignal, validSignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::readyStr, wrapVR.ready());
  rewriter.replaceOp(wrap, {wrapVR.chanOutput()});
  return success();
}

namespace {
/// Lower an unwrap interface to just extract the wires and feed them into an
/// `unwrap.vr`.
struct UnwrapInterfaceLower : public OpConversionPattern<UnwrapSVInterface> {
public:
  UnwrapInterfaceLower(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnwrapSVInterface wrap, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult UnwrapInterfaceLower::matchAndRewrite(
    UnwrapSVInterface unwrap, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (operands.size() != 2)
    return rewriter.notifyMatchFailure(unwrap, [&operands](Diagnostic &d) {
      d << "Unwrap.iface has 2 arguments. Got " << operands.size()
        << "operands";
    });

  auto sourceModport = dyn_cast<GetModportOp>(operands[1].getDefiningOp());
  if (!sourceModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sourceModport.iface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = unwrap.getLoc();
  auto readySignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::readyStr);
  auto unwrapVR =
      rewriter.create<UnwrapValidReady>(loc, operands[0], readySignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::validStr, unwrapVR.valid());
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIRTLBuilder::dataStr, unwrapVR.rawOutput());
  rewriter.eraseOp(unwrap);
  return success();
}

namespace {
/// Lower `CosimEndpoint` ops to a SystemVerilog extern module and a Capnp
/// gasket op.
struct CosimLowering : public OpConversionPattern<CosimEndpoint> {
public:
  CosimLowering(ESIRTLBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimEndpoint, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIRTLBuilder &builder;
};
} // anonymous namespace

LogicalResult
CosimLowering::matchAndRewrite(CosimEndpoint ep, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
#ifndef CAPNP
  (void)builder;
  return rewriter.notifyMatchFailure(
      ep, "Cosim lowering requires the ESI capnp plugin, which was disabled.");
#else
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  Value clk = operands[0];
  Value rstn = operands[1];
  Value send = operands[2];

  circt::BackedgeBuilder bb(rewriter, loc);
  builder.declareCosimEndpoint();
  Type ui64Type =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);
  capnp::TypeSchema sendTypeSchema(send.getType());
  if (!sendTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Send type not supported yet");
  capnp::TypeSchema recvTypeSchema(ep.recv().getType());
  if (!recvTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Recv type not supported yet");

  // Set all the parameters.
  NamedAttrList params;
  params.set("ENDPOINT_ID", rewriter.getI32IntegerAttr(ep.endpointID()));
  params.set("SEND_TYPE_ID",
             IntegerAttr::get(ui64Type, sendTypeSchema.capnpTypeID()));
  params.set("SEND_TYPE_SIZE_BITS",
             rewriter.getI32IntegerAttr(sendTypeSchema.size()));
  params.set("RECV_TYPE_ID",
             IntegerAttr::get(ui64Type, recvTypeSchema.capnpTypeID()));
  params.set("RECV_TYPE_SIZE_BITS",
             rewriter.getI32IntegerAttr(recvTypeSchema.size()));

  // Set up the egest route to drive the EP's send ports.
  ArrayType egestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), sendTypeSchema.size());
  auto sendReady = bb.get(rewriter.getI1Type());
  UnwrapValidReady unwrapSend =
      rewriter.create<UnwrapValidReady>(loc, send, sendReady);
  auto encodeData = rewriter.create<CapnpEncode>(
      loc, egestBitArrayType, clk, unwrapSend.valid(), unwrapSend.rawOutput());

  // Get information necessary for injest path.
  auto recvReady = bb.get(rewriter.getI1Type());
  ArrayType ingestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), recvTypeSchema.size());

  // Create replacement Cosim_Endpoint instance.
  StringAttr nameAttr = ep->getAttr("name").dyn_cast_or_null<StringAttr>();
  StringRef name = nameAttr ? nameAttr.getValue() : "cosimEndpoint";
  Value epInstInputs[] = {
      clk, rstn, recvReady, unwrapSend.valid(), encodeData.capnpBits(),
  };
  Type epInstOutputs[] = {rewriter.getI1Type(), ingestBitArrayType,
                          rewriter.getI1Type()};
  auto cosimEpModule =
      rewriter.create<InstanceOp>(loc, epInstOutputs, name, "Cosim_Endpoint",
                                  epInstInputs, params.getDictionary(ctxt));
  sendReady.setValue(cosimEpModule.getResult(2));

  // Set up the injest path.
  Value recvDataFromCosim = cosimEpModule.getResult(1);
  Value recvValidFromCosim = cosimEpModule.getResult(0);
  auto decodeData =
      rewriter.create<CapnpDecode>(loc, recvTypeSchema.getType(), clk,
                                   recvValidFromCosim, recvDataFromCosim);
  WrapValidReady wrapRecv = rewriter.create<WrapValidReady>(
      loc, decodeData.decodedData(), recvValidFromCosim);
  recvReady.setValue(wrapRecv.ready());

  // Replace the CosimEndpoint op.
  rewriter.replaceOp(ep, wrapRecv.chanOutput());

  return success();
#endif // CAPNP
}

namespace {
/// Lower the encode gasket to SV/RTL.
struct EncoderLowering : public OpConversionPattern<CapnpEncode> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpEncode enc, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(enc,
                                       "encode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema encodeType(enc.dataToEncode().getType());
    if (!encodeType.isSupported())
      return rewriter.notifyMatchFailure(enc, "Type not supported yet");
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
/// Lower the decode gasket to SV/RTL.
struct DecoderLowering : public OpConversionPattern<CapnpDecode> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpDecode dec, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(dec,
                                       "decode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::TypeSchema decodeType(dec.decodedData().getType());
    if (!decodeType.isSupported())
      return rewriter.notifyMatchFailure(dec, "Type not supported yet");
    Value decoderOutput = decodeType.buildDecoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(decoderOutput && "Error in TypeSchema.buildDecoder()");
    rewriter.replaceOp(dec, decoderOutput);
    return success();
#endif
  }
};
} // namespace

void ESItoRTLPass::runOnOperation() {
  auto top = getOperation();
  auto ctxt = &getContext();

  // Set up a conversion and give it a set of laws.
  ConversionTarget pass1Target(*ctxt);
  pass1Target.addLegalDialect<CombDialect>();
  pass1Target.addLegalDialect<RTLDialect>();
  pass1Target.addLegalDialect<SVDialect>();
  pass1Target.addLegalOp<WrapValidReady, UnwrapValidReady>();
  pass1Target.addLegalOp<CapnpDecode, CapnpEncode>();

  pass1Target.addIllegalOp<WrapSVInterface, UnwrapSVInterface>();
  pass1Target.addIllegalOp<PipelineStage>();

  // Add all the conversion patterns.
  ESIRTLBuilder esiBuilder(top);
  OwningRewritePatternList pass1Patterns;
  pass1Patterns.insert<PipelineStageLowering>(esiBuilder, ctxt);
  pass1Patterns.insert<WrapInterfaceLower>(ctxt);
  pass1Patterns.insert<UnwrapInterfaceLower>(ctxt);
  pass1Patterns.insert<CosimLowering>(esiBuilder);

  // Run the conversion.
  if (failed(
          applyPartialConversion(top, pass1Target, std::move(pass1Patterns))))
    signalPassFailure();

  ConversionTarget pass2Target(*ctxt);
  pass2Target.addLegalDialect<CombDialect>();
  pass2Target.addLegalDialect<RTLDialect>();
  pass2Target.addLegalDialect<SVDialect>();
  pass2Target.addIllegalDialect<ESIDialect>();

  OwningRewritePatternList pass2Patterns;
  pass2Patterns.insert<RemoveWrapUnwrap>();
  pass2Patterns.insert<EncoderLowering>(ctxt);
  pass2Patterns.insert<DecoderLowering>(ctxt);
  if (failed(
          applyPartialConversion(top, pass2Target, std::move(pass2Patterns))))
    signalPassFailure();
}

namespace circt {
namespace esi {
std::unique_ptr<OperationPass<ModuleOp>> createESIPhysicalLoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createESItoRTLPass() {
  return std::make_unique<ESItoRTLPass>();
}

} // namespace esi
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace

void circt::esi::registerESIPasses() { registerPasses(); }
