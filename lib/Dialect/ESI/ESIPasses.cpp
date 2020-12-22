//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Types.h"
#include "circt/Dialect/SV/Ops.h"
#include "circt/Support/ImplicitLocOpBuilder.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace mlir;
using namespace circt::esi;
using namespace circt::rtl;
using namespace circt::sv;

/// Figure out the number of bits a type takes on the wire in verilog. Doing it
/// here is probably a giant hack. TODO: Establish a canonical method to get
/// this information.
static size_t getNumBits(Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Case<ArrayType>([](ArrayType a) {
        return a.getSize() * getNumBits(a.getInnerType());
      })
      .Default([](Type) { return 0; });
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
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStage>(loc, type, buffer.clk(),
                                                buffer.rstn(), input);
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
// Lower to RTL/SV conversions and pass.
//===----------------------------------------------------------------------===//

namespace {
/// Assist the lowering steps for conversions which need to create auxiliary IR.
class ESIRTLBuilder : public circt::ImplicitLocOpBuilder {
public:
  ESIRTLBuilder(Operation *top);

  RTLExternModuleOp declareStage();

  InterfaceOp getOrConstructInterface(ChannelPort);
  InterfaceOp constructInterface(ChannelPort);

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr clk, rstn, input, output;
  const Identifier width;
  const FlatSymbolRefAttr data, valid, ready, source, sink;

private:
  RTLExternModuleOp declaredStage;
  llvm::DenseMap<Type, InterfaceOp> portTypeLookup;
};
} // anonymous namespace

ESIRTLBuilder::ESIRTLBuilder(Operation *top)
    : ImplicitLocOpBuilder(UnknownLoc::get(getContext()), top),
      a(StringAttr::get("a", getContext())),
      aValid(StringAttr::get("a_valid", getContext())),
      aReady(StringAttr::get("a_ready", getContext())),
      x(StringAttr::get("x", getContext())),
      xValid(StringAttr::get("x_valid", getContext())),
      xReady(StringAttr::get("x_ready", getContext())),
      clk(StringAttr::get("clk", getContext())),
      rstn(StringAttr::get("rstn", getContext())),
      input(StringAttr::get("input", getContext())),
      output(StringAttr::get("output", getContext())),
      width(Identifier::get("WIDTH", getContext())),
      data(FlatSymbolRefAttr::get("data", getContext())),
      valid(FlatSymbolRefAttr::get("valid", getContext())),
      ready(FlatSymbolRefAttr::get("ready", getContext())),
      source(FlatSymbolRefAttr::get("source", getContext())),
      sink(FlatSymbolRefAttr::get("sink", getContext())),
      declaredStage(nullptr) {

  auto regions = top->getRegions();
  if (regions.size() == 0) {
    top->emitError("ESI RTL Builder needs a region to insert RTL.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
RTLExternModuleOp ESIRTLBuilder::declareStage() {
  if (declaredStage)
    return declaredStage;

  auto name = StringAttr::get("ESI_PipelineStage", getContext());
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
  declaredStage = create<RTLExternModuleOp>(name, ports);
  return declaredStage;
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
  auto *ctxt = getContext();

  InterfaceOp iface = create<InterfaceOp>("IDataVR");
  ImplicitLocOpBuilder ib(getLoc(), iface.getRegion());
  ib.createBlock(&iface.getRegion());

  InterfaceSignalOp s;
  ib.create<InterfaceSignalOp>(valid.getRootReference(),
                               TypeAttr::get(getI1Type()));
  ib.create<InterfaceSignalOp>(ready.getRootReference(),
                               TypeAttr::get(getI1Type()));
  ib.create<InterfaceSignalOp>(data.getRootReference(),
                               TypeAttr::get(chan.getInner()));
  ib.create<InterfaceModportOp>(
      source.getRootReference(),
      ArrayAttr::get({ModportStructAttr::get(input, ready, ctxt),
                      ModportStructAttr::get(output, valid, ctxt),
                      ModportStructAttr::get(output, data, ctxt)},
                     ctxt));
  ib.create<InterfaceModportOp>(
      sink.getRootReference(),
      ArrayAttr::get({ModportStructAttr::get(output, ready, ctxt),
                      ModportStructAttr::get(input, valid, ctxt),
                      ModportStructAttr::get(input, data, ctxt)},
                     ctxt));
  ib.create<TypeDeclTerminatorOp>();
  return iface;
}

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

  NamedAttrList stageAttrs = stage.getAttrs();
  size_t width = getNumBits(chPort.getInner());
  stageAttrs.set(builder.width, rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet, so
  // create a temp value and replace it later. Give this constant an odd-looking
  // type to make debugging easier.
  auto tempConstant = rewriter.create<mlir::ConstantIntOp>(loc, 0, 1234);
  auto unwrap =
      rewriter.create<UnwrapValidReady>(loc, stage.input(), tempConstant);

  // Instantiate the "ESI_PipelineStage" external module.
  Value operands[] = {stage.clk(), stage.rstn(), unwrap.rawOutput(),
                      unwrap.valid(), tempConstant};
  Type resultTypes[] = {rewriter.getI1Type(), unwrap.rawOutput().getType(),
                        rewriter.getI1Type()};
  auto stageInst = rewriter.create<InstanceOp>(
      loc, resultTypes, "pipelineStage", stageModule.getName(), operands,
      stageAttrs.getDictionary(rewriter.getContext()));
  auto stageInstResults = stageInst.getResults();
  Value aReady = stageInstResults[0];
  Value x = stageInstResults[1];
  Value xValid = stageInstResults[2];

  // Wrap up the output of the RTL stage module.
  auto wrap = rewriter.create<WrapValidReady>(loc, chPort, rewriter.getI1Type(),
                                              x, xValid);

  // Set back edges correctly and erase temp value.
  unwrap.readyMutable().assign(aReady);
  stageInst.setOperand(4, wrap.ready());
  rewriter.eraseOp(tempConstant);

  rewriter.replaceOp(stage, wrap.chanOutput());
  return success();
}

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV interfaces
/// for now. In the future, it may be possible to select a different format.
struct ESIPortsPass : public LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation();

private:
  void updateFunc(RTLExternModuleOp mod);
  void updateInstance(RTLExternModuleOp mod, InstanceOp inst);
  ESIRTLBuilder *build;
};
} // anonymous namespace

/// Iterate through the `rtl.externmodule`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIRTLBuilder b(top);
  build = &b;

  for (auto mod : top.getOps<RTLExternModuleOp>()) {
    updateFunc(mod);
  }

  build = nullptr;
}

/// Convert all input and output ChannelPorts into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Then call the
/// function which adapts the instances (the hard part).
void ESIPortsPass::updateFunc(RTLExternModuleOp mod) {
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

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(build->sink));
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
    ModportType sourcePort = iface.getModportType(build->source);
    newArgTypes.push_back(sourcePort);
    argAttrs.push_back(resAttrs[resNum]);
    updated = true;
  }

  if (!updated)
    return;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  mod.setAllArgAttrs(argAttrs);
  mod.setAllResultAttrs(newResultAttrs);

  // Find all instances and update them.
  Operation *scope = SymbolTable::getNearestSymbolTable(mod);
  llvm::Optional<SymbolTable::UseRange> instances =
      SymbolTable::getSymbolUses(mod, scope);
  if (!instances)
    return;
  for (auto symUse : *instances) {
    auto instOp = dyn_cast<InstanceOp>(symUse.getUser());
    if (instOp)
      updateInstance(mod, instOp);
  }
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(RTLExternModuleOp mod, InstanceOp inst) {
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
    if (iface.getModportType(build->sink) != funcTy.getInput(opNum)) {
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
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, build->source);
    instBuilder.create<UnwrapSVInterface>(op, sourceModport);
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, build->sink);
    // Finally, add the correct modport to the list of operands.
    newOperands.push_back(sinkModport);
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
    if (iface.getModportType(build->source) != funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelPort (result #")
          << resNum << ", op #" << opNum << ") doesn't match module!";
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
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, build->sink);
    auto newChannel =
        instBuilder.create<WrapSVInterface>(res.getType(), sinkModport);
    // Connect all the old users of the output channel with the newly wrapped
    // replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, build->source);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sourceModport);
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

namespace {
struct ESItoRTLPass : public LowerESItoRTLBase<ESItoRTLPass> {
  void runOnOperation() override;
};
} // anonymous namespace

// static bool esiModulePortFree(RTLExternModuleOp mod) {
//   auto funcType = mod.getType();
//   for (auto arg : funcType.getInputs())
//     if (arg.isa<ChannelPort>())
//       return false;
//   for (auto res : funcType.getResults())
//     if (res.isa<ChannelPort>())
//       return false;
//   return true;
// }

void ESItoRTLPass::runOnOperation() {
  auto top = getOperation();

  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.addLegalDialect<RTLDialect>();
  target.addLegalOp<WrapValidReady, UnwrapValidReady>();
  target.addLegalOp<WrapSVInterface, UnwrapSVInterface>();
  target.addIllegalOp<PipelineStage>();

  // Add all the conversion patterns.
  ESIRTLBuilder esiBuilder(top);
  OwningRewritePatternList patterns;
  patterns.insert<PipelineStageLowering>(esiBuilder, &getContext());

  // Run the conversion.
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
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
