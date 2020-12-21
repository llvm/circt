//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Types.h"
#include "circt/Support/BackedgeBuilder.h"

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

/// Figure out the number of bits a type takes on the wire in verilog. Doing it
/// here is probably a giant hack. TODO: Establish a canonical method to get
/// this information.
static size_t getNumBits(Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Case<circt::rtl::ArrayType>([](circt::rtl::ArrayType a) {
        return a.getSize() * getNumBits(a.getElementType());
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
class ESIRTLBuilder : public OpBuilder {
public:
  ESIRTLBuilder(Operation *top);

  circt::rtl::RTLExternModuleOp declareStage();

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr clk, rstn;
  const Identifier width;

private:
  circt::rtl::RTLExternModuleOp declaredStage;
  const Location loc;
};
} // anonymous namespace

ESIRTLBuilder::ESIRTLBuilder(Operation *top)
    : OpBuilder(top->getContext()), a(StringAttr::get("a", getContext())),
      aValid(StringAttr::get("a_valid", getContext())),
      aReady(StringAttr::get("a_ready", getContext())),
      x(StringAttr::get("x", getContext())),
      xValid(StringAttr::get("x_valid", getContext())),
      xReady(StringAttr::get("x_ready", getContext())),
      clk(StringAttr::get("clk", getContext())),
      rstn(StringAttr::get("rstn", getContext())),
      width(Identifier::get("WIDTH", getContext())), declaredStage(nullptr),
      loc(UnknownLoc::get(getContext())) {

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
circt::rtl::RTLExternModuleOp ESIRTLBuilder::declareStage() {
  using namespace circt::rtl;

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
  declaredStage = create<RTLExternModuleOp>(loc, name, ports);
  return declaredStage;
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
  circt::BackedgeBuilder back(rewriter, loc);
  circt::Backedge wrapReady = back.get(rewriter.getI1Type());
  auto unwrap =
      rewriter.create<UnwrapValidReady>(loc, stage.input(), wrapReady);

  // Instantiate the "ESI_PipelineStage" external module.
  circt::Backedge stageReady = back.get(rewriter.getI1Type());
  Value operands[] = {stage.clk(), stage.rstn(), unwrap.rawOutput(),
                      unwrap.valid(), stageReady};
  Type resultTypes[] = {rewriter.getI1Type(), unwrap.rawOutput().getType(),
                        rewriter.getI1Type()};
  auto stageInst = rewriter.create<circt::rtl::InstanceOp>(
      loc, resultTypes, "pipelineStage", stageModule.getName(), operands,
      stageAttrs.getDictionary(rewriter.getContext()));
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
struct ESItoRTLPass : public LowerESItoRTLBase<ESItoRTLPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESItoRTLPass::runOnOperation() {
  auto top = getOperation();

  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.addLegalDialect<circt::rtl::RTLDialect>();
  target.addLegalOp<WrapValidReady, UnwrapValidReady>();
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
