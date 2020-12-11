//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"
#include "circt/Dialect/RTL/Types.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
  size_t w = 0;
  llvm::TypeSwitch<::mlir::Type>(type)
      .Case<IntegerType>([&](IntegerType t) { w = t.getIntOrFloatBitWidth(); })
      .Case<circt::rtl::ArrayType>([&](circt::rtl::ArrayType a) {
        w = a.getSize() * getNumBits(a.getInnerType());
      });
  return w;
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
  Value input = buffer.input();

  // Expand 'abstract' buffer into 'physical' stages.
  auto stages = opts.stages();
  uint64_t numStages = 1;
  if (stages) {
    // Guaranteed positive by the parser.
    numStages = stages.getValue().getLimitedValue();
  }
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStage>(loc, type, input);
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
class ESIRTLBuilder : public TypeConverter, OpBuilder {
public:
  ESIRTLBuilder(Operation *top);

  circt::rtl::RTLExternModuleOp declareStage();

  const StringAttr a, a_valid, a_ready, x, x_valid, x_ready;
  const StringAttr clk, rstn;
  const Identifier parameters, WIDTH;

private:
  circt::rtl::RTLExternModuleOp declaredStage;
  const Location loc;
};
} // anonymous namespace

ESIRTLBuilder::ESIRTLBuilder(Operation *top)
    : OpBuilder(top->getContext()),
      a(StringAttr::get("a", getContext())),
      a_valid(StringAttr::get("a_valid", getContext())),
      a_ready(StringAttr::get("a_ready", getContext())),
      x(StringAttr::get("x", getContext())),
      x_valid(StringAttr::get("x_valid", getContext())),
      x_ready(StringAttr::get("x_ready", getContext())),
      clk(StringAttr::get("clk", getContext())),
      rstn(StringAttr::get("rstn", getContext())),
      parameters(Identifier::get("parameters", getContext())),
      WIDTH(Identifier::get("WIDTH", getContext())),
      declaredStage(nullptr),
      loc(UnknownLoc::get(getContext())) {

  auto regions = top->getRegions();
  if (regions.size() == 0) {
    top->emitError("ESI RTL Builder needs a region to insert RTL");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());

  addConversion([](ChannelPort ch) -> Type { return ch; });
}

circt::rtl::RTLExternModuleOp ESIRTLBuilder::declareStage() {
  using namespace circt::rtl;

  if (declaredStage)
    return declaredStage;

  auto name = StringAttr::get("ESI_PipelineStage", getContext());
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // things.
  ArrayRef<ModulePortInfo> ports = {
      {clk, PortDirection::INPUT, getI1Type(), 0},
      {rstn, PortDirection::INPUT, getI1Type(), 1},
      {a, PortDirection::INPUT, getNoneType(), 2},
      {a_valid, PortDirection::INPUT, getI1Type(), 3},
      {a_ready, PortDirection::OUTPUT, getI1Type(), 2},
      {x, PortDirection::OUTPUT, getNoneType(), 0},
      {x_valid, PortDirection::OUTPUT, getI1Type(), 1},
      {x_ready, PortDirection::INPUT, getI1Type(), 4}};
  declaredStage = create<RTLExternModuleOp>(loc, name, ports);
  return declaredStage;
}

namespace {
/// Construct RTL/SV to act as pipeline stages. They should be double-buffered
/// to support pipelined backpressure. Build a pipeline stage module for each
/// type since not all synthesis engines support modules parameterized by type.
struct PipelineStageLowering : public OpConversionPattern<PipelineStage> {
public:
  PipelineStageLowering(ESIRTLBuilder &builder, MLIRContext *ctxt)
      : OpConversionPattern(builder, ctxt), builder(builder) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PipelineStage stage, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIRTLBuilder &builder;
};
} // anonymous namespace

LogicalResult PipelineStageLowering::matchAndRewrite(
    PipelineStage stage, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  using namespace circt::sv;
  auto *ctxt = rewriter.getContext();
  auto loc = stage.getLoc();
  auto chPort = stage.input().getType().dyn_cast<ChannelPort>();
  if (!chPort)
    return failure();
  auto stageModule = builder.declareStage();

  MutableDictionaryAttr stageAttrs;
  size_t width = getNumBits(chPort.getInner());
  stageAttrs.set(builder.WIDTH, rewriter.getUI32IntegerAttr(width));

  auto stageInst = rewriter.create<circt::rtl::InstanceOp>(
      loc, stage.getType(), "pipelineStage", stageModule.getName(), operands,
      stageAttrs.getDictionary(ctxt));
  rewriter.replaceOp(stage, stageInst.getResults());
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
  target.addLegalDialect<circt::sv::SVDialect>();
  target.addLegalOp<WrapSVInterface, UnwrapSVInterface>();
  // target.addIllegalDialect<circt::esi::ESIDialect>();
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
