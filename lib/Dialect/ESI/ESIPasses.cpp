//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "esi-to-rtl"

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace mlir;
using namespace circt::esi;

namespace {

struct ChannelBufferLowering : public ConversionPattern {
  ChannelBufferLowering(MLIRContext *ctx)
      : ConversionPattern(ChannelBuffer::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ChannelBuffer buffer;
    if (!(buffer = dyn_cast<ChannelBuffer>(op)))
      return failure();

    ChannelBufferOptions opts = buffer.options();
    auto type = buffer.getType();
    Value input = buffer.input();

    // Expand 'abstract' buffer into 'physical' stages.
    auto stages = opts.stages();
    size_t numStages = 1;
    if (stages) {
      // Guaranteed positive by the parser.
      numStages = (size_t)stages.getInt();
    }

    for (size_t i = 0; i < numStages; ++i) {
      auto stage = rewriter.create<PipelineStage>(loc, type, input);
      input = stage;
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};

void ESIToPhysicalPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<ESIDialect>();
  target.addIllegalOp<ChannelBuffer>();

  OwningRewritePatternList patterns;
  patterns.insert<ChannelBufferLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
};

struct ESIToRTLPass : public LowerESIToRTLBase<ESIToRTLPass> {

  void runOnOperation() override;
};

void ESIToRTLPass::runOnOperation() { llvm::outs() << "test!\n"; }

} // end anonymous namespace.

namespace circt {
namespace esi {
std::unique_ptr<OperationPass<ModuleOp>> createESILoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createESIToRTLPass() {
  return std::make_unique<ESIToRTLPass>();
}
} // namespace esi
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace

void circt::esi::registerESIPasses() { registerPasses(); }
