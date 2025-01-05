#include "circt/Conversion/ICE40ToBLIF.h"
#include "circt/Dialect/BLIF/BLIFDialect.h"
#include "circt/Dialect/BLIF/BLIFOps.h"
#include "circt/Dialect/ICE40/ICE40Dialect.h"
#include "circt/Dialect/ICE40/ICE40Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_ICE40TOBLIF
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace mlir;

namespace {
struct ICE40ToBLIFPass
    : public ::circt::impl::ICE40ToBLIFBase<ICE40ToBLIFPass> {
  void populateICE40ToBLIFPatterns(RewritePatternSet &patterns);

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Define the conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<blif::BLIFDialect>();
    target.addIllegalDialect<ice40::ICE40Dialect>();

    // Define the type converter
    TypeConverter typeConverter;

    // Define the rewrite patterns
    RewritePatternSet patterns(&getContext());
    populateICE40ToBLIFPatterns(patterns);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Define the patterns to convert ICE40 operations to BLIF operations

namespace {
class SBLut4OpLowering : public OpConversionPattern<ice40::SBLut4Op> {
public:
  SBLut4OpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(ice40::SBLut4Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new BLIF LibraryGateOp with the same operands and attributes
    rewriter.replaceOpWithNewOp<blif::LibraryGateOp>(
        op, rewriter.getIntegerType(1), "SB_LUT4", adaptor.getOperands());
    return success();
  }
};

class SBCarryOpLowering : public OpConversionPattern<ice40::SBCarryOp> {
public:
  SBCarryOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(ice40::SBCarryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new BLIF LibraryGateOp with the same operands and attributes
    rewriter.replaceOpWithNewOp<blif::LibraryGateOp>(
        op, rewriter.getIntegerType(1), "SB_CARRY", adaptor.getOperands());
    return success();
  }
};

class SBLut4CarryOpLowering : public OpConversionPattern<ice40::SBLut4CarryOp> {
public:
  SBLut4CarryOpLowering(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(ice40::SBLut4CarryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new BLIF LibraryGateOp with the same operands and attributes
    auto newFunc = rewriter.create<blif::LibraryGateOp>(
        op.getLoc(), rewriter.getIntegerType(1), "SB_LUT4",
        adaptor.getOperands().take_front(4));
    Value carry[] = {adaptor.getOperands()[0], adaptor.getOperands()[1],
                     adaptor.getOperands()[2], adaptor.getOperands()[4]};
    auto newCarry = rewriter.create<blif::LibraryGateOp>(
        op.getLoc(), rewriter.getIntegerType(1), "SB_CARRY", carry);
    rewriter.replaceOpWithMultiple(op,
                                   {newFunc.getResult(), newCarry.getResult()});
    return success();
  }
};
} // end anonymous namespace

void ICE40ToBLIFPass::populateICE40ToBLIFPatterns(RewritePatternSet &patterns) {
  patterns.add<SBLut4OpLowering, SBCarryOpLowering, SBLut4CarryOpLowering>(
      &getContext());
}

} // end anonymous namespace

std::unique_ptr<Pass> circt::createICE40ToBLIFPass() {
  return std::make_unique<ICE40ToBLIFPass>();
}
