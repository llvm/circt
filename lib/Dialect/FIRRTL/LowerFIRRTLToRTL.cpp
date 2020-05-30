//===- LowerFIRRTLToRTL.cpp - Lower FIRRTL -> RTL dialect -----------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/FIRRTL/Ops.h"
#include "cirt/Dialect/FIRRTL/Passes.h"
#include "cirt/Dialect/RTL/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace cirt;
using namespace firrtl;
using namespace rtl;

/// Utility class for operation conversions targeting that
/// match exactly one source operation.
template <typename OpTy>
class ConvertOpToRTLPattern : public ConversionPattern {
public:
  ConvertOpToRTLPattern(MLIRContext *ctx)
      : ConversionPattern(OpTy::getOperationName(), /*benefit*/ 1, ctx) {}
};

// Lower ConstantOp to ConstantOp.
struct LowerConstantOp : public ConvertOpToRTLPattern<firrtl::ConstantOp> {
  using ConvertOpToRTLPattern::ConvertOpToRTLPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cstOp = cast<firrtl::ConstantOp>(op);
    rewriter.replaceOpWithNewOp<rtl::ConstantOp>(op, cstOp.value());
    return success();
  }
};

namespace {
class RTLConversionTarget : public ConversionTarget {
public:
  explicit RTLConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<RTLDialect>();
    this->addIllegalOp<firrtl::ConstantOp>();
  }
};
} // end anonymous namespace

#define GEN_PASS_CLASSES
#include "cirt/Dialect/FIRRTL/FIRRTLPasses.h.inc"

namespace {
struct FIRRTLLowering : public LowerFIRRTLToRTLBase<FIRRTLLowering> {
  /// Run the dialect converter on the FModule.
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    patterns.insert<LowerConstantOp>(&getContext());

    RTLConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> cirt::firrtl::createLowerFIRRTLToRTLPass() {
  return std::make_unique<FIRRTLLowering>();
}
