//===- LowerFIRRTLToRTL.cpp - Lower FIRRTL -> RTL dialect -----------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/FIRRTL/Ops.h"
#include "cirt/Dialect/FIRRTL/Passes.h"
#include "cirt/Dialect/RTL/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace cirt;
using namespace firrtl;
using namespace rtl;

namespace {
/// Utility class for operation conversions targeting that
/// match exactly one source operation.
template <typename CRTPClass, typename OpTy>
class RTLRewriter : public ConversionPattern {
public:
  RTLRewriter(MLIRContext *ctx)
      : ConversionPattern(OpTy::getOperationName(), /*benefit*/ 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    static_cast<const CRTPClass *>(this)->doIt(OpTy(op), operands, rewriter);
    return success();
  }
};
} // end anonymous namespace

struct LowerConstantOp
    : public RTLRewriter<LowerConstantOp, firrtl::ConstantOp> {
  using RTLRewriter::RTLRewriter;

  void doIt(firrtl::ConstantOp op, ArrayRef<Value> operands,
            ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<rtl::ConstantOp>(op, op.value());
  }
};

struct LowerAddOp : public RTLRewriter<LowerAddOp, firrtl::AddPrimOp> {
  using RTLRewriter::RTLRewriter;

  void doIt(firrtl::AddPrimOp op, ArrayRef<Value> operands,
            ConversionPatternRewriter &rewriter) const {
    // FIXME: Not handling extensions correctly.
    rewriter.replaceOpWithNewOp<rtl::AddOp>(op, operands[0], operands[1]);
  }
};

//===----------------------------------------------------------------------===//
// RTLTypeConverter
//===----------------------------------------------------------------------===//

namespace {
class RTLTypeConverter : public TypeConverter {
public:
  RTLTypeConverter();

  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value> inputs,
                                   Location loc) override;

private:
  static Optional<Type> convertIntType(IntType type);
};
} // end anonymous namespace

RTLTypeConverter::RTLTypeConverter() { addConversion(convertIntType); }

// Lower SInt/UInt to builtin integer type.
Optional<Type> RTLTypeConverter::convertIntType(IntType type) {
  auto width = type.getWidthOrSentinel();
  if (width == -1)
    return None;
  return IntegerType::get(width, type.getContext());
}

Operation *RTLTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                   Type resultType,
                                                   ArrayRef<Value> inputs,
                                                   Location loc) {

  assert(0 && "xx");
}

//===----------------------------------------------------------------------===//
// RTLConversionTarget
//===----------------------------------------------------------------------===//

namespace {
class RTLConversionTarget : public ConversionTarget {
public:
  explicit RTLConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<RTLDialect>();
    this->addIllegalOp<firrtl::ConstantOp>();
    this->addIllegalOp<firrtl::AddPrimOp>();
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
    patterns.insert<LowerConstantOp, LowerAddOp>(&getContext());

    RTLConversionTarget target(getContext());
    RTLTypeConverter typeConverter;
    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      &typeConverter)))
      signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> cirt::firrtl::createLowerFIRRTLToRTLPass() {
  return std::make_unique<FIRRTLLowering>();
}
