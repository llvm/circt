//===- LowerToRTL.cpp - Lower FIRRTL -> RTL dialect -----------------------===//
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
// Helpers
//===----------------------------------------------------------------------===//

static Value mapOperand(Value operand, Operation *op,
                        ConversionPatternRewriter &rewriter) {
  auto opType = operand.getType();
  if (auto intType = opType.dyn_cast<IntType>()) {
    auto destType = RTLTypeConverter::convertIntType(intType);
    if (!destType.hasValue())
      return operand;

    // Cast firrtl -> standard type.
    return rewriter.create<firrtl::StdIntCast>(op->getLoc(),
                                               destType.getValue(), operand);
  }

  return operand;
}

static LogicalResult lower(firrtl::ConstantOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<rtl::ConstantOp>(op, op.value());
  return success();
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

static LogicalResult lower(firrtl::AsUIntPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  auto operand = mapOperand(operands[0], op, rewriter);

  auto resultType = op.getType().cast<UIntType>();
  auto srcWidth = operand.getType().getIntOrFloatBitWidth();
  auto destWidth = resultType.getWidthOrSentinel();
  if (destWidth == -1)
    return failure();

  // Noop cast.
  if (srcWidth == unsigned(destWidth))
    return rewriter.replaceOp(op, operand), success();

  return failure();
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

static LogicalResult lower(firrtl::AddPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  // FIXME: Not handling extensions correctly.
  rewriter.replaceOpWithNewOp<rtl::AddOp>(
      op, mapOperand(operands[0], op, rewriter),
      mapOperand(operands[1], op, rewriter));
  return success();
}

static LogicalResult lower(firrtl::SubPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  // FIXME: Not handling extensions correctly.
  rewriter.replaceOpWithNewOp<rtl::SubOp>(
      op, mapOperand(operands[0], op, rewriter),
      mapOperand(operands[1], op, rewriter));
  return success();
}

namespace {
/// Utility class for operation conversions targeting that
/// match exactly one source operation.
template <typename OpTy>
class RTLRewriter : public ConversionPattern {
public:
  RTLRewriter(MLIRContext *ctx)
      : ConversionPattern(OpTy::getOperationName(), /*benefit*/ 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return lower(OpTy(op), operands, rewriter);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// RTLConversionTarget
//===----------------------------------------------------------------------===//

namespace {
class RTLConversionTarget : public ConversionTarget {
public:
  explicit RTLConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<RTLDialect>();
    addLegalOp<firrtl::StdIntCast>();
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
    patterns
        .insert<RTLRewriter<firrtl::ConstantOp>, RTLRewriter<firrtl::AddPrimOp>,
                RTLRewriter<firrtl::SubPrimOp>,

                // Unary Operations
                RTLRewriter<firrtl::AsUIntPrimOp>>(&getContext());

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
