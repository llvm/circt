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

/// Map in the input value into an expected type, using the type converter to
/// do this.  This would ideally be handled by the conversion framework.
static Value mapOperand(Value operand, Operation *op,
                        ConversionPatternRewriter &rewriter) {
  auto opType = operand.getType();
  if (auto intType = opType.dyn_cast<IntType>()) {
    auto destType = RTLTypeConverter::convertIntType(intType);
    if (!destType.hasValue())
      return {};

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

// Lower a cast that is a noop at the RTL level.
static LogicalResult lowerNoopCast(Operation *op, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) {
  auto operand = mapOperand(operands[0], op, rewriter);
  if (!operand)
    return failure();

  // Noop cast.
  rewriter.replaceOp(op, operand);
  return success();
}

static LogicalResult lower(firrtl::AsSIntPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  return lowerNoopCast(op, operands, rewriter);
}

static LogicalResult lower(firrtl::AsUIntPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  return lowerNoopCast(op, operands, rewriter);
}

// Pad is a noop or extension operation.
static LogicalResult lower(firrtl::PadPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  auto operand = mapOperand(operands[0], op, rewriter);
  if (!operand)
    return failure();

  auto srcWidth = operand.getType().cast<IntegerType>().getWidth();
  auto destType = op.getType().cast<IntType>();
  auto destWidth = destType.getWidthOrSentinel();
  if (destWidth == -1)
    return failure();

  auto resultType = rewriter.getIntegerType(destWidth);
  if (srcWidth == unsigned(destWidth)) // Noop cast.
    rewriter.replaceOp(op, operand);
  else if (destType.isSigned())
    rewriter.replaceOpWithNewOp<rtl::SExtOp>(op, resultType, operand);
  else
    rewriter.replaceOpWithNewOp<rtl::ZExtOp>(op, resultType, operand);
  return success();
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

static LogicalResult lower(firrtl::AddPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  auto lhs = mapOperand(operands[0], op, rewriter);
  auto rhs = mapOperand(operands[1], op, rewriter);
  if (!lhs || !rhs)
    return failure();

  // FIXME: Not handling extensions correctly.
  rewriter.replaceOpWithNewOp<rtl::AddOp>(op, lhs, rhs);
  return success();
}

static LogicalResult lower(firrtl::SubPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  auto lhs = mapOperand(operands[0], op, rewriter);
  auto rhs = mapOperand(operands[1], op, rewriter);
  if (!lhs || !rhs)
    return failure();

  // FIXME: Not handling extensions correctly.
  rewriter.replaceOpWithNewOp<rtl::SubOp>(op, lhs, rhs);
  return success();
}

static LogicalResult lower(firrtl::XorPrimOp op, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) {
  auto lhs = mapOperand(operands[0], op, rewriter);
  auto rhs = mapOperand(operands[1], op, rewriter);
  if (!lhs || !rhs)
    return failure();

  // FIXME: Not handling extensions correctly.
  rewriter.replaceOpWithNewOp<rtl::XorOp>(op, lhs, rhs);
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
    patterns.insert<
        RTLRewriter<firrtl::ConstantOp>,
        // Binary Operations
        RTLRewriter<firrtl::AddPrimOp>, RTLRewriter<firrtl::SubPrimOp>,
        RTLRewriter<firrtl::XorPrimOp>,

        // Unary Operations
        RTLRewriter<firrtl::PadPrimOp>, RTLRewriter<firrtl::AsSIntPrimOp>,
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
