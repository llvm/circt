//===- OpFolds.cpp - Implement folds and canonicalizations for ops --------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Ops.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace firrtl;

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(value.getBitWidth(), context),
                          value);
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<ConstantOp>(op)) {
      value = cst.value();
      return true;
    }
    return false;
  }
};
} // end anonymous namespace

static inline ConstantIntMatcher m_FConstant(APInt &value) {
  return ConstantIntMatcher(value);
}

//===----------------------------------------------------------------------===//
// Fold Hooks
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

// TODO: Move to DRR.
OpFoldResult AndPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// and(x, 0) -> 0
  if (matchPattern(rhs(), m_FConstant(value)) && value.isNullValue() &&
      rhs().getType() == getType())
    return rhs();

  /// and(x, -1) -> x
  if (matchPattern(rhs(), m_FConstant(value)) && value.isAllOnesValue() &&
      lhs().getType() == getType() && rhs().getType() == getType())
    return lhs();

  /// and(x, x) -> x
  if (lhs() == rhs() && rhs().getType() == getType())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

OpFoldResult OrPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// or(x, 0) -> x
  if (matchPattern(rhs(), m_FConstant(value)) && value.isNullValue() &&
      lhs().getType() == getType())
    return lhs();

  /// or(x, -1) -> -1
  if (matchPattern(rhs(), m_FConstant(value)) && value.isAllOnesValue() &&
      rhs().getType() == getType() && lhs().getType() == getType())
    return rhs();

  /// or(x, x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

OpFoldResult XorPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_FConstant(value)) && value.isNullValue() &&
      lhs().getType() == getType())
    return lhs();

  /// xor(x, x) -> 0
  if (lhs() == rhs()) {
    auto width = getType().cast<IntType>().getWidthOrSentinel();
    if (width == -1)
      width = 1;
    auto type = IntegerType::get(width, getContext());
    return Builder(getContext()).getZeroAttr(type);
  }

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

OpFoldResult EQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  if (matchPattern(rhs(), m_FConstant(value))) {
    APInt lhsCst;
    // Constant fold.
    if (matchPattern(lhs(), m_FConstant(lhsCst)) &&
        value.getBitWidth() == lhsCst.getBitWidth()) {
      return getIntAttr(APInt(1, value == lhsCst), getContext());
    }

    /// eq(x, 1) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (value.isAllOnesValue() && lhs().getType() == getType())
      return lhs();

    /// TODO: eq(x, 0) -> not(x) when x is 1 bit.
    /// TODO: eq(x, 0) -> not(orr(x)) when x is >1 bit
    /// TODO: eq(x, ~0) -> andr(x)) when x is >1 bit
  }

  return {};
}

OpFoldResult NEQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  if (matchPattern(rhs(), m_FConstant(value))) {
    APInt lhsCst;
    // Constant fold.
    if (matchPattern(lhs(), m_FConstant(lhsCst)) &&
        value.getBitWidth() == lhsCst.getBitWidth()) {
      return getIntAttr(APInt(1, value != lhsCst), getContext());
    }

    /// neq(x, 0) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (value.isNullValue() && lhs().getType() == getType())
      return lhs();

    /// TODO: neq(x, 0) -> not(orr(x)) when x is >1 bit
    /// TODO: neq(x, 1) -> not(x) when x is 1 bit.
    /// TODO: neq(x, ~0) -> andr(x)) when x is >1 bit
  }

  return {};
}

void CatPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  struct Folder final : public OpRewritePattern<CatPrimOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(CatPrimOp op,
                                  PatternRewriter &rewriter) const override {
      // cat(bits(x, ...), bits(x, ...)) -> bits(x ...) when the two ...'s are
      // consequtive in the input.
      if (auto lhsBits =
              dyn_cast_or_null<BitsPrimOp>(op.lhs().getDefiningOp())) {
        if (auto rhsBits =
                dyn_cast_or_null<BitsPrimOp>(op.rhs().getDefiningOp())) {
          if (lhsBits.input() == rhsBits.input() &&
              lhsBits.lo() - 1 == rhsBits.hi()) {
            rewriter.replaceOpWithNewOp<BitsPrimOp>(
                op, op.getType(), lhsBits.input(), lhsBits.hi(), rhsBits.lo());
            return success();
          }
        }
      }
      return failure();
    }
  };

  results.insert<Folder>(context);
}

OpFoldResult BitsPrimOp::fold(ArrayRef<Attribute> operands) {
  auto inputType = input().getType().cast<FIRRTLType>();
  // If we are extracting the entire input, then return it.
  if (inputType == getType() &&
      inputType.cast<IntType>().getWidthOrSentinel() != -1)
    return input();

  // Constant fold.
  APInt value;
  if (inputType.cast<IntType>().hasWidth() &&
      matchPattern(input(), m_FConstant(value)))
    return getIntAttr(value.lshr(lo()).trunc(hi() - lo() + 1), getContext());

  return {};
}

void BitsPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  struct Folder final : public OpRewritePattern<BitsPrimOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(BitsPrimOp op,
                                  PatternRewriter &rewriter) const override {

      auto inputOp = op.input().getDefiningOp();
      // bits(bits(x, ...), ...) -> bits(x, ...).
      if (auto innerBits = dyn_cast_or_null<BitsPrimOp>(inputOp)) {
        auto newLo = op.lo() + innerBits.lo();
        auto newHi = newLo + op.hi() - op.lo();
        rewriter.replaceOpWithNewOp<BitsPrimOp>(op, innerBits.input(), newHi,
                                                newLo);
        return success();
      }
      return failure();
    }
  };

  results.insert<Folder>(context);
}

/// Replace the specified operation with a 'bits' op from the specified hi/lo
/// bits.  Insert a cast to handle the case where the original operation
/// returned a signed integer.
static void replaceWithBits(Operation *op, Value input, unsigned hiBit,
                            unsigned loBit, PatternRewriter &rewriter) {
  auto resultType = op->getResult(0).getType();
  if (bool isUnsigned = resultType.cast<IntType>().isUnsigned()) {
    rewriter.replaceOpWithNewOp<BitsPrimOp>(op, input, hiBit, loBit);
  } else {
    auto bits = rewriter.create<BitsPrimOp>(op->getLoc(), input, hiBit, loBit);
    rewriter.replaceOpWithNewOp<AsSIntPrimOp>(op, resultType, bits);
  }
}

void HeadPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  struct Folder final : public OpRewritePattern<HeadPrimOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(HeadPrimOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputWidth =
          op.input().getType().cast<IntType>().getWidthOrSentinel();
      if (inputWidth == -1)
        return failure();

      // If we know the input width, we can canonicalize this into a BitsPrimOp.
      unsigned keepAmount = op.amount();
      replaceWithBits(op, op.input(), inputWidth - 1, inputWidth - keepAmount,
                      rewriter);
      return success();
    }
  };

  results.insert<Folder>(context);
}

OpFoldResult MuxPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// mux(0/1, x, y) -> x or y
  if (matchPattern(sel(), m_FConstant(value))) {
    if (value.isNullValue() && low().getType() == getType())
      return low();
    if (!value.isNullValue() && high().getType() == getType())
      return high();
  }

  // mux(cond, x, x) -> x
  if (high() == low())
    return high();

  // mux(cond, x, cst)
  if (matchPattern(low(), m_FConstant(value))) {
    APInt c1;
    // mux(cond, c1, c2)
    if (matchPattern(high(), m_FConstant(c1))) {
      // mux(cond, 1, 0) -> cond
      if (c1.isOneValue() && value.isNullValue() &&
          getType() == sel().getType())
        return sel();

      // TODO: x ? ~0 : 0 -> sext(x)
      // TODO: "x ? c1 : c2" -> many tricks
    }
    // TODO: "x ? a : 0" -> sext(x) & a
  }

  // TODO: "x ? c1 : y" -> "~x ? y : c1"

  return {};
}

OpFoldResult PadPrimOp::fold(ArrayRef<Attribute> operands) {
  auto input = this->input();

  // pad(x) -> x  if the width doesn't change.
  if (input.getType() == getType())
    return input;

  // Need to know the input width.
  auto inputType = input.getType().cast<IntType>();
  int32_t width = inputType.getWidthOrSentinel();
  if (width == -1)
    return {};

  // Constant fold.
  APInt value;
  if (matchPattern(input, m_FConstant(value))) {
    auto destWidth = getType().cast<IntType>().getWidthOrSentinel();
    if (destWidth == -1)
      return {};

    if (inputType.isSigned())
      return getIntAttr(value.sext(destWidth), getContext());
    return getIntAttr(value.zext(destWidth), getContext());
  }

  return {};
}

OpFoldResult ShlPrimOp::fold(ArrayRef<Attribute> operands) {
  auto input = this->input();
  auto inputType = input.getType().cast<IntType>();
  int shiftAmount = amount();

  // shl(x, 0) -> x
  if (shiftAmount == 0)
    return input;

  // Constant fold.
  APInt value;
  if (matchPattern(input, m_FConstant(value))) {
    auto inputWidth = inputType.getWidthOrSentinel();
    if (inputWidth != -1) {
      auto resultWidth = inputWidth + shiftAmount;
      return getIntAttr(value.zext(resultWidth).shl(shiftAmount), getContext());
    }
  }
  return {};
}

OpFoldResult ShrPrimOp::fold(ArrayRef<Attribute> operands) {
  auto input = this->input();
  auto inputType = input.getType().cast<IntType>();
  int shiftAmount = amount();

  // shl(x, 0) -> x
  if (shiftAmount == 0)
    return input;

  auto inputWidth = inputType.getWidthOrSentinel();
  if (inputWidth == -1)
    return {};

  // shr(x, cst) where cst is all of x's bits and x is unsigned is 0.
  // If x is signed, it is the sign bit.
  if (shiftAmount >= inputWidth && inputType.isUnsigned())
    return getIntAttr(APInt(1, 0), getContext());

  // Constant fold.
  APInt value;
  if (matchPattern(input, m_FConstant(value))) {
    auto resultWidth = std::max(inputWidth - shiftAmount, 1);
    if (!inputType.isSigned())
      value = value.lshr(shiftAmount);
    else
      value = value.ashr(shiftAmount);
    return getIntAttr(value.trunc(resultWidth), getContext());
  }
  return {};
}

void ShrPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  struct Folder final : public OpRewritePattern<ShrPrimOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ShrPrimOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputWidth =
          op.input().getType().cast<IntType>().getWidthOrSentinel();
      if (inputWidth == -1)
        return failure();

      // If we know the input width, we can canonicalize this into a BitsPrimOp.
      unsigned shiftAmount = op.amount();
      if (int(shiftAmount) >= inputWidth) {
        // shift(x, 32) => 0 when x has 32 bits.  This is handled by fold().
        if (op.getType().cast<IntType>().isUnsigned())
          return failure();

        // Shifting a signed value by the full width is actually taking the sign
        // bit. If the shift amount is greater than the input width, it is
        // equivalent to shifting by the input width.
        shiftAmount = inputWidth - 1;
      }

      replaceWithBits(op, op.input(), inputWidth - 1, shiftAmount, rewriter);
      return success();
    }
  };

  results.insert<Folder>(context);
}

void TailPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  struct Folder final : public OpRewritePattern<TailPrimOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(TailPrimOp op,
                                  PatternRewriter &rewriter) const override {
      auto inputWidth =
          op.input().getType().cast<IntType>().getWidthOrSentinel();
      if (inputWidth == -1)
        return failure();

      // If we know the input width, we can canonicalize this into a
      // BitsPrimOp.
      unsigned dropAmount = op.amount();
      replaceWithBits(op, op.input(), inputWidth - dropAmount - 1, 0, rewriter);
      return success();
    }
  };

  results.insert<Folder>(context);
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

OpFoldResult StdIntCast::fold(ArrayRef<Attribute> operands) {
  if (auto castInput =
          dyn_cast_or_null<StdIntCast>(getOperand().getDefiningOp()))
    if (castInput.getOperand().getType() == getType())
      return castInput.getOperand();

  return {};
}

OpFoldResult AsPassivePrimOp::fold(ArrayRef<Attribute> operands) {
  if (auto castInput =
          dyn_cast_or_null<AsNonPassivePrimOp>(getOperand().getDefiningOp()))
    if (castInput.getOperand().getType() == getType())
      return castInput.getOperand();

  return {};
}

OpFoldResult AsNonPassivePrimOp::fold(ArrayRef<Attribute> operands) {
  if (auto castInput =
          dyn_cast_or_null<AsPassivePrimOp>(getOperand().getDefiningOp()))
    if (castInput.getOperand().getType() == getType())
      return castInput.getOperand();

  return {};
}
