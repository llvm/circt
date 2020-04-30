//===- OpFolds.cpp - Implement folds and canonicalizations for ops --------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/FIRRTL/Ops.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace cirt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Fold Hooks
//===----------------------------------------------------------------------===//

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

static inline ConstantIntMatcher m_FConstant(APInt &value) {
  return ConstantIntMatcher(value);
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
      lhs().getType() == getType())
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
      rhs().getType() == getType())
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
      auto result = value == lhsCst;
      return IntegerAttr::get(IntegerType::get(1, getContext()),
                              APInt(1, result));
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
      auto result = value != lhsCst;
      return IntegerAttr::get(IntegerType::get(1, getContext()),
                              APInt(1, result));
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

namespace {
struct CatFolder final : public OpRewritePattern<CatPrimOp> {
  CatFolder(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(CatPrimOp op,
                                PatternRewriter &rewriter) const override {
    // cat(bits(x, ...), bits(x, ...)) -> bits(x ...) when the two ...'s are
    // consequtive in the input.
    if (auto lhsBits = dyn_cast_or_null<BitsPrimOp>(op.lhs().getDefiningOp())) {
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
} // end anonymous namespace

void CatPrimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<CatFolder>(context);
}

OpFoldResult BitsPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  // If we are extracting the entire input, then return it.
  if (input().getType() == getType() &&
      getType().cast<IntType>().getWidthOrSentinel() != -1)
    return input();

  return {};
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
  auto inputType = input.getType().cast<IntType>();

  // pad(x) -> x  if the width doesn't change.
  if (input.getType() == getType())
    return input;

  // Need to know the input width.
  int32_t width = inputType.getWidthOrSentinel();
  if (width == -1)
    return {};

  APInt value;

  /// pad(cst1) -> cst2
  if (matchPattern(input, m_FConstant(value))) {
    auto destWidth = getType().cast<IntType>().getWidthOrSentinel();
    if (inputType.isSigned())
      value = value.sext(destWidth);
    else
      value = value.zext(destWidth);

    return IntegerAttr::get(IntegerType::get(destWidth, getContext()), value);
  }

  return {};
}