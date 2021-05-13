//===- FIRRTLFolds.cpp - Implement folds and canonicalizations for ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the folding and canonicalizations for FIRRTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

// Declarative canonicalization patterns
namespace circt {
namespace firrtl {
namespace patterns {
#include "circt/Dialect/FIRRTL/FIRRTLCanonicalization.h.inc"
} // namespace patterns
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

static IntegerAttr getIntAttr(Type type, const APInt &value) {
  auto firType = type.cast<IntType>();
  assert((!firType.hasWidth() ||
          (unsigned)firType.getWidthOrSentinel() == value.getBitWidth()) &&
         "value / type width mismatch");
  auto intSign =
      firType.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  auto intType =
      IntegerType::get(type.getContext(), value.getBitWidth(), intSign);
  return IntegerAttr::get(intType, value);
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

/// Implicitly replace the operand to a constant folding operation with a const
/// 0 in case the operand is non-constant but has a bit width 0.
///
/// This makes constant folding significantly easier, as we can simply pass the
/// operands to an operation through this function to appropriately replace any
/// zero-width dynamic values with a constant of value 0.
static IntegerAttr elideZeroWidthFoldOperand(Value operand,
                                             Attribute foldOperand) {
  if (foldOperand)
    return foldOperand.dyn_cast<IntegerAttr>();

  if (auto type = operand.getType().dyn_cast<IntType>()) {
    if (type.getWidth() == 0) {
      auto intSign =
          type.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
      return IntegerAttr::get(
          IntegerType::get(operand.getContext(), 1, intSign), APInt(1, 0));
    }
  }
  return {};
}

/// Check if the operands and results of \p op are of integer type and with
/// known bitwidth. Can be used to determine if any fold is legal.
static bool hasKnownWidths(Operation *op) {
  auto resTy = op->getResultTypes().front().cast<FIRRTLType>();
  // The result must be of integer type and the bitwidth must be known and
  // non-zero. Unkown bitwidths are handled after width inference.
  if (!resTy.isa<IntType>() || resTy.getBitWidthOrSentinel() <= 0)
    return false;

  for (auto opTy : op->getOperandTypes()) {
    auto ty = opTy.cast<FIRRTLType>();
    // Operand bitwidth must be known. Unkown bitwidths are handled after width
    // inference.
    if (ty.getBitWidthOrSentinel() == -1)
      return false;
  }
  return true;
}

/// Applies the constant folding function `calculate` to the given operands.
///
/// Sign or zero extends the operands appropriately to the bitwidth of the
/// result type if \p useDstWidth is true, else to the larger of the two operand
/// bit widths and depending on whether the operation is to be performed on
/// signed or unsigned operands.
static Attribute
constFoldFIRRTLBinaryOp(Operation *op, ArrayRef<Attribute> operands,
                        const function_ref<APInt(APInt, APInt)> &calculate,
                        bool useDstWidth = false) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!hasKnownWidths(op) && useDstWidth)
    return {};
  IntegerAttr lhs = elideZeroWidthFoldOperand(op->getOperand(0), operands[0]);
  IntegerAttr rhs = elideZeroWidthFoldOperand(op->getOperand(1), operands[1]);
  if (!lhs || !rhs)
    return {};
  auto srcType = op->getOperandTypes().front().cast<IntType>();
  auto dstType = op->getResultTypes().front().cast<IntType>();
  auto commonWidth = useDstWidth
                         ? dstType.getBitWidthOrSentinel()
                         : std::max<int32_t>(lhs.getValue().getBitWidth(),
                                             rhs.getValue().getBitWidth());
  auto extOrSelf =
      srcType.isUnsigned() ? &APInt::zextOrTrunc : &APInt::sextOrTrunc;
  return getIntAttr(op->getResult(0).getType(),
                    calculate((lhs.getValue().*extOrSelf)(commonWidth),
                              (rhs.getValue().*extOrSelf)(commonWidth)));
}

/// Get the largest unsigned value of a given bit width. Returns a 1-bit zero
/// value if `bitWidth` is 0.
static APInt getMaxUnsignedValue(unsigned bitWidth) {
  return bitWidth > 0 ? APInt::getMaxValue(bitWidth) : APInt();
}

/// Get the smallest signed value of a given bit width. Returns a 1-bit zero
/// value if `bitWidth` is 0.
static APInt getMinSignedValue(unsigned bitWidth) {
  return bitWidth > 0 ? APInt::getSignedMinValue(bitWidth) : APInt();
}

/// Get the largest signed value of a given bit width. Returns a 1-bit zero
/// value if `bitWidth` is 0.
static APInt getMaxSignedValue(unsigned bitWidth) {
  return bitWidth > 0 ? APInt::getSignedMaxValue(bitWidth) : APInt();
}

//===----------------------------------------------------------------------===//
// Fold Hooks
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AddPrimOp::fold(ArrayRef<Attribute> operands) {
  /// Any folding here requires a bitwidth extension.

  /// If both operands are constant, and the result is integer with known
  /// widths, then perform constant folding.
  return constFoldFIRRTLBinaryOp(
      *this, operands, [=](APInt a, APInt b) { return a + b; }, true);
  return {};
}

OpFoldResult SubPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(
      *this, operands, [=](APInt a, APInt b) { return a - b; }, true);
  return {};
}

OpFoldResult MulPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(
      *this, operands, [=](APInt a, APInt b) { return a * b; }, true);
  return {};
}

OpFoldResult DivPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// div(x, x) -> 1
  ///
  /// Division by zero is undefined in the FIRRTL specification. This
  /// fold exploits that fact to optimize self division to one.
  if (lhs() == rhs()) {
    auto width = getType().getWidthOrSentinel();
    if (width == -1)
      width = 2;
    if (width != 0)
      return getIntAttr(getType(), APInt(width, 1));
  }

  /// div(x, 1) -> x : (uint, uint) -> uint
  ///
  /// UInt division by one returns the numerator. SInt division can't
  /// be folded here because it increases the return type bitwidth by
  /// one and requires sign extension (a new op).
  if (matchPattern(rhs(), m_FConstant(value)) && value.isOneValue() &&
      lhs().getType() == getType())
    return lhs();

  return {};
}

OpFoldResult RemPrimOp::fold(ArrayRef<Attribute> operands) { return {}; }

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

  return constFoldFIRRTLBinaryOp(
      *this, operands, [](APInt a, APInt b) { return a & b; }, true);
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
  if (lhs() == rhs() && rhs().getType() == getType())
    return rhs();

  return constFoldFIRRTLBinaryOp(
      *this, operands, [](APInt a, APInt b) { return a | b; }, true);
}

OpFoldResult XorPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_FConstant(value)) && value.isNullValue() &&
      lhs().getType() == getType())
    return lhs();

  /// xor(x, x) -> 0
  if (lhs() == rhs()) {
    auto width = abs(getType().getWidthOrSentinel());
    if (width != 0) // We cannot create a zero bit APInt.
      return getIntAttr(getType(), APInt(width, 0));
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, [](APInt a, APInt b) { return a ^ b; }, true);
}

void LEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::LEQWithConstLHS>(context);
}

OpFoldResult LEQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // leq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (matchPattern(rhs(), m_FConstant(value))) {
      auto commonWidth = std::max<int32_t>(*width, value.getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // leq(x, const) -> 0 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // leq(x, const) -> 0 where const < minValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .slt(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // leq(x, const) -> 1 where const >= maxValue of the unsigned type of x
      if (isUnsigned &&
          value.zextOrSelf(commonWidth)
              .uge(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // leq(x, const) -> 1 where const >= maxValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sge(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(*this, operands, [=](APInt a, APInt b) {
    return APInt(1, isUnsigned ? a.ule(b) : a.sle(b));
  });
}

void LTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::LTWithConstLHS>(context);
}

OpFoldResult LTPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // lt(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  // lt(x, 0) -> 0 when x is unsigned
  if (matchPattern(rhs(), m_FConstant(value))) {
    if (value.isNullValue() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 0));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (matchPattern(rhs(), m_FConstant(value))) {
      auto commonWidth = std::max<int32_t>(*width, value.getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // lt(x, const) -> 0 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // lt(x, const) -> 0 where const <= minValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sle(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // lt(x, const) -> 1 where const > maxValue of the unsigned type of x
      if (isUnsigned &&
          value.zextOrSelf(commonWidth)
              .ugt(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // lt(x, const) -> 1 where const > maxValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sgt(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(*this, operands, [=](APInt a, APInt b) {
    return APInt(1, isUnsigned ? a.ult(b) : a.slt(b));
  });
}

void GEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::GEQWithConstLHS>(context);
}

OpFoldResult GEQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // geq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  // geq(x, 0) -> 1 when x is unsigned
  if (matchPattern(rhs(), m_FConstant(value))) {
    if (value.isNullValue() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 1));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (matchPattern(rhs(), m_FConstant(value))) {
      auto commonWidth = std::max<int32_t>(*width, value.getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // geq(x, const) -> 0 where const > maxValue of the unsigned type of x
      if (isUnsigned &&
          value.zextOrSelf(commonWidth)
              .ugt(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 0 where const > maxValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sgt(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 1 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // geq(x, const) -> 1 where const <= minValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sle(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(*this, operands, [=](APInt a, APInt b) {
    return APInt(1, isUnsigned ? a.uge(b) : a.sge(b));
  });
}

void GTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::GTWithConstLHS>(context);
}

OpFoldResult GTPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // gt(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (matchPattern(rhs(), m_FConstant(value))) {
      auto commonWidth = std::max<int32_t>(*width, value.getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // gt(x, const) -> 0 where const >= maxValue of the unsigned type of x
      if (isUnsigned &&
          value.zextOrSelf(commonWidth)
              .uge(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 0 where const >= maxValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .sge(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 1 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // gt(x, const) -> 1 where const < minValue of the signed type of x
      if (!isUnsigned &&
          value.sextOrSelf(commonWidth)
              .slt(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(*this, operands, [=](APInt a, APInt b) {
    return APInt(1, isUnsigned ? a.ugt(b) : a.sgt(b));
  });
}

OpFoldResult EQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  // eq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  if (matchPattern(rhs(), m_FConstant(value))) {
    /// eq(x, 1) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (value.isAllOnesValue() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();

    /// TODO: eq(x, 0) -> not(x) when x is 1 bit.
    /// TODO: eq(x, 0) -> not(orr(x)) when x is >1 bit
    /// TODO: eq(x, ~0) -> andr(x)) when x is >1 bit
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, [=](APInt a, APInt b) { return APInt(1, a.eq(b)); });
}

OpFoldResult NEQPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  // neq(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  if (matchPattern(rhs(), m_FConstant(value))) {
    /// neq(x, 0) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (value.isNullValue() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();

    /// TODO: neq(x, 0) -> not(orr(x)) when x is >1 bit
    /// TODO: neq(x, 1) -> not(x) when x is 1 bit.
    /// TODO: neq(x, ~0) -> andr(x)) when x is >1 bit
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, [=](APInt a, APInt b) { return APInt(1, a.ne(b)); });
}

OpFoldResult CatPrimOp::fold(ArrayRef<Attribute> operands) { return {}; }

OpFoldResult DShlPrimOp::fold(ArrayRef<Attribute> operands) { return {}; }

OpFoldResult DShlwPrimOp::fold(ArrayRef<Attribute> operands) { return {}; }

OpFoldResult DShrPrimOp::fold(ArrayRef<Attribute> operands) { return {}; }

//===----------------------------------------------------------------------===//
// Unary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AsSIntPrimOp::fold(ArrayRef<Attribute> operands) {
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(), attr.getValue());
  return {};
}

OpFoldResult AsUIntPrimOp::fold(ArrayRef<Attribute> operands) {
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(), attr.getValue());
  return {};
}

//===----------------------------------------------------------------------===//
// Other Operators
//===----------------------------------------------------------------------===//

LogicalResult CatPrimOp::canonicalize(CatPrimOp op, PatternRewriter &rewriter) {
  // cat(bits(x, ...), bits(x, ...)) -> bits(x ...) when the two ...'s are
  // consequtive in the input.
  if (auto lhsBits = dyn_cast_or_null<BitsPrimOp>(op.lhs().getDefiningOp())) {
    if (auto rhsBits = dyn_cast_or_null<BitsPrimOp>(op.rhs().getDefiningOp())) {
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
    return getIntAttr(getType(), value.lshr(lo()).truncOrSelf(hi() - lo() + 1));

  return {};
}

LogicalResult BitsPrimOp::canonicalize(BitsPrimOp op,
                                       PatternRewriter &rewriter) {
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

/// Replace the specified operation with a 'bits' op from the specified hi/lo
/// bits.  Insert a cast to handle the case where the original operation
/// returned a signed integer.
static void replaceWithBits(Operation *op, Value value, unsigned hiBit,
                            unsigned loBit, PatternRewriter &rewriter) {
  auto resType = op->getResult(0).getType().cast<IntType>();
  if (value.getType().cast<IntType>().getWidth() != resType.getWidth())
    value = rewriter.create<BitsPrimOp>(op->getLoc(), value, hiBit, loBit);

  if (resType.isSigned() && !value.getType().cast<IntType>().isSigned()) {
    value = rewriter.createOrFold<AsSIntPrimOp>(op->getLoc(), resType, value);
  } else if (resType.isUnsigned() &&
             !value.getType().cast<IntType>().isUnsigned()) {
    value = rewriter.createOrFold<AsUIntPrimOp>(op->getLoc(), resType, value);
  }
  rewriter.replaceOp(op, value);
}

LogicalResult HeadPrimOp::canonicalize(HeadPrimOp op,
                                       PatternRewriter &rewriter) {
  auto inputWidth = op.input().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth == -1)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned keepAmount = op.amount();
  if (keepAmount)
    replaceWithBits(op, op.input(), inputWidth - 1, inputWidth - keepAmount,
                    rewriter);
  return success();
}

OpFoldResult MuxPrimOp::fold(ArrayRef<Attribute> operands) {
  APInt value;

  // mux(cond, x, invalid) -> x
  // mux(cond, invalid, x) -> x
  if (high().getDefiningOp<InvalidValueOp>())
    return low();
  if (low().getDefiningOp<InvalidValueOp>())
    return high();

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

void MuxPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::MuxSameCondLow>(context);
  results.insert<patterns::MuxSameCondHigh>(context);
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
    auto destWidth = getType().getWidthOrSentinel();
    if (destWidth == -1)
      return {};

    if (inputType.isSigned())
      return getIntAttr(getType(), value.sext(destWidth));
    return getIntAttr(getType(), value.zext(destWidth));
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
      shiftAmount = std::min(shiftAmount, resultWidth);
      return getIntAttr(getType(), value.zext(resultWidth).shl(shiftAmount));
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
    return getIntAttr(getType(), APInt(1, 0));

  // Constant fold.
  APInt value;
  if (matchPattern(input, m_FConstant(value))) {
    if (!inputType.isSigned())
      value = value.lshr(std::min(shiftAmount, inputWidth));
    else
      value = value.ashr(std::min(shiftAmount, inputWidth - 1));
    auto resultWidth = std::max(inputWidth - shiftAmount, 1);
    return getIntAttr(getType(), value.truncOrSelf(resultWidth));
  }
  return {};
}

LogicalResult ShrPrimOp::canonicalize(ShrPrimOp op, PatternRewriter &rewriter) {
  auto inputWidth = op.input().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth == -1)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned shiftAmount = op.amount();
  if (int(shiftAmount) >= inputWidth) {
    // shift(x, 32) => 0 when x has 32 bits.  This is handled by fold().
    if (op.getType().isUnsigned())
      return failure();

    // Shifting a signed value by the full width is actually taking the
    // sign bit. If the shift amount is greater than the input width, it
    // is equivalent to shifting by the input width.
    shiftAmount = inputWidth - 1;
  }

  replaceWithBits(op, op.input(), inputWidth - 1, shiftAmount, rewriter);
  return success();
}

LogicalResult TailPrimOp::canonicalize(TailPrimOp op,
                                       PatternRewriter &rewriter) {
  auto inputWidth = op.input().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth == -1)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned dropAmount = op.amount();
  if (dropAmount != unsigned(inputWidth))
    replaceWithBits(op, op.input(), inputWidth - dropAmount - 1, 0, rewriter);
  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Scan all the uses of the specified value, checking to see if there is
/// exactly one connect that sets the value as its destination.  This returns
/// the operation if found and if all the other users are "reads" from the
/// value.
static bool isOnlyConnectToValue(ConnectOp connect, Value value) {
  for (Operation *user : value.getUsers()) {
    // If we see a partial connect or attach, just conservatively fail.
    if (isa<PartialConnectOp>(user) || isa<AttachOp>(user))
      return {};

    if (auto aConnect = dyn_cast<ConnectOp>(user)) {
      if (aConnect.dest() == value && aConnect != connect)
        return false;
    }
  }
  return true;
}

// Forward simple values through wire's and reg's.
static LogicalResult foldSingleSetConnect(ConnectOp op,
                                          PatternRewriter &rewriter) {
  //
  // While we can do this for nearly all wires, we currently limit it to simple
  // things.
  Operation *connectedDecl = op.dest().getDefiningOp();
  if (!connectedDecl)
    return failure();

  // Only support wire and reg for now.
  if (!isa<WireOp>(connectedDecl) && !isa<RegOp>(connectedDecl))
    return failure();

  // Only forward if the types exactly match and there is one connect.
  if (op.dest().getType() != op.src().getType() ||
      !isOnlyConnectToValue(op, op.dest()))
    return failure();

  // Only do this if the connectee and the declaration are in the same block.
  auto *declBlock = connectedDecl->getBlock();
  auto *srcValueOp = op.src().getDefiningOp();
  if (!srcValueOp) {
    // Ports are ok for wires but not registers.
    if (!isa<WireOp>(connectedDecl))
      return failure();

  } else {
    // Constants/invalids in the same block are ok to forward, even through
    // reg's since the clocking doesn't matter for constants.
    if (!isa<ConstantOp>(srcValueOp) && !isa<InvalidValueOp>(srcValueOp))
      return failure();
    if (srcValueOp->getBlock() != declBlock)
      return failure();
  }

  // Ok, we know we are doing the transformation.

  // Make sure the constant dominates all users.
  if (srcValueOp && srcValueOp != &declBlock->front())
    srcValueOp->moveBefore(&declBlock->front());

  // Replace all things *using* the decl with the constant/port, and
  // remove the declaration.
  rewriter.replaceOp(connectedDecl, op.src());

  // Remove the connect.
  rewriter.eraseOp(op);

  return success();
}

LogicalResult ConnectOp::canonicalize(ConnectOp op, PatternRewriter &rewriter) {
  // TODO: Canonicalize towards explicit extensions and flips here.

  // If there is a simple value connected to a foldable decl like a wire or reg,
  // see if we can eliminate the decl.
  if (succeeded(foldSingleSetConnect(op, rewriter)))
    return success();

  return failure();
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

OpFoldResult StdIntCastOp::fold(ArrayRef<Attribute> operands) {
  if (auto castInput =
          dyn_cast_or_null<StdIntCastOp>(getOperand().getDefiningOp()))
    if (castInput.getOperand().getType() == getType())
      return castInput.getOperand();

  return {};
}

OpFoldResult AnalogInOutCastOp::fold(ArrayRef<Attribute> operands) {
  if (auto castInput =
          dyn_cast_or_null<AnalogInOutCastOp>(getOperand().getDefiningOp()))
    if (castInput.getOperand().getType() == getType())
      return castInput.getOperand();

  return {};
}

OpFoldResult AsPassivePrimOp::fold(ArrayRef<Attribute> operands) {
  // If the input is already passive, then we don't need a conversion.
  if (getOperand().getType() == getType())
    return getOperand();

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

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// If the specified value has an AttachOp user strictly dominating by
/// "dominatingAttach" then return it.
static AttachOp getDominatingAttachUser(Value value, AttachOp dominatedAttach) {
  for (auto *user : value.getUsers()) {
    auto attach = dyn_cast<AttachOp>(user);
    if (!attach || attach == dominatedAttach)
      continue;
    if (attach->isBeforeInBlock(dominatedAttach))
      return attach;
  }
  return {};
}

LogicalResult AttachOp::canonicalize(AttachOp op, PatternRewriter &rewriter) {
  // Single operand attaches are a noop.
  if (op.getNumOperands() <= 1) {
    rewriter.eraseOp(op);
    return success();
  }

  for (auto operand : op.getOperands()) {
    // Check to see if any of our operands has other attaches to it:
    //    attach x, y
    //      ...
    //    attach x, z
    // If so, we can merge these into "attach x, y, z".
    if (auto attach = getDominatingAttachUser(operand, op)) {
      SmallVector<Value> newOperands(op.getOperands());
      for (auto newOperand : attach.getOperands())
        if (newOperand != operand) // Don't add operand twice.
          newOperands.push_back(newOperand);
      rewriter.create<AttachOp>(op->getLoc(), newOperands);
      rewriter.eraseOp(attach);
      rewriter.eraseOp(op);
      return success();
    }

    // If this wire is *only* used by an attach then we can just delete
    // it.
    // TODO: May need to be sensitive to "don't touch" or other
    // annotations.
    if (auto wire = dyn_cast_or_null<WireOp>(operand.getDefiningOp())) {
      if (wire->hasOneUse()) {
        SmallVector<Value> newOperands;
        for (auto newOperand : op.getOperands())
          if (newOperand != operand) // Don't the add wire.
            newOperands.push_back(newOperand);

        rewriter.create<AttachOp>(op->getLoc(), newOperands);
        rewriter.eraseOp(op);
        rewriter.eraseOp(wire);
        return success();
      }
    }
  }
  return failure();
}

LogicalResult PartialConnectOp::canonicalize(PartialConnectOp op,
                                             PatternRewriter &rewriter) {
  // If a partial connect exists from a longer int to a shorter int, simplify
  // to a truncation and connect.
  auto destType =
      op.getOperand(0).getType().cast<FIRRTLType>().getPassiveType();
  auto srcType = op.getOperand(1).getType().cast<FIRRTLType>();
  if (destType == srcType)
    return failure();

  auto srcWidth = srcType.getBitWidthOrSentinel();
  auto destWidth = destType.getBitWidthOrSentinel();

  if (destType.isa<IntType>() && srcType.isa<IntType>() && srcWidth > 0 &&
      destWidth > 0 && destWidth < srcWidth) {
    auto shortened = rewriter.createOrFold<TailPrimOp>(
        op.getLoc(), destType, op.getOperand(1), srcWidth - destWidth);
    rewriter.create<ConnectOp>(op.getLoc(), op.getOperand(0), shortened);
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

void NodeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<patterns::EmptyNode>(context);
}
