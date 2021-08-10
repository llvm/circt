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

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"

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
  auto intType = type.cast<IntType>();
  assert((!intType.hasWidth() ||
          (unsigned)intType.getWidthOrSentinel() == value.getBitWidth()) &&
         "value / type width mismatch");
  auto intSign =
      intType.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  auto attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), intSign);
  return IntegerAttr::get(attrType, value);
}

/// Return an IntegerAttr filled with zeros for the specified FIRRTL integer
/// type.  This handles both the known width and unknown width case, but returns
/// a null attribute for zero width results.
static IntegerAttr getIntZerosAttr(Type type) {
  int32_t width = abs(type.cast<IntType>().getWidthOrSentinel());
  if (width == 0)
    return {};
  return getIntAttr(type, APInt(width, 0));
}

/// Return true if this operation's operands and results all have known width,
/// or if the result has zero width result (which we cannot constant fold).
/// This only works for integer types.
static bool hasKnownWidthIntTypesAndNonZeroResult(Operation *op) {
  auto resultType = op->getResult(0).getType().cast<IntType>();
  if (!resultType.hasWidth() || resultType.getWidth() == 0)
    return false;
  for (Value operand : op->getOperands())
    if (!operand.getType().cast<IntType>().hasWidth())
      return false;
  return true;
}

/// Return true if this value is 1 bit UInt.
static bool isUInt1(Type type) {
  auto t = type.dyn_cast<UIntType>();
  if (!t || !t.hasWidth() || t.getWidth() != 1)
    return false;
  return true;
}

/// Implicitly replace the operand to a constant folding operation with a const
/// 0 in case the operand is non-constant but has a bit width 0.
///
/// This makes constant folding significantly easier, as we can simply pass the
/// operands to an operation through this function to appropriately replace any
/// zero-width dynamic values with a constant of value 0.
static Optional<APSInt> getExtendedConstant(Value operand, Attribute constant,
                                            int32_t destWidth) {
  // We never support constant folding to unknown or zero width values: APInt
  // can't do it.
  if (destWidth <= 0)
    return {};

  // Extension signedness follows the operand sign.
  if (IntegerAttr result = constant.dyn_cast_or_null<IntegerAttr>())
    return result.getAPSInt().extOrTrunc(destWidth);

  // If the operand is zero bits, then we can return a zero of the result
  // type.
  if (operand.getType().cast<IntType>().getWidth() == 0)
    return APSInt(destWidth, operand.getType().cast<IntType>().isUnsigned());
  return {};
}

/// This is the policy for folding, which depends on the sort of operator we're
/// processing.
enum class BinOpKind {
  Normal,
  Compare,
  DivideOrShift,
};

/// Applies the constant folding function `calculate` to the given operands.
///
/// Sign or zero extends the operands appropriately to the bitwidth of the
/// result type if \p useDstWidth is true, else to the larger of the two operand
/// bit widths and depending on whether the operation is to be performed on
/// signed or unsigned operands.
static Attribute
constFoldFIRRTLBinaryOp(Operation *op, ArrayRef<Attribute> operands,
                        BinOpKind opKind,
                        const function_ref<APInt(APSInt, APSInt)> &calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");

  // We cannot fold something to an unknown or zero width.
  auto resultType = op->getResult(0).getType().cast<IntType>();
  if (resultType.getWidthOrSentinel() <= 0)
    return {};

  // Determine the operand widths. This is either dictated by the operand type,
  // or if that type is an unsized integer, by the actual bits necessary to
  // represent the constant value.
  auto lhsWidth =
      op->getOperand(0).getType().cast<IntType>().getWidthOrSentinel();
  auto rhsWidth =
      op->getOperand(1).getType().cast<IntType>().getWidthOrSentinel();
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    lhsWidth = std::max<int32_t>(lhsWidth, lhs.getValue().getBitWidth());
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>())
    rhsWidth = std::max<int32_t>(rhsWidth, rhs.getValue().getBitWidth());

  // Compares extend the operands to the widest of the operand types, not to the
  // result type.
  int32_t operandWidth;
  switch (opKind) {
  case BinOpKind::Normal:
    operandWidth = resultType.getWidthOrSentinel();
    break;
  case BinOpKind::Compare:
    // Compares compute with the widest operand, not at the destination type
    // (which is always i1).
    operandWidth = std::max(1, std::max(lhsWidth, rhsWidth));
    break;
  case BinOpKind::DivideOrShift:
    operandWidth =
        std::max(std::max(lhsWidth, rhsWidth), resultType.getWidthOrSentinel());
    break;
  }

  auto lhs = getExtendedConstant(op->getOperand(0), operands[0], operandWidth);
  if (!lhs.hasValue())
    return {};
  auto rhs = getExtendedConstant(op->getOperand(1), operands[1], operandWidth);
  if (!rhs.hasValue())
    return {};

  APInt resultValue = calculate(lhs.getValue(), rhs.getValue());

  // If the result type is smaller than the computation then we need to
  // narrow the constant after the calculation.
  if (opKind == BinOpKind::DivideOrShift)
    resultValue = resultValue.truncOrSelf(resultType.getWidthOrSentinel());

  assert((unsigned)resultType.getWidthOrSentinel() ==
         resultValue.getBitWidth());
  return getIntAttr(resultType, resultValue);
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

OpFoldResult SpecialConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

OpFoldResult InvalidValueOp::fold(ArrayRef<Attribute> operands) {
  return InvalidValueAttr::get(getType());
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AddPrimOp::fold(ArrayRef<Attribute> operands) {
  /// Any folding here requires a bitwidth extension.

  /// If both operands are constant, and the result is integer with known
  /// widths, then perform constant folding.
  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::Normal,
                                 [=](APSInt a, APSInt b) { return a + b; });
}

void AddPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::AddWithInvalidOp>(context);
}

OpFoldResult SubPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::Normal,
                                 [=](APSInt a, APSInt b) { return a - b; });
}

void SubPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::SubWithInvalidOp>(context);
}

OpFoldResult MulPrimOp::fold(ArrayRef<Attribute> operands) {
  // mul(x, invalid) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // multiplication this way and will emit "x * 0".
  if (operands[1].dyn_cast_or_null<InvalidValueAttr>() ||
      operands[0].dyn_cast_or_null<InvalidValueAttr>())
    return getIntZerosAttr(getType());

  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::Normal,
                                 [=](APSInt a, APSInt b) { return a * b; });
}

OpFoldResult DivPrimOp::fold(ArrayRef<Attribute> operands) {
  /// div(x, x) -> 1
  ///
  /// Division by zero is undefined in the FIRRTL specification.  This fold
  /// exploits that fact to optimize self division to one.  Note: this should
  /// supersede any division with invalid or zero.  Division of invalid by
  /// invalid should be one.
  if (lhs() == rhs()) {
    auto width = getType().getWidthOrSentinel();
    if (width == -1)
      width = 2;
    if (width != 0)
      return getIntAttr(getType(), APInt(width, 1));
  }

  // div(invalid, x) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // division this way and will emit "0 / x".
  if (operands[0].dyn_cast_or_null<InvalidValueAttr>() &&
      !operands[1].dyn_cast_or_null<InvalidValueAttr>())
    return getIntZerosAttr(getType());

  /// div(x, 1) -> x : (uint, uint) -> uint
  ///
  /// UInt division by one returns the numerator. SInt division can't
  /// be folded here because it increases the return type bitwidth by
  /// one and requires sign extension (a new op).
  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhsCst.getValue().isOneValue() && lhs().getType() == getType())
      return lhs();

  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::DivideOrShift,
                                 [=](APSInt a, APSInt b) -> APInt {
                                   if (!!b)
                                     return a / b;
                                   return APInt(a.getBitWidth(), 0);
                                 });
}

OpFoldResult RemPrimOp::fold(ArrayRef<Attribute> operands) {
  // rem(invalid, x) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // division this way and will emit "0 % x".
  if (operands[0].dyn_cast_or_null<InvalidValueAttr>() &&
      !operands[1].dyn_cast_or_null<InvalidValueAttr>())
    return getIntZerosAttr(getType());

  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::DivideOrShift,
                                 [=](APSInt a, APSInt b) -> APInt {
                                   if (!!b)
                                     return a % b;
                                   return APInt(a.getBitWidth(), 0);
                                 });
}

OpFoldResult DShlPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::DivideOrShift,
      [=](APSInt a, APSInt b) -> APInt { return a.shl(b); });
}

OpFoldResult DShlwPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::DivideOrShift,
      [=](APSInt a, APSInt b) -> APInt { return a.shl(b); });
}

OpFoldResult DShrPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::DivideOrShift,
                                 [=](APSInt a, APSInt b) -> APInt {
                                   return getType().isUnsigned() ? a.lshr(b)
                                                                 : a.ashr(b);
                                 });
}

// TODO: Move to DRR.
OpFoldResult AndPrimOp::fold(ArrayRef<Attribute> operands) {
  // and(x, invalid) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.
  if (operands[1].dyn_cast_or_null<InvalidValueAttr>() ||
      operands[0].dyn_cast_or_null<InvalidValueAttr>())
    return getIntZerosAttr(getType());

  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    /// and(x, 0) -> 0
    if (rhsCst.getValue().isNullValue() && rhs().getType() == getType())
      return rhs();

    /// and(x, -1) -> x
    if (rhsCst.getValue().isAllOnesValue() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();
  }

  /// and(x, x) -> x
  if (lhs() == rhs() && rhs().getType() == getType())
    return rhs();

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Normal,
      [](APSInt a, APSInt b) -> APInt { return a & b; });
}

OpFoldResult OrPrimOp::fold(ArrayRef<Attribute> operands) {
  // or(x, invalid) -> x
  // or(invalid, x) -> x
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.
  if (operands[0].dyn_cast_or_null<InvalidValueAttr>() &&
      rhs().getType() == getType())
    return rhs();
  if (operands[1].dyn_cast_or_null<InvalidValueAttr>() &&
      lhs().getType() == getType())
    return lhs();

  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    /// or(x, 0) -> x
    if (rhsCst.getValue().isNullValue() && lhs().getType() == getType())
      return lhs();

    /// or(x, -1) -> -1
    if (rhsCst.getValue().isAllOnesValue() && rhs().getType() == getType() &&
        lhs().getType() == getType())
      return rhs();
  }

  /// or(x, x) -> x
  if (lhs() == rhs() && rhs().getType() == getType())
    return rhs();

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Normal,
      [](APSInt a, APSInt b) -> APInt { return a | b; });
}

OpFoldResult XorPrimOp::fold(ArrayRef<Attribute> operands) {
  // xor(x, invalid) -> x
  // xor(invalid, x) -> x
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.
  if (operands[0].dyn_cast_or_null<InvalidValueAttr>() &&
      rhs().getType() == getType())
    return rhs();
  if (operands[1].dyn_cast_or_null<InvalidValueAttr>() &&
      lhs().getType() == getType())
    return lhs();

  /// xor(x, 0) -> x
  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhsCst.getValue().isNullValue() && lhs().getType() == getType())
      return lhs();

  /// xor(x, x) -> 0
  if (lhs() == rhs()) {
    auto width = abs(getType().getWidthOrSentinel());
    if (width != 0) // We cannot create a zero bit APInt.
      return getIntAttr(getType(), APInt(width, 0));
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Normal,
      [](APSInt a, APSInt b) -> APInt { return a ^ b; });
}

void LEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::LEQWithConstLHS>(context);
}

OpFoldResult LEQPrimOp::fold(ArrayRef<Attribute> operands) {
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // leq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // leq(x, const) -> 0 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // leq(x, const) -> 0 where const < minValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .slt(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // leq(x, const) -> 1 where const >= maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .uge(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // leq(x, const) -> 1 where const >= maxValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sge(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a <= b); });
}

void LTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::LTWithConstLHS>(context);
}

OpFoldResult LTPrimOp::fold(ArrayRef<Attribute> operands) {
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // lt(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  // lt(x, 0) -> 0 when x is unsigned
  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhsCst.getValue().isNullValue() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 0));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // lt(x, const) -> 0 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // lt(x, const) -> 0 where const <= minValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sle(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // lt(x, const) -> 1 where const > maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .ugt(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // lt(x, const) -> 1 where const > maxValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sgt(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a < b); });
}

void GEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::GEQWithConstLHS>(context);
}

OpFoldResult GEQPrimOp::fold(ArrayRef<Attribute> operands) {
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // geq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  // geq(x, 0) -> 1 when x is unsigned
  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhsCst.getValue().isNullValue() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 1));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // geq(x, const) -> 0 where const > maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .ugt(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 0 where const > maxValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sgt(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 1 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // geq(x, const) -> 1 where const <= minValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sle(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a >= b); });
}

void GTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::GTWithConstLHS>(context);
}

OpFoldResult GTPrimOp::fold(ArrayRef<Attribute> operands) {
  bool isUnsigned = lhs().getType().isa<UIntType>();

  // gt(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // gt(x, const) -> 0 where const >= maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .uge(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 0 where const >= maxValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .sge(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 1 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // gt(x, const) -> 1 where const < minValue of the signed type of x
      if (!isUnsigned &&
          rhsCst.getValue()
              .sextOrSelf(commonWidth)
              .slt(getMinSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a > b); });
}

OpFoldResult EQPrimOp::fold(ArrayRef<Attribute> operands) {
  // eq(x, x) -> 1
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 1));

  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    /// eq(x, 1) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst.getValue().isAllOnesValue() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a == b); });
}

LogicalResult EQPrimOp::canonicalize(EQPrimOp op, PatternRewriter &rewriter) {

  if (auto rhsCst = dyn_cast_or_null<ConstantOp>(op.rhs().getDefiningOp())) {
    auto width = op.lhs().getType().cast<IntType>().getBitWidthOrSentinel();

    // eq(x, 0) ->  not(x) when x is 1 bit.
    if (rhsCst.value().isNullValue() && op.lhs().getType() == op.getType() &&
        op.rhs().getType() == op.getType()) {
      rewriter.replaceOpWithNewOp<NotPrimOp>(op, op.lhs());
      return success();
    }

    // eq(x, 0) -> not(orr(x)) when x is >1 bit
    if (rhsCst.value().isNullValue() && width > 1) {
      auto orrOp = rewriter.create<OrRPrimOp>(op.getLoc(), op.lhs());
      rewriter.replaceOpWithNewOp<NotPrimOp>(op, orrOp);
      return success();
    }

    // eq(x, ~0) -> andr(x) when x is >1 bit
    if (rhsCst.value().isAllOnesValue() && width > 1 &&
        op.lhs().getType() == op.rhs().getType()) {
      rewriter.replaceOpWithNewOp<AndRPrimOp>(op, op.lhs());
      return success();
    }
  }

  return failure();
}

OpFoldResult NEQPrimOp::fold(ArrayRef<Attribute> operands) {
  // neq(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    /// neq(x, 0) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst.getValue().isNullValue() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a != b); });
}

LogicalResult NEQPrimOp::canonicalize(NEQPrimOp op, PatternRewriter &rewriter) {
  if (auto rhsCst = dyn_cast_or_null<ConstantOp>(op.rhs().getDefiningOp())) {
    auto width = op.lhs().getType().cast<IntType>().getBitWidthOrSentinel();
    // neq(x, 1) -> not(x) when x is 1 bit
    if (rhsCst.value().isAllOnesValue() && op.lhs().getType() == op.getType() &&
        op.rhs().getType() == op.getType()) {
      rewriter.replaceOpWithNewOp<NotPrimOp>(op, op.lhs());
      return success();
    }

    // neq(x, 0) -> orr(x) when x is >1 bit
    if (rhsCst.value().isNullValue() && width > 1) {
      rewriter.replaceOpWithNewOp<OrRPrimOp>(op, op.lhs());
      return success();
    }

    // neq(x, ~0) -> not(andr(x))) when x is >1 bit
    if (rhsCst.value().isAllOnesValue() && width > 1 &&
        op.lhs().getType() == op.rhs().getType()) {
      auto andrOp = rewriter.create<AndRPrimOp>(op.getLoc(), op.lhs());
      rewriter.replaceOpWithNewOp<NotPrimOp>(op, andrOp);
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Unary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AsSIntPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  if (!operands[0])
    return {};

  // Constant clocks and resets are bool attributes.
  if (auto attr = operands[0].dyn_cast<BoolAttr>())
    return getIntAttr(getType(), APInt(/*bitWidth*/ 1, attr.getValue()));

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (auto attr = operands[0].dyn_cast<IntegerAttr>())
    if (getType().hasWidth())
      return getIntAttr(getType(), attr.getValue());

  return {};
}

OpFoldResult AsUIntPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  if (!operands[0])
    return {};

  // Constant clocks and resets are bool attributes.
  if (auto attr = operands[0].dyn_cast<BoolAttr>())
    return getIntAttr(getType(), APInt(/*bitWidth*/ 1, attr.getValue()));

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (auto attr = operands[0].dyn_cast<IntegerAttr>())
    if (getType().hasWidth())
      return getIntAttr(getType(), attr.getValue());

  return {};
}

OpFoldResult AsAsyncResetPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return BoolAttr::get(getContext(), attr.getValue().getBoolValue());

  return {};
}

OpFoldResult AsClockPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return BoolAttr::get(getContext(), attr.getValue().getBoolValue());

  return {};
}

OpFoldResult CvtPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // Signed to signed is a noop, unsigned operands prepend a zero bit.
  auto cst = getExtendedConstant(getOperand(), operands[0],
                                 getType().getWidthOrSentinel());
  if (cst.hasValue())
    return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult NegPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // FIRRTL negate always adds a bit.
  // -x ---> 0-sext(x) or 0-zext(x)
  auto cst = getExtendedConstant(getOperand(), operands[0],
                                 getType().getWidthOrSentinel());
  if (cst.hasValue())
    return getIntAttr(getType(), APInt((*cst).getBitWidth(), 0) - *cst);

  return {};
}

OpFoldResult NotPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(), ~attr.getValue());

  return {};
}

OpFoldResult AndRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // x == -1
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(), APInt(1, attr.getValue().isAllOnesValue()));

  // one bit is identity.  Only applies to UInt since we cann't make a cast
  // here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

OpFoldResult OrRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // x != 0
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(), APInt(1, !attr.getValue().isNullValue()));

  // one bit is identity.  Only applies to UInt since we cann't make a cast
  // here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

OpFoldResult XorRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // popcount(x) & 1
  if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(getType(),
                      APInt(1, attr.getValue().countPopulation() & 1));

  // one bit is identity.  Only applies to UInt since we can't make a cast here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

//===----------------------------------------------------------------------===//
// Other Operators
//===----------------------------------------------------------------------===//

OpFoldResult CatPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(*this))
    return {};

  // Constant fold cat.
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      auto destWidth = getType().getWidthOrSentinel();
      APInt tmp1 = lhs.getValue().zext(destWidth)
                   << rhs.getValue().getBitWidth();
      APInt tmp2 = rhs.getValue().zext(destWidth);
      return getIntAttr(getType(), tmp1 | tmp2);
    }

  return {};
}

LogicalResult DShlPrimOp::canonicalize(DShlPrimOp op,
                                       PatternRewriter &rewriter) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(op))
    return failure();

  // dshl(x, cst) -> shl(x, cst).  The result size is generally much wider than
  // what is needed for the constant.
  if (auto rhsCst = dyn_cast_or_null<ConstantOp>(op.rhs().getDefiningOp())) {
    // Shift amounts are always unsigned, but shift only takes a 32-bit amount.
    uint64_t shiftAmt = rhsCst.value().getLimitedValue(1ULL << 31);
    auto result =
        rewriter.createOrFold<ShlPrimOp>(op.getLoc(), op.lhs(), shiftAmt);
    rewriter.replaceOpWithNewOp<PadPrimOp>(
        op, result, op.getType().cast<IntType>().getWidthOrSentinel());
    return success();
  }
  return failure();
}

LogicalResult DShrPrimOp::canonicalize(DShrPrimOp op,
                                       PatternRewriter &rewriter) {
  if (!hasKnownWidthIntTypesAndNonZeroResult(op))
    return failure();

  // dshr(x, cst) -> shr(x, cst).  The result size is generally much wider than
  // what is needed for the constant.
  if (auto rhsCst = dyn_cast_or_null<ConstantOp>(op.rhs().getDefiningOp())) {
    // Shift amounts are always unsigned, but shift only takes a 32-bit amount.
    uint64_t shiftAmt = rhsCst.value().getLimitedValue(1ULL << 31);
    auto result =
        rewriter.createOrFold<ShrPrimOp>(op.getLoc(), op.lhs(), shiftAmt);
    rewriter.replaceOpWithNewOp<PadPrimOp>(
        op, result, op.getType().cast<IntType>().getWidthOrSentinel());
    return success();
  }
  return failure();
}

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
  if (inputType == getType() && getType().hasWidth())
    return input();

  // Constant fold.
  if (hasKnownWidthIntTypesAndNonZeroResult(*this))
    if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
      return getIntAttr(
          getType(), attr.getValue().lshr(lo()).truncOrSelf(hi() - lo() + 1));

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

OpFoldResult MuxPrimOp::fold(ArrayRef<Attribute> operands) {
  // mux(cond, x, invalid) -> x
  // mux(cond, invalid, x) -> x
  //
  // These are NOT optimizations that the Scala FIRRTL Compiler makes.  However,
  // these agree with the interpretation of mux with an invalid true of false
  // condition as a conditionally valid statement.
  if (operands[2].dyn_cast_or_null<InvalidValueAttr>())
    return getOperand(1);
  if (operands[1].dyn_cast_or_null<InvalidValueAttr>())
    return getOperand(2);

  // mux(cond, x, x) -> x
  if (high() == low())
    return high();

  // The following folds require that the result has a known width. Otherwise
  // the mux requires an additional padding operation to be inserted, which is
  // not possible in a fold.
  if (getType().getBitWidthOrSentinel() < 0)
    return {};

  // mux(0/1, x, y) -> x or y
  if (auto cond = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (cond.getValue().isNullValue() && low().getType() == getType())
      return low();
    if (!cond.getValue().isNullValue() && high().getType() == getType())
      return high();
  }

  // mux(cond, x, cst)
  if (auto lowCst = operands[2].dyn_cast_or_null<IntegerAttr>()) {
    // mux(cond, c1, c2)
    if (auto highCst = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      // mux(cond, 1, 0) -> cond
      if (highCst.getValue().isOneValue() && lowCst.getValue().isNullValue() &&
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

static LogicalResult canonicalizeMux(MuxPrimOp op, PatternRewriter &rewriter) {
  // If the mux has a known output width, pad the operands up to this width.
  // Most folds on mux require that folded operands are of the same width as
  // the mux itself.
  auto width = op.getType().getBitWidthOrSentinel();
  if (width < 0)
    return failure();

  auto pad = [&](Value input) {
    auto inputWidth =
        input.getType().cast<FIRRTLType>().getBitWidthOrSentinel();
    if (inputWidth < 0 || width == inputWidth)
      return input;
    return rewriter.create<PadPrimOp>(op.getLoc(), op.getType(), input, width)
        .getResult();
  };

  auto newHigh = pad(op.high());
  auto newLow = pad(op.low());
  if (newHigh == op.high() && newLow == op.low())
    return failure();

  rewriter.replaceOpWithNewOp<MuxPrimOp>(
      op, op.getType(), ValueRange{op.sel(), newHigh, newLow}, op->getAttrs());
  return success();
}

void MuxPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add(canonicalizeMux);
  results.add<patterns::MuxSameCondLow>(context);
  results.add<patterns::MuxSameCondHigh>(context);
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
  if (auto cst = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    auto destWidth = getType().getWidthOrSentinel();
    if (destWidth == -1)
      return {};

    if (inputType.isSigned())
      return getIntAttr(getType(), cst.getValue().sext(destWidth));
    return getIntAttr(getType(), cst.getValue().zext(destWidth));
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
  if (auto cst = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    auto inputWidth = inputType.getWidthOrSentinel();
    if (inputWidth != -1) {
      auto resultWidth = inputWidth + shiftAmount;
      shiftAmount = std::min(shiftAmount, resultWidth);
      return getIntAttr(getType(),
                        cst.getValue().zext(resultWidth).shl(shiftAmount));
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
  if (auto cst = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    APInt value;
    if (inputType.isSigned())
      value = cst.getValue().ashr(std::min(shiftAmount, inputWidth - 1));
    else
      value = cst.getValue().lshr(std::min(shiftAmount, inputWidth));
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

OpFoldResult HeadPrimOp::fold(ArrayRef<Attribute> operands) {
  if (hasKnownWidthIntTypesAndNonZeroResult(*this))
    if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>()) {
      int shiftAmount =
          input().getType().cast<IntType>().getWidthOrSentinel() - amount();
      return getIntAttr(
          getType(), attr.getValue().lshr(shiftAmount).truncOrSelf(amount()));
    }

  return {};
}

OpFoldResult TailPrimOp::fold(ArrayRef<Attribute> operands) {
  if (hasKnownWidthIntTypesAndNonZeroResult(*this))
    if (auto attr = operands[0].dyn_cast_or_null<IntegerAttr>())
      return getIntAttr(getType(), attr.getValue().truncOrSelf(
                                       getType().getWidthOrSentinel()));
  return {};
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

LogicalResult SubaccessOp::canonicalize(SubaccessOp op,
                                        PatternRewriter &rewriter) {
  if (auto index = op.index().getDefiningOp()) {
    if (auto constIndex = dyn_cast<ConstantOp>(index)) {
      // The SubindexOp require the index value to be unsigned 32-bits
      // integer.
      auto value = constIndex.value().getExtValue();
      auto valueAttr = rewriter.getI32IntegerAttr(value);
      rewriter.replaceOpWithNewOp<SubindexOp>(op, op.result().getType(),
                                              op.input(), valueAttr);
      return success();
    }
  }
  return failure();
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
  // While we can do this for nearly all wires, we currently limit it to simple
  // things.
  Operation *connectedDecl = op.dest().getDefiningOp();
  if (!connectedDecl)
    return failure();

  // Only support wire and reg for now.
  if (!isa<WireOp>(connectedDecl) && !isa<RegOp>(connectedDecl))
    return failure();
  if (AnnotationSet(connectedDecl).hasDontTouch())
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
      if (!AnnotationSet(wire).hasDontTouch() && wire->hasOneUse()) {
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
    // firrtl.tail always returns uint even for sint operands.
    IntType tmpType = destType.cast<IntType>();
    if (tmpType.isSigned())
      tmpType = UIntType::get(destType.getContext(), destWidth);
    auto shortened = rewriter.createOrFold<TailPrimOp>(
        op.getLoc(), tmpType, op.getOperand(1), srcWidth - destWidth);
    // Insert the cast back to signed if needed.
    if (tmpType != destType)
      shortened =
          rewriter.createOrFold<AsSIntPrimOp>(op.getLoc(), destType, shortened);
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

void RegResetOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::RegresetWithZeroReset,
                 patterns::RegresetWithInvalidReset,
                 patterns::RegresetWithInvalidResetValue>(context);
}

LogicalResult MemOp::canonicalize(MemOp op, PatternRewriter &rewriter) {
  // If memory has known, but zero width, eliminate it.
  if (op.getDataType().getBitWidthOrSentinel() != 0)
    return failure();
  // Make sure are users are safe to replace
  for (auto port : op->getResults())
    for (auto user : port.getUsers())
      if (!isa<SubfieldOp>(user))
        return failure();

  // Annoyingly, there isn't a good replacement for the port as a whole, since
  // they have an outer flip type.
  for (auto port : op->getResults()) {
    for (auto user : llvm::make_early_inc_range(port.getUsers())) {
      SubfieldOp sfop = cast<SubfieldOp>(user);
      rewriter.replaceOpWithNewOp<WireOp>(sfop, sfop.result().getType());
    }
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

// Turn synchronous reset looking register updates to registers with resets.
// Also, const prop registers that are driven by a mux tree containing only
// instances of one constant or self-assigns.
static LogicalResult foldHiddenReset(RegOp reg, PatternRewriter &rewriter) {
  // reg ; connect(reg, mux(port, const, val)) ->
  // reg.reset(port, const); connect(reg, val)

  // Find the one true connect, or bail
  ConnectOp con;
  for (Operation *user : reg->getUsers()) {
    // If we see a partial connect or attach, just conservatively fail.
    if (isa<PartialConnectOp>(user) || isa<AttachOp>(user))
      return failure();

    auto aConnect = dyn_cast<ConnectOp>(user);
    if (aConnect && aConnect.dest().getDefiningOp() == reg) {
      if (con)
        return failure();
      con = aConnect;
    }
  }
  if (!con)
    return failure();

  auto mux = dyn_cast_or_null<MuxPrimOp>(con.src().getDefiningOp());
  if (!mux)
    return failure();
  auto high = mux.high().getDefiningOp();
  auto low = mux.low().getDefiningOp();
  // Reset value must be constant
  auto constOp = dyn_cast_or_null<ConstantOp>(high);

  // Detect the case if a register only has two possible drivers:
  // (1) itself/uninit and (2) constant.
  // The mux can then be replaced with the constant.
  // r = mux(cond, r, 3) --> r = 3
  // r = mux(cond, 3, r) --> r = 3
  bool constReg = false;

  if (constOp && low == reg)
    constReg = true;
  else if (dyn_cast_or_null<ConstantOp>(low) && high == reg) {
    constReg = true;
    constOp = dyn_cast<ConstantOp>(low);
  }
  if (!constOp)
    return failure();

  // reset should be a module port (heuristic to limit to intended reset lines).
  if (!mux.sel().isa<BlockArgument>())
    return failure();

  // Check all types should be typed by now
  auto regTy = reg.getType();
  if (con.dest().getType() != regTy || con.src().getType() != regTy ||
      mux.high().getType() != regTy || mux.low().getType() != regTy ||
      regTy.getBitWidthOrSentinel() < 1)
    return failure();

  // Ok, we know we are doing the transformation.

  // Make sure the constant dominates all users.
  if (constOp != &con->getBlock()->front())
    constOp->moveBefore(&con->getBlock()->front());

  if (!constReg)
    rewriter.replaceOpWithNewOp<RegResetOp>(reg, reg.getType(), reg.clockVal(),
                                            mux.sel(), mux.high(), reg.name(),
                                            reg.annotations());
  auto pt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(con);
  rewriter.replaceOpWithNewOp<ConnectOp>(con, con.dest(),
                                         constReg ? constOp : mux.low());
  rewriter.restoreInsertionPoint(pt);
  return success();
}

LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  if (!(AnnotationSet(op).hasDontTouch()) &&
      succeeded(foldHiddenReset(op, rewriter)))
    return success();

  return failure();
}
