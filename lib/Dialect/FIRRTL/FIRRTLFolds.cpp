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
#include "circt/Support/APInt.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// Forward Decl for patterns.
static bool isUselessName(circt::StringRef name);

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
/// type. This handles both the known width and unknown width case.
static IntegerAttr getIntZerosAttr(Type type) {
  int32_t width = abs(type.cast<IntType>().getWidthOrSentinel());
  return getIntAttr(type, APInt(width, 0));
}

/// Return true if this operation's operands and results all have a known width.
/// This only works for integer types.
static bool hasKnownWidthIntTypes(Operation *op) {
  auto resultType = op->getResult(0).getType().cast<IntType>();
  if (!resultType.hasWidth())
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

/// Return true if this is a useless temporary name produced by FIRRTL.  We
/// drop these as they don't convey semantic meaning.
static bool isUselessName(StringRef name) {
  // Ignore _T and _T_123
  if (name.startswith("_T")) {
    if (name.size() == 2)
      return true;
    return name.size() > 3 && name[2] == '_' && llvm::isDigit(name[3]);
  }

  // Ignore _GEN and _GEN_123, these are produced by Namespace.scala.
  if (name.startswith("_GEN")) {
    if (name.size() == 4)
      return true;
    return name.size() > 5 && name[4] == '_' && llvm::isDigit(name[5]);
  }
  return false;
}

/// Implicitly replace the operand to a constant folding operation with a const
/// 0 in case the operand is non-constant but has a bit width 0, or if the
/// operand is an invalid value.
///
/// This makes constant folding significantly easier, as we can simply pass the
/// operands to an operation through this function to appropriately replace any
/// zero-width dynamic values or invalid values with a constant of value 0.
static Optional<APSInt> getExtendedConstant(Value operand, Attribute constant,
                                            int32_t destWidth) {
  assert(operand.getType().isa<IntType>() &&
         "getExtendedConstant is limited to integer types");

  // We never support constant folding to unknown width values.
  if (destWidth < 0)
    return {};

  // InvalidValue inputs simply read as zero.
  if (auto result = constant.dyn_cast_or_null<InvalidValueAttr>())
    return APSInt(destWidth, operand.getType().cast<IntType>().isUnsigned());

  // Extension signedness follows the operand sign.
  if (IntegerAttr result = constant.dyn_cast_or_null<IntegerAttr>())
    return extOrTruncZeroWidth(result.getAPSInt(), destWidth);

  // If the operand is zero bits, then we can return a zero of the result
  // type.
  if (operand.getType().cast<IntType>().getWidth() == 0)
    return APSInt(destWidth, operand.getType().cast<IntType>().isUnsigned());
  return {};
}

/// Determine the value of a constant operand for the sake of constant folding.
/// This will map `invalidvalue` to a zero value of the corresopnding type,
/// which aligns with how the Scala FIRRTL compiler handles invalids in most
/// cases. For a full discussion of this see the FIRRTL Rationale document.
static Optional<APSInt> getConstant(Attribute operand) {
  if (!operand)
    return {};
  if (auto attr = operand.dyn_cast<InvalidValueAttr>()) {
    if (auto type = attr.getType().dyn_cast<IntType>())
      return APSInt(type.getWidth().getValueOr(1), type.isUnsigned());
    if (attr.getType().isa<ClockType, ResetType, AsyncResetType>())
      return APSInt(1);
  }
  if (auto attr = operand.dyn_cast<BoolAttr>())
    return APSInt(APInt(1, attr.getValue()));
  if (auto attr = operand.dyn_cast<IntegerAttr>())
    return attr.getAPSInt();
  return {};
}

/// Determine whether a constant operand is a zero value for the sake of
/// constant folding. This considers `invalidvalue` to be zero.
static bool isConstantZero(Attribute operand) {
  if (auto cst = getConstant(operand))
    return cst->isZero();
  return false;
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

  // We cannot fold something to an unknown width.
  auto resultType = op->getResult(0).getType().cast<IntType>();
  if (resultType.getWidthOrSentinel() < 0)
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

/// Applies the canonicalization function `canonicalize` to the given operation.
///
/// Determines which (if any) of the operation's operands are constants, and
/// provides them as arguments to the callback function. Any `invalidvalue` in
/// the input is mapped to a constant zero. The value returned from the callback
/// is used as the replacement for `op`, and an additional pad operation is
/// inserted if necessary. Does nothing if the result of `op` is of unknown
/// width, in which case the necessity of a pad cannot be determined.
static LogicalResult canonicalizePrimOp(
    Operation *op, PatternRewriter &rewriter,
    const function_ref<OpFoldResult(ArrayRef<Attribute>)> &canonicalize) {
  // Can only operate on FIRRTL primitive operations.
  if (op->getNumResults() != 1)
    return failure();
  auto type = op->getResult(0).getType().dyn_cast<FIRRTLType>();
  if (!type)
    return failure();

  // Can only operate on operations with a known result width.
  auto width = type.getBitWidthOrSentinel();
  if (width < 0)
    return failure();

  // Determine which of the operands are constants.
  SmallVector<Attribute, 3> constOperands;
  constOperands.reserve(op->getNumOperands());
  for (auto operand : op->getOperands()) {
    Attribute attr;
    if (auto *defOp = operand.getDefiningOp())
      TypeSwitch<Operation *>(defOp)
          .Case<ConstantOp, SpecialConstantOp, InvalidValueOp>(
              [&](auto op) { attr = op.fold({}).template get<Attribute>(); });
    constOperands.push_back(attr);
  }

  // Perform the canonicalization and materialize the result if it is a
  // constant.
  auto result = canonicalize(constOperands);
  if (!result)
    return failure();
  Value resultValue;
  if (auto cst = result.dyn_cast<Attribute>())
    resultValue = op->getDialect()
                      ->materializeConstant(rewriter, cst, type, op->getLoc())
                      ->getResult(0);
  else
    resultValue = result.get<Value>();

  // Insert a pad if the type widths disagree.
  if (width != resultValue.getType().cast<FIRRTLType>().getBitWidthOrSentinel())
    resultValue = rewriter.create<PadPrimOp>(op->getLoc(), resultValue, width);

  // Insert a cast if this is a uint vs. sint or vice versa.
  if (type.isa<SIntType>() && resultValue.getType().isa<UIntType>())
    resultValue = rewriter.create<AsSIntPrimOp>(op->getLoc(), resultValue);
  else if (type.isa<UIntType>() && resultValue.getType().isa<SIntType>())
    resultValue = rewriter.create<AsUIntPrimOp>(op->getLoc(), resultValue);

  assert(type == resultValue.getType() && "canonicalization changed type");
  rewriter.replaceOp(op, resultValue);
  return success();
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
  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::Normal,
                                 [=](APSInt a, APSInt b) { return a + b; });
}

LogicalResult AddPrimOp::canonicalize(AddPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(op, rewriter,
                            [&](ArrayRef<Attribute> operands) -> OpFoldResult {
                              // add(x, 0) -> x
                              if (isConstantZero(operands[1]))
                                return op.getOperand(0);
                              // add(0, x) -> x
                              if (isConstantZero(operands[0]))
                                return op.getOperand(1);
                              return {};
                            });
}

OpFoldResult SubPrimOp::fold(ArrayRef<Attribute> operands) {
  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::Normal,
                                 [=](APSInt a, APSInt b) { return a - b; });
}

LogicalResult SubPrimOp::canonicalize(SubPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        // sub(x, 0) -> x
        if (isConstantZero(operands[1]))
          return op.getOperand(0);
        // sub(0, x) -> neg(x)  if x is signed
        // sub(0, x) -> asUInt(neg(x))  if x is unsigned
        if (isConstantZero(operands[0])) {
          Value value =
              rewriter.create<NegPrimOp>(op.getLoc(), op.getOperand(1));
          return value;
        }
        return {};
      });
}

OpFoldResult MulPrimOp::fold(ArrayRef<Attribute> operands) {
  // mul(x, 0) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // multiplication this way and will emit "x * 0".
  if (isConstantZero(operands[1]) || isConstantZero(operands[0]))
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
    // Only fold if we have at least 1 bit of width to represent the `1` value.
    if (width != 0)
      return getIntAttr(getType(), APInt(width, 1));
  }

  // div(0, x) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // division this way and will emit "0 / x".
  if (isConstantZero(operands[0]) && !isConstantZero(operands[1]))
    return getIntZerosAttr(getType());

  /// div(x, 1) -> x : (uint, uint) -> uint
  ///
  /// UInt division by one returns the numerator. SInt division can't
  /// be folded here because it increases the return type bitwidth by
  /// one and requires sign extension (a new op).
  if (auto rhsCst = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhsCst.getValue().isOne() && lhs().getType() == getType())
      return lhs();

  return constFoldFIRRTLBinaryOp(*this, operands, BinOpKind::DivideOrShift,
                                 [=](APSInt a, APSInt b) -> APInt {
                                   if (!!b)
                                     return a / b;
                                   return APInt(a.getBitWidth(), 0);
                                 });
}

OpFoldResult RemPrimOp::fold(ArrayRef<Attribute> operands) {
  // rem(x, x) -> 0
  //
  // Division by zero is undefined in the FIRRTL specification.  This fold
  // exploits that fact to optimize self division remainder to zero.  Note:
  // this should supersede any division with invalid or zero.  Remainder of
  // division of invalid by invalid should be zero.
  if (lhs() == rhs())
    return getIntZerosAttr(getType());

  // rem(0, x) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // division this way and will emit "0 % x".
  if (isConstantZero(operands[0]))
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
                                   return getType().isUnsigned() ||
                                                  !a.getBitWidth()
                                              ? a.lshr(b)
                                              : a.ashr(b);
                                 });
}

// TODO: Move to DRR.
OpFoldResult AndPrimOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhsCst = getConstant(operands[1])) {
    /// and(x, 0) -> 0
    if (rhsCst->isZero() && rhs().getType() == getType())
      return getIntZerosAttr(getType());

    /// and(x, -1) -> x
    if (rhsCst->isAllOnes() && lhs().getType() == getType() &&
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
  if (auto rhsCst = getConstant(operands[1])) {
    /// or(x, 0) -> x
    if (rhsCst.getValue().isZero() && lhs().getType() == getType())
      return lhs();

    /// or(x, -1) -> -1
    if (rhsCst.getValue().isAllOnes() && rhs().getType() == getType() &&
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
  /// xor(x, 0) -> x
  if (auto rhsCst = getConstant(operands[1]))
    if (rhsCst.getValue().isZero() && lhs().getType() == getType())
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
    if (auto rhsCst = getConstant(operands[1])) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 0);

      // leq(x, const) -> 0 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // leq(x, const) -> 0 where const < minValue of the signed type of x
      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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
  if (auto rhsCst = getConstant(operands[1])) {
    if (rhsCst.getValue().isZero() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 0));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(operands[1])) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 0);

      // lt(x, const) -> 0 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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
  if (auto rhsCst = getConstant(operands[1])) {
    if (rhsCst.getValue().isZero() && lhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 1));
  }

  // Comparison against constant outside type bounds.
  if (auto width = lhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(operands[1])) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 0);

      // geq(x, const) -> 0 where const > maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .ugt(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 0 where const > maxValue of the signed type of x
      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
              .sgt(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 1 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // geq(x, const) -> 1 where const <= minValue of the signed type of x
      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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
    if (auto rhsCst = getConstant(operands[1])) {
      auto commonWidth =
          std::max<int32_t>(*width, rhsCst.getValue().getBitWidth());
      commonWidth = std::max(commonWidth, 0);

      // gt(x, const) -> 0 where const >= maxValue of the unsigned type of x
      if (isUnsigned &&
          rhsCst.getValue()
              .zextOrSelf(commonWidth)
              .uge(getMaxUnsignedValue(*width).zextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 0 where const >= maxValue of the signed type of x
      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
              .sge(getMaxSignedValue(*width).sextOrSelf(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 1 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // gt(x, const) -> 1 where const < minValue of the signed type of x
      if (!isUnsigned &&
          sextOrSelfZeroWidth(rhsCst.getValue(), commonWidth)
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

  if (auto rhsCst = getConstant(operands[1])) {
    /// eq(x, 1) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst.getValue().isAllOnes() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a == b); });
}

LogicalResult EQPrimOp::canonicalize(EQPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        if (auto rhsCst = getConstant(operands[1])) {
          auto width =
              op.lhs().getType().cast<IntType>().getBitWidthOrSentinel();

          // eq(x, 0) ->  not(x) when x is 1 bit.
          if (rhsCst->isZero() && op.lhs().getType() == op.getType() &&
              op.rhs().getType() == op.getType()) {
            return rewriter.create<NotPrimOp>(op.getLoc(), op.lhs())
                .getResult();
          }

          // eq(x, 0) -> not(orr(x)) when x is >1 bit
          if (rhsCst->isZero() && width > 1) {
            auto orrOp = rewriter.create<OrRPrimOp>(op.getLoc(), op.lhs());
            return rewriter.create<NotPrimOp>(op.getLoc(), orrOp).getResult();
          }

          // eq(x, ~0) -> andr(x) when x is >1 bit
          if (rhsCst->isAllOnes() && width > 1 &&
              op.lhs().getType() == op.rhs().getType()) {
            return rewriter.create<AndRPrimOp>(op.getLoc(), op.lhs())
                .getResult();
          }
        }

        return {};
      });
}

OpFoldResult NEQPrimOp::fold(ArrayRef<Attribute> operands) {
  // neq(x, x) -> 0
  if (lhs() == rhs())
    return getIntAttr(getType(), APInt(1, 0));

  if (auto rhsCst = getConstant(operands[1])) {
    /// neq(x, 0) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst.getValue().isZero() && lhs().getType() == getType() &&
        rhs().getType() == getType())
      return lhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, operands, BinOpKind::Compare,
      [=](APSInt a, APSInt b) -> APInt { return APInt(1, a != b); });
}

LogicalResult NEQPrimOp::canonicalize(NEQPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        if (auto rhsCst = getConstant(operands[1])) {
          auto width =
              op.lhs().getType().cast<IntType>().getBitWidthOrSentinel();

          // neq(x, 1) -> not(x) when x is 1 bit
          if (rhsCst->isAllOnes() && op.lhs().getType() == op.getType() &&
              op.rhs().getType() == op.getType()) {
            return rewriter.create<NotPrimOp>(op.getLoc(), op.lhs())
                .getResult();
          }

          // neq(x, 0) -> orr(x) when x is >1 bit
          if (rhsCst->isZero() && width > 1) {
            return rewriter.create<OrRPrimOp>(op.getLoc(), op.lhs())
                .getResult();
          }

          // neq(x, ~0) -> not(andr(x))) when x is >1 bit
          if (rhsCst->isAllOnes() && width > 1 &&
              op.lhs().getType() == op.rhs().getType()) {
            auto andrOp = rewriter.create<AndRPrimOp>(op.getLoc(), op.lhs());
            return rewriter.create<NotPrimOp>(op.getLoc(), andrOp).getResult();
          }
        }

        return {};
      });
}

//===----------------------------------------------------------------------===//
// Unary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AsSIntPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (getType().hasWidth())
    if (auto cst = getConstant(operands[0]))
      return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult AsUIntPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (getType().hasWidth())
    if (auto cst = getConstant(operands[0]))
      return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult AsAsyncResetPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto cst = getConstant(operands[0]))
    return BoolAttr::get(getContext(), cst->getBoolValue());

  return {};
}

OpFoldResult AsClockPrimOp::fold(ArrayRef<Attribute> operands) {
  // No effect.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto cst = getConstant(operands[0]))
    return BoolAttr::get(getContext(), cst->getBoolValue());

  return {};
}

OpFoldResult CvtPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // Signed to signed is a noop, unsigned operands prepend a zero bit.
  auto cst = getExtendedConstant(getOperand(), operands[0],
                                 getType().getWidthOrSentinel());
  if (cst.hasValue())
    return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult NegPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // FIRRTL negate always adds a bit.
  // -x ---> 0-sext(x) or 0-zext(x)
  if (auto cst = getExtendedConstant(getOperand(), operands[0],
                                     getType().getWidthOrSentinel()))
    return getIntAttr(getType(), APInt((*cst).getBitWidth(), 0) - *cst);

  return {};
}

OpFoldResult NotPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  if (auto cst = getExtendedConstant(getOperand(), operands[0],
                                     getType().getWidthOrSentinel()))
    return getIntAttr(getType(), ~*cst);

  return {};
}

OpFoldResult AndRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // x == -1
  if (auto cst = getConstant(operands[0]))
    return getIntAttr(getType(), APInt(1, cst->isAllOnes()));

  // one bit is identity.  Only applies to UInt since we can't make a cast
  // here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

OpFoldResult OrRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // x != 0
  if (auto cst = getConstant(operands[0]))
    return getIntAttr(getType(), APInt(1, !cst->isZero()));

  // one bit is identity.  Only applies to UInt since we can't make a cast
  // here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

OpFoldResult XorRPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // popcount(x) & 1
  if (auto cst = getConstant(operands[0]))
    return getIntAttr(getType(), APInt(1, cst->countPopulation() & 1));

  // one bit is identity.  Only applies to UInt since we can't make a cast here.
  if (isUInt1(input().getType()))
    return input();

  return {};
}

//===----------------------------------------------------------------------===//
// Other Operators
//===----------------------------------------------------------------------===//

OpFoldResult CatPrimOp::fold(ArrayRef<Attribute> operands) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // Constant fold cat.
  if (auto lhs = getConstant(operands[0]))
    if (auto rhs = getConstant(operands[1])) {
      auto destWidth = getType().getWidthOrSentinel();
      APInt tmp1 = lhs->zextOrSelf(destWidth) << rhs->getBitWidth();
      APInt tmp2 = rhs->zextOrSelf(destWidth);
      return getIntAttr(getType(), tmp1 | tmp2);
    }

  return {};
}

LogicalResult DShlPrimOp::canonicalize(DShlPrimOp op,
                                       PatternRewriter &rewriter) {
  if (!hasKnownWidthIntTypes(op))
    return failure();

  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        // dshl(x, cst) -> shl(x, cst).  The result size is generally much wider
        // than what is needed for the constant.
        if (auto rhsCst = getConstant(operands[1])) {
          // Shift amounts are always unsigned, but shift only takes a 32-bit
          // amount.
          uint64_t shiftAmt = rhsCst->getLimitedValue(1ULL << 31);
          return rewriter.createOrFold<ShlPrimOp>(op.getLoc(), op.lhs(),
                                                  shiftAmt);
        }
        return {};
      });
}

LogicalResult DShrPrimOp::canonicalize(DShrPrimOp op,
                                       PatternRewriter &rewriter) {
  if (!hasKnownWidthIntTypes(op))
    return failure();

  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        // dshr(x, cst) -> shr(x, cst).  The result size is generally much wider
        // than what is needed for the constant.
        if (auto rhsCst = getConstant(operands[1])) {
          // Shift amounts are always unsigned, but shift only takes a 32-bit
          // amount.
          uint64_t shiftAmt = rhsCst->getLimitedValue(1ULL << 31);
          return rewriter.createOrFold<ShrPrimOp>(op.getLoc(), op.lhs(),
                                                  shiftAmt);
        }
        return {};
      });
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

OpFoldResult BitCastOp::fold(ArrayRef<Attribute> operands) {
  auto op = (*this);
  // BitCast is redundant if input and result types are same.
  if (op.getType() == op.input().getType())
    return op.input();

  // Two consecutive BitCasts are redundant if first bitcast type is same as the
  // final result type.
  if (BitCastOp in = dyn_cast_or_null<BitCastOp>(op.input().getDefiningOp()))
    if (op.getType() == in.input().getType())
      return in.input();

  return {};
}

OpFoldResult BitsPrimOp::fold(ArrayRef<Attribute> operands) {
  auto inputType = input().getType().cast<FIRRTLType>();
  // If we are extracting the entire input, then return it.
  if (inputType == getType() && getType().hasWidth())
    return input();

  // Constant fold.
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(operands[0]))
      return getIntAttr(getType(),
                        cst->lshr(lo()).truncOrSelf(hi() - lo() + 1));

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

  // mux(cond, x, x) -> x
  if (high() == low())
    return high();

  // The following folds require that the result has a known width. Otherwise
  // the mux requires an additional padding operation to be inserted, which is
  // not possible in a fold.
  if (getType().getBitWidthOrSentinel() < 0)
    return {};

  // mux(0/1, x, y) -> x or y
  if (auto cond = getConstant(operands[0])) {
    if (cond->isZero() && low().getType() == getType())
      return low();
    if (!cond->isZero() && high().getType() == getType())
      return high();
  }

  // mux(cond, x, cst)
  if (auto lowCst = getConstant(operands[2])) {
    // mux(cond, c1, c2)
    if (auto highCst = getConstant(operands[1])) {
      // mux(cond, cst, cst) -> cst
      if (highCst->getBitWidth() == lowCst->getBitWidth() &&
          *highCst == *lowCst)
        return getIntAttr(getType(), *highCst);
      // mux(cond, 1, 0) -> cond
      if (highCst->isOne() && lowCst->isZero() && getType() == sel().getType())
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
  if (auto cst = getConstant(operands[0])) {
    auto destWidth = getType().getWidthOrSentinel();
    if (destWidth == -1)
      return {};

    if (inputType.isSigned() && cst->getBitWidth())
      return getIntAttr(getType(), cst->sext(destWidth));
    return getIntAttr(getType(), cst->zext(destWidth));
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
  if (auto cst = getConstant(operands[0])) {
    auto inputWidth = inputType.getWidthOrSentinel();
    if (inputWidth != -1) {
      auto resultWidth = inputWidth + shiftAmount;
      shiftAmount = std::min(shiftAmount, resultWidth);
      return getIntAttr(getType(), cst->zext(resultWidth).shl(shiftAmount));
    }
  }
  return {};
}

OpFoldResult ShrPrimOp::fold(ArrayRef<Attribute> operands) {
  auto input = this->input();
  auto inputType = input.getType().cast<IntType>();
  int shiftAmount = amount();

  // shr(x, 0) -> x
  if (shiftAmount == 0)
    return input;

  auto inputWidth = inputType.getWidthOrSentinel();
  if (inputWidth == -1)
    return {};
  if (inputWidth == 0)
    return getIntZerosAttr(getType());

  // shr(x, cst) where cst is all of x's bits and x is unsigned is 0.
  // If x is signed, it is the sign bit.
  if (shiftAmount >= inputWidth && inputType.isUnsigned())
    return getIntAttr(getType(), APInt(1, 0));

  // Constant fold.
  if (auto cst = getConstant(operands[0])) {
    APInt value;
    if (inputType.isSigned())
      value = cst->ashr(std::min(shiftAmount, inputWidth - 1));
    else
      value = cst->lshr(std::min(shiftAmount, inputWidth));
    auto resultWidth = std::max(inputWidth - shiftAmount, 1);
    return getIntAttr(getType(), value.truncOrSelf(resultWidth));
  }
  return {};
}

LogicalResult ShrPrimOp::canonicalize(ShrPrimOp op, PatternRewriter &rewriter) {
  auto inputWidth = op.input().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth <= 0)
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
  if (inputWidth <= 0)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned keepAmount = op.amount();
  if (keepAmount)
    replaceWithBits(op, op.input(), inputWidth - 1, inputWidth - keepAmount,
                    rewriter);
  return success();
}

OpFoldResult HeadPrimOp::fold(ArrayRef<Attribute> operands) {
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(operands[0])) {
      int shiftAmount =
          input().getType().cast<IntType>().getWidthOrSentinel() - amount();
      return getIntAttr(getType(),
                        cst->lshr(shiftAmount).truncOrSelf(amount()));
    }

  return {};
}

OpFoldResult TailPrimOp::fold(ArrayRef<Attribute> operands) {
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(operands[0]))
      return getIntAttr(getType(),
                        cst->truncOrSelf(getType().getWidthOrSentinel()));
  return {};
}

LogicalResult TailPrimOp::canonicalize(TailPrimOp op,
                                       PatternRewriter &rewriter) {
  auto inputWidth = op.input().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth <= 0)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned dropAmount = op.amount();
  if (dropAmount != unsigned(inputWidth))
    replaceWithBits(op, op.input(), inputWidth - dropAmount - 1, 0, rewriter);
  return success();
}

LogicalResult SubaccessOp::canonicalize(SubaccessOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        if (auto constIndex = getConstant(operands[1])) {
          // The SubindexOp require the index value to be unsigned 32-bits
          // integer.
          auto value = constIndex->getExtValue();
          auto valueAttr = rewriter.getI32IntegerAttr(value);
          return rewriter.createOrFold<SubindexOp>(
              op.getLoc(), op.result().getType(), op.input(), valueAttr);
        }
        return {};
      });
}

OpFoldResult MultibitMuxOp::fold(ArrayRef<Attribute> operands) {
  // If there is only one input, just return it.
  if (operands.size() == 2)
    return getOperand(1);

  if (auto constIndex = getConstant(operands[0])) {
    auto index = constIndex->getExtValue();
    // operands[0] is index so (index + 1) is the index we want.
    if (index >= 0 && index + 1 < static_cast<int64_t>(operands.size()))
      return getOperand(index + 1);
  }

  return {};
}

LogicalResult MultibitMuxOp::canonicalize(MultibitMuxOp op,
                                          PatternRewriter &rewriter) {
  // If all operands are equal, just canonicalize to it. We can add this
  // canonicalization as a folder but it costly to look through all inputs so it
  // is added here.
  if (llvm::all_of(op.inputs().drop_front(),
                   [&](auto input) { return input == op.inputs().front(); })) {
    rewriter.replaceOp(op, op.inputs().front());
    return success();
  }

  // If the size is 2, canonicalize into a normal mux to introduce more folds.
  if (op.inputs().size() != 2)
    return failure();

  // multibit_mux(index, {lhs, rhs}) -> mux(index==0, lhs, rhs)
  Value zero = rewriter.create<ConstantOp>(
      op.getLoc(), op.index().getType().cast<IntType>(), APInt(1, 0));
  Value cond = rewriter.createOrFold<EQPrimOp>(op.getLoc(), op.index(), zero);
  rewriter.replaceOpWithNewOp<MuxPrimOp>(op, cond, op.inputs()[0],
                                         op.inputs()[1]);
  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Scan all the uses of the specified value, checking to see if there is
/// exactly one connect that sets the value as its destination.  This returns
/// the operation if found and if all the other users are "reads" from the
/// value.
static ConnectOp getSingleConnectUserOf(Value value) {
  ConnectOp connect;
  for (Operation *user : value.getUsers()) {
    // If we see a partial connect or attach, just conservatively fail.
    if (isa<PartialConnectOp>(user) || isa<AttachOp>(user))
      return {};

    if (auto aConnect = dyn_cast<ConnectOp>(user))
      if (aConnect.dest() == value) {
        if (!connect)
          connect = aConnect;
        else
          return {};
      }
  }
  return connect;
}

// Forward simple values through wire's and reg's.
static LogicalResult canonicalizeSingleSetConnect(ConnectOp op,
                                                  PatternRewriter &rewriter) {
  // While we can do this for nearly all wires, we currently limit it to simple
  // things.
  Operation *connectedDecl = op.dest().getDefiningOp();
  if (!connectedDecl)
    return failure();

  // Only support wire and reg for now.
  if (!isa<WireOp>(connectedDecl) && !isa<RegOp>(connectedDecl))
    return failure();
  if (hasDontTouch(connectedDecl))
    return failure();

  // Only forward if the types exactly match and there is one connect.
  if (op.dest().getType() != op.src().getType() ||
      getSingleConnectUserOf(op.dest()) != op)
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

  auto replacement = op.src();
  if (srcValueOp) {
    // Replace with constant zero.
    if (isa<InvalidValueOp>(srcValueOp)) {
      if (op.dest().getType().isa<ClockType, AsyncResetType, ResetType>())
        replacement = rewriter.create<SpecialConstantOp>(
            op.src().getLoc(), op.dest().getType(),
            rewriter.getBoolAttr(false));
      else
        replacement =
            rewriter.create<ConstantOp>(op.src().getLoc(), op.dest().getType(),
                                        getIntZerosAttr(op.dest().getType()));
    }
    // This will be replaced with the constant source.  First, make sure the
    // constant dominates all users.
    else if (srcValueOp != &declBlock->front()) {
      srcValueOp->moveBefore(&declBlock->front());
    }
  }

  // Replace all things *using* the decl with the constant/port, and
  // remove the declaration.
  rewriter.replaceOp(connectedDecl, replacement);

  // Remove the connect.
  rewriter.eraseOp(op);

  return success();
}

LogicalResult ConnectOp::canonicalize(ConnectOp op, PatternRewriter &rewriter) {
  // TODO: Canonicalize towards explicit extensions and flips here.

  // If there is a simple value connected to a foldable decl like a wire or reg,
  // see if we can eliminate the decl.
  if (succeeded(canonicalizeSingleSetConnect(op, rewriter)))
    return success();

  return failure();
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
      if (!hasDontTouch(wire.getOperation()) && wire->hasOneUse()) {
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

  if (destType.isa<IntType>() && srcType.isa<IntType>() && srcWidth >= 0 &&
      destWidth >= 0 && destWidth < srcWidth) {
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
  results.insert<patterns::EmptyNode, patterns::DropNameNode>(context);
}

void WireOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<patterns::DropNameWire>(context);
}

// A register with constant reset and all connection to either itself or the
// same constant, must be replaced by the constant.
struct foldResetMux : public mlir::RewritePattern {
  foldResetMux(MLIRContext *context)
      : RewritePattern(RegResetOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reg = cast<RegResetOp>(op);
    auto reset = dyn_cast_or_null<ConstantOp>(reg.resetValue().getDefiningOp());
    if (!reset || hasDontTouch(reg.getOperation()))
      return failure();
    // Find the one true connect, or bail
    ConnectOp con = getSingleConnectUserOf(reg.result());
    if (!con)
      return failure();

    auto mux = dyn_cast_or_null<MuxPrimOp>(con.src().getDefiningOp());
    if (!mux)
      return failure();
    auto high = mux.high().getDefiningOp();
    auto low = mux.low().getDefiningOp();
    auto constOp = dyn_cast_or_null<ConstantOp>(high);

    if (constOp && low != reg)
      return failure();
    if (dyn_cast_or_null<ConstantOp>(low) && high == reg)
      constOp = dyn_cast<ConstantOp>(low);

    if (!constOp || constOp.getType() != reset.getType() ||
        constOp.value() != reset.value())
      return failure();

    // Check all types should be typed by now
    auto regTy = reg.getType();
    if (con.dest().getType() != regTy || con.src().getType() != regTy ||
        mux.high().getType() != regTy || mux.low().getType() != regTy ||
        regTy.getBitWidthOrSentinel() < 0)
      return failure();

    // Ok, we know we are doing the transformation.

    // Make sure the constant dominates all users.
    if (constOp != &con->getBlock()->front())
      constOp->moveBefore(&con->getBlock()->front());

    // Replace the register with the constant.
    rewriter.replaceOp(reg, constOp.getResult());
    // Remove the connect.
    rewriter.eraseOp(con);
    return success();
  }
};

void RegResetOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::RegResetWithZeroReset,
                 patterns::RegResetWithInvalidReset,
                 patterns::RegResetWithInvalidResetValue,
                 patterns::DropNameRegReset, foldResetMux>(context);
}

LogicalResult MemOp::canonicalize(MemOp op, PatternRewriter &rewriter) {
  patterns::DropNameMem dnm(op.getContext());
  if (succeeded(dnm.matchAndRewrite(op, rewriter)))
    return success();

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
  ConnectOp con = getSingleConnectUserOf(reg.result());
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
      regTy.getBitWidthOrSentinel() < 0)
    return failure();

  // Ok, we know we are doing the transformation.

  // Make sure the constant dominates all users.
  if (constOp != &con->getBlock()->front())
    constOp->moveBefore(&con->getBlock()->front());

  if (!constReg)
    rewriter.replaceOpWithNewOp<RegResetOp>(
        reg, reg.getType(), reg.clockVal(), mux.sel(), mux.high(), reg.name(),
        reg.annotations(), reg.inner_symAttr());
  auto pt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(con);
  rewriter.replaceOpWithNewOp<ConnectOp>(con, con.dest(),
                                         constReg ? constOp : mux.low());
  rewriter.restoreInsertionPoint(pt);
  return success();
}

LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  if (!hasDontTouch(op.getOperation()) &&
      succeeded(foldHiddenReset(op, rewriter)))
    return success();

  patterns::DropNameReg dnr(op.getContext());
  if (succeeded(dnr.matchAndRewrite(op, rewriter)))
    return success();

  return failure();
}
