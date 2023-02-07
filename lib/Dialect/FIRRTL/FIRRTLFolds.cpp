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
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/APInt.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

// Drop writes to old and pass through passthrough to make patterns easier to
// write.
static Value dropWrite(PatternRewriter &rewriter, OpResult old,
                       Value passthrough) {
  for (auto *user : llvm::make_early_inc_range(old.getUsers())) {
    if (isa<StrictConnectOp, ConnectOp>(user)) {
      if (user->getOperand(0) == old)
        rewriter.eraseOp(user);
    }
  }
  return passthrough;
}

// Move a name hint from a soon to be deleted operation to a new operation.
// Pass through the new operation to make patterns easier to write.  This cannot
// move a name to a port (block argument), doing so would require rewriting all
// instance sites as well as the module.
static Value moveNameHint(OpResult old, Value passthrough) {
  Operation *op = passthrough.getDefiningOp();
  // This should handle ports, but it isn't clear we can change those in
  // canonicalizers.
  assert(op && "passthrough must be an operation");
  Operation *oldOp = old.getOwner();
  auto name = oldOp->getAttrOfType<StringAttr>("name");
  if (name && !name.getValue().empty())
    op->setAttr("name", name);
  return passthrough;
}

// Declarative canonicalization patterns
namespace circt {
namespace firrtl {
namespace patterns {
#include "circt/Dialect/FIRRTL/FIRRTLCanonicalization.h.inc"
} // namespace patterns
} // namespace firrtl
} // namespace circt

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

/// A wrapper of `PatternRewriter::replaceOp` to propagate "name" attribute.
/// If a replaced op has a "name" attribute, this function propagates the name
/// to the new value.
static void replaceOpAndCopyName(PatternRewriter &rewriter, Operation *op,
                                 Value newValue) {
  if (auto *newOp = newValue.getDefiningOp()) {
    auto name = op->getAttrOfType<StringAttr>("name");
    if (name && !name.getValue().empty()) {
      auto newOpName = newOp->getAttrOfType<StringAttr>("name");
      if (!newOpName || isUselessName(newOpName))
        rewriter.updateRootInPlace(newOp,
                                   [&] { newOp->setAttr("name", name); });
    }
  }
  rewriter.replaceOp(op, newValue);
}

/// A wrapper of `PatternRewriter::replaceOpWithNewOp` to propagate "name"
/// attribute. If a replaced op has a "name" attribute, this function propagates
/// the name to the new value.
template <typename OpTy, typename... Args>
static OpTy replaceOpWithNewOpAndCopyName(PatternRewriter &rewriter,
                                          Operation *op, Args &&...args) {
  auto name = op->getAttrOfType<StringAttr>("name");
  auto newOp =
      rewriter.replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
  if (name && !name.getValue().empty()) {
    auto newOpName = newOp->template getAttrOfType<StringAttr>("name");
    if (!newOpName || isUselessName(newOpName))
      rewriter.updateRootInPlace(newOp, [&] { newOp->setAttr("name", name); });
  }
  return newOp;
}

/// Return true if this is a useless temporary name produced by FIRRTL.  We
/// drop these as they don't convey semantic meaning.
bool circt::firrtl::isUselessName(StringRef name) {
  if (name.empty())
    return true;
  // Ignore _.*
  return name.startswith("_");
}

/// Return true if the name is droppable. Note that this is different from
/// `isUselessName` because non-useless names may be also droppable.
bool circt::firrtl::hasDroppableName(Operation *op) {
  if (auto namableOp = dyn_cast<firrtl::FNamableOp>(op))
    return namableOp.hasDroppableName();
  return false;
}

/// Implicitly replace the operand to a constant folding operation with a const
/// 0 in case the operand is non-constant but has a bit width 0, or if the
/// operand is an invalid value.
///
/// This makes constant folding significantly easier, as we can simply pass the
/// operands to an operation through this function to appropriately replace any
/// zero-width dynamic values or invalid values with a constant of value 0.
static std::optional<APSInt>
getExtendedConstant(Value operand, Attribute constant, int32_t destWidth) {
  assert(operand.getType().isa<IntType>() &&
         "getExtendedConstant is limited to integer types");

  // We never support constant folding to unknown width values.
  if (destWidth < 0)
    return {};

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
static std::optional<APSInt> getConstant(Attribute operand) {
  if (!operand)
    return {};
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
static Attribute constFoldFIRRTLBinaryOp(
    Operation *op, ArrayRef<Attribute> operands, BinOpKind opKind,
    const function_ref<APInt(const APSInt &, const APSInt &)> &calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");

  // We cannot fold something to an unknown width.
  auto resultType = op->getResult(0).getType().cast<IntType>();
  if (resultType.getWidthOrSentinel() < 0)
    return {};

  // Any binary op returning i0 is 0.
  if (resultType.getWidthOrSentinel() == 0)
    return getIntAttr(resultType, APInt(0, 0, resultType.isSigned()));

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
  if (!lhs)
    return {};
  auto rhs = getExtendedConstant(op->getOperand(1), operands[1], operandWidth);
  if (!rhs)
    return {};

  APInt resultValue = calculate(*lhs, *rhs);

  // If the result type is smaller than the computation then we need to
  // narrow the constant after the calculation.
  if (opKind == BinOpKind::DivideOrShift)
    resultValue = resultValue.trunc(resultType.getWidthOrSentinel());

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
  auto type = op->getResult(0).getType().dyn_cast<FIRRTLBaseType>();
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
      TypeSwitch<Operation *>(defOp).Case<ConstantOp, SpecialConstantOp>(
          [&](auto op) { attr = op.getValueAttr(); });
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
  if (width !=
      resultValue.getType().cast<FIRRTLBaseType>().getBitWidthOrSentinel())
    resultValue = rewriter.create<PadPrimOp>(op->getLoc(), resultValue, width);

  // Insert a cast if this is a uint vs. sint or vice versa.
  if (type.isa<SIntType>() && resultValue.getType().isa<UIntType>())
    resultValue = rewriter.create<AsSIntPrimOp>(op->getLoc(), resultValue);
  else if (type.isa<UIntType>() && resultValue.getType().isa<SIntType>())
    resultValue = rewriter.create<AsUIntPrimOp>(op->getLoc(), resultValue);

  assert(type == resultValue.getType() && "canonicalization changed type");
  replaceOpAndCopyName(rewriter, op, resultValue);
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

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

OpFoldResult SpecialConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

OpFoldResult AggregateConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getFieldsAttr();
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

OpFoldResult AddPrimOp::fold(FoldAdaptor adaptor) {
  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [=](const APSInt &a, const APSInt &b) { return a + b; });
}

void AddPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results
      .insert<patterns::moveConstAdd, patterns::AddOfZero, patterns::AddOfSelf>(
          context);
}

OpFoldResult SubPrimOp::fold(FoldAdaptor adaptor) {
  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [=](const APSInt &a, const APSInt &b) { return a - b; });
}

void SubPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::SubOfZero, patterns::SubFromZeroSigned,
                 patterns::SubFromZeroUnsigned, patterns::SubOfSelf>(context);
}

OpFoldResult MulPrimOp::fold(FoldAdaptor adaptor) {
  // mul(x, 0) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // multiplication this way and will emit "x * 0".
  if (isConstantZero(adaptor.getRhs()) || isConstantZero(adaptor.getLhs()))
    return getIntZerosAttr(getType());

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [=](const APSInt &a, const APSInt &b) { return a * b; });
}

OpFoldResult DivPrimOp::fold(FoldAdaptor adaptor) {
  /// div(x, x) -> 1
  ///
  /// Division by zero is undefined in the FIRRTL specification.  This fold
  /// exploits that fact to optimize self division to one.  Note: this should
  /// supersede any division with invalid or zero.  Division of invalid by
  /// invalid should be one.
  if (getLhs() == getRhs()) {
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
  if (isConstantZero(adaptor.getLhs()) && !isConstantZero(adaptor.getRhs()))
    return getIntZerosAttr(getType());

  /// div(x, 1) -> x : (uint, uint) -> uint
  ///
  /// UInt division by one returns the numerator. SInt division can't
  /// be folded here because it increases the return type bitwidth by
  /// one and requires sign extension (a new op).
  if (auto rhsCst = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>())
    if (rhsCst.getValue().isOne() && getLhs().getType() == getType())
      return getLhs();

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::DivideOrShift,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        if (!!b)
          return a / b;
        return APInt(a.getBitWidth(), 0);
      });
}

OpFoldResult RemPrimOp::fold(FoldAdaptor adaptor) {
  // rem(x, x) -> 0
  //
  // Division by zero is undefined in the FIRRTL specification.  This fold
  // exploits that fact to optimize self division remainder to zero.  Note:
  // this should supersede any division with invalid or zero.  Remainder of
  // division of invalid by invalid should be zero.
  if (getLhs() == getRhs())
    return getIntZerosAttr(getType());

  // rem(0, x) -> 0
  //
  // This is legal because it aligns with the Scala FIRRTL Compiler
  // interpretation of lowering invalid to constant zero before constant
  // propagation.  Note: the Scala FIRRTL Compiler does NOT currently optimize
  // division this way and will emit "0 % x".
  if (isConstantZero(adaptor.getLhs()))
    return getIntZerosAttr(getType());

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::DivideOrShift,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        if (!!b)
          return a % b;
        return APInt(a.getBitWidth(), 0);
      });
}

OpFoldResult DShlPrimOp::fold(FoldAdaptor adaptor) {
  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::DivideOrShift,
      [=](const APSInt &a, const APSInt &b) -> APInt { return a.shl(b); });
}

OpFoldResult DShlwPrimOp::fold(FoldAdaptor adaptor) {
  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::DivideOrShift,
      [=](const APSInt &a, const APSInt &b) -> APInt { return a.shl(b); });
}

OpFoldResult DShrPrimOp::fold(FoldAdaptor adaptor) {
  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::DivideOrShift,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return getType().isUnsigned() || !a.getBitWidth() ? a.lshr(b)
                                                          : a.ashr(b);
      });
}

// TODO: Move to DRR.
OpFoldResult AndPrimOp::fold(FoldAdaptor adaptor) {
  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    /// and(x, 0) -> 0, 0 is largest or is implicit zero extended
    if (rhsCst->isZero())
      return getIntZerosAttr(getType());

    /// and(x, -1) -> x
    if (rhsCst->isAllOnes() && getLhs().getType() == getType() &&
        getRhs().getType() == getType())
      return getLhs();
  }

  if (auto lhsCst = getConstant(adaptor.getLhs())) {
    /// and(0, x) -> 0, 0 is largest or is implicit zero extended
    if (lhsCst->isZero())
      return getIntZerosAttr(getType());

    /// and(-1, x) -> x
    if (lhsCst->isAllOnes() && getLhs().getType() == getType() &&
        getRhs().getType() == getType())
      return getRhs();
  }

  /// and(x, x) -> x
  if (getLhs() == getRhs() && getRhs().getType() == getType())
    return getRhs();

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [](const APSInt &a, const APSInt &b) -> APInt { return a & b; });
}

void AndPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results
      .insert<patterns::extendAnd, patterns::moveConstAnd, patterns::AndOfZero,
              patterns::AndOfAllOne, patterns::AndOfSelf>(context);
}

OpFoldResult OrPrimOp::fold(FoldAdaptor adaptor) {
  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    /// or(x, 0) -> x
    if (rhsCst->isZero() && getLhs().getType() == getType())
      return getLhs();

    /// or(x, -1) -> -1
    if (rhsCst->isAllOnes() && getRhs().getType() == getType() &&
        getLhs().getType() == getType())
      return getRhs();
  }

  if (auto lhsCst = getConstant(adaptor.getLhs())) {
    /// or(0, x) -> x
    if (lhsCst->isZero() && getRhs().getType() == getType())
      return getRhs();

    /// or(-1, x) -> -1
    if (lhsCst->isAllOnes() && getLhs().getType() == getType() &&
        getRhs().getType() == getType())
      return getLhs();
  }

  /// or(x, x) -> x
  if (getLhs() == getRhs() && getRhs().getType() == getType())
    return getRhs();

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [](const APSInt &a, const APSInt &b) -> APInt { return a | b; });
}

void OrPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::extendOr, patterns::moveConstOr, patterns::OrOfZero,
                 patterns::OrOfAllOne, patterns::OrOfSelf>(context);
}

OpFoldResult XorPrimOp::fold(FoldAdaptor adaptor) {
  /// xor(x, 0) -> x
  if (auto rhsCst = getConstant(adaptor.getRhs()))
    if (rhsCst->isZero() && getLhs().getType() == getType())
      return getLhs();

  /// xor(x, 0) -> x
  if (auto lhsCst = getConstant(adaptor.getLhs()))
    if (lhsCst->isZero() && getRhs().getType() == getType())
      return getRhs();

  /// xor(x, x) -> 0
  if (getLhs() == getRhs())
    return getIntAttr(getType(),
                      APInt(std::max(getType().getWidthOrSentinel(), 0), 0));

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Normal,
      [](const APSInt &a, const APSInt &b) -> APInt { return a ^ b; });
}

void XorPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::extendXor, patterns::moveConstXor,
                 patterns::XorOfZero, patterns::XorOfSelf>(context);
}

void LEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::LEQWithConstLHS>(context);
}

OpFoldResult LEQPrimOp::fold(FoldAdaptor adaptor) {
  bool isUnsigned = getLhs().getType().isa<UIntType>();

  // leq(x, x) -> 1
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 1));

  // Comparison against constant outside type bounds.
  if (auto width = getLhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(adaptor.getRhs())) {
      auto commonWidth = std::max<int32_t>(*width, rhsCst->getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // leq(x, const) -> 0 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // leq(x, const) -> 0 where const < minValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .slt(getMinSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // leq(x, const) -> 1 where const >= maxValue of the unsigned type of x
      if (isUnsigned && rhsCst->zext(commonWidth)
                            .uge(getMaxUnsignedValue(*width).zext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // leq(x, const) -> 1 where const >= maxValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sge(getMaxSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a <= b);
      });
}

void LTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::LTWithConstLHS>(context);
}

OpFoldResult LTPrimOp::fold(FoldAdaptor adaptor) {
  bool isUnsigned = getLhs().getType().isa<UIntType>();

  // lt(x, x) -> 0
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 0));

  // lt(x, 0) -> 0 when x is unsigned
  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    if (rhsCst->isZero() && getLhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 0));
  }

  // Comparison against constant outside type bounds.
  if (auto width = getLhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(adaptor.getRhs())) {
      auto commonWidth = std::max<int32_t>(*width, rhsCst->getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // lt(x, const) -> 0 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // lt(x, const) -> 0 where const <= minValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sle(getMinSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // lt(x, const) -> 1 where const > maxValue of the unsigned type of x
      if (isUnsigned && rhsCst->zext(commonWidth)
                            .ugt(getMaxUnsignedValue(*width).zext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));

      // lt(x, const) -> 1 where const > maxValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sgt(getMaxSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a < b);
      });
}

void GEQPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::GEQWithConstLHS>(context);
}

OpFoldResult GEQPrimOp::fold(FoldAdaptor adaptor) {
  bool isUnsigned = getLhs().getType().isa<UIntType>();

  // geq(x, x) -> 1
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 1));

  // geq(x, 0) -> 1 when x is unsigned
  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    if (rhsCst->isZero() && getLhs().getType().isa<UIntType>())
      return getIntAttr(getType(), APInt(1, 1));
  }

  // Comparison against constant outside type bounds.
  if (auto width = getLhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(adaptor.getRhs())) {
      auto commonWidth = std::max<int32_t>(*width, rhsCst->getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // geq(x, const) -> 0 where const > maxValue of the unsigned type of x
      if (isUnsigned && rhsCst->zext(commonWidth)
                            .ugt(getMaxUnsignedValue(*width).zext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 0 where const > maxValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sgt(getMaxSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // geq(x, const) -> 1 where const <= minValue of the unsigned type of x
      // Handled explicitly above.

      // geq(x, const) -> 1 where const <= minValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sle(getMinSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a >= b);
      });
}

void GTPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<patterns::GTWithConstLHS>(context);
}

OpFoldResult GTPrimOp::fold(FoldAdaptor adaptor) {
  bool isUnsigned = getLhs().getType().isa<UIntType>();

  // gt(x, x) -> 0
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 0));

  // Comparison against constant outside type bounds.
  if (auto width = getLhs().getType().cast<IntType>().getWidth()) {
    if (auto rhsCst = getConstant(adaptor.getRhs())) {
      auto commonWidth = std::max<int32_t>(*width, rhsCst->getBitWidth());
      commonWidth = std::max(commonWidth, 1);

      // gt(x, const) -> 0 where const >= maxValue of the unsigned type of x
      if (isUnsigned && rhsCst->zext(commonWidth)
                            .uge(getMaxUnsignedValue(*width).zext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 0 where const >= maxValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .sge(getMaxSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 0));

      // gt(x, const) -> 1 where const < minValue of the unsigned type of x
      // This can never occur since const is unsigned and cannot be less than 0.

      // gt(x, const) -> 1 where const < minValue of the signed type of x
      if (!isUnsigned && sextZeroWidth(*rhsCst, commonWidth)
                             .slt(getMinSignedValue(*width).sext(commonWidth)))
        return getIntAttr(getType(), APInt(1, 1));
    }
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a > b);
      });
}

OpFoldResult EQPrimOp::fold(FoldAdaptor adaptor) {
  // eq(x, x) -> 1
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 1));

  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    /// eq(x, 1) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst->isAllOnes() && getLhs().getType() == getType() &&
        getRhs().getType() == getType())
      return getLhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a == b);
      });
}

LogicalResult EQPrimOp::canonicalize(EQPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        if (auto rhsCst = getConstant(operands[1])) {
          auto width =
              op.getLhs().getType().cast<IntType>().getBitWidthOrSentinel();

          // eq(x, 0) ->  not(x) when x is 1 bit.
          if (rhsCst->isZero() && op.getLhs().getType() == op.getType() &&
              op.getRhs().getType() == op.getType()) {
            return rewriter.create<NotPrimOp>(op.getLoc(), op.getLhs())
                .getResult();
          }

          // eq(x, 0) -> not(orr(x)) when x is >1 bit
          if (rhsCst->isZero() && width > 1) {
            auto orrOp = rewriter.create<OrRPrimOp>(op.getLoc(), op.getLhs());
            return rewriter.create<NotPrimOp>(op.getLoc(), orrOp).getResult();
          }

          // eq(x, ~0) -> andr(x) when x is >1 bit
          if (rhsCst->isAllOnes() && width > 1 &&
              op.getLhs().getType() == op.getRhs().getType()) {
            return rewriter.create<AndRPrimOp>(op.getLoc(), op.getLhs())
                .getResult();
          }
        }

        return {};
      });
}

OpFoldResult NEQPrimOp::fold(FoldAdaptor adaptor) {
  // neq(x, x) -> 0
  if (getLhs() == getRhs())
    return getIntAttr(getType(), APInt(1, 0));

  if (auto rhsCst = getConstant(adaptor.getRhs())) {
    /// neq(x, 0) -> x when x is 1 bit.
    /// TODO: Support SInt<1> on the LHS etc.
    if (rhsCst->isZero() && getLhs().getType() == getType() &&
        getRhs().getType() == getType())
      return getLhs();
  }

  return constFoldFIRRTLBinaryOp(
      *this, adaptor.getOperands(), BinOpKind::Compare,
      [=](const APSInt &a, const APSInt &b) -> APInt {
        return APInt(1, a != b);
      });
}

LogicalResult NEQPrimOp::canonicalize(NEQPrimOp op, PatternRewriter &rewriter) {
  return canonicalizePrimOp(
      op, rewriter, [&](ArrayRef<Attribute> operands) -> OpFoldResult {
        if (auto rhsCst = getConstant(operands[1])) {
          auto width =
              op.getLhs().getType().cast<IntType>().getBitWidthOrSentinel();

          // neq(x, 1) -> not(x) when x is 1 bit
          if (rhsCst->isAllOnes() && op.getLhs().getType() == op.getType() &&
              op.getRhs().getType() == op.getType()) {
            return rewriter.create<NotPrimOp>(op.getLoc(), op.getLhs())
                .getResult();
          }

          // neq(x, 0) -> orr(x) when x is >1 bit
          if (rhsCst->isZero() && width > 1) {
            return rewriter.create<OrRPrimOp>(op.getLoc(), op.getLhs())
                .getResult();
          }

          // neq(x, ~0) -> not(andr(x))) when x is >1 bit
          if (rhsCst->isAllOnes() && width > 1 &&
              op.getLhs().getType() == op.getRhs().getType()) {
            auto andrOp = rewriter.create<AndRPrimOp>(op.getLoc(), op.getLhs());
            return rewriter.create<NotPrimOp>(op.getLoc(), andrOp).getResult();
          }
        }

        return {};
      });
}

//===----------------------------------------------------------------------===//
// Unary Operators
//===----------------------------------------------------------------------===//

OpFoldResult SizeOfIntrinsicOp::fold(FoldAdaptor) {
  auto base = getInput().getType().cast<FIRRTLBaseType>();
  auto w = base.getBitWidthOrSentinel();
  if (w >= 0)
    return getIntAttr(getType(), APInt(32, w));
  return {};
}

OpFoldResult IsXIntrinsicOp::fold(FoldAdaptor adaptor) {
  // No constant can be 'x' by definition.
  if (auto cst = getConstant(adaptor.getArg()))
    return getIntAttr(getType(), APInt(1, 0));
  return {};
}

OpFoldResult AsSIntPrimOp::fold(FoldAdaptor adaptor) {
  // No effect.
  if (getInput().getType() == getType())
    return getInput();

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (getType().hasWidth())
    if (auto cst = getConstant(adaptor.getInput()))
      return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult AsUIntPrimOp::fold(FoldAdaptor adaptor) {
  // No effect.
  if (getInput().getType() == getType())
    return getInput();

  // Be careful to only fold the cast into the constant if the size is known.
  // Otherwise width inference may produce differently-sized constants if the
  // sign changes.
  if (getType().hasWidth())
    if (auto cst = getConstant(adaptor.getInput()))
      return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult AsAsyncResetPrimOp::fold(FoldAdaptor adaptor) {
  // No effect.
  if (getInput().getType() == getType())
    return getInput();

  // Constant fold.
  if (auto cst = getConstant(adaptor.getInput()))
    return BoolAttr::get(getContext(), cst->getBoolValue());

  return {};
}

OpFoldResult AsClockPrimOp::fold(FoldAdaptor adaptor) {
  // No effect.
  if (getInput().getType() == getType())
    return getInput();

  // Constant fold.
  if (auto cst = getConstant(adaptor.getInput()))
    return BoolAttr::get(getContext(), cst->getBoolValue());

  return {};
}

OpFoldResult CvtPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // Signed to signed is a noop, unsigned operands prepend a zero bit.
  if (auto cst = getExtendedConstant(getOperand(), adaptor.getInput(),
                                     getType().getWidthOrSentinel()))
    return getIntAttr(getType(), *cst);

  return {};
}

OpFoldResult NegPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  // FIRRTL negate always adds a bit.
  // -x ---> 0-sext(x) or 0-zext(x)
  if (auto cst = getExtendedConstant(getOperand(), adaptor.getInput(),
                                     getType().getWidthOrSentinel()))
    return getIntAttr(getType(), APInt((*cst).getBitWidth(), 0) - *cst);

  return {};
}

OpFoldResult NotPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  if (auto cst = getExtendedConstant(getOperand(), adaptor.getInput(),
                                     getType().getWidthOrSentinel()))
    return getIntAttr(getType(), ~*cst);

  return {};
}

OpFoldResult AndRPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  if (getInput().getType().getBitWidthOrSentinel() == 0)
    return getIntAttr(getType(), APInt(1, 1));

  // x == -1
  if (auto cst = getConstant(adaptor.getInput()))
    return getIntAttr(getType(), APInt(1, cst->isAllOnes()));

  // one bit is identity.  Only applies to UInt since we can't make a cast
  // here.
  if (isUInt1(getInput().getType()))
    return getInput();

  return {};
}

OpFoldResult OrRPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  if (getInput().getType().getBitWidthOrSentinel() == 0)
    return getIntAttr(getType(), APInt(1, 0));

  // x != 0
  if (auto cst = getConstant(adaptor.getInput()))
    return getIntAttr(getType(), APInt(1, !cst->isZero()));

  // one bit is identity.  Only applies to UInt since we can't make a cast
  // here.
  if (isUInt1(getInput().getType()))
    return getInput();

  return {};
}

OpFoldResult XorRPrimOp::fold(FoldAdaptor adaptor) {
  if (!hasKnownWidthIntTypes(*this))
    return {};

  if (getInput().getType().getBitWidthOrSentinel() == 0)
    return getIntAttr(getType(), APInt(1, 0));

  // popcount(x) & 1
  if (auto cst = getConstant(adaptor.getInput()))
    return getIntAttr(getType(), APInt(1, cst->countPopulation() & 1));

  // one bit is identity.  Only applies to UInt since we can't make a cast here.
  if (isUInt1(getInput().getType()))
    return getInput();

  return {};
}

//===----------------------------------------------------------------------===//
// Other Operators
//===----------------------------------------------------------------------===//

OpFoldResult CatPrimOp::fold(FoldAdaptor adaptor) {
  // cat(x, 0-width) -> x
  // cat(0-width, x) -> x
  // Limit to unsigned (result type), as cannot insert cast here.
  if (getLhs().getType().getBitWidthOrSentinel() == 0 &&
      getRhs().getType().isUnsigned())
    return getRhs();
  if (getRhs().getType().getBitWidthOrSentinel() == 0 &&
      getLhs().getType().isUnsigned())
    return getLhs();

  if (!hasKnownWidthIntTypes(*this))
    return {};

  // Constant fold cat.
  if (auto lhs = getConstant(adaptor.getLhs()))
    if (auto rhs = getConstant(adaptor.getRhs()))
      return getIntAttr(getType(), lhs->concat(*rhs));

  return {};
}

void DShlPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::DShlOfConstant>(context);
}

void DShrPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::DShrOfConstant>(context);
}

namespace {
// cat(bits(x, ...), bits(x, ...)) -> bits(x ...) when the two ...'s are
// consequtive in the input.
struct CatBitsBits : public mlir::RewritePattern {
  CatBitsBits(MLIRContext *context)
      : RewritePattern(CatPrimOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto cat = cast<CatPrimOp>(op);
    if (auto lhsBits =
            dyn_cast_or_null<BitsPrimOp>(cat.getLhs().getDefiningOp())) {
      if (auto rhsBits =
              dyn_cast_or_null<BitsPrimOp>(cat.getRhs().getDefiningOp())) {
        if (lhsBits.getInput() == rhsBits.getInput() &&
            lhsBits.getLo() - 1 == rhsBits.getHi()) {
          replaceOpWithNewOpAndCopyName<BitsPrimOp>(
              rewriter, cat, cat.getType(), lhsBits.getInput(), lhsBits.getHi(),
              rhsBits.getLo());
          return success();
        }
      }
    }
    return failure();
  }
};
} // namespace

void CatPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CatBitsBits>(context);
}

OpFoldResult BitCastOp::fold(FoldAdaptor adaptor) {
  auto op = (*this);
  // BitCast is redundant if input and result types are same.
  if (op.getType() == op.getInput().getType())
    return op.getInput();

  // Two consecutive BitCasts are redundant if first bitcast type is same as the
  // final result type.
  if (BitCastOp in = dyn_cast_or_null<BitCastOp>(op.getInput().getDefiningOp()))
    if (op.getType() == in.getInput().getType())
      return in.getInput();

  return {};
}

OpFoldResult BitsPrimOp::fold(FoldAdaptor adaptor) {
  auto inputType = getInput().getType().cast<FIRRTLBaseType>();
  // If we are extracting the entire input, then return it.
  if (inputType == getType() && getType().hasWidth())
    return getInput();

  // Constant fold.
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(adaptor.getInput()))
      return getIntAttr(getType(),
                        cst->extractBits(getHi() - getLo() + 1, getLo()));

  return {};
}

void BitsPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::BitsOfBits>(context);
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

OpFoldResult MuxPrimOp::fold(FoldAdaptor adaptor) {

  // mux : UInt<0> -> 0
  if (getType().getBitWidthOrSentinel() == 0)
    return getIntAttr(getType(), APInt(0, 0, getType().isSignedInteger()));

  // mux(cond, x, x) -> x
  if (getHigh() == getLow())
    return getHigh();

  // The following folds require that the result has a known width. Otherwise
  // the mux requires an additional padding operation to be inserted, which is
  // not possible in a fold.
  if (getType().getBitWidthOrSentinel() < 0)
    return {};

  // mux(0/1, x, y) -> x or y
  if (auto cond = getConstant(adaptor.getSel())) {
    if (cond->isZero() && getLow().getType() == getType())
      return getLow();
    if (!cond->isZero() && getHigh().getType() == getType())
      return getHigh();
  }

  // mux(cond, x, cst)
  if (auto lowCst = getConstant(adaptor.getLow())) {
    // mux(cond, c1, c2)
    if (auto highCst = getConstant(adaptor.getHigh())) {
      // mux(cond, cst, cst) -> cst
      if (highCst->getBitWidth() == lowCst->getBitWidth() &&
          *highCst == *lowCst)
        return getIntAttr(getType(), *highCst);
      // mux(cond, 1, 0) -> cond
      if (highCst->isOne() && lowCst->isZero() &&
          getType() == getSel().getType())
        return getSel();

      // TODO: x ? ~0 : 0 -> sext(x)
      // TODO: "x ? c1 : c2" -> many tricks
    }
    // TODO: "x ? a : 0" -> sext(x) & a
  }

  // TODO: "x ? c1 : y" -> "~x ? y : c1"
  return {};
}

namespace {

// If the mux has a known output width, pad the operands up to this width.
// Most folds on mux require that folded operands are of the same width as
// the mux itself.
class MuxPad : public mlir::RewritePattern {
public:
  MuxPad(MLIRContext *context)
      : RewritePattern(MuxPrimOp::getOperationName(), 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mux = cast<MuxPrimOp>(op);
    auto width = mux.getType().getBitWidthOrSentinel();
    if (width < 0)
      return failure();

    auto pad = [&](Value input) -> Value {
      auto inputWidth = input.getType()
                            .template cast<FIRRTLBaseType>()
                            .getBitWidthOrSentinel();
      if (inputWidth < 0 || width == inputWidth)
        return input;
      return rewriter
          .create<PadPrimOp>(mux.getLoc(), mux.getType(), input, width)
          .getResult();
    };

    auto newHigh = pad(mux.getHigh());
    auto newLow = pad(mux.getLow());
    if (newHigh == mux.getHigh() && newLow == mux.getLow())
      return failure();

    replaceOpWithNewOpAndCopyName<MuxPrimOp>(
        rewriter, op, mux.getType(), ValueRange{mux.getSel(), newHigh, newLow},
        mux->getAttrs());
    return success();
  }
};
} // namespace

void MuxPrimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<MuxPad, patterns::MuxSameCondLow, patterns::MuxSameCondHigh,
              patterns::MuxSameTrue, patterns::MuxSameFalse,
              patterns::NarrowMuxLHS, patterns::NarrowMuxRHS>(context);
}

OpFoldResult PadPrimOp::fold(FoldAdaptor adaptor) {
  auto input = this->getInput();

  // pad(x) -> x  if the width doesn't change.
  if (input.getType() == getType())
    return input;

  // Need to know the input width.
  auto inputType = input.getType().cast<IntType>();
  int32_t width = inputType.getWidthOrSentinel();
  if (width == -1)
    return {};

  // Constant fold.
  if (auto cst = getConstant(adaptor.getInput())) {
    auto destWidth = getType().getWidthOrSentinel();
    if (destWidth == -1)
      return {};

    if (inputType.isSigned() && cst->getBitWidth())
      return getIntAttr(getType(), cst->sext(destWidth));
    return getIntAttr(getType(), cst->zext(destWidth));
  }

  return {};
}

OpFoldResult ShlPrimOp::fold(FoldAdaptor adaptor) {
  auto input = this->getInput();
  auto inputType = input.getType().cast<IntType>();
  int shiftAmount = getAmount();

  // shl(x, 0) -> x
  if (shiftAmount == 0)
    return input;

  // Constant fold.
  if (auto cst = getConstant(adaptor.getInput())) {
    auto inputWidth = inputType.getWidthOrSentinel();
    if (inputWidth != -1) {
      auto resultWidth = inputWidth + shiftAmount;
      shiftAmount = std::min(shiftAmount, resultWidth);
      return getIntAttr(getType(), cst->zext(resultWidth).shl(shiftAmount));
    }
  }
  return {};
}

OpFoldResult ShrPrimOp::fold(FoldAdaptor adaptor) {
  auto input = this->getInput();
  auto inputType = input.getType().cast<IntType>();
  int shiftAmount = getAmount();

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
  if (auto cst = getConstant(adaptor.getInput())) {
    APInt value;
    if (inputType.isSigned())
      value = cst->ashr(std::min(shiftAmount, inputWidth - 1));
    else
      value = cst->lshr(std::min(shiftAmount, inputWidth));
    auto resultWidth = std::max(inputWidth - shiftAmount, 1);
    return getIntAttr(getType(), value.trunc(resultWidth));
  }
  return {};
}

LogicalResult ShrPrimOp::canonicalize(ShrPrimOp op, PatternRewriter &rewriter) {
  auto inputWidth =
      op.getInput().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth <= 0)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned shiftAmount = op.getAmount();
  if (int(shiftAmount) >= inputWidth) {
    // shift(x, 32) => 0 when x has 32 bits.  This is handled by fold().
    if (op.getType().isUnsigned())
      return failure();

    // Shifting a signed value by the full width is actually taking the
    // sign bit. If the shift amount is greater than the input width, it
    // is equivalent to shifting by the input width.
    shiftAmount = inputWidth - 1;
  }

  replaceWithBits(op, op.getInput(), inputWidth - 1, shiftAmount, rewriter);
  return success();
}

LogicalResult HeadPrimOp::canonicalize(HeadPrimOp op,
                                       PatternRewriter &rewriter) {
  auto inputWidth =
      op.getInput().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth <= 0)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned keepAmount = op.getAmount();
  if (keepAmount)
    replaceWithBits(op, op.getInput(), inputWidth - 1, inputWidth - keepAmount,
                    rewriter);
  return success();
}

OpFoldResult HeadPrimOp::fold(FoldAdaptor adaptor) {
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(adaptor.getInput())) {
      int shiftAmount =
          getInput().getType().cast<IntType>().getWidthOrSentinel() -
          getAmount();
      return getIntAttr(getType(), cst->lshr(shiftAmount).trunc(getAmount()));
    }

  return {};
}

OpFoldResult TailPrimOp::fold(FoldAdaptor adaptor) {
  if (hasKnownWidthIntTypes(*this))
    if (auto cst = getConstant(adaptor.getInput()))
      return getIntAttr(getType(), cst->trunc(getType().getWidthOrSentinel()));
  return {};
}

LogicalResult TailPrimOp::canonicalize(TailPrimOp op,
                                       PatternRewriter &rewriter) {
  auto inputWidth =
      op.getInput().getType().cast<IntType>().getWidthOrSentinel();
  if (inputWidth <= 0)
    return failure();

  // If we know the input width, we can canonicalize this into a BitsPrimOp.
  unsigned dropAmount = op.getAmount();
  if (dropAmount != unsigned(inputWidth))
    replaceWithBits(op, op.getInput(), inputWidth - dropAmount - 1, 0,
                    rewriter);
  return success();
}

void SubaccessOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<patterns::SubaccessOfConstant>(context);
}

OpFoldResult MultibitMuxOp::fold(FoldAdaptor adaptor) {
  // If there is only one input, just return it.
  if (adaptor.getInputs().size() == 1)
    return getOperand(1);

  if (auto constIndex = getConstant(adaptor.getIndex())) {
    auto index = constIndex->getZExtValue();
    if (index >= 0 && index < getInputs().size())
      return getInputs()[getInputs().size() - 1 - index];
  }

  return {};
}

LogicalResult MultibitMuxOp::canonicalize(MultibitMuxOp op,
                                          PatternRewriter &rewriter) {
  // If all operands are equal, just canonicalize to it. We can add this
  // canonicalization as a folder but it costly to look through all inputs so it
  // is added here.
  if (llvm::all_of(op.getInputs().drop_front(), [&](auto input) {
        return input == op.getInputs().front();
      })) {
    replaceOpAndCopyName(rewriter, op, op.getInputs().front());
    return success();
  }

  // If the op is a vector indexing (e.g. `multbit_mux idx, a[n-1], a[n-2], ...,
  // a[0]`), we can fold the op into subaccess op `a[idx]`.
  if (auto lastSubindex = op.getInputs().back().getDefiningOp<SubindexOp>()) {
    if (llvm::all_of(llvm::enumerate(op.getInputs()), [&](auto e) {
          auto subindex = e.value().template getDefiningOp<SubindexOp>();
          return subindex && lastSubindex.getInput() == subindex.getInput() &&
                 subindex.getIndex() + e.index() + 1 == op.getInputs().size();
        })) {
      replaceOpWithNewOpAndCopyName<SubaccessOp>(
          rewriter, op, lastSubindex.getInput(), op.getIndex());
      return success();
    }
  }

  // If the size is 2, canonicalize into a normal mux to introduce more folds.
  if (op.getInputs().size() != 2)
    return failure();

  // TODO: Handle even when `index` doesn't have uint<1>.
  auto uintType =
      op.getIndex().getType().cast<FIRRTLBaseType>().dyn_cast<UIntType>();
  if (!uintType || uintType.getBitWidthOrSentinel() != 1)
    return failure();

  // multibit_mux(index, {lhs, rhs}) -> mux(index, lhs, rhs)
  replaceOpWithNewOpAndCopyName<MuxPrimOp>(
      rewriter, op, op.getIndex(), op.getInputs()[0], op.getInputs()[1]);
  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

/// Scan all the uses of the specified value, checking to see if there is
/// exactly one connect that has the value as its destination. This returns the
/// operation if found and if all the other users are "reads" from the value.
/// Returns null if there are no connects, or multiple connects to the value, or
/// if the value is involved in an `AttachOp`.
///
/// Note that this will simply return the connect, which is located *anywhere*
/// after the definition of the value. Users of this function are likely
/// interested in the source side of the returned connect, the definition of
/// which does likely not dominate the original value.
StrictConnectOp firrtl::getSingleConnectUserOf(Value value) {
  StrictConnectOp connect;
  for (Operation *user : value.getUsers()) {
    // If we see an attach, just conservatively fail.
    if (isa<AttachOp>(user))
      return {};

    if (auto aConnect = dyn_cast<StrictConnectOp>(user))
      if (aConnect.getDest() == value) {
        if (!connect)
          connect = aConnect;
        else
          return {};
      }
  }
  return connect;
}

// Forward simple values through wire's and reg's.
static LogicalResult canonicalizeSingleSetConnect(StrictConnectOp op,
                                                  PatternRewriter &rewriter) {
  // While we can do this for nearly all wires, we currently limit it to simple
  // things.
  Operation *connectedDecl = op.getDest().getDefiningOp();
  if (!connectedDecl)
    return failure();

  // Only support wire and reg for now.
  if (!isa<WireOp>(connectedDecl) && !isa<RegOp>(connectedDecl))
    return failure();
  if (hasDontTouch(connectedDecl) || !AnnotationSet(connectedDecl).empty() ||
      !hasDroppableName(connectedDecl))
    return failure();

  // Only forward if the types exactly match and there is one connect.
  if (getSingleConnectUserOf(op.getDest()) != op)
    return failure();

  // Only foward if there is more than one use
  if (connectedDecl->hasOneUse())
    return failure();

  // Only do this if the connectee and the declaration are in the same block.
  auto *declBlock = connectedDecl->getBlock();
  auto *srcValueOp = op.getSrc().getDefiningOp();
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

  auto replacement = op.getSrc();
  if (srcValueOp) {
    // Replace with constant zero.
    if (isa<InvalidValueOp>(srcValueOp)) {
      if (op.getDest().getType().isa<BundleType, FVectorType>())
        return failure();
      if (op.getDest().getType().isa<ClockType, AsyncResetType, ResetType>())
        replacement = rewriter.create<SpecialConstantOp>(
            op.getSrc().getLoc(), op.getDest().getType(),
            rewriter.getBoolAttr(false));
      else
        replacement = rewriter.create<ConstantOp>(
            op.getSrc().getLoc(), op.getDest().getType(),
            getIntZerosAttr(op.getDest().getType()));
    }
    // This will be replaced with the constant source.  First, make sure the
    // constant dominates all users.
    else if (srcValueOp != &declBlock->front()) {
      srcValueOp->moveBefore(&declBlock->front());
    }
  }

  // Replace all things *using* the decl with the constant/port, and
  // remove the declaration.
  replaceOpAndCopyName(rewriter, connectedDecl, replacement);

  // Remove the connect
  rewriter.eraseOp(op);
  return success();
}

void ConnectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<patterns::ConnectExtension, patterns::ConnectSameType>(
      context);
}

LogicalResult StrictConnectOp::canonicalize(StrictConnectOp op,
                                            PatternRewriter &rewriter) {
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

namespace {
// Remove private nodes.  If they have an interesting names, move the name to
// the source expression.
struct FoldNodeName : public mlir::RewritePattern {
  FoldNodeName(MLIRContext *context)
      : RewritePattern(NodeOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto node = cast<NodeOp>(op);
    auto name = node.getNameAttr();
    if (!node.hasDroppableName() || node.getInnerSym() ||
        !node.getAnnotations().empty())
      return failure();
    auto *newOp = node.getInput().getDefiningOp();
    // Best effort
    if (name && !name.getValue().empty() && newOp) {
      auto newOpName = newOp->getAttrOfType<StringAttr>("name");
      if (!newOpName || isUselessName(newOpName))
        rewriter.updateRootInPlace(newOp,
                                   [&] { newOp->setAttr("name", name); });
    }
    rewriter.replaceOp(node, node.getInput());
    return success();
  }
};

// Bypass nodes.
struct NodeBypass : public mlir::RewritePattern {
  NodeBypass(MLIRContext *context)
      : RewritePattern(NodeOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto node = cast<NodeOp>(op);
    if (node.getInnerSym() || !node.getAnnotations().empty() ||
        node.use_empty())
      return failure();
    rewriter.startRootUpdate(node);
    node.replaceAllUsesWith(node.getInput());
    rewriter.finalizeRootUpdate(node);
    return success();
  }
};
} // namespace

// Interesting names and symbols and don't touch force nodes to stick around.
OpFoldResult NodeOp::fold(FoldAdaptor adaptor) {
  if (!hasDroppableName())
    return {};
  if (hasDontTouch(getResult())) // handles inner symbols
    return {};
  if (getAnnotationsAttr() && !getAnnotationsAttr().empty())
    return {};
  return adaptor.getInput();
}

void NodeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<FoldNodeName>(context);
}

namespace {
// For a lhs, find all the writers of fields of the aggregate type.  If there
// is one writer for each field, merge the writes
struct AggOneShot : public mlir::RewritePattern {
  AggOneShot(StringRef name, uint32_t weight, MLIRContext *context)
      : RewritePattern(name, 0, context) {}

  SmallVector<Value> getCompleteWrite(Operation *lhs) const {
    auto lhsTy = lhs->getResult(0).getType();
    if (!lhsTy.isa<BundleType>() && !lhsTy.isa<FVectorType>())
      return {};

    DenseMap<uint32_t, Value> fields;
    for (Operation *user : lhs->getResult(0).getUsers()) {
      if (user->getParentOp() != lhs->getParentOp())
        return {};
      if (auto aConnect = dyn_cast<StrictConnectOp>(user)) {
        if (aConnect.getDest() == lhs->getResult(0))
          return {};
      } else if (auto subField = dyn_cast<SubfieldOp>(user)) {
        for (Operation *subuser : subField.getResult().getUsers()) {
          if (auto aConnect = dyn_cast<StrictConnectOp>(subuser)) {
            if (aConnect.getDest() == subField) {
              if (fields.count(subField.getFieldIndex())) // duplicate write
                return {};
              fields[subField.getFieldIndex()] = aConnect.getSrc();
            }
            continue;
          }
          return {};
        }
      } else if (auto subIndex = dyn_cast<SubindexOp>(user)) {
        for (Operation *subuser : subIndex.getResult().getUsers()) {
          if (auto aConnect = dyn_cast<StrictConnectOp>(subuser)) {
            if (aConnect.getDest() == subIndex) {
              if (fields.count(subIndex.getIndex())) // duplicate write
                return {};
              fields[subIndex.getIndex()] = aConnect.getSrc();
            }
            continue;
          }
          return {};
        }
      } else {
        return {};
      }
    }

    SmallVector<Value> values;
    uint32_t total = lhsTy.isa<BundleType>()
                         ? lhsTy.cast<BundleType>().getNumElements()
                         : lhsTy.cast<FVectorType>().getNumElements();
    for (uint32_t i = 0; i < total; ++i) {
      if (!fields.count(i))
        return {};
      values.push_back(fields[i]);
    }
    return values;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto values = getCompleteWrite(op);
    if (values.empty())
      return failure();
    rewriter.setInsertionPointToEnd(op->getBlock());
    Value newVal = op->getResult(0).getType().isa<BundleType>()
                       ? rewriter.createOrFold<BundleCreateOp>(
                             op->getLoc(), op->getResult(0).getType(), values)
                       : rewriter.createOrFold<VectorCreateOp>(
                             op->getLoc(), op->getResult(0).getType(), values);
    rewriter.createOrFold<StrictConnectOp>(op->getLoc(), op->getResult(0),
                                           newVal);
    for (Operation *user : op->getResult(0).getUsers()) {
      if (auto subIndex = dyn_cast<SubindexOp>(user)) {
        for (Operation *subuser :
             llvm::make_early_inc_range(subIndex.getResult().getUsers()))
          if (auto aConnect = dyn_cast<StrictConnectOp>(subuser))
            if (aConnect.getDest() == subIndex)
              rewriter.eraseOp(aConnect);
      } else if (auto subField = dyn_cast<SubfieldOp>(user)) {
        for (Operation *subuser :
             llvm::make_early_inc_range(subField.getResult().getUsers()))
          if (auto aConnect = dyn_cast<StrictConnectOp>(subuser))
            if (aConnect.getDest() == subField)
              rewriter.eraseOp(aConnect);
      }
    }
    return success();
  }
};

struct WireAggOneShot : public AggOneShot {
  WireAggOneShot(MLIRContext *context)
      : AggOneShot(WireOp::getOperationName(), 0, context) {}
};
struct SubindexAggOneShot : public AggOneShot {
  SubindexAggOneShot(MLIRContext *context)
      : AggOneShot(SubindexOp::getOperationName(), 0, context) {}
};
struct SubfieldAggOneShot : public AggOneShot {
  SubfieldAggOneShot(MLIRContext *context)
      : AggOneShot(SubfieldOp::getOperationName(), 0, context) {}
};
} // namespace

void WireOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<WireAggOneShot>(context);
}

void SubindexOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SubindexAggOneShot>(context);
}

OpFoldResult SubindexOp::fold(FoldAdaptor adaptor) {
  auto attr = adaptor.getInput().dyn_cast_or_null<ArrayAttr>();
  if (!attr)
    return {};
  return attr[getIndex()];
}

OpFoldResult SubfieldOp::fold(FoldAdaptor adaptor) {
  auto attr = adaptor.getInput().dyn_cast_or_null<ArrayAttr>();
  if (!attr)
    return {};
  auto index = getFieldIndex();
  return attr[index];
}

void SubfieldOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SubfieldAggOneShot>(context);
}

static Attribute collectFields(MLIRContext *context,
                               ArrayRef<Attribute> operands) {
  for (auto operand : operands)
    if (!operand)
      return {};
  return ArrayAttr::get(context, operands);
}

OpFoldResult BundleCreateOp::fold(FoldAdaptor adaptor) {
  return collectFields(getContext(), adaptor.getOperands());
}

OpFoldResult VectorCreateOp::fold(FoldAdaptor adaptor) {
  return collectFields(getContext(), adaptor.getOperands());
}

namespace {
// A register with constant reset and all connection to either itself or the
// same constant, must be replaced by the constant.
struct FoldResetMux : public mlir::RewritePattern {
  FoldResetMux(MLIRContext *context)
      : RewritePattern(RegResetOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reg = cast<RegResetOp>(op);
    auto reset =
        dyn_cast_or_null<ConstantOp>(reg.getResetValue().getDefiningOp());
    if (!reset || hasDontTouch(reg.getOperation()) ||
        !reg.getAnnotations().empty())
      return failure();
    // Find the one true connect, or bail
    auto con = getSingleConnectUserOf(reg.getResult());
    if (!con)
      return failure();

    auto mux = dyn_cast_or_null<MuxPrimOp>(con.getSrc().getDefiningOp());
    if (!mux)
      return failure();
    auto *high = mux.getHigh().getDefiningOp();
    auto *low = mux.getLow().getDefiningOp();
    auto constOp = dyn_cast_or_null<ConstantOp>(high);

    if (constOp && low != reg)
      return failure();
    if (dyn_cast_or_null<ConstantOp>(low) && high == reg)
      constOp = dyn_cast<ConstantOp>(low);

    if (!constOp || constOp.getType() != reset.getType() ||
        constOp.getValue() != reset.getValue())
      return failure();

    // Check all types should be typed by now
    auto regTy = reg.getType();
    if (con.getDest().getType() != regTy || con.getSrc().getType() != regTy ||
        mux.getHigh().getType() != regTy || mux.getLow().getType() != regTy ||
        regTy.getBitWidthOrSentinel() < 0)
      return failure();

    // Ok, we know we are doing the transformation.

    // Make sure the constant dominates all users.
    if (constOp != &con->getBlock()->front())
      constOp->moveBefore(&con->getBlock()->front());

    // Replace the register with the constant.
    replaceOpAndCopyName(rewriter, reg, constOp.getResult());
    // Remove the connect.
    rewriter.eraseOp(con);
    return success();
  }
};
} // namespace

void RegResetOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<patterns::RegResetWithZeroReset,
                 patterns::RegResetWithOneReset, FoldResetMux>(context);
}

// Returns true if the enable field of a port is set to false.
static bool isPortDisabled(Value port) {
  auto portTy = port.getType().cast<BundleType>();
  auto enableIndex = portTy.getElementIndex("en");
  assert(enableIndex && "missing enable flag on memory port");

  Value en = {};
  for (auto *op : port.getUsers()) {
    auto portAccess = cast<SubfieldOp>(op);
    if (enableIndex != portAccess.getFieldIndex())
      continue;
    auto conn = getSingleConnectUserOf(portAccess);
    if (!conn || en)
      return false;
    en = conn.getSrc();
  }
  if (!en)
    return false;

  auto portConst = en.getDefiningOp<ConstantOp>();
  if (!portConst)
    return false;
  return portConst.getValue().isZero();
}

// Returns true if the data output is unused.
static bool isPortUnused(Value port, StringRef data) {
  auto portTy = port.getType().cast<BundleType>();
  auto fieldIndex = portTy.getElementIndex(data);
  assert(fieldIndex && "missing enable flag on memory port");

  for (auto *op : port.getUsers()) {
    auto portAccess = cast<SubfieldOp>(op);
    if (fieldIndex != portAccess.getFieldIndex())
      continue;
    if (!portAccess.use_empty())
      return false;
  }

  return true;
}

// Remove accesses to a port which is used.
static void erasePort(PatternRewriter &rewriter, Value port) {
  // Helper to create a dummy 0 clock for the dummy registers.
  Value clock;
  auto getClock = [&] {
    if (!clock)
      clock = rewriter.create<SpecialConstantOp>(
          port.getLoc(), ClockType::get(rewriter.getContext()), false);
    return clock;
  };

  // Find the clock field of the port and determine whether the port is
  // accessed only through its subfields or as a whole wire.  If the port
  // is used in its entirety, replace it with a wire.  Otherwise,
  // eliminate individual subfields and replace with reasonable defaults.
  for (auto *op : port.getUsers()) {
    auto subfield = dyn_cast<SubfieldOp>(op);
    if (!subfield) {
      auto ty = port.getType();
      auto reg = rewriter.create<RegOp>(port.getLoc(), ty, getClock());
      port.replaceAllUsesWith(reg);
      return;
    }
  }

  // Remove all connects to field accesses as they are no longer relevant.
  // If field values are used anywhere, which should happen solely for read
  // ports, a dummy register is introduced which replicates the behaviour of
  // memory that is never written, but might be read.
  for (auto *accessOp : llvm::make_early_inc_range(port.getUsers())) {
    auto access = cast<SubfieldOp>(accessOp);
    for (auto *user : llvm::make_early_inc_range(access->getUsers())) {
      auto connect = dyn_cast<FConnectLike>(user);
      if (connect && connect.getDest() == access) {
        rewriter.eraseOp(user);
        continue;
      }
    }
    if (access.use_empty()) {
      rewriter.eraseOp(access);
      continue;
    }

    // Replace read values with a register that is never written, handing off
    // the canonicalization of such a register to another canonicalizer.
    auto ty = access.getType();
    Value reg = rewriter.create<RegOp>(access.getLoc(), ty, getClock());
    rewriter.replaceOp(access, reg);
  }
  assert(port.use_empty() && "port should have no remaining uses");
}

namespace {
// If memory has known, but zero width, eliminate it.
struct FoldZeroWidthMemory : public mlir::RewritePattern {
  FoldZeroWidthMemory(MLIRContext *context)
      : RewritePattern(MemOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    MemOp mem = cast<MemOp>(op);
    if (hasDontTouch(mem))
      return failure();

    if (mem.getDataType().getBitWidthOrSentinel() != 0)
      return failure();

    // Make sure are users are safe to replace
    for (auto port : mem.getResults())
      for (auto *user : port.getUsers())
        if (!isa<SubfieldOp>(user))
          return failure();

    // Annoyingly, there isn't a good replacement for the port as a whole,
    // since they have an outer flip type.
    for (auto port : op->getResults()) {
      for (auto *user : llvm::make_early_inc_range(port.getUsers())) {
        SubfieldOp sfop = cast<SubfieldOp>(user);
        replaceOpWithNewOpAndCopyName<WireOp>(rewriter, sfop,
                                              sfop.getResult().getType());
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// If memory has no write ports and no file initialization, eliminate it.
struct FoldReadOrWriteOnlyMemory : public mlir::RewritePattern {
  FoldReadOrWriteOnlyMemory(MLIRContext *context)
      : RewritePattern(MemOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    MemOp mem = cast<MemOp>(op);
    if (hasDontTouch(mem))
      return failure();
    bool isRead = false, isWritten = false;
    for (unsigned i = 0; i < mem.getNumResults(); ++i) {
      switch (mem.getPortKind(i)) {
      case MemOp::PortKind::Read:
        isRead = true;
        if (isWritten)
          return failure();
        continue;
      case MemOp::PortKind::Write:
        isWritten = true;
        if (isRead)
          return failure();
        continue;
      case MemOp::PortKind::Debug:
      case MemOp::PortKind::ReadWrite:
        return failure();
      }
      llvm_unreachable("unknown port kind");
    }
    assert((!isWritten || !isRead) && "memory is in use");

    // If the memory is read only, but has a file initialization, then we can't
    // remove it.  A write only memory with file initialization is okay to
    // remove.
    if (isRead && mem.getInit())
      return failure();

    for (auto port : mem.getResults())
      erasePort(rewriter, port);

    rewriter.eraseOp(op);
    return success();
  }
};

// Eliminate the dead ports of memories.
struct FoldUnusedPorts : public mlir::RewritePattern {
  FoldUnusedPorts(MLIRContext *context)
      : RewritePattern(MemOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    MemOp mem = cast<MemOp>(op);
    if (hasDontTouch(mem))
      return failure();
    // Identify the dead and changed ports.
    llvm::SmallBitVector deadPorts(mem.getNumResults());
    for (auto [i, port] : llvm::enumerate(mem.getResults())) {
      // Do not simplify annotated ports.
      if (!mem.getPortAnnotation(i).empty())
        continue;

      // Skip debug ports.
      auto kind = mem.getPortKind(i);
      if (kind == MemOp::PortKind::Debug)
        continue;

      // If a port is disabled, always eliminate it.
      if (isPortDisabled(port)) {
        deadPorts.set(i);
        continue;
      }
      // Eliminate read ports whose outputs are not used.
      if (kind == MemOp::PortKind::Read && isPortUnused(port, "data")) {
        deadPorts.set(i);
        continue;
      }
    }
    if (deadPorts.none())
      return failure();

    // Rebuild the new memory with the altered ports.
    SmallVector<Type> resultTypes;
    SmallVector<StringRef> portNames;
    SmallVector<Attribute> portAnnotations;
    for (auto [i, port] : llvm::enumerate(mem.getResults())) {
      if (deadPorts[i])
        continue;
      resultTypes.push_back(port.getType());
      portNames.push_back(mem.getPortName(i));
      portAnnotations.push_back(mem.getPortAnnotation(i));
    }

    MemOp newOp;
    if (!resultTypes.empty())
      newOp = rewriter.create<MemOp>(
          mem.getLoc(), resultTypes, mem.getReadLatency(),
          mem.getWriteLatency(), mem.getDepth(), mem.getRuw(),
          rewriter.getStrArrayAttr(portNames), mem.getName(), mem.getNameKind(),
          mem.getAnnotations(), rewriter.getArrayAttr(portAnnotations),
          mem.getInnerSymAttr(), mem.getGroupIDAttr(), mem.getInitAttr());

    // Replace the dead ports with dummy wires.
    unsigned nextPort = 0;
    for (auto [i, port] : llvm::enumerate(mem.getResults())) {
      if (deadPorts[i])
        erasePort(rewriter, port);
      else
        port.replaceAllUsesWith(newOp.getResult(nextPort++));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Rewrite write-only read-write ports to write ports.
struct FoldReadWritePorts : public mlir::RewritePattern {
  FoldReadWritePorts(MLIRContext *context)
      : RewritePattern(MemOp::getOperationName(), 0, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    MemOp mem = cast<MemOp>(op);
    if (hasDontTouch(mem))
      return failure();

    // Identify read-write ports whose read end is unused.
    llvm::SmallBitVector deadReads(mem.getNumResults());
    for (auto [i, port] : llvm::enumerate(mem.getResults())) {
      if (mem.getPortKind(i) != MemOp::PortKind::ReadWrite)
        continue;
      if (!mem.getPortAnnotation(i).empty())
        continue;
      if (isPortUnused(port, "rdata")) {
        deadReads.set(i);
        continue;
      }
    }
    if (deadReads.none())
      return failure();

    SmallVector<Type> resultTypes;
    SmallVector<StringRef> portNames;
    SmallVector<Attribute> portAnnotations;
    for (auto [i, port] : llvm::enumerate(mem.getResults())) {
      if (deadReads[i])
        resultTypes.push_back(
            MemOp::getTypeForPort(mem.getDepth(), mem.getDataType(),
                                  MemOp::PortKind::Write, mem.getMaskBits()));
      else
        resultTypes.push_back(port.getType());

      portNames.push_back(mem.getPortName(i));
      portAnnotations.push_back(mem.getPortAnnotation(i));
    }

    auto newOp = rewriter.create<MemOp>(
        mem.getLoc(), resultTypes, mem.getReadLatency(), mem.getWriteLatency(),
        mem.getDepth(), mem.getRuw(), rewriter.getStrArrayAttr(portNames),
        mem.getName(), mem.getNameKind(), mem.getAnnotations(),
        rewriter.getArrayAttr(portAnnotations), mem.getInnerSymAttr(),
        mem.getGroupIDAttr(), mem.getInitAttr());

    for (unsigned i = 0, n = mem.getNumResults(); i < n; ++i) {
      auto result = mem.getResult(i);
      auto newResult = newOp.getResult(i);
      if (deadReads[i]) {
        // Create a wire to replace the old result. Wire the sub-fields of the
        // old result to the relevant sub-fields of the write port.
        auto wire = rewriter.create<WireOp>(result.getLoc(), result.getType());
        result.replaceAllUsesWith(wire);

        auto connect = [&](Value to, StringRef toName, Value from,
                           StringRef fromName) {
          auto toField = rewriter.create<SubfieldOp>(to.getLoc(), to, toName);
          auto fromField =
              rewriter.create<SubfieldOp>(from.getLoc(), from, fromName);
          rewriter.create<StrictConnectOp>(result.getLoc(), toField, fromField);
        };

        connect(newResult, "addr", wire, "addr");
        connect(newResult, "en", wire, "en");
        connect(newResult, "clk", wire, "clk");
        connect(newResult, "data", wire, "wdata");
        connect(newResult, "mask", wire, "wmask");
      } else {
        result.replaceAllUsesWith(newResult);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void MemOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<FoldZeroWidthMemory, FoldReadOrWriteOnlyMemory,
                 FoldReadWritePorts, FoldUnusedPorts>(context);
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
  auto con = getSingleConnectUserOf(reg.getResult());
  if (!con)
    return failure();

  auto mux = dyn_cast_or_null<MuxPrimOp>(con.getSrc().getDefiningOp());
  if (!mux)
    return failure();
  auto *high = mux.getHigh().getDefiningOp();
  auto *low = mux.getLow().getDefiningOp();
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
  if (!mux.getSel().isa<BlockArgument>())
    return failure();

  // Check all types should be typed by now
  auto regTy = reg.getType();
  if (con.getDest().getType() != regTy || con.getSrc().getType() != regTy ||
      mux.getHigh().getType() != regTy || mux.getLow().getType() != regTy ||
      regTy.getBitWidthOrSentinel() < 0)
    return failure();

  // Ok, we know we are doing the transformation.

  // Make sure the constant dominates all users.
  if (constOp != &con->getBlock()->front())
    constOp->moveBefore(&con->getBlock()->front());

  if (!constReg) {
    SmallVector<NamedAttribute, 2> attrs(reg->getDialectAttrs());
    auto newReg = replaceOpWithNewOpAndCopyName<RegResetOp>(
        rewriter, reg, reg.getType(), reg.getClockVal(), mux.getSel(),
        mux.getHigh(), reg.getName(), reg.getNameKind(), reg.getAnnotations(),
        reg.getInnerSymAttr());
    newReg->setDialectAttrs(attrs);
  }
  auto pt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(con);
  auto v = constReg ? (Value)constOp.getResult() : (Value)mux.getLow();
  replaceOpWithNewOpAndCopyName<ConnectOp>(rewriter, con, con.getDest(), v);
  rewriter.restoreInsertionPoint(pt);
  return success();
}

LogicalResult RegOp::canonicalize(RegOp op, PatternRewriter &rewriter) {
  if (!hasDontTouch(op.getOperation()) &&
      succeeded(foldHiddenReset(op, rewriter)))
    return success();

  return failure();
}

//===----------------------------------------------------------------------===//
// Verification Ops.
//===----------------------------------------------------------------------===//

static LogicalResult eraseIfZeroOrNotZero(Operation *op, Value predicate,
                                          Value enable,
                                          PatternRewriter &rewriter,
                                          bool eraseIfZero) {
  // If the verification op is never enabled, delete it.
  if (auto constant = enable.getDefiningOp<firrtl::ConstantOp>()) {
    if (constant.getValue().isZero()) {
      rewriter.eraseOp(op);
      return success();
    }
  }

  // If the verification op is never triggered, delete it.
  if (auto constant = predicate.getDefiningOp<firrtl::ConstantOp>()) {
    if (constant.getValue().isZero() == eraseIfZero) {
      rewriter.eraseOp(op);
      return success();
    }
  }

  return failure();
}

template <class Op, bool EraseIfZero = false>
static LogicalResult canonicalizeImmediateVerifOp(Op op,
                                                  PatternRewriter &rewriter) {
  return eraseIfZeroOrNotZero(op, op.getPredicate(), op.getEnable(), rewriter,
                              EraseIfZero);
}

void AssertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssertOp>);
}

void AssumeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<AssumeOp>);
}

void CoverOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add(canonicalizeImmediateVerifOp<CoverOp, /* EraseIfZero = */ true>);
}
