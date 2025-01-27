//===- CombToAIG.cpp - Comb to AIG Conversion Pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to AIG Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOAIG
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// A wrapper for comb::extractBits that returns a SmallVector<Value>.
static SmallVector<Value> extractBits(OpBuilder &builder, Value val) {
  SmallVector<Value> bits;
  comb::extractBits(builder, val, bits);
  return bits;
}

// Construct a mux tree for shift operations. `isLeftShift` controls the
// direction of the shift operation and is used to determine order of the
// padding and extracted bits. Callbacks `getPadding` and `getExtract` are used
// to get the padding and extracted bits for each shift amount. `getPadding`
// could return a nullptr as i0 value but except for that, these callbacks must
// return a valid value for each shift amount in the range [0, maxShiftAmount].
// The value for `maxShiftAmount` is used as the out-of-bounds value.
template <bool isLeftShift>
static Value createShiftLogic(ConversionPatternRewriter &rewriter, Location loc,
                              Value shiftAmount, int64_t maxShiftAmount,
                              llvm::function_ref<Value(int64_t)> getPadding,
                              llvm::function_ref<Value(int64_t)> getExtract) {
  // Extract individual bits from shift amount
  auto bits = extractBits(rewriter, shiftAmount);

  // Create nodes for each possible shift amount
  SmallVector<Value> nodes;
  nodes.reserve(maxShiftAmount);
  for (int64_t i = 0; i < maxShiftAmount; ++i) {
    Value extract = getExtract(i);
    Value padding = getPadding(i);

    if (!padding) {
      nodes.push_back(extract);
      continue;
    }

    // Concatenate extracted bits with padding
    if (isLeftShift)
      nodes.push_back(
          rewriter.createOrFold<comb::ConcatOp>(loc, extract, padding));
    else
      nodes.push_back(
          rewriter.createOrFold<comb::ConcatOp>(loc, padding, extract));
  }

  // Create out-of-bounds value
  auto outOfBoundsValue = getPadding(maxShiftAmount);
  assert(outOfBoundsValue && "outOfBoundsValue must be valid");

  // Construct mux tree for shift operation
  auto result =
      comb::constructMuxTree(rewriter, loc, bits, nodes, outOfBoundsValue);

  // Add bounds checking
  auto inBound = rewriter.createOrFold<comb::ICmpOp>(
      loc, ICmpPredicate::ult, shiftAmount,
      rewriter.create<hw::ConstantOp>(loc, shiftAmount.getType(),
                                      maxShiftAmount));

  return rewriter.createOrFold<comb::MuxOp>(loc, inBound, result,
                                            outOfBoundsValue);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower a comb::AndOp operation to aig::AndInverterOp
struct CombAndOpConversion : OpConversionPattern<AndOp> {
  using OpConversionPattern<AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<bool> nonInverts(adaptor.getInputs().size(), false);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, adaptor.getInputs(),
                                                    nonInverts);
    return success();
  }
};

/// Lower a comb::OrOp operation to aig::AndInverterOp with invert flags
struct CombOrOpConversion : OpConversionPattern<OrOp> {
  using OpConversionPattern<OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement Or using And and invert flags: a | b = ~(~a & ~b)
    SmallVector<bool> allInverts(adaptor.getInputs().size(), true);
    auto andOp = rewriter.create<aig::AndInverterOp>(
        op.getLoc(), adaptor.getInputs(), allInverts);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, andOp,
                                                    /*invert=*/true);
    return success();
  }
};

/// Lower a comb::XorOp operation to AIG operations
struct CombXorOpConversion : OpConversionPattern<XorOp> {
  using OpConversionPattern<XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 2)
      return failure();
    // Xor using And with invert flags: a ^ b = (a | b) & (~a | ~b)

    // (a | b) = ~(~a & ~b)
    // (~a | ~b) = ~(a & b)
    auto inputs = adaptor.getInputs();
    SmallVector<bool> allInverts(inputs.size(), true);
    SmallVector<bool> allNotInverts(inputs.size(), false);

    auto notAAndNotB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allInverts);
    auto aAndB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allNotInverts);

    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, notAAndNotB, aAndB,
                                                    /*lhs_invert=*/true,
                                                    /*rhs_invert=*/true);
    return success();
  }
};

template <typename OpTy>
struct CombLowerVariadicOp : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = lowerFullyAssociativeOp(op, op.getOperands(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

  static Value lowerFullyAssociativeOp(OpTy op, OperandRange operands,
                                       ConversionPatternRewriter &rewriter) {
    Value lhs, rhs;
    switch (operands.size()) {
    case 0:
      assert(false && "cannot be called with empty operand range");
      break;
    case 1:
      return operands[0];
    case 2:
      lhs = operands[0];
      rhs = operands[1];
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    default:
      auto firstHalf = operands.size() / 2;
      lhs =
          lowerFullyAssociativeOp(op, operands.take_front(firstHalf), rewriter);
      rhs =
          lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), rewriter);
      return rewriter.create<OpTy>(op.getLoc(), ValueRange{lhs, rhs}, true);
    }
  }
};

// Lower comb::MuxOp to AIG operations.
struct CombMuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement: c ? a : b = (replicate(c) & a) | (~replicate(c) & b)

    Value cond = op.getCond();
    auto trueVal = op.getTrueValue();
    auto falseVal = op.getFalseValue();

    if (!op.getType().isInteger()) {
      // If the type of the mux is not integer, bitcast the operands first.
      auto widthType = rewriter.getIntegerType(hw::getBitWidth(op.getType()));
      trueVal =
          rewriter.create<hw::BitcastOp>(op->getLoc(), widthType, trueVal);
      falseVal =
          rewriter.create<hw::BitcastOp>(op->getLoc(), widthType, falseVal);
    }

    // Replicate condition if needed
    if (!trueVal.getType().isInteger(1))
      cond = rewriter.create<comb::ReplicateOp>(op.getLoc(), trueVal.getType(),
                                                cond);

    // c ? a : b => (replicate(c) & a) | (~replicate(c) & b)
    auto lhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, trueVal);
    auto rhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, falseVal,
                                                   true, false);

    Value result = rewriter.create<comb::OrOp>(op.getLoc(), lhs, rhs);
    // Insert the bitcast if the type of the mux is not integer.
    if (result.getType() != op.getType())
      result =
          rewriter.create<hw::BitcastOp>(op.getLoc(), op.getType(), result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombAddOpConversion : OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getInputs();
    // Lower only when there are two inputs.
    // Variadic operands must be lowered in a different pattern.
    if (inputs.size() != 2)
      return failure();

    auto width = op.getType().getIntOrFloatBitWidth();
    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    // Implement a naive Ripple-carry full adder.
    Value carry;

    auto aBits = extractBits(rewriter, inputs[0]);
    auto bBits = extractBits(rewriter, inputs[1]);
    SmallVector<Value> results;
    results.resize(width);
    for (int64_t i = 0; i < width; ++i) {
      SmallVector<Value> xorOperands = {aBits[i], bBits[i]};
      if (carry)
        xorOperands.push_back(carry);

      // sum[i] = xor(carry[i-1], a[i], b[i])
      // NOTE: The result is stored in reverse order.
      results[width - i - 1] =
          rewriter.create<comb::XorOp>(op.getLoc(), xorOperands, true);

      // If this is the last bit, we are done.
      if (i == width - 1) {
        break;
      }

      // carry[i] = (carry[i-1] & (a[i] ^ b[i])) | (a[i] & b[i])
      Value nextCarry = rewriter.create<comb::AndOp>(
          op.getLoc(), ValueRange{aBits[i], bBits[i]}, true);
      if (!carry) {
        // This is the first bit, so the carry is the next carry.
        carry = nextCarry;
        continue;
      }

      auto aXnorB = rewriter.create<comb::XorOp>(
          op.getLoc(), ValueRange{aBits[i], bBits[i]}, true);
      auto andOp = rewriter.create<comb::AndOp>(
          op.getLoc(), ValueRange{carry, aXnorB}, true);
      carry = rewriter.create<comb::OrOp>(op.getLoc(),
                                          ValueRange{andOp, nextCarry}, true);
    }

    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, results);
    return success();
  }
};

struct CombSubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    // Since `-rhs = ~rhs + 1` holds, rewrite `sub(lhs, rhs)` to:
    // sub(lhs, rhs) => add(lhs, -rhs) => add(lhs, add(~rhs, 1))
    // => add(lhs, ~rhs, 1)
    auto notRhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), rhs,
                                                      /*invert=*/true);
    auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), 1);
    rewriter.replaceOpWithNewOp<comb::AddOp>(op, ValueRange{lhs, notRhs, one},
                                             true);
    return success();
  }
};

struct CombMulOpConversion : OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<MulOp>::OpAdaptor;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 2)
      return failure();

    // FIXME: Currently it's lowered to a really naive implementation that
    // chains add operations.

    // a_{n}a_{n-1}...a_0 * b
    // = sum_{i=0}^{n} a_i * 2^i * b
    // = sum_{i=0}^{n} (a_i ? b : 0) << i
    int64_t width = op.getType().getIntOrFloatBitWidth();
    auto aBits = extractBits(rewriter, adaptor.getInputs()[0]);
    SmallVector<Value> results;
    auto rhs = op.getInputs()[1];
    auto zero = rewriter.create<hw::ConstantOp>(op.getLoc(),
                                                llvm::APInt::getZero(width));
    for (int64_t i = 0; i < width; ++i) {
      auto aBit = aBits[i];
      auto andBit =
          rewriter.createOrFold<comb::MuxOp>(op.getLoc(), aBit, rhs, zero);
      auto upperBits = rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), andBit, 0, width - i);
      if (i == 0) {
        results.push_back(upperBits);
        continue;
      }

      auto lowerBits =
          rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(i));

      auto shifted = rewriter.createOrFold<comb::ConcatOp>(
          op.getLoc(), op.getType(), ValueRange{upperBits, lowerBits});
      results.push_back(shifted);
    }

    rewriter.replaceOpWithNewOp<comb::AddOp>(op, results, true);
    return success();
  }
};

struct CombICmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;
  static Value constructUnsignedCompare(ICmpOp op, ArrayRef<Value> aBits,
                                        ArrayRef<Value> bBits, bool isLess,
                                        bool includeEq,
                                        ConversionPatternRewriter &rewriter) {
    // Construct following unsigned comparison expressions.
    // a <= b  ==> (~a[n] &  b[n]) | (a[n] == b[n] & a[n-1:0] <= b[n-1:0])
    // a <  b  ==> (~a[n] &  b[n]) | (a[n] == b[n] & a[n-1:0] < b[n-1:0])
    // a >= b  ==> ( a[n] & ~b[n]) | (a[n] == b[n] & a[n-1:0] >= b[n-1:0])
    // a >  b  ==> ( a[n] & ~b[n]) | (a[n] == b[n] & a[n-1:0] > b[n-1:0])
    Value acc =
        rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), includeEq);

    for (auto [aBit, bBit] : llvm::zip(aBits, bBits)) {
      auto aBitXorBBit =
          rewriter.createOrFold<comb::XorOp>(op.getLoc(), aBit, bBit, true);
      auto aEqualB = rewriter.createOrFold<aig::AndInverterOp>(
          op.getLoc(), aBitXorBBit, true);
      auto pred = rewriter.createOrFold<aig::AndInverterOp>(
          op.getLoc(), aBit, bBit, isLess, !isLess);

      auto aBitAndBBit = rewriter.createOrFold<comb::AndOp>(
          op.getLoc(), ValueRange{aEqualB, acc}, true);
      acc = rewriter.createOrFold<comb::OrOp>(op.getLoc(), pred, aBitAndBBit,
                                              true);
    }
    return acc;
  }

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    switch (op.getPredicate()) {
    default:
      return failure();

    case ICmpPredicate::eq:
    case ICmpPredicate::ceq: {
      // a == b  ==> ~(a[n] ^ b[n]) & ~(a[n-1] ^ b[n-1]) & ...
      auto xorOp = rewriter.createOrFold<comb::XorOp>(op.getLoc(), lhs, rhs);
      auto xorBits = extractBits(rewriter, xorOp);
      SmallVector<bool> allInverts(xorBits.size(), true);
      rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, xorBits, allInverts);
      return success();
    }

    case ICmpPredicate::ne:
    case ICmpPredicate::cne: {
      // a != b  ==> (a[n] ^ b[n]) | (a[n-1] ^ b[n-1]) | ...
      auto xorOp = rewriter.createOrFold<comb::XorOp>(op.getLoc(), lhs, rhs);
      rewriter.replaceOpWithNewOp<comb::OrOp>(op, extractBits(rewriter, xorOp),
                                              true);
      return success();
    }

    case ICmpPredicate::uge:
    case ICmpPredicate::ugt:
    case ICmpPredicate::ule:
    case ICmpPredicate::ult: {
      bool isLess = op.getPredicate() == ICmpPredicate::ult ||
                    op.getPredicate() == ICmpPredicate::ule;
      bool includeEq = op.getPredicate() == ICmpPredicate::uge ||
                       op.getPredicate() == ICmpPredicate::ule;
      auto aBits = extractBits(rewriter, lhs);
      auto bBits = extractBits(rewriter, rhs);
      rewriter.replaceOp(op, constructUnsignedCompare(op, aBits, bBits, isLess,
                                                      includeEq, rewriter));
      return success();
    }
    case ICmpPredicate::slt:
    case ICmpPredicate::sle:
    case ICmpPredicate::sgt:
    case ICmpPredicate::sge: {
      if (lhs.getType().getIntOrFloatBitWidth() == 0)
        return rewriter.notifyMatchFailure(
            op.getLoc(), "i0 signed comparison is unsupported");
      bool isLess = op.getPredicate() == ICmpPredicate::slt ||
                    op.getPredicate() == ICmpPredicate::sle;
      bool includeEq = op.getPredicate() == ICmpPredicate::sge ||
                       op.getPredicate() == ICmpPredicate::sle;

      auto aBits = extractBits(rewriter, lhs);
      auto bBits = extractBits(rewriter, rhs);

      // Get a sign bit
      auto signA = aBits.back();
      auto signB = bBits.back();

      // Compare magnitudes (all bits except sign)
      auto sameSignResult = constructUnsignedCompare(
          op, ArrayRef(aBits).drop_back(), ArrayRef(bBits).drop_back(), isLess,
          includeEq, rewriter);

      // XOR of signs: true if signs are different
      auto signsDiffer =
          rewriter.create<comb::XorOp>(op.getLoc(), signA, signB);

      // Result when signs are different
      Value diffSignResult = isLess ? signA : signB;

      // Final result: choose based on whether signs differ
      rewriter.replaceOpWithNewOp<comb::MuxOp>(op, signsDiffer, diffSignResult,
                                               sameSignResult);
      return success();
    }
    }
  }
};

struct CombParityOpConversion : OpConversionPattern<ParityOp> {
  using OpConversionPattern<ParityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Parity is the XOR of all bits.
    rewriter.replaceOpWithNewOp<comb::XorOp>(
        op, extractBits(rewriter, adaptor.getInput()), true);
    return success();
  }
};

struct CombShlOpConversion : OpConversionPattern<comb::ShlOp> {
  using OpConversionPattern<comb::ShlOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto width = op.getType().getIntOrFloatBitWidth();
    auto lhs = adaptor.getLhs();
    auto result = createShiftLogic</*isLeftShift=*/true>(
        rewriter, op.getLoc(), adaptor.getRhs(), width,
        /*getPadding=*/
        [&](int64_t index) {
          // Don't create zero width value.
          if (index == 0)
            return Value();
          // Padding is 0 for left shift.
          return rewriter.createOrFold<hw::ConstantOp>(
              op.getLoc(), rewriter.getIntegerType(index), 0);
        },
        /*getExtract=*/
        [&](int64_t index) {
          assert(index < width && "index out of bounds");
          // Exract the bits from LSB.
          return rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, 0,
                                                        width - index);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombShrUOpConversion : OpConversionPattern<comb::ShrUOp> {
  using OpConversionPattern<comb::ShrUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::ShrUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto width = op.getType().getIntOrFloatBitWidth();
    auto lhs = adaptor.getLhs();
    auto result = createShiftLogic</*isLeftShift=*/false>(
        rewriter, op.getLoc(), adaptor.getRhs(), width,
        /*getPadding=*/
        [&](int64_t index) {
          // Don't create zero width value.
          if (index == 0)
            return Value();
          // Padding is 0 for right shift.
          return rewriter.createOrFold<hw::ConstantOp>(
              op.getLoc(), rewriter.getIntegerType(index), 0);
        },
        /*getExtract=*/
        [&](int64_t index) {
          assert(index < width && "index out of bounds");
          // Exract the bits from MSB.
          return rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, index,
                                                        width - index);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CombShrSOpConversion : OpConversionPattern<comb::ShrSOp> {
  using OpConversionPattern<comb::ShrSOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::ShrSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto width = op.getType().getIntOrFloatBitWidth();
    if (width == 0)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "i0 signed shift is unsupported");
    auto lhs = adaptor.getLhs();
    // Get the sign bit.
    auto sign =
        rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, width - 1, 1);

    // NOTE: The max shift amount is width - 1 because the sign bit is already
    // shifted out.
    auto result = createShiftLogic</*isLeftShift=*/false>(
        rewriter, op.getLoc(), adaptor.getRhs(), width - 1,
        /*getPadding=*/
        [&](int64_t index) {
          return rewriter.createOrFold<comb::ReplicateOp>(op.getLoc(), sign,
                                                          index + 1);
        },
        /*getExtract=*/
        [&](int64_t index) {
          return rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, index,
                                                        width - index - 1);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Consider this as Comb canonicalization.
template <typename OpTy, bool isDivOp>
struct CombDivModUOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitWarning() << *op << "\n";
    hw::ConstantOp rhsConstantOp =
        adaptor.getRhs().template getDefiningOp<hw::ConstantOp>();
    APInt rhsValue;
    // TODO: Support non-constant divisor.
    if (rhsConstantOp) {
      rhsValue = rhsConstantOp.getValue();
    } else {
      /*
      auto concat = adaptor.getRhs().template getDefiningOp<comb::ConcatOp>();
      if (!concat)
        return rewriter.notifyMatchFailure(
            op, "currently divisor that has non-power-of-two constant as "
                "rhs is not supported");

      // If the rhs is known to be either zero or a constant power of two,
      // we can ignore the zero division since it's undefined.
      int64_t nonZeroBit = -1;
      int64_t numBits = 0;
      for (auto bit : llvm::reverse(concat.getOperands())) {
        if (auto constOp = bit.template getDefiningOp<hw::ConstantOp>()) {
          if (!constOp.getValue().isZero())
            return rewriter.notifyMatchFailure(
                op, "currently divisor that has non-power-of-two constant as "
                    "rhs is not supported");
        } else if (bit.getType().getIntOrFloatBitWidth() == 1) {
          if (nonZeroBit != -1)
            // If there are two unknown bits, unsupported.
            return rewriter.notifyMatchFailure(
                op, "currently divisor with non-constant rhs is not supported");
          nonZeroBit = numBits;
        } else {
          return rewriter.notifyMatchFailure(
              op, "currently divisor with non-constant rhs is not supported");
        }
        numBits += bit.getType().getIntOrFloatBitWidth();
      }

      if (nonZeroBit == -1) {
        rhsValue = APInt::getZero(op.getType().getIntOrFloatBitWidth());
      } else {
        rhsValue =
            APInt(op.getType().getIntOrFloatBitWidth(), 1).shl(nonZeroBit);
      }
      */
    }

    if (!rhsValue.isPowerOf2()) {
      return rewriter.notifyMatchFailure(
          op, "currently divisor with non-power-of-2 is not supported");
    }

    size_t shiftAmount = rhsValue.ceilLogBase2();
    size_t width = op.getType().getIntOrFloatBitWidth();

    if (isDivOp) {
      // Extract upper bits.
      Value upperBits = rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), adaptor.getLhs(), shiftAmount, width - shiftAmount);
      Value constZero = rewriter.create<hw::ConstantOp>(
          op.getLoc(), APInt::getZero(shiftAmount));
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(
          op, op.getType(), ArrayRef<Value>{constZero, upperBits});
    } else {
      // Extract lower bits.
      Value lowerBits = rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), adaptor.getLhs(), 0, shiftAmount);
      Value constZero = rewriter.create<hw::ConstantOp>(
          op.getLoc(), APInt::getZero(width - shiftAmount));

      rewriter.replaceOpWithNewOp<comb::ConcatOp>(
          op, op.getType(), ArrayRef<Value>{constZero, lowerBits});
    }
    return success();
  }
};

// When rhs is not power of two and the number of unknown bits are small (<=6),
// create a mux tree that emulates all possible cases.
// TODO: This is probably not most efficient so consider revisiting this.
template <typename OpTy, bool isDivOp>
struct CombDivModUOpTableConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto rhsConstantOp =
            adaptor.getRhs().template getDefiningOp<hw::ConstantOp>())
      // Let the other pattern matches this.
      if (rhsConstantOp.getValue().isPowerOf2())
        return failure();

    auto width = op.getType().getIntOrFloatBitWidth();

    SmallVector<Value> lhsBits, rhsBits;
    auto getBits = [&](auto &&recurse, llvm::SmallVectorImpl<Value> &results,
                       Value elem) -> size_t {
      if (auto concat = elem.getDefiningOp<comb::ConcatOp>()) {
        size_t numUnknownBits = 0;
        for (auto c : llvm::reverse(concat.getInputs())) {
          if (c.getType().isInteger(0))
            continue;
          numUnknownBits += recurse(recurse, results, c);
        }
        return numUnknownBits;
      }
      results.push_back(elem);
      return elem.getDefiningOp<hw::ConstantOp>()
                 ? 0
                 : elem.getType().getIntOrFloatBitWidth();
    };

    auto numLhsUnknownBits = getBits(getBits, lhsBits, op.getLhs());
    auto numRhsUnknownBits = getBits(getBits, rhsBits, op.getRhs());
    // std::reverse(lhsBits.begin(), lhsBits.end());
    // std::reverse(rhsBits.begin(), rhsBits.end());

    // Give up if there are too many unknown bits.
    if (numLhsUnknownBits + numRhsUnknownBits > 8)
      return failure();

    auto subStituteMask = [width](llvm::SmallVectorImpl<Value> &bits,
                                  size_t mask) -> APInt {
      size_t pos = 0, unknownPos = 0;
      APInt result(width, 0);
      for (auto elem : bits) {
        auto elemWidth = elem.getType().getIntOrFloatBitWidth();
        if (auto constant = elem.getDefiningOp<hw::ConstantOp>()) {
          result |= constant.getValue().zext(width) << APInt(width, pos);
        } else {
          size_t usedMask = (1 << (elemWidth + unknownPos)) - (1 << unknownPos);
          result |= APInt(width, ((usedMask & mask) >> unknownPos) << pos);
          unknownPos += elemWidth;
        }
        pos += elemWidth;
      }
      return result;
    };

    SmallVector<Value> results;
    auto allZero =
        rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(width));
    for (size_t lhsMask = 0, e = 1 << numLhsUnknownBits; lhsMask < e;
         ++lhsMask) {
      auto lhsValue = subStituteMask(lhsBits, lhsMask);
      for (size_t rhsMask = 0, e2 = 1 << numRhsUnknownBits; rhsMask < e2;
           ++rhsMask) {
        APInt rhsValue = subStituteMask(rhsBits, rhsMask);
        llvm::errs() << lhsValue.getZExtValue() << " "
                     << rhsValue.getZExtValue() << "\n";
        if (rhsValue.isZero()) {
          // Undefined value. Just use zero for now.
          results.push_back(allZero);
        } else {
          APInt result =
              isDivOp ? lhsValue.udiv(rhsValue) : lhsValue.urem(rhsValue);
          results.push_back(
              rewriter.create<hw::ConstantOp>(op.getLoc(), result));
        }
      }
    }
    SmallVector<Value> tableInputs;
    for (auto &bits : {lhsBits, rhsBits}) {
      SmallVector<Value> tmp;
      for (auto bit : bits) {
        if (!bit.getDefiningOp<hw::ConstantOp>()) {
          auto extracted = extractBits(rewriter, bit);
          tmp.append(extracted);
        }
      }
      std::reverse(tmp.begin(), tmp.end());
      tableInputs.append(tmp.begin(), tmp.end());
    }
    std::reverse(tableInputs.begin(), tableInputs.end());
    assert(numLhsUnknownBits + numRhsUnknownBits == tableInputs.size());
    auto muxed =
        constructMuxTree(rewriter, op->getLoc(), tableInputs, results, allZero);

    rewriter.replaceOp(op, muxed);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToAIGPass
    : public impl::ConvertCombToAIGBase<ConvertCombToAIGPass> {
  void runOnOperation() override;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::ConvertCombToAIGBase;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::additionalLegalOps;
};
} // namespace

static void populateCombToAIGConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      // Bitwise Logical Ops
      CombAndOpConversion, CombOrOpConversion, CombXorOpConversion,
      CombMuxOpConversion, CombParityOpConversion,
      // Arithmetic Ops
      CombAddOpConversion, CombSubOpConversion, CombMulOpConversion,
      CombICmpOpConversion,
      // Shift Ops
      CombShlOpConversion, CombShrUOpConversion, CombShrSOpConversion,
      CombDivModUOpConversion<DivUOp, true>,
      CombDivModUOpConversion<ModUOp, false>,
      CombDivModUOpTableConversion<DivUOp, true>,
      CombDivModUOpTableConversion<ModUOp, false>,
      // Variadic ops that must be lowered to binary operations
      CombLowerVariadicOp<XorOp>, CombLowerVariadicOp<AddOp>,
      CombLowerVariadicOp<MulOp>>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  if (!getOperation().getModuleName().starts_with("SiFive_") ||
      getOperation().getNumOutputPorts() == 0)
    return markAllAnalysesPreserved();
  // if (!getOperation().getModuleName().starts_with("SiFive_") ||
  //     getOperation().getNumOutputPorts() == 0)
  //   return markAllAnalysesPreserved();
  ConversionTarget target(getContext());

  // Comb is source dialect.
  target.addIllegalDialect<comb::CombDialect>();
  // Keep data movement operations like Extract, Concat and Replicate.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::BitcastOp, hw::ConstantOp>();

  // Treat array operations as illegal. Strictly speaking, other than array get
  // operation with non-const index are legal in AIG but array types prevent a
  // bunch of optimizations so just lower them to integer operations. It's
  // required to run HWAggregateToComb pass before this pass.
  target.addIllegalOp<hw::ArrayGetOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
                      hw::AggregateConstantOp>();

  // AIG is target dialect.
  target.addLegalDialect<aig::AIGDialect>();

  // This is a test only option to add logical ops.
  if (!additionalLegalOps.empty())
    for (const auto &opName : additionalLegalOps)
      target.addLegalOp(OperationName(opName, &getContext()));

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
