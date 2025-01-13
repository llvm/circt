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

// Extract individual bits from a value
static SmallVector<Value> extractBits(ConversionPatternRewriter &rewriter,
                                      Value val) {
  assert(val.getType().isInteger() && "expected integer");
  auto width = val.getType().getIntOrFloatBitWidth();
  SmallVector<Value> bits;
  bits.reserve(width);

  // Check if we can reuse concat operands
  if (auto concat = val.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getNumOperands() == width &&
        llvm::all_of(concat.getOperandTypes(), [](Type type) {
          return type.getIntOrFloatBitWidth() == 1;
        })) {
      // Reverse the operands to match the bit order
      bits.append(std::make_reverse_iterator(concat.getOperands().end()),
                  std::make_reverse_iterator(concat.getOperands().begin()));
      return bits;
    }
  }

  // Extract individual bits
  for (int64_t i = 0; i < width; ++i)
    bits.push_back(
        rewriter.createOrFold<comb::ExtractOp>(val.getLoc(), val, i, 1));

  return bits;
}

// Construct a mux tree for given leaf nodes. `selectors` is the selector for
// each level of the tree. Currently the selector is tested from MSB to LSB.
static Value constructMuxTree(ConversionPatternRewriter &rewriter, Location loc,
                              ArrayRef<Value> selectors,
                              ArrayRef<Value> leafNodes,
                              Value outOfBoundsValue) {
  // Recursive helper function to construct the mux tree
  std::function<Value(size_t, size_t)> constructTreeHelper =
      [&](size_t id, size_t level) -> Value {
    // Base case: at the lowest level, return the result
    if (level == 0) {
      // Return the result for the given index. If the index is out of bounds,
      // return the out-of-bound value.
      return id < leafNodes.size() ? leafNodes[id] : outOfBoundsValue;
    }

    auto selector = selectors[level - 1];

    // Recursive case: create muxes for true and false branches
    auto trueVal = constructTreeHelper(2 * id + 1, level - 1);
    auto falseVal = constructTreeHelper(2 * id, level - 1);

    // Combine the results with a mux
    return rewriter.createOrFold<comb::MuxOp>(loc, selector, trueVal, falseVal);
  };

  return constructTreeHelper(0, llvm::Log2_64_Ceil(leafNodes.size()));
}

// Construct a mux tree for shift operations. `isLeftShift` controls the
// direction of the shift operation and is used to determine order of the
// padding and extracted bits. Callbacks `getPadding` and `getExtract` are used
// to get the padding and extracted bits for each shift amount.  One of the
// callbacks must return a valid value for each shift amount in the range [0,
// maxShiftAmount]. The value for `maxShiftAmount` is used as the out-of-bounds
// value.
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

    if (!padding || !extract) {
      nodes.push_back(extract ? extract : padding);
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

  // Construct mux tree for shift operation
  auto result = constructMuxTree(rewriter, loc, bits, nodes, outOfBoundsValue);

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
      // Variadic ops that must be lowered to binary operations
      CombLowerVariadicOp<XorOp>, CombLowerVariadicOp<AddOp>,
      CombLowerVariadicOp<MulOp>>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  // Keep data movement operations like Extract, Concat and Replicate.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::BitcastOp, hw::ConstantOp>();
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
