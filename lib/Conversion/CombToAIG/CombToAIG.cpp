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
    bits.push_back(rewriter.create<comb::ExtractOp>(val.getLoc(), val, i, 1));

  return bits;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

// Convert comb.mux to AIG operations
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

    // Implement: c ? a : b = (replicate(c) & a) | (~replicate(c) & b)
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
    // Variable operands are lowered in a different pattern.
    if (inputs.size() != 2)
      return failure();

    auto width = op.getType().getIntOrFloatBitWidth();
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

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

template <typename OpTy, bool isLeftShift>
struct CombShiftOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto width = op.getType().getIntOrFloatBitWidth();
    if (width <= 0)
      return failure();
    // For left shift, construct the mux tree from the most significant bit to
    // the least significant bit. For right shift, construct the mux tree from
    // the least significant bit to the most significant bit.

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto bits = extractBits(rewriter, rhs);
    auto allZero = rewriter.create<hw::ConstantOp>(
        op.getLoc(), IntegerType::get(rewriter.getContext(), width), 0);

    SmallVector<Value> nodes = createShiftNodes(rewriter, op, lhs, width);

    auto result =
        constructMuxTree(rewriter, op.getLoc(), width, bits, nodes, allZero);
    // Add a gate to check if the shift amount is within bounds.
    auto inBound = rewriter.createOrFold<comb::ICmpOp>(
        op.getLoc(), ICmpPredicate::ult, rhs,
        rewriter.create<hw::ConstantOp>(op.getLoc(), rhs.getType(), width));
    result = rewriter.createOrFold<comb::MuxOp>(op.getLoc(), inBound, result,
                                                allZero);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static SmallVector<Value>
  createShiftNodes(ConversionPatternRewriter &rewriter, OpTy op, Value lhs,
                   int64_t width) {
    SmallVector<Value> nodes;
    nodes.reserve(width);

    for (int64_t i = 0; i < width; ++i) {
      // Switch the extract operation based on the shift direction.
      // For left shift, extract lower bits. For right shift, extract higher
      // bits.
      Value extract = isLeftShift ? rewriter.createOrFold<comb::ExtractOp>(
                                        op.getLoc(), lhs, 0, width - i)
                                  : rewriter.createOrFold<comb::ExtractOp>(
                                        op.getLoc(), lhs, i, width - i);

      if (i == 0) {
        nodes.push_back(extract);
        continue;
      }

      auto zeros = rewriter.createOrFold<hw::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), i), 0);

      // Concatenate the extracted bits with zeros
      auto concat = isLeftShift
                        ? rewriter.createOrFold<comb::ConcatOp>(op.getLoc(),
                                                                extract, zeros)
                        : rewriter.createOrFold<comb::ConcatOp>(op.getLoc(),
                                                                zeros, extract);
      nodes.push_back(concat);
    }

    return nodes;
  }

  static Value constructMuxTree(ConversionPatternRewriter &rewriter,
                                Location loc, int64_t width,
                                ArrayRef<Value> bits, ArrayRef<Value> nodes,
                                Value allZero) {
    // Lambda to get the result for a given index
    auto getResult = [&](size_t id) -> Value {
      // Out of bounds index is undefined, so return all-zero for now.
      return id < nodes.size() ? nodes[id] : allZero;
    };

    // Recursive helper function to construct the mux tree
    std::function<Value(size_t, size_t)> constructTreeHelper =
        [&](size_t id, size_t level) -> Value {
      // Base case: at the lowest level, return the result
      if (level == 0)
        return getResult(id);

      auto selector = bits[level - 1];

      // Recursive case: create muxes for true and false branches
      auto trueVal = constructTreeHelper(2 * id + 1, level - 1);
      auto falseVal = constructTreeHelper(2 * id, level - 1);

      // Combine the results with a mux
      return rewriter.createOrFold<comb::MuxOp>(loc, selector, trueVal,
                                                falseVal);
    };

    return constructTreeHelper(0, llvm::Log2_64_Ceil(width));
  }
};

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

struct CombSubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    // Since `-rhs = ~rhs + 1`, we can rewrite `sub(lhs, rhs)` as
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

// Consider this as Comb canonicalization.
template <typename OpTy, bool isDivOp>
struct CombDivModUOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    hw::ConstantOp rhsConstantOp =
        adaptor.getRhs().template getDefiningOp<hw::ConstantOp>();
    APInt rhsValue;
    // TODO: Support non-constant divisor.
    if (rhsConstantOp) {
      rhsValue = rhsConstantOp.getValue();
    } else {
      auto concat = adaptor.getRhs().template getDefiningOp<comb::ConcatOp>();
      if (!concat)
        return failure();

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
            return failure();
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
    }

    if (!rhsValue.isPowerOf2())
      return failure();

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
    // = sum_{i=0}^{n} (a_i * 2^i * b
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

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToAIGPass
    : public impl::ConvertCombToAIGBase<ConvertCombToAIGPass> {

  void runOnOperation() override;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::ConvertCombToAIGBase;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::keepBitwiseLogicalOps;
};
} // namespace

static void populateCombToAIGConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      // Bitwise Logical Ops
      CombAndOpConversion, CombOrOpConversion, CombXorOpConversion,
      CombMuxOpConversion, CombParityOpConversion,
      // Arithmatic Ops
      CombAddOpConversion, CombSubOpConversion, CombICmpOpConversion,
      CombMulOpConversion, CombDivModUOpConversion<DivUOp, true>,
      CombDivModUOpConversion<ModUOp, false>,
      // Shift Ops
      CombShiftOpConversion<ShlOp, true>, CombShiftOpConversion<ShrUOp, false>,
      // This is not Correct
      CombShiftOpConversion<ShrSOp, false>,
      // Variadic ops that must be lowered to binary operations
      CombLowerVariadicOp<AddOp>, CombLowerVariadicOp<MulOp>,
      CombLowerVariadicOp<XorOp>>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  if (/*!getOperation().getModuleName().starts_with("SiFive_") ||*/
      getOperation().getNumOutputPorts() == 0)
    return markAllAnalysesPreserved();
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  // Extract, Concat and Replicate just pass through so we can keep them.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::ConstantOp>();

  target.addLegalDialect<aig::AIGDialect, hw::HWDialect>();

  // This is a test only option to keep bitwise logical ops instead of lowering
  // to AIG.
  if (keepBitwiseLogicalOps)
    target.addLegalOp<comb::MuxOp, comb::XorOp, comb::AndOp, comb::OrOp>();

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
