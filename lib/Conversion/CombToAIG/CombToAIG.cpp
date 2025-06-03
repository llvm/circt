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
#include "circt/Support/Naming.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "comb-to-aig"

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

namespace {
// A union of Value and IntegerAttr to cleanly handle constant values.
using ConstantOrValue = llvm::PointerUnion<Value, mlir::IntegerAttr>;
} // namespace

// Return the number of unknown bits and populate the concatenated values.
static int64_t getNumUnknownBitsAndPopulateValues(
    Value value, llvm::SmallVectorImpl<ConstantOrValue> &values) {
  // Constant or zero width value are all known.
  if (value.getType().isInteger(0))
    return 0;

  // Recursively count unknown bits for concat.
  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    int64_t totalUnknownBits = 0;
    for (auto concatInput : llvm::reverse(concat.getInputs())) {
      auto unknownBits =
          getNumUnknownBitsAndPopulateValues(concatInput, values);
      if (unknownBits < 0)
        return unknownBits;
      totalUnknownBits += unknownBits;
    }
    return totalUnknownBits;
  }

  // Constant value is known.
  if (auto constant = value.getDefiningOp<hw::ConstantOp>()) {
    values.push_back(constant.getValueAttr());
    return 0;
  }

  // Consider other operations as unknown bits.
  // TODO: We can handle replicate, extract, etc.
  values.push_back(value);
  return hw::getBitWidth(value.getType());
}

// Return a value that substitutes the unknown bits with the mask.
static APInt
substitueMaskToValues(size_t width,
                      llvm::SmallVectorImpl<ConstantOrValue> &constantOrValues,
                      uint32_t mask) {
  uint32_t bitPos = 0, unknownPos = 0;
  APInt result(width, 0);
  for (auto constantOrValue : constantOrValues) {
    int64_t elemWidth;
    if (auto constant = dyn_cast<IntegerAttr>(constantOrValue)) {
      elemWidth = constant.getValue().getBitWidth();
      result.insertBits(constant.getValue(), bitPos);
    } else {
      elemWidth = hw::getBitWidth(cast<Value>(constantOrValue).getType());
      assert(elemWidth >= 0 && "unknown bit width");
      assert(elemWidth + unknownPos < 32 && "unknown bit width too large");
      // Create a mask for the unknown bits.
      uint32_t usedBits = (mask >> unknownPos) & ((1 << elemWidth) - 1);
      result.insertBits(APInt(elemWidth, usedBits), bitPos);
      unknownPos += elemWidth;
    }
    bitPos += elemWidth;
  }

  return result;
}

// Emulate a binary operation with unknown bits using a table lookup.
// This function enumerates all possible combinations of unknown bits and
// emulates the operation for each combination.
static LogicalResult emulateBinaryOpForUnknownBits(
    ConversionPatternRewriter &rewriter, int64_t maxEmulationUnknownBits,
    Operation *op,
    llvm::function_ref<APInt(const APInt &, const APInt &)> emulate) {
  SmallVector<ConstantOrValue> lhsValues, rhsValues;

  assert(op->getNumResults() == 1 && op->getNumOperands() == 2 &&
         "op must be a single result binary operation");

  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto width = op->getResult(0).getType().getIntOrFloatBitWidth();
  auto loc = op->getLoc();
  auto numLhsUnknownBits = getNumUnknownBitsAndPopulateValues(lhs, lhsValues);
  auto numRhsUnknownBits = getNumUnknownBitsAndPopulateValues(rhs, rhsValues);

  // If unknown bit width is detected, abort the lowering.
  if (numLhsUnknownBits < 0 || numRhsUnknownBits < 0)
    return failure();

  int64_t totalUnknownBits = numLhsUnknownBits + numRhsUnknownBits;
  if (totalUnknownBits > maxEmulationUnknownBits)
    return failure();

  SmallVector<Value> emulatedResults;
  emulatedResults.reserve(1 << totalUnknownBits);

  // Emulate all possible cases.
  DenseMap<IntegerAttr, hw::ConstantOp> constantPool;
  auto getConstant = [&](const APInt &value) -> hw::ConstantOp {
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(width), value);
    auto it = constantPool.find(attr);
    if (it != constantPool.end())
      return it->second;
    auto constant = rewriter.create<hw::ConstantOp>(loc, value);
    constantPool[attr] = constant;
    return constant;
  };

  for (uint32_t lhsMask = 0, lhsMaskEnd = 1 << numLhsUnknownBits;
       lhsMask < lhsMaskEnd; ++lhsMask) {
    APInt lhsValue = substitueMaskToValues(width, lhsValues, lhsMask);
    for (uint32_t rhsMask = 0, rhsMaskEnd = 1 << numRhsUnknownBits;
         rhsMask < rhsMaskEnd; ++rhsMask) {
      APInt rhsValue = substitueMaskToValues(width, rhsValues, rhsMask);
      // Emulate.
      emulatedResults.push_back(getConstant(emulate(lhsValue, rhsValue)));
    }
  }

  // Create selectors for mux tree.
  SmallVector<Value> selectors;
  selectors.reserve(totalUnknownBits);
  for (auto &concatedValues : {rhsValues, lhsValues})
    for (auto valueOrConstant : concatedValues) {
      auto value = dyn_cast<Value>(valueOrConstant);
      if (!value)
        continue;
      extractBits(rewriter, value, selectors);
    }

  assert(totalUnknownBits == static_cast<int64_t>(selectors.size()) &&
         "number of selectors must match");
  auto muxed = constructMuxTree(rewriter, loc, selectors, emulatedResults,
                                getConstant(APInt::getZero(width)));

  replaceOpAndCopyNamehint(rewriter, op, muxed);
  return success();
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
    replaceOpWithNewOpAndCopyNamehint<aig::AndInverterOp>(
        rewriter, op, adaptor.getInputs(), nonInverts);
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
    replaceOpWithNewOpAndCopyNamehint<aig::AndInverterOp>(rewriter, op, andOp,
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

    replaceOpWithNewOpAndCopyNamehint<aig::AndInverterOp>(rewriter, op,
                                                          notAAndNotB, aAndB,
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
    replaceOpAndCopyNamehint(rewriter, op, result);
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
    replaceOpAndCopyNamehint(rewriter, op, result);
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
      replaceOpWithNewOpAndCopyNamehint<hw::ConstantOp>(rewriter, op,
                                                        op.getType(), 0);
      return success();
    }

    if (width < 8)
      lowerRippleCarryAdder(op, inputs, rewriter);
    else
      lowerParallelPrefixAdder(op, inputs, rewriter);

    return success();
  }

  // Implement a basic ripple-carry adder for small bitwidths.
  void lowerRippleCarryAdder(comb::AddOp op, ValueRange inputs,
                             ConversionPatternRewriter &rewriter) const {
    auto width = op.getType().getIntOrFloatBitWidth();
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
      if (i == width - 1)
        break;

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
    LLVM_DEBUG(llvm::dbgs() << "Lower comb.add to Ripple-Carry Adder of width "
                            << width << "\n");

    replaceOpWithNewOpAndCopyNamehint<comb::ConcatOp>(rewriter, op, results);
  }

  // Implement a parallel prefix adder - with Kogge-Stone or Brent-Kung trees
  // Will introduce unused signals for the carry bits but these will be removed
  // by the AIG pass.
  void lowerParallelPrefixAdder(comb::AddOp op, ValueRange inputs,
                                ConversionPatternRewriter &rewriter) const {
    auto width = op.getType().getIntOrFloatBitWidth();

    auto aBits = extractBits(rewriter, inputs[0]);
    auto bBits = extractBits(rewriter, inputs[1]);
    // Construct propagate (p) and generate (g) signals
    SmallVector<Value> p, g;
    p.reserve(width);
    g.reserve(width);

    for (auto [aBit, bBit] : llvm::zip(aBits, bBits)) {
      // p_i = a_i XOR b_i
      p.push_back(rewriter.create<comb::XorOp>(op.getLoc(), aBit, bBit));
      // g_i = a_i AND b_i
      g.push_back(rewriter.create<comb::AndOp>(op.getLoc(), aBit, bBit));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Lower comb.add to Parallel-Prefix of width " << width
                   << "\n--------------------------------------- Init\n";

      for (int64_t i = 0; i < width; ++i) {
        // p_i = a_i XOR b_i
        llvm::dbgs() << "P0" << i << " = A" << i << " XOR B" << i << "\n";
        // g_i = a_i AND b_i
        llvm::dbgs() << "G0" << i << " = A" << i << " AND B" << i << "\n";
      }
    });

    // Create copies of p and g for the prefix computation
    SmallVector<Value> pPrefix = p;
    SmallVector<Value> gPrefix = g;
    if (width < 32)
      lowerKoggeStonePrefixTree(op, inputs, rewriter, pPrefix, gPrefix);
    else
      lowerBrentKungPrefixTree(op, inputs, rewriter, pPrefix, gPrefix);

    // Generate result sum bits
    // NOTE: The result is stored in reverse order.
    SmallVector<Value> results;
    results.resize(width);
    // Sum bit 0 is just p[0] since carry_in = 0
    results[width - 1] = p[0];

    // For remaining bits, sum_i = p_i XOR c_(i-1)
    // The carry into position i is the group generate from position i-1
    for (int64_t i = 1; i < width; ++i)
      results[width - 1 - i] =
          rewriter.create<comb::XorOp>(op.getLoc(), p[i], gPrefix[i - 1]);

    replaceOpWithNewOpAndCopyNamehint<comb::ConcatOp>(rewriter, op, results);

    LLVM_DEBUG({
      llvm::dbgs() << "--------------------------------------- Completion\n"
                   << "RES0 = P0\n";
      for (int64_t i = 1; i < width; ++i)
        llvm::dbgs() << "RES" << i << " = P" << i << " XOR G" << i - 1 << "\n";
    });
  }

  // Implement the Kogge-Stone parallel prefix tree
  // Described in https://en.wikipedia.org/wiki/Kogge%E2%80%93Stone_adder
  // Slightly better delay than Brent-Kung, but more area.
  void lowerKoggeStonePrefixTree(comb::AddOp op, ValueRange inputs,
                                 ConversionPatternRewriter &rewriter,
                                 SmallVector<Value> &pPrefix,
                                 SmallVector<Value> &gPrefix) const {
    auto width = op.getType().getIntOrFloatBitWidth();

    // Kogge-Stone parallel prefix computation
    for (int64_t stride = 1; stride < width; stride *= 2) {
      for (int64_t i = stride; i < width; ++i) {
        int64_t j = i - stride;
        // Group generate: g_i OR (p_i AND g_j)
        Value andPG =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], gPrefix[j]);
        gPrefix[i] =
            rewriter.create<comb::OrOp>(op.getLoc(), gPrefix[i], andPG);

        // Group propagate: p_i AND p_j
        pPrefix[i] =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], pPrefix[j]);
      }
    }
    LLVM_DEBUG({
      int64_t stage = 0;
      for (int64_t stride = 1; stride < width; stride *= 2) {
        llvm::dbgs()
            << "--------------------------------------- Kogge-Stone Stage "
            << stage << "\n";
        for (int64_t i = stride; i < width; ++i) {
          int64_t j = i - stride;
          // Group generate: g_i OR (p_i AND g_j)
          llvm::dbgs() << "G" << i << stage + 1 << " = G" << i << stage
                       << " OR (P" << i << stage << " AND G" << j << stage
                       << ")\n";

          // Group propagate: p_i AND p_j
          llvm::dbgs() << "P" << i << stage + 1 << " = P" << i << stage
                       << " AND P" << j << stage << "\n";
        }
        ++stage;
      }
    });
  }

  // Implement the Brent-Kung parallel prefix tree
  // Described in https://en.wikipedia.org/wiki/Brent%E2%80%93Kung_adder
  // Slightly worse delay than Kogge-Stone, but less area.
  void lowerBrentKungPrefixTree(comb::AddOp op, ValueRange inputs,
                                ConversionPatternRewriter &rewriter,
                                SmallVector<Value> &pPrefix,
                                SmallVector<Value> &gPrefix) const {
    auto width = op.getType().getIntOrFloatBitWidth();

    // Brent-Kung parallel prefix computation
    // Forward phase
    int64_t stride;
    for (stride = 1; stride < width; stride *= 2) {
      for (int64_t i = stride * 2 - 1; i < width; i += stride * 2) {
        int64_t j = i - stride;

        // Group generate: g_i OR (p_i AND g_j)
        Value andPG =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], gPrefix[j]);
        gPrefix[i] =
            rewriter.create<comb::OrOp>(op.getLoc(), gPrefix[i], andPG);

        // Group propagate: p_i AND p_j
        pPrefix[i] =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], pPrefix[j]);
      }
    }

    // Backward phase
    for (; stride > 0; stride /= 2) {
      for (int64_t i = stride * 3 - 1; i < width; i += stride * 2) {
        int64_t j = i - stride;

        // Group generate: g_i OR (p_i AND g_j)
        Value andPG =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], gPrefix[j]);
        gPrefix[i] = rewriter.create<OrOp>(op.getLoc(), gPrefix[i], andPG);

        // Group propagate: p_i AND p_j
        pPrefix[i] =
            rewriter.create<comb::AndOp>(op.getLoc(), pPrefix[i], pPrefix[j]);
      }
    }

    LLVM_DEBUG({
      int64_t stage = 0;
      for (stride = 1; stride < width; stride *= 2) {
        llvm::dbgs() << "--------------------------------------- Brent-Kung FW "
                     << stage << " : Stride " << stride << "\n";
        for (int64_t i = stride * 2 - 1; i < width; i += stride * 2) {
          int64_t j = i - stride;

          // Group generate: g_i OR (p_i AND g_j)
          llvm::dbgs() << "G" << i << stage + 1 << " = G" << i << stage
                       << " OR (P" << i << stage << " AND G" << j << stage
                       << ")\n";

          // Group propagate: p_i AND p_j
          llvm::dbgs() << "P" << i << stage + 1 << " = P" << i << stage
                       << " AND P" << j << stage << "\n";
        }
        ++stage;
      }

      for (; stride > 0; stride /= 2) {
        if (stride * 3 - 1 < width)
          llvm::dbgs()
              << "--------------------------------------- Brent-Kung BW "
              << stage << " : Stride " << stride << "\n";

        for (int64_t i = stride * 3 - 1; i < width; i += stride * 2) {
          int64_t j = i - stride;

          // Group generate: g_i OR (p_i AND g_j)
          llvm::dbgs() << "G" << i << stage + 1 << " = G" << i << stage
                       << " OR (P" << i << stage << " AND G" << j << stage
                       << ")\n";

          // Group propagate: p_i AND p_j
          llvm::dbgs() << "P" << i << stage + 1 << " = P" << i << stage
                       << " AND P" << j << stage << "\n";
        }
        --stage;
      }
    });
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
    replaceOpWithNewOpAndCopyNamehint<comb::AddOp>(
        rewriter, op, ValueRange{lhs, notRhs, one}, true);
    return success();
  }
};

// Construct a full adder for three 1-bit inputs.
std::pair<Value, Value> fullAdder(ConversionPatternRewriter &rewriter,
                                  Location loc, Value a, Value b, Value c) {

  Value sum =
      rewriter.createOrFold<comb::XorOp>(loc, ArrayRef<Value>{a, b, c}, true);

  auto carry = rewriter.createOrFold<comb::OrOp>(
      loc,
      ArrayRef<Value>{rewriter.createOrFold<comb::AndOp>(loc, a, b),
                      rewriter.createOrFold<comb::AndOp>(loc, b, c),
                      rewriter.createOrFold<comb::AndOp>(loc, c, a)},
      true);

  return {sum, carry};
}

struct CombMulOpConversion : OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<MulOp>::OpAdaptor;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 2)
      return failure();

    Location loc = op.getLoc();
    Value a = adaptor.getInputs()[0];
    Value b = adaptor.getInputs()[1];
    unsigned width = op.getType().getIntOrFloatBitWidth();

    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    // Extract individual bits from operands
    SmallVector<Value> aBits = extractBits(rewriter, a);
    SmallVector<Value> bBits = extractBits(rewriter, b);

    auto falseValue = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));

    // Generate partial products
    SmallVector<SmallVector<Value>> partialProducts;
    partialProducts.reserve(width);
    for (unsigned i = 0; i < width; ++i) {
      SmallVector<Value> row(i, falseValue);
      row.reserve(width);
      // Generate partial product bits
      for (unsigned j = 0; i + j < width; ++j)
        row.push_back(
            rewriter.createOrFold<comb::AndOp>(loc, aBits[j], bBits[i]));

      partialProducts.push_back(row);
    }

    // If the width is 1, we are done.
    if (width == 1) {
      rewriter.replaceOp(op, partialProducts[0][0]);
      return success();
    }

    // Wallace tree reduction
    replaceOpAndCopyNamehint(
        rewriter, op,
        wallaceReduction(falseValue, width, rewriter, loc, partialProducts));
    return success();
  }

private:
  // Perform Wallace tree reduction on partial products
  static Value
  wallaceReduction(Value falseValue, size_t width,
                   ConversionPatternRewriter &rewriter, Location loc,
                   SmallVector<SmallVector<Value>> &partialProducts) {
    SmallVector<SmallVector<Value>> newPartialProducts;
    newPartialProducts.reserve(partialProducts.size());
    // Continue reduction until we have only two rows. The length of
    // `partialProducts` is reduced by 1/3 in each iteration.
    while (partialProducts.size() > 2) {
      newPartialProducts.clear();
      // Take three rows at a time and reduce to two rows(sum and carry).
      for (unsigned i = 0; i < partialProducts.size(); i += 3) {
        if (i + 2 < partialProducts.size()) {
          // We have three rows to reduce
          auto &row1 = partialProducts[i];
          auto &row2 = partialProducts[i + 1];
          auto &row3 = partialProducts[i + 2];

          assert(row1.size() == width && row2.size() == width &&
                 row3.size() == width);

          SmallVector<Value> sumRow, carryRow;
          sumRow.reserve(width);
          carryRow.reserve(width);
          carryRow.push_back(falseValue);

          // Process each bit position
          for (unsigned j = 0; j < width; ++j) {
            // Full adder logic
            auto [sum, carry] =
                fullAdder(rewriter, loc, row1[j], row2[j], row3[j]);
            sumRow.push_back(sum);
            if (j + 1 < width)
              carryRow.push_back(carry);
          }

          newPartialProducts.push_back(std::move(sumRow));
          newPartialProducts.push_back(std::move(carryRow));
        } else {
          // Add remaining rows as is
          newPartialProducts.append(partialProducts.begin() + i,
                                    partialProducts.end());
        }
      }

      std::swap(newPartialProducts, partialProducts);
    }

    assert(partialProducts.size() == 2);
    // Reverse the order of the bits
    std::reverse(partialProducts[0].begin(), partialProducts[0].end());
    std::reverse(partialProducts[1].begin(), partialProducts[1].end());
    auto lhs = rewriter.create<comb::ConcatOp>(loc, partialProducts[0]);
    auto rhs = rewriter.create<comb::ConcatOp>(loc, partialProducts[1]);

    // Use comb.add for the final addition.
    return rewriter.create<comb::AddOp>(loc, ArrayRef<Value>{lhs, rhs}, true);
  }
};

template <typename OpTy>
struct DivModOpConversionBase : OpConversionPattern<OpTy> {
  DivModOpConversionBase(MLIRContext *context, int64_t maxEmulationUnknownBits)
      : OpConversionPattern<OpTy>(context),
        maxEmulationUnknownBits(maxEmulationUnknownBits) {
    assert(maxEmulationUnknownBits < 32 &&
           "maxEmulationUnknownBits must be less than 32");
  }
  const int64_t maxEmulationUnknownBits;
};

struct CombDivUOpConversion : DivModOpConversionBase<DivUOp> {
  using DivModOpConversionBase<DivUOp>::DivModOpConversionBase;
  LogicalResult
  matchAndRewrite(DivUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the divisor is a power of two.
    if (auto rhsConstantOp = adaptor.getRhs().getDefiningOp<hw::ConstantOp>())
      if (rhsConstantOp.getValue().isPowerOf2()) {
        // Extract upper bits.
        size_t extractAmount = rhsConstantOp.getValue().ceilLogBase2();
        size_t width = op.getType().getIntOrFloatBitWidth();
        Value upperBits = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(), adaptor.getLhs(), extractAmount,
            width - extractAmount);
        Value constZero = rewriter.create<hw::ConstantOp>(
            op.getLoc(), APInt::getZero(extractAmount));
        replaceOpWithNewOpAndCopyNamehint<comb::ConcatOp>(
            rewriter, op, op.getType(), ArrayRef<Value>{constZero, upperBits});
        return success();
      }

    // When rhs is not power of two and the number of unknown bits are small,
    // create a mux tree that emulates all possible cases.
    return emulateBinaryOpForUnknownBits(
        rewriter, maxEmulationUnknownBits, op,
        [](const APInt &lhs, const APInt &rhs) {
          // Division by zero is undefined, just return zero.
          if (rhs.isZero())
            return APInt::getZero(rhs.getBitWidth());
          return lhs.udiv(rhs);
        });
  }
};

struct CombModUOpConversion : DivModOpConversionBase<ModUOp> {
  using DivModOpConversionBase<ModUOp>::DivModOpConversionBase;
  LogicalResult
  matchAndRewrite(ModUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the divisor is a power of two.
    if (auto rhsConstantOp = adaptor.getRhs().getDefiningOp<hw::ConstantOp>())
      if (rhsConstantOp.getValue().isPowerOf2()) {
        // Extract lower bits.
        size_t extractAmount = rhsConstantOp.getValue().ceilLogBase2();
        size_t width = op.getType().getIntOrFloatBitWidth();
        Value lowerBits = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(), adaptor.getLhs(), 0, extractAmount);
        Value constZero = rewriter.create<hw::ConstantOp>(
            op.getLoc(), APInt::getZero(width - extractAmount));
        replaceOpWithNewOpAndCopyNamehint<comb::ConcatOp>(
            rewriter, op, op.getType(), ArrayRef<Value>{constZero, lowerBits});
        return success();
      }

    // When rhs is not power of two and the number of unknown bits are small,
    // create a mux tree that emulates all possible cases.
    return emulateBinaryOpForUnknownBits(
        rewriter, maxEmulationUnknownBits, op,
        [](const APInt &lhs, const APInt &rhs) {
          // Division by zero is undefined, just return zero.
          if (rhs.isZero())
            return APInt::getZero(rhs.getBitWidth());
          return lhs.urem(rhs);
        });
  }
};

struct CombDivSOpConversion : DivModOpConversionBase<DivSOp> {
  using DivModOpConversionBase<DivSOp>::DivModOpConversionBase;

  LogicalResult
  matchAndRewrite(DivSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently only lower with emulation.
    // TODO: Implement a signed division lowering at least for power of two.
    return emulateBinaryOpForUnknownBits(
        rewriter, maxEmulationUnknownBits, op,
        [](const APInt &lhs, const APInt &rhs) {
          // Division by zero is undefined, just return zero.
          if (rhs.isZero())
            return APInt::getZero(rhs.getBitWidth());
          return lhs.sdiv(rhs);
        });
  }
};

struct CombModSOpConversion : DivModOpConversionBase<ModSOp> {
  using DivModOpConversionBase<ModSOp>::DivModOpConversionBase;
  LogicalResult
  matchAndRewrite(ModSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently only lower with emulation.
    // TODO: Implement a signed modulus lowering at least for power of two.
    return emulateBinaryOpForUnknownBits(
        rewriter, maxEmulationUnknownBits, op,
        [](const APInt &lhs, const APInt &rhs) {
          // Division by zero is undefined, just return zero.
          if (rhs.isZero())
            return APInt::getZero(rhs.getBitWidth());
          return lhs.srem(rhs);
        });
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
      replaceOpWithNewOpAndCopyNamehint<aig::AndInverterOp>(
          rewriter, op, xorBits, allInverts);
      return success();
    }

    case ICmpPredicate::ne:
    case ICmpPredicate::cne: {
      // a != b  ==> (a[n] ^ b[n]) | (a[n-1] ^ b[n-1]) | ...
      auto xorOp = rewriter.createOrFold<comb::XorOp>(op.getLoc(), lhs, rhs);
      replaceOpWithNewOpAndCopyNamehint<comb::OrOp>(
          rewriter, op, extractBits(rewriter, xorOp), true);
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
      replaceOpAndCopyNamehint(rewriter, op,
                               constructUnsignedCompare(op, aBits, bBits,
                                                        isLess, includeEq,
                                                        rewriter));
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
      replaceOpWithNewOpAndCopyNamehint<comb::MuxOp>(
          rewriter, op, signsDiffer, diffSignResult, sameSignResult);
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
    replaceOpWithNewOpAndCopyNamehint<comb::XorOp>(
        rewriter, op, extractBits(rewriter, adaptor.getInput()), true);
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

    replaceOpAndCopyNamehint(rewriter, op, result);
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

    replaceOpAndCopyNamehint(rewriter, op, result);
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

    // NOTE: The max shift amount is width - 1 because the sign bit is
    // already shifted out.
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

    replaceOpAndCopyNamehint(rewriter, op, result);
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
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::maxEmulationUnknownBits;
};
} // namespace

static void
populateCombToAIGConversionPatterns(RewritePatternSet &patterns,
                                    uint32_t maxEmulationUnknownBits) {
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

  // Add div/mod patterns with a threshold given by the pass option.
  patterns.add<CombDivUOpConversion, CombModUOpConversion, CombDivSOpConversion,
               CombModSOpConversion>(patterns.getContext(),
                                     maxEmulationUnknownBits);
}

void ConvertCombToAIGPass::runOnOperation() {
  ConversionTarget target(getContext());

  // Comb is source dialect.
  target.addIllegalDialect<comb::CombDialect>();
  // Keep data movement operations like Extract, Concat and Replicate.
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::BitcastOp, hw::ConstantOp>();

  // Treat array operations as illegal. Strictly speaking, other than array
  // get operation with non-const index are legal in AIG but array types
  // prevent a bunch of optimizations so just lower them to integer
  // operations. It's required to run HWAggregateToComb pass before this pass.
  target.addIllegalOp<hw::ArrayGetOp, hw::ArrayCreateOp, hw::ArrayConcatOp,
                      hw::AggregateConstantOp>();

  // AIG is target dialect.
  target.addLegalDialect<aig::AIGDialect>();

  // If additional legal ops are specified, add them to the target.
  if (!additionalLegalOps.empty())
    for (const auto &opName : additionalLegalOps)
      target.addLegalOp(OperationName(opName, &getContext()));

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns, maxEmulationUnknownBits);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
