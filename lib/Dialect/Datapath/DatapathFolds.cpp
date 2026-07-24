//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>

using namespace mlir;
using namespace circt;
using namespace datapath;
using namespace matchers;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//
static FailureOr<size_t> calculateNonZeroBits(Value operand,
                                              size_t numResults) {
  // If the extracted bits are all known, then return the result.
  auto knownBits = comb::computeKnownBits(operand);
  if (knownBits.isUnknown())
    return failure(); // Skip if we don't know anything about the bits

  size_t nonZeroBits = operand.getType().getIntOrFloatBitWidth() -
                       knownBits.Zero.countLeadingOnes();

  // If all bits non-zero we will not reduce the number of results
  if (nonZeroBits == numResults)
    return failure();

  return nonZeroBits;
}

// This pattern commonly arrises when inverting zext: ~zext(x) = {1,...1, ~x}
// Check if the operand is {ones, base} and return the unextended operand:
static FailureOr<Value> isOneExt(Value operand) {
  // Check if operand is a concat operation
  auto concatOp = operand.getDefiningOp<comb::ConcatOp>();
  if (!concatOp)
    return failure();

  auto operands = concatOp.getOperands();
  // ConcatOp must have exactly 2 operands
  if (operands.size() != 2)
    return failure();

  APInt value;
  if (matchPattern(operands[0], m_ConstantInt(&value)) && value.isAllOnes())
    // Return the base unextended value
    return success(operands[1]);

  return failure();
}

// zext(input<<trailingZeros) to targetWidth
static Value zeroPad(PatternRewriter &rewriter, Location loc, Value input,
                     size_t targetWidth, size_t trailingZeros) {
  assert(trailingZeros > 0 && "zeroPad called with zero trailing zeros");
  auto trailingZerosValue =
      hw::ConstantOp::create(rewriter, loc, APInt::getZero(trailingZeros));
  auto padTrailing = comb::ConcatOp::create(
      rewriter, loc, ValueRange{input, trailingZerosValue});
  return comb::createZExt(rewriter, loc, padTrailing, targetWidth);
}

//===----------------------------------------------------------------------===//
// Compress Operation
//===----------------------------------------------------------------------===//
// Check that all compressor results are included in this list of operands
// If not we must take care as manipulating compressor results independently
// could easily introduce a non-equivalent representation.
static bool areAllCompressorResultsSummed(ValueRange compressResults,
                                          ValueRange operands) {
  for (auto result : compressResults) {
    if (!llvm::is_contained(operands, result))
      return false;
  }
  return true;
}

struct FoldCompressIntoCompress
    : public OpRewritePattern<datapath::CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  // compress(compress(a,b,c), add(e,f)) -> compress(a,b,c,e,f)
  LogicalResult matchAndRewrite(datapath::CompressOp compOp,
                                PatternRewriter &rewriter) const override {
    auto operands = compOp.getOperands();
    llvm::SmallSetVector<Value, 8> processedCompressorResults;
    SmallVector<Value, 8> newCompressOperands;

    for (Value operand : operands) {

      // Skip if already processed this compressor
      if (processedCompressorResults.contains(operand))
        continue;

      // If the operand has multiple uses, we do not fold it into a compress
      // operation, so we treat it as a regular operand to maintain sharing.
      if (!operand.hasOneUse()) {
        newCompressOperands.push_back(operand);
        continue;
      }

      // Found a compress op - add its operands to our new list
      if (auto compressOp = operand.getDefiningOp<datapath::CompressOp>()) {

        // Check that all results of the compressor are summed in this add
        if (!areAllCompressorResultsSummed(compressOp.getResults(), operands))
          return failure();

        llvm::append_range(newCompressOperands, compressOp.getOperands());
        // Only process each compressor once as multiple operands will point
        // to the same defining operation
        processedCompressorResults.insert(compressOp.getResults().begin(),
                                          compressOp.getResults().end());
        continue;
      }

      if (auto addOp = operand.getDefiningOp<comb::AddOp>()) {
        llvm::append_range(newCompressOperands, addOp.getOperands());
        continue;
      }

      // Regular operand - just add it to our list
      newCompressOperands.push_back(operand);
    }

    // If unable to collect more operands then this pattern doesn't apply
    if (newCompressOperands.size() <= compOp.getNumOperands())
      return failure();

    // Create a new CompressOp with all collected operands
    rewriter.replaceOpWithNewOp<datapath::CompressOp>(
        compOp, newCompressOperands, compOp.getNumResults());
    return success();
  }
};

struct FoldAddIntoCompress : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  // add(compress(a,b,c),d) -> add(compress(a,b,c,d))
  // FIXME: This should be implemented as a canonicalization pattern for
  // compress op. Currently `hasDatapathOperand` flag prevents introducing
  // datapath operations from comb operations.
  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // comb.add canonicalization patterns handle folding add operations
    if (addOp.getNumOperands() <= 2)
      return failure();

    // Get operands of the AddOp
    auto operands = addOp.getOperands();
    llvm::SmallSetVector<Value, 8> processedCompressorResults;
    SmallVector<Value, 8> newCompressOperands;
    // Only construct compressor if can form a larger compressor than what
    // is currently an input of this add. Also check that there is at least
    // one datapath operand.
    bool shouldFold = false, hasDatapathOperand = false;

    for (Value operand : operands) {

      // Skip if already processed this compressor
      if (processedCompressorResults.contains(operand))
        continue;

      if (auto *op = operand.getDefiningOp())
        if (isa_and_nonnull<datapath::DatapathDialect>(op->getDialect()))
          hasDatapathOperand = true;

      // If the operand has multiple uses, we do not fold it into a compress
      // operation, so we treat it as a regular operand.
      if (!operand.hasOneUse()) {
        shouldFold |= !newCompressOperands.empty();
        newCompressOperands.push_back(operand);
        continue;
      }

      // Found a compress op - add its operands to our new list
      if (auto compressOp = operand.getDefiningOp<datapath::CompressOp>()) {

        // Check that all results of the compressor are summed in this add
        if (!areAllCompressorResultsSummed(compressOp.getResults(), operands))
          return failure();

        // If we've already added one operand it should be folded
        shouldFold |= !newCompressOperands.empty();
        llvm::append_range(newCompressOperands, compressOp.getOperands());
        // Only process each compressor once
        processedCompressorResults.insert(compressOp.getResults().begin(),
                                          compressOp.getResults().end());
        continue;
      }

      if (auto addOp = operand.getDefiningOp<comb::AddOp>()) {
        shouldFold |= !newCompressOperands.empty();
        llvm::append_range(newCompressOperands, addOp.getOperands());
        continue;
      }

      // Regular operand - just add it to our list
      shouldFold |= !newCompressOperands.empty();
      newCompressOperands.push_back(operand);
    }

    // Only fold if we have constructed a larger compressor than what was
    // already there
    if (!shouldFold || !hasDatapathOperand)
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = datapath::CompressOp::create(rewriter, addOp.getLoc(),
                                                      newCompressOperands, 2);

    // Replace the original AddOp with a new add(compress(inputs))
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
    return success();
  }
};

// compress(..., sext(x),...) ->
// compress(..., zext({~x[p-1], x[p-2:0]}), (-1) << (width(x)-1), ...)
// Justification:
// sext(x) = {x[p-1], x[p-1], ...,  x[p-1], x[p-2], ..., x[0]} =
//         = {       0,    0, ..., ~x[p-1], x[p-2], ..., x[0]} +
//           {       1,    1, ...,       1,      0, ...,    0} =
//         = zext({~x[p-1], x[p-2], ..., x[0]}) + ((-1) << (width(x)-1))
//
// Note that we are adding arguments to the compressor, but we are reducing the
// number of unknown bits in the compressor array
struct SextCompress : public OpRewritePattern<CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompressOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto opSize = inputs[0].getType().getIntOrFloatBitWidth();
    auto size = inputs.size();

    APInt value;
    SmallVector<Value> newInputs;
    for (auto input : inputs) {
      Value replBits;
      // Check for sext of the inverted value
      if (!matchPattern(input, comb::m_SextBy(m_Any(&replBits)))) {
        newInputs.push_back(input);
        continue;
      }
      auto baseWidth = opSize - replBits.getType().getIntOrFloatBitWidth();
      auto sextInput =
          comb::ExtractOp::create(rewriter, op.getLoc(), input, 0, baseWidth);

      // Need a separate sign-bit that gets extended by at least two bits to
      // be beneficial
      if (baseWidth <= 1 || (opSize - baseWidth) <= 1) {
        newInputs.push_back(input);
        continue;
      }

      // x[p-2:0]
      auto base = comb::ExtractOp::create(rewriter, op.getLoc(), sextInput, 0,
                                          baseWidth - 1);
      // x[p-1]
      auto signBit = comb::ExtractOp::create(rewriter, op.getLoc(), sextInput,
                                             baseWidth - 1, 1);
      auto invSign =
          comb::createOrFoldNot(rewriter, op.getLoc(), signBit, true);
      // {~x[p-1], x[p-2:0]}
      auto newOp = comb::ConcatOp::create(rewriter, op.getLoc(),
                                          ValueRange{invSign, base});
      auto newOpZExt = comb::createZExt(rewriter, op.getLoc(), newOp, opSize);

      newInputs.push_back(newOpZExt);

      // (-1) << (width(x)-1)
      auto ones = APInt::getAllOnes(opSize);
      auto correction = hw::ConstantOp::create(rewriter, op.getLoc(),
                                               ones << (baseWidth - 1));

      newInputs.push_back(correction);
    }

    // If no sext inputs have not updated any arguments
    if (newInputs.size() == size)
      return failure();

    auto newCompress = datapath::CompressOp::create(
        rewriter, op.getLoc(), newInputs, op.getNumResults());
    rewriter.replaceOp(op, newCompress.getResults());
    return success();
  }
};

// compress(..., oneExt(x),...) ->
// compress(..., zext(x), (-1) << (width(x)-1), ...)
// Justification:
//           {1, 1, ..., 1, x}
//         = zext(x) + ((-1) << (width(x)-1))
//
// Note that we are adding arguments to the compressor, but these can be
// constant folded should other constants arise
//
// A pattern encountered when we convert subtraction to addition:
// zext(a)-zext(b) = zext(a) + ~zext(b) + 1
//                 = zext(a) + oneExt(~b) + 1
// TODO: use knownBits to extract all constant ones
struct OnesExtCompress : public OpRewritePattern<CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompressOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto opType = inputs[0].getType();
    auto opSize = opType.getIntOrFloatBitWidth();

    SmallVector<Value> newInputs;
    for (auto input : inputs) {
      // Check for replication of ones leading
      auto baseInput = isOneExt(input);
      if (failed(baseInput)) {
        newInputs.push_back(input);
        continue;
      }

      // Separate {ones, x} -> zext(x) + (ones << baseWidth)
      auto newOp = comb::createZExt(rewriter, op.getLoc(), *baseInput, opSize);
      newInputs.push_back(newOp);

      APInt ones = APInt::getAllOnes(opSize);
      auto baseWidth = baseInput->getType().getIntOrFloatBitWidth();
      auto correction =
          hw::ConstantOp::create(rewriter, op.getLoc(), ones << baseWidth);
      newInputs.push_back(correction);
    }

    if (newInputs.size() == inputs.size())
      return failure();

    auto newCompress = datapath::CompressOp::create(
        rewriter, op.getLoc(), newInputs, op.getNumResults());
    rewriter.replaceOp(op, newCompress.getResults());
    return success();
  }
};

struct ConstantFoldCompress : public OpRewritePattern<CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompressOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto size = inputs.size();

    APInt value;

    // compress(..., 0) -> compress(...) -- identity
    if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isZero()) {

      // If only reducing by one row and contains zero - pass through operands
      if (size - 1 == op.getNumResults()) {
        rewriter.replaceOp(op, inputs.drop_back());
        return success();
      }

      // Default create a compressor with fewer arguments
      rewriter.replaceOpWithNewOp<CompressOp>(op, inputs.drop_back(),
                                              op.getNumResults());
      return success();
    }

    APInt value1, value2;
    // compress(...c1, c2) -> compress(..., c1+c2)
    assert(size >= 3 &&
           "compress op has 3 or more operands ensured by a verifier");
    if (matchPattern(inputs.back(), m_ConstantInt(&value1)) &&
        matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {

      SmallVector<Value> newInputs(inputs.drop_back(2));
      auto summedValue = value1 + value2;
      auto constOp = hw::ConstantOp::create(rewriter, op.getLoc(), summedValue);
      newInputs.push_back(constOp);
      // If reducing by one row and constant folding - pass through operands
      if (size - 1 == op.getNumResults()) {
        rewriter.replaceOp(op, newInputs);
        return success();
      }

      // Default create a compressor with fewer arguments
      rewriter.replaceOpWithNewOp<CompressOp>(op, newInputs,
                                              op.getNumResults());
      return success();
    }

    return failure();
  }
};

void CompressOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<FoldCompressIntoCompress, FoldAddIntoCompress,
              ConstantFoldCompress, SextCompress, OnesExtCompress>(context);
}

//===----------------------------------------------------------------------===//
// Partial Product Operation
//===----------------------------------------------------------------------===//
struct ReduceNumPartialProducts : public OpRewritePattern<PartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // pp(concat(0,a), concat(0,b)) -> reduced number of results
  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {
    auto operands = op.getOperands();
    unsigned inputWidth = operands[0].getType().getIntOrFloatBitWidth();

    // TODO: implement a constant multiplication for the PartialProductOp

    auto op0NonZeroBits = calculateNonZeroBits(operands[0], op.getNumResults());
    auto op1NonZeroBits = calculateNonZeroBits(operands[1], op.getNumResults());

    if (failed(op0NonZeroBits) || failed(op1NonZeroBits))
      return failure();

    // Need the +1 for the carry-out
    size_t maxNonZeroBits = std::max(*op0NonZeroBits, *op1NonZeroBits);

    auto newPP = datapath::PartialProductOp::create(
        rewriter, op.getLoc(), op.getOperands(), maxNonZeroBits);

    auto zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                       APInt::getZero(inputWidth));

    // Collect newPP results and pad with zeros if needed
    SmallVector<Value> newResults(newPP.getResults().begin(),
                                  newPP.getResults().end());

    newResults.append(op.getNumResults() - newResults.size(), zero);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct SignedPartialProducts : public OpRewritePattern<PartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // Based on the classical Baugh-Wooley algorithm for signed mulitplication.
  // Paper: A Two's Complement Parallel Array Multiplication Algorithm
  //
  // Consider a p-bit by q-bit signed multiplier - producing a (p+q)-bit result:
  // a_sign = a[p-1], a_mag = a[p-2:0],
  // b_sign = b[q-1], b_mag = b[q-2:0]
  // sext(a) * sext(b) = a_mag * b_mag                    [unsigned product]
  //                     - 2^(p-1) * a_sign * b_mag       [sign correction]
  //                     - 2^(q-1) * b_sign * a_mag       [sign correction]
  //                     + 2^(p+q-2) * a_sign * b_sign    [sign * sign]
  //
  // We implement optimizations to turn the subtractions into bitwise
  // negations with constant corrections that can be folded together.
  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {
    // Booth encoding will automatically handle signed multiplications
    if (comb::shouldUseBoothEncoding(op.getLhs(), op.getRhs()))
      return failure();

    auto inputWidth = op.getLhs().getType().getIntOrFloatBitWidth();
    Value lhsReplBits;
    Value rhsReplBits;
    if (!matchPattern(op.getLhs(), comb::m_SextBy(m_Any(&lhsReplBits))) ||
        !matchPattern(op.getRhs(), comb::m_SextBy(m_Any(&rhsReplBits))))
      return failure();

    size_t lhsWidth =
        inputWidth - lhsReplBits.getType().getIntOrFloatBitWidth();
    size_t rhsWidth =
        inputWidth - rhsReplBits.getType().getIntOrFloatBitWidth();
    // Subtract 1 as will handle sign-bit separately
    size_t maxRows = std::max(lhsWidth, rhsWidth) - 1;

    // TODO: add support for different width inputs
    // Need to have a sign bit in both inputs
    if (lhsWidth != rhsWidth || lhsWidth <= 1 || rhsWidth <= 1)
      return failure();

    // No further reduction possible
    if (maxRows >= op.getNumResults())
      return failure();

    // Pull off the sign bits
    auto lhsBaseWidth = lhsWidth - 1;
    auto rhsBaseWidth = rhsWidth - 1;
    auto lhsSignBit = comb::ExtractOp::create(rewriter, op.getLoc(),
                                              op.getLhs(), lhsBaseWidth, 1);
    auto rhsSignBit = comb::ExtractOp::create(rewriter, op.getLoc(),
                                              op.getRhs(), rhsBaseWidth, 1);
    auto lhsBase = comb::ExtractOp::create(rewriter, op.getLoc(), op.getLhs(),
                                           0, lhsBaseWidth);
    auto rhsBase = comb::ExtractOp::create(rewriter, op.getLoc(), op.getRhs(),
                                           0, rhsBaseWidth);

    // Create the unsigned partial product of the unextended inputs
    auto lhsBaseZext =
        comb::createZExt(rewriter, op.getLoc(), lhsBase, inputWidth);
    auto rhsBaseZext =
        comb::createZExt(rewriter, op.getLoc(), rhsBase, inputWidth);
    auto newPP = datapath::PartialProductOp::create(
        rewriter, op.getLoc(), ValueRange{lhsBaseZext, rhsBaseZext}, maxRows);

    // Optimization (similar for second sign correction), ext to (p+q)-bits:
    // -2^(p-1)*sign(lhs)*rhsBase = ~((sign(lhs) * rhsBase) << (p-1)) + 1
    //                            = (~(replicate(sign(lhs)) & rhsBase)) << (p-1)
    //                            + (-1) << (p+q-2)      [msb correction]
    //                            + (1<<(p-1)) - 1 + 1   [lsb correction]

    // Create ~(replicate(sign(lhs)) & rhsBase)
    auto lhsSignReplicate = comb::ReplicateOp::create(rewriter, op.getLoc(),
                                                      lhsSignBit, rhsBaseWidth);
    auto lhsSignAndRhs =
        comb::AndOp::create(rewriter, op.getLoc(), lhsSignReplicate, rhsBase);
    auto lhsSignCorrection =
        comb::createOrFoldNot(rewriter, op.getLoc(), lhsSignAndRhs, true);

    // zext({lhsSignCorrection, lhsBaseWidth{1'b0}})
    auto alignLhsSignCorrection = zeroPad(
        rewriter, op.getLoc(), lhsSignCorrection, inputWidth, lhsBaseWidth);

    // Create ~(replicate(sign(rhs)) & lhsBase)
    auto rhsSignReplicate = comb::ReplicateOp::create(rewriter, op.getLoc(),
                                                      rhsSignBit, lhsBaseWidth);
    auto rhsSignAndLhs =
        comb::AndOp::create(rewriter, op.getLoc(), rhsSignReplicate, lhsBase);
    auto rhsSignCorrection =
        comb::createOrFoldNot(rewriter, op.getLoc(), rhsSignAndLhs, true);

    // zext({rhsSignCorrection, rhsBaseWidth{1'b0}})
    auto alignRhsSignCorrection = zeroPad(
        rewriter, op.getLoc(), rhsSignCorrection, inputWidth, rhsBaseWidth);

    // 2^(p+q-2) * sign(lhs) * sign(rhs) = (sign(lhs) & sign(rhs)) << (p+q-2)
    // Create sign(lhs) & sign(rhs)
    auto signAnd =
        comb::AndOp::create(rewriter, op.getLoc(), lhsSignBit, rhsSignBit);
    // zext({sign(lhs) & sign(rhs), lhsBaseWidth+rhsBaseWidth{1'b0}})
    auto alignSignAndZext = zeroPad(rewriter, op.getLoc(), signAnd, inputWidth,
                                    lhsBaseWidth + rhsBaseWidth);

    // Gather constant corrections together (once for each sign correction):
    // (-1) << (p+q-2) + (1<<(p-1)) - 1 + 1
    auto ones = APInt::getAllOnes(inputWidth);
    auto lowerLhs = APInt::getOneBitSet(inputWidth, lhsBaseWidth);
    auto lowerRhs = APInt::getOneBitSet(inputWidth, rhsBaseWidth);
    auto msbCorrection = ones << (lhsBaseWidth + rhsBaseWidth);
    auto correction = lowerLhs + lowerRhs + 2 * msbCorrection;

    auto constantCorrection =
        hw::ConstantOp::create(rewriter, op.getLoc(), correction);

    auto zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                       APInt::getZero(inputWidth));
    // Collect newPP results and pad with zeros if needed
    SmallVector<Value> newResults(newPP.getResults().begin(),
                                  newPP.getResults().end());

    // ~(replicate(sign(lhs)) & rhsBase) * 2^(p-1)
    newResults.push_back(alignLhsSignCorrection);
    // ~(replicate(sign(rhs)) & lhsBase) * 2^(q-1)
    newResults.push_back(alignRhsSignCorrection);
    // sign(lhs)*sign(rhs) * 2^(p+q-2)
    newResults.push_back(alignSignAndZext);
    // Constant correction
    newResults.push_back(constantCorrection);
    // Zero pad if necessary
    newResults.append(op.getNumResults() - newResults.size(), zero);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct PosPartialProducts : public OpRewritePattern<PartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // pp(add(a,b),c) -> pos_pp(a,b,c)
  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {
    auto width = op.getType(0).getIntOrFloatBitWidth();

    assert(op.getNumOperands() == 2);

    // Detect if any input is an AddOp
    auto lhsAdder = op.getOperand(0).getDefiningOp<comb::AddOp>();
    auto rhsAdder = op.getOperand(1).getDefiningOp<comb::AddOp>();
    if ((lhsAdder && rhsAdder) || !(lhsAdder || rhsAdder))
      return failure();
    auto addInput = lhsAdder ? lhsAdder : rhsAdder;
    auto otherInput = lhsAdder ? op.getOperand(1) : op.getOperand(0);

    if (addInput->getNumOperands() != 2)
      return failure();

    Value addend0 = addInput->getOperand(0);
    Value addend1 = addInput->getOperand(1);

    rewriter.replaceOpWithNewOp<PosPartialProductOp>(
        op, ValueRange{addend0, addend1, otherInput}, width);
    return success();
  }
};

void PartialProductOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results
      .add<ReduceNumPartialProducts, SignedPartialProducts, PosPartialProducts>(
          context);
}

//===----------------------------------------------------------------------===//
// Pos Partial Product Operation
//===----------------------------------------------------------------------===//
struct ReduceNumPosPartialProducts
    : public OpRewritePattern<PosPartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // pos_pp(concat(0,a), concat(0,b), c) -> reduced number of results
  LogicalResult matchAndRewrite(PosPartialProductOp op,
                                PatternRewriter &rewriter) const override {
    unsigned inputWidth = op.getAddend0().getType().getIntOrFloatBitWidth();
    auto addend0NonZero =
        calculateNonZeroBits(op.getAddend0(), op.getNumResults());
    auto addend1NonZero =
        calculateNonZeroBits(op.getAddend1(), op.getNumResults());

    if (failed(addend0NonZero) || failed(addend1NonZero))
      return failure();

    // Need the +1 for the carry-out
    size_t maxNonZeroBits = std::max(*addend0NonZero, *addend1NonZero) + 1;

    if (maxNonZeroBits >= op.getNumResults())
      return failure();

    auto newPP = datapath::PosPartialProductOp::create(
        rewriter, op.getLoc(), op.getOperands(), maxNonZeroBits);

    auto zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                       APInt::getZero(inputWidth));

    // Collect newPP results and pad with zeros if needed
    SmallVector<Value> newResults(newPP.getResults().begin(),
                                  newPP.getResults().end());

    newResults.append(op.getNumResults() - newResults.size(), zero);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct SignedPosPartialProducts : public OpRewritePattern<PosPartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // Inspired by the classical Baugh-Wooley algorithm for signed mulitplication.
  // Paper: A Two's Complement Parallel Array Multiplication Algorithm
  //
  // Consider a p-bit signed pos - producing a 2p-bit result:
  // a_sign = a[p-1], a_mag = a[p-2:0],
  // b_sign = b[p-1], b_mag = b[p-2:0]
  // (sext(a) + sext(b)) * c = (a_mag + b_mag) * c               [unsigned pos]
  //                           - 2^(p-1) * (a_sign + b_sign) * c [sign correct]
  //
  // We implement optimizations to turn the subtractions into bitwise
  // negations with constant corrections that can be folded together.
  LogicalResult matchAndRewrite(PosPartialProductOp op,
                                PatternRewriter &rewriter) const override {

    auto a = op.getAddend0();
    auto b = op.getAddend1();
    auto c = op.getMultiplicand();
    auto loc = op.getLoc();
    auto inputWidth = a.getType().getIntOrFloatBitWidth();
    Value aReplBits;
    Value bReplBits;
    if (!matchPattern(a, comb::m_Sext(m_Any(&aReplBits))) ||
        !matchPattern(b, comb::m_Sext(m_Any(&bReplBits))))
      return failure();

    size_t aWidth = inputWidth - aReplBits.getType().getIntOrFloatBitWidth();
    size_t bWidth = inputWidth - bReplBits.getType().getIntOrFloatBitWidth();

    // TODO: add support for different width inputs
    // Need to have a sign bit in both inputs
    if (aWidth != bWidth || aWidth <= 1 || bWidth <= 1)
      return failure();

    // Minus 1 as we handle the sign-bits separately
    auto maxPartialProducts = aWidth - 1;
    // No further reduction possible - already reduced to min partial products
    if (maxPartialProducts >= op.getNumResults())
      return failure();

    // Pull off the sign bits
    auto baseWidth = aWidth - 1;
    auto aSign = comb::ExtractOp::create(rewriter, loc, a, baseWidth, 1);
    auto bSign = comb::ExtractOp::create(rewriter, loc, b, baseWidth, 1);
    auto aBase = comb::ExtractOp::create(rewriter, loc, a, 0, baseWidth);
    auto bBase = comb::ExtractOp::create(rewriter, loc, b, 0, baseWidth);

    // Create the unsigned pos partial product of the unextended inputs
    auto aBaseZext = comb::createZExt(rewriter, loc, aBase, inputWidth);
    auto bBaseZext = comb::createZExt(rewriter, loc, bBase, inputWidth);
    auto newPP = datapath::PosPartialProductOp::create(
        rewriter, loc, ValueRange{aBaseZext, bBaseZext, op.getMultiplicand()},
        maxPartialProducts);

    // Optimization:
    // -2^(p-1)*(a_sign + b_sign) * c =
    // ~(((a_sign & b_sign)*2c | (a_sign ^ b_sign)*c)  << (p-1)) + 1
    //          CARRY                   SAVE

    auto cWidth = c.getType().getIntOrFloatBitWidth();
    auto carry = rewriter.createOrFold<comb::AndOp>(loc, aSign, bSign);
    auto save = rewriter.createOrFold<comb::XorOp>(loc, aSign, bSign);
    auto one = hw::ConstantOp::create(rewriter, loc, APInt(cWidth, 1));
    auto twoC = rewriter.createOrFold<comb::ShlOp>(loc, c, one);
    auto replSave = rewriter.createOrFold<comb::ReplicateOp>(loc, save, cWidth);
    auto replCarry =
        rewriter.createOrFold<comb::ReplicateOp>(loc, carry, cWidth);
    auto carryAnd = rewriter.createOrFold<comb::AndOp>(loc, replCarry, twoC);
    auto saveAnd = rewriter.createOrFold<comb::AndOp>(loc, replSave, c);
    auto ppRow = rewriter.createOrFold<comb::OrOp>(loc, carryAnd, saveAnd);
    auto shiftBy =
        hw::ConstantOp::create(rewriter, loc, APInt(cWidth, baseWidth));
    auto ppRowShift = rewriter.createOrFold<comb::ShlOp>(loc, ppRow, shiftBy);
    auto ppRowNot = comb::createOrFoldNot(rewriter, loc, ppRowShift);

    // Collect newPP results and pad with zeros if needed
    SmallVector<Value> newResults(newPP.getResults().begin(),
                                  newPP.getResults().end());

    newResults.push_back(ppRowNot);
    newResults.push_back(one); // Constant correction for the sign correction
    // Zero pad if necessary
    auto zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                       APInt::getZero(inputWidth));
    newResults.append(op.getNumResults() - newResults.size(), zero);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

void PosPartialProductOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ReduceNumPosPartialProducts, SignedPosPartialProducts>(context);
}
