//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>

#define DEBUG_TYPE "datapath-to-comb"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace datapath;

// A wrapper for comb::extractBits that returns a SmallVector<Value>.
static SmallVector<Value> extractBits(OpBuilder &builder, Value val) {
  SmallVector<Value> bits;
  comb::extractBits(builder, val, bits);
  return bits;
}

// Check whether a value is zero-extended or sign-extended - and return the
// unextended base value and whether it was sign-extended.
static std::pair<bool, Value> getBaseOfExt(PatternRewriter &rewriter,
                                           Location loc, Value val) {

  Value replBits;
  // Check for zext
  if (matchPattern(val, comb::m_ZextBy(mlir::matchers::m_Any(&replBits)))) {
    auto baseWidth = val.getType().getIntOrFloatBitWidth() -
                     replBits.getType().getIntOrFloatBitWidth();
    auto valBase =
        rewriter.createOrFold<comb::ExtractOp>(loc, val, 0, baseWidth);
    return {false, valBase};
  }

  // Check for sext of the value
  if (matchPattern(val, comb::m_SextBy(mlir::matchers::m_Any(&replBits)))) {
    auto baseWidth = val.getType().getIntOrFloatBitWidth() -
                     replBits.getType().getIntOrFloatBitWidth();
    auto valBase =
        rewriter.createOrFold<comb::ExtractOp>(loc, val, 0, baseWidth);
    return {true, valBase};
  }

  // Not extended, return original value
  return {false, val};
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
// Replace compressor by an adder of the inputs and zero for the other results:
// compress(a,b,c,d) -> {a+b+c+d, 0}
// Facilitates use of downstream compression algorithms e.g. Yosys
struct DatapathCompressOpAddConversion : mlir::OpRewritePattern<CompressOp> {
  using mlir::OpRewritePattern<CompressOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(CompressOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();
    // Sum all the inputs - set that to result value 0
    auto addOp = comb::AddOp::create(rewriter, loc, inputs, true);
    // Replace remaining results with zeros
    auto zeroOp = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
    SmallVector<Value> results(op.getNumResults() - 1, zeroOp);
    results.push_back(addOp);
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Replace compressor by a wallace tree of full-adders
struct DatapathCompressOpConversion : mlir::OpRewritePattern<CompressOp> {
  DatapathCompressOpConversion(MLIRContext *context,
                               synth::IncrementalLongestPathAnalysis *analysis)
      : mlir::OpRewritePattern<CompressOp>(context), analysis(analysis) {}

  LogicalResult
  matchAndRewrite(CompressOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();

    SmallVector<SmallVector<Value>> addends;
    for (auto input : inputs) {
      addends.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    }

    // Compressor tree reduction
    auto width = inputs[0].getType().getIntOrFloatBitWidth();
    auto targetAddends = op.getNumResults();
    datapath::CompressorTree comp(width, addends, loc);

    if (analysis) {
      // Update delay information with arrival times
      if (failed(comp.withInputDelays(
              [&](Value v) { return analysis->getMaxDelay(v, 0); })))
        return failure();
    }

    rewriter.replaceOp(op, comp.compressToHeight(rewriter, targetAddends));
    return success();
  }

private:
  synth::IncrementalLongestPathAnalysis *analysis = nullptr;
};

struct DatapathPartialProductOpConversion : OpRewritePattern<PartialProductOp> {
  using OpRewritePattern<PartialProductOp>::OpRewritePattern;

  DatapathPartialProductOpConversion(MLIRContext *context, bool forceBooth)
      : OpRewritePattern<PartialProductOp>(context), forceBooth(forceBooth){};

  const bool forceBooth;

  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {

    Value a = op.getLhs();
    Value b = op.getRhs();
    unsigned width = a.getType().getIntOrFloatBitWidth();

    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(0), 0);
      return success();
    }

    // Square partial product array can be reduced to upper triangular array.
    // For example: AND array for a 4-bit squarer:
    //    0    0    0 a0a3 a0a2 a0a1 a0a0
    //    0    0 a1a3 a1a2 a1a1 a1a0    0
    //    0 a2a3 a2a2 a2a1 a2a0    0    0
    // a3a3 a3a2 a3a1 a3a0    0    0    0
    //
    // Can be reduced to:
    //    0    0 a0a3 a0a2 a0a1    0   a0
    //    0 a1a3 a1a2    0   a1    0    0
    // a2a3    0   a2    0    0    0    0
    //   a3    0    0    0    0    0    0
    if (a == b)
      return lowerSqrAndArray(rewriter, a, op, width);

    // Use result rows as a heuristic to guide partial product
    // implementation
    if (comb::boothEncode(a, b) || forceBooth)
      return lowerBoothArray(rewriter, a, b, op, width);
    else
      return lowerAndArray(rewriter, a, b, op, width);
  }

private:
  static LogicalResult lowerAndArray(PatternRewriter &rewriter, Value a,
                                     Value b, PartialProductOp op,
                                     unsigned width) {

    Location loc = op.getLoc();
    // Keep a as a bitvector - multiply by each digit of b
    SmallVector<Value> bBits = extractBits(rewriter, b);

    auto rowWidth = width;
    auto knownBitsA = comb::computeKnownBits(a);
    if (!knownBitsA.Zero.isZero()) {
      if (knownBitsA.Zero.countLeadingOnes() > 1) {
        rowWidth -= knownBitsA.Zero.countLeadingOnes();
        a = rewriter.createOrFold<comb::ExtractOp>(loc, a, 0, rowWidth);
      }
    }

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);
    // AND Array Construction:
    // partialProducts[i] = ({b[i],..., b[i]} & a) << i
    assert(op.getNumResults() <= width &&
           "Cannot return more results than the operator width");

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto repl =
          rewriter.createOrFold<comb::ReplicateOp>(loc, bBits[i], rowWidth);
      auto ppRow = rewriter.createOrFold<comb::AndOp>(loc, repl, a);
      if (rowWidth < width) {
        auto padding = width - rowWidth;
        auto zeroPad = hw::ConstantOp::create(rewriter, loc, APInt(padding, 0));
        ppRow = rewriter.createOrFold<comb::ConcatOp>(
            loc, ValueRange{zeroPad, ppRow}); // Pad to full width
      }

      if (i == 0) {
        partialProducts.push_back(ppRow);
        continue;
      }
      auto shiftBy = hw::ConstantOp::create(rewriter, loc, APInt(i, 0));
      auto ppAlign =
          comb::ConcatOp::create(rewriter, loc, ValueRange{ppRow, shiftBy});
      auto ppAlignTrunc = rewriter.createOrFold<comb::ExtractOp>(
          loc, ppAlign, 0, width); // Truncate to width+i bits
      partialProducts.push_back(ppAlignTrunc);
    }

    rewriter.replaceOp(op, partialProducts);
    return success();
  }

  static LogicalResult lowerSqrAndArray(PatternRewriter &rewriter, Value a,
                                        PartialProductOp op, unsigned width) {

    Location loc = op.getLoc();
    SmallVector<Value> aBits = extractBits(rewriter, a);

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);
    // AND Array Construction - reducing to upper triangle:
    // partialProducts[i] = ({a[i],..., a[i]} & a) << i
    // optimised to: {a[i] & a[n-1], ..., a[i] & a[i+1], 0, a[i], 0, ..., 0}
    assert(op.getNumResults() <= width &&
           "Cannot return more results than the operator width");
    auto zeroFalse = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      SmallVector<Value> row;
      row.reserve(width);

      if (2 * i >= width) {
        // Pad the remaining rows with zeros
        auto zeroWidth = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
        partialProducts.push_back(zeroWidth);
        continue;
      }

      if (i > 0) {
        auto shiftBy = hw::ConstantOp::create(rewriter, loc, APInt(2 * i, 0));
        row.push_back(shiftBy);
      }
      row.push_back(aBits[i]);

      // Track width of constructed row
      unsigned rowWidth = 2 * i + 1;
      if (rowWidth < width) {
        row.push_back(zeroFalse);
        ++rowWidth;
      }

      for (unsigned j = i + 1; j < width; ++j) {
        // Stop when we reach the required width
        if (rowWidth == width)
          break;

        // Otherwise pad with zeros or partial product bits
        ++rowWidth;
        // Number of results indicates number of non-zero bits in input
        if (j >= op.getNumResults()) {
          row.push_back(zeroFalse);
          continue;
        }

        auto ppBit =
            rewriter.createOrFold<comb::AndOp>(loc, aBits[i], aBits[j]);
        row.push_back(ppBit);
      }
      std::reverse(row.begin(), row.end());
      auto ppRow = comb::ConcatOp::create(rewriter, loc, row);
      partialProducts.push_back(ppRow);
    }

    rewriter.replaceOp(op, partialProducts);
    return success();
  }

  static LogicalResult lowerBoothArray(PatternRewriter &rewriter, Value a,
                                       Value b, PartialProductOp op,
                                       unsigned width) {
    // TODO: sort a and b based on non-zero bits to encode the smaller input
    Location loc = op.getLoc();
    auto zeroFalse = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));

    auto [aSigned, aBase] = getBaseOfExt(rewriter, loc, op.getLhs());
    auto [bSigned, bBase] = getBaseOfExt(rewriter, loc, op.getRhs());

    auto aBaseWidth = aBase.getType().getIntOrFloatBitWidth();
    auto bBaseWidth = bBase.getType().getIntOrFloatBitWidth();

    // Detect leading zeros in multiplicand due to zero-extension
    // and truncate to reduce partial product bits {'0, a} * {'0, b}
    auto rowWidth = width;
    if (aBaseWidth < width) {
      // Retain one leading zero/sign-bit to represent 2*a
      rowWidth = aBaseWidth + 1;
      a = rewriter.createOrFold<comb::ExtractOp>(loc, a, 0, rowWidth);
    }

    // Booth encoding will select each row from {-2a, -1a, 0, 1a, 2a}
    Value twoAPre =
        rewriter.createOrFold<comb::ConcatOp>(loc, ValueRange{a, zeroFalse});
    Value twoA = rewriter.createOrFold<comb::ExtractOp>(
        loc, twoAPre, 0, rowWidth); // Truncate to width bits

    // Encode based on the bits of b

    SmallVector<Value> bBits = extractBits(rewriter, b);
    // Pad with two zeros - for case where there's no extensions
    bBits.push_back(zeroFalse); // Add a zero bit for the first row
    bBits.push_back(zeroFalse); // Add a zero bit for the last row

    // Retain two leading zeros as when b has an even number of bits we just
    // need to retain two leading zeros
    if (!bSigned)
      bBits.resize(bBaseWidth + 2);

    // If b is signed, we need to sign-extend with a single sign-bit
    if (bSigned)
      bBits.resize(bBaseWidth + 1);

    SmallVector<Value> partialProducts;
    partialProducts.reserve(op.getNumResults());

    // Booth encoding halves array height by grouping three bits at a time:
    // partialProducts[i] = a * (-2*b[2*i+1] + b[2*i] + b[2*i-1]) << 2*i
    // encNeg \approx (-2*b[2*i+1] + b[2*i] + b[2*i-1]) <= 0
    // encOne = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 1
    // encTwo = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 2
    SmallVector<Value> encNegs;
    Value encNegPrev;

    // For even width - additional row contains the final sign correction
    for (unsigned i = 0; i + 1 < bBits.size(); i += 2) {
      // Get Booth bits: b[i+1], b[i], b[i-1] (b[-1] = 0)
      Value bim1 = (i == 0) ? zeroFalse : bBits[i - 1];
      Value bi = bBits[i];
      Value bip1 = bBits[i + 1];

      // Is the encoding zero or negative (an approximation)
      Value encNeg = bip1;
      encNegs.push_back(encNeg); // Store for sign-extension optimisation
      // Is the encoding one = b[i] xor b[i-1]
      Value encOne = rewriter.createOrFold<comb::XorOp>(loc, bi, bim1, true);
      // Is the encoding two = (bip1 & ~bi & ~bim1) | (~bip1 & bi & bim1)
      Value constOne = hw::ConstantOp::create(rewriter, loc, APInt(1, 1));
      Value biInv = rewriter.createOrFold<comb::XorOp>(loc, bi, constOne, true);
      Value bip1Inv =
          rewriter.createOrFold<comb::XorOp>(loc, bip1, constOne, true);
      Value bim1Inv =
          rewriter.createOrFold<comb::XorOp>(loc, bim1, constOne, true);

      Value andLeft = rewriter.createOrFold<comb::AndOp>(
          loc, ValueRange{bip1Inv, bi, bim1}, true);
      Value andRight = rewriter.createOrFold<comb::AndOp>(
          loc, ValueRange{bip1, biInv, bim1Inv}, true);
      Value encTwo =
          rewriter.createOrFold<comb::OrOp>(loc, andLeft, andRight, true);

      Value encNegRepl =
          rewriter.createOrFold<comb::ReplicateOp>(loc, encNeg, rowWidth);
      Value encOneRepl =
          rewriter.createOrFold<comb::ReplicateOp>(loc, encOne, rowWidth);
      Value encTwoRepl =
          rewriter.createOrFold<comb::ReplicateOp>(loc, encTwo, rowWidth);

      // Select between 2*a or 1*a or 0*a
      Value selTwoA = rewriter.createOrFold<comb::AndOp>(loc, encTwoRepl, twoA);
      Value selOneA = rewriter.createOrFold<comb::AndOp>(loc, encOneRepl, a);
      Value magA =
          rewriter.createOrFold<comb::OrOp>(loc, selTwoA, selOneA, true);

      // Conditionally invert the row
      Value ppRow =
          rewriter.createOrFold<comb::XorOp>(loc, magA, encNegRepl, true);

      // No sign-correction in the first row
      if (i == 0) {
        partialProducts.push_back(ppRow);
        encNegPrev = encNeg;
        continue;
      }

      if (i == 2) {
        Value withSignCorrection = rewriter.createOrFold<comb::ConcatOp>(
            loc, ValueRange{ppRow, zeroFalse, encNegPrev});
        partialProducts.push_back(withSignCorrection);
        encNegPrev = encNeg;
        continue;
      }

      // Insert a sign-correction from the previous row
      // {ppRow, 0, encNegPrev} << (i-2)
      Value shiftBy = hw::ConstantOp::create(rewriter, loc, APInt(i - 2, 0));
      Value withSignCorrection = rewriter.createOrFold<comb::ConcatOp>(
          loc, ValueRange{ppRow, zeroFalse, encNegPrev, shiftBy});
      partialProducts.push_back(withSignCorrection);
      encNegPrev = encNeg;

      if (partialProducts.size() == op.getNumResults())
        break;
    }

    // Add the final sign-correction row for signed multiplication
    // Not necessary for unsigned multiplication as the final row is positive
    if (bSigned) {
      auto numPP = partialProducts.size();
      Value shiftByFinal =
          hw::ConstantOp::create(rewriter, loc, APInt((numPP - 1) * 2, 0));
      Value finalSignCorrection = rewriter.createOrFold<comb::ConcatOp>(
          loc, ValueRange{zeroFalse, encNegPrev, shiftByFinal});
      partialProducts.push_back(finalSignCorrection);
      encNegs.push_back(zeroFalse); // No sign-extension for the final row
    }

    // Sign-extension:
    // { s1, s1, s1, s1, s1, p1}
    // { s2, s2, s2,   p2      }
    // { s3,       p3          }
    // TODO: optimize by only replicating the sign bit once using
    // typical sign-extension trick - can be handled by separate
    // canonicalization patterns
    for (unsigned i = 0; i < partialProducts.size(); ++i) {
      auto ppRow = partialProducts[i];

      auto ppWidth = ppRow.getType().getIntOrFloatBitWidth();
      if (ppWidth < width) {
        auto padding = width - ppWidth;
        auto encNeg = encNegs[i];
        if (aSigned)
          encNeg = rewriter.createOrFold<comb::ExtractOp>(loc, ppRow,
                                                          ppWidth - 1, 1);

        // Replicate the encNeg bit for sign-extension
        Value encNegPad =
            rewriter.createOrFold<comb::ReplicateOp>(loc, encNeg, padding);
        ppRow = rewriter.createOrFold<comb::ConcatOp>(
            loc, ValueRange{encNegPad, ppRow}); // Pad to full width
      }

      // Truncate any excess bits
      ppWidth = ppRow.getType().getIntOrFloatBitWidth();
      if (ppWidth > width) {
        ppRow = rewriter.createOrFold<comb::ExtractOp>(loc, ppRow, 0, width);
      }
      partialProducts[i] = ppRow;
      assert(partialProducts[i].getType().getIntOrFloatBitWidth() == width &&
             "Expected sign-extended partial product to be full width");
    }

    // Zero-pad to match the required output width
    auto zeroWidth = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
    while (partialProducts.size() < op.getNumResults())
      partialProducts.push_back(zeroWidth);

    assert(partialProducts.size() == op.getNumResults() &&
           "Expected number of booth partial products to match results");

    rewriter.replaceOp(op, partialProducts);
    return success();
  }
};

struct DatapathPosPartialProductOpConversion
    : OpRewritePattern<PosPartialProductOp> {
  using OpRewritePattern<PosPartialProductOp>::OpRewritePattern;

  DatapathPosPartialProductOpConversion(MLIRContext *context, bool forceBooth)
      : OpRewritePattern<PosPartialProductOp>(context),
        forceBooth(forceBooth){};

  const bool forceBooth;

  LogicalResult matchAndRewrite(PosPartialProductOp op,
                                PatternRewriter &rewriter) const override {

    Value a = op.getAddend0();
    Value b = op.getAddend1();
    Value c = op.getMultiplicand();
    unsigned width = a.getType().getIntOrFloatBitWidth();

    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(0), 0);
      return success();
    }

    // TODO: Implement Booth lowering
    return lowerAndArray(rewriter, a, b, c, op, width);
  }

private:
  static LogicalResult lowerAndArray(PatternRewriter &rewriter, Value a,
                                     Value b, Value c, PosPartialProductOp op,
                                     unsigned width) {

    Location loc = op.getLoc();
    // Encode (a+b) by implementing a half-adder - then note the following
    // fact carry[i] & save[i] == false
    auto carry = rewriter.createOrFold<comb::AndOp>(loc, a, b);
    auto save = rewriter.createOrFold<comb::XorOp>(loc, a, b);

    SmallVector<Value> carryBits = extractBits(rewriter, carry);
    SmallVector<Value> saveBits = extractBits(rewriter, save);

    // Reduce c width based on leading zeros
    auto rowWidth = width;
    auto knownBitsC = comb::computeKnownBits(c);
    if (!knownBitsC.Zero.isZero()) {
      if (knownBitsC.Zero.countLeadingOnes() > 1) {
        // Retain one leading zero to represent 2*{1'b0, c} = {c, 1'b0}
        // {'0, c} -> {1'b0, c}
        rowWidth -= knownBitsC.Zero.countLeadingOnes() - 1;
        c = rewriter.createOrFold<comb::ExtractOp>(loc, c, 0, rowWidth);
      }
    }

    // Compute 2*c for use in array construction
    Value zero = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    Value twoCWider =
        comb::ConcatOp::create(rewriter, loc, ValueRange{c, zero});
    Value twoC = comb::ExtractOp::create(rewriter, loc, twoCWider, 0, rowWidth);

    // AND Array Construction:
    // pp[i] = ( (carry[i] * (c<<1)) | (save[i] * c) ) << i
    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);

    assert(op.getNumResults() <= width &&
           "Cannot return more results than the operator width");

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto replSave =
          rewriter.createOrFold<comb::ReplicateOp>(loc, saveBits[i], rowWidth);
      auto replCarry =
          rewriter.createOrFold<comb::ReplicateOp>(loc, carryBits[i], rowWidth);

      auto ppRowSave = rewriter.createOrFold<comb::AndOp>(loc, replSave, c);
      auto ppRowCarry =
          rewriter.createOrFold<comb::AndOp>(loc, replCarry, twoC);
      auto ppRow =
          rewriter.createOrFold<comb::OrOp>(loc, ppRowSave, ppRowCarry);
      auto ppAlign = ppRow;
      if (i > 0) {
        auto shiftBy = hw::ConstantOp::create(rewriter, loc, APInt(i, 0));
        ppAlign =
            comb::ConcatOp::create(rewriter, loc, ValueRange{ppRow, shiftBy});
      }

      // May need to truncate shifted value
      if (rowWidth + i > width) {
        auto ppAlignTrunc =
            rewriter.createOrFold<comb::ExtractOp>(loc, ppAlign, 0, width);
        partialProducts.push_back(ppAlignTrunc);
        continue;
      }
      // May need to zero pad to approriate width
      if (rowWidth + i < width) {
        auto padding = width - rowWidth - i;
        Value zeroPad =
            hw::ConstantOp::create(rewriter, loc, APInt(padding, 0));
        partialProducts.push_back(rewriter.createOrFold<comb::ConcatOp>(
            loc, ValueRange{zeroPad, ppAlign})); // Pad to full width
        continue;
      }

      partialProducts.push_back(ppAlign);
    }

    rewriter.replaceOp(op, partialProducts);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Datapath to Comb pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDatapathToCombPass
    : public impl::ConvertDatapathToCombBase<ConvertDatapathToCombPass> {
  void runOnOperation() override;
  using ConvertDatapathToCombBase<
      ConvertDatapathToCombPass>::ConvertDatapathToCombBase;
};
} // namespace

static LogicalResult applyPatternsGreedilyWithTimingInfo(
    Operation *op, RewritePatternSet &&patterns,
    synth::IncrementalLongestPathAnalysis *analysis) {
  // TODO: Topologically sort the operations in the module to ensure that all
  // dependencies are processed before their users.
  mlir::GreedyRewriteConfig config;
  // Set the listener to update timing information
  // HACK: Setting max iterations to 2 to ensure that the patterns are
  // one-shot, making sure target operations are datapath operations are
  // replaced.
  config.setMaxIterations(2).setListener(analysis).setUseTopDownTraversal(true);

  // Apply the patterns greedily
  if (failed(mlir::applyPatternsGreedily(op, std::move(patterns), config)))
    return failure();

  return success();
}

void ConvertDatapathToCombPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  patterns.add<DatapathPartialProductOpConversion,
               DatapathPosPartialProductOpConversion>(patterns.getContext(),
                                                      forceBooth);
  synth::IncrementalLongestPathAnalysis *analysis = nullptr;
  if (timingAware)
    analysis = &getAnalysis<synth::IncrementalLongestPathAnalysis>();

  if (lowerCompressToAdd)
    // Lower compressors to simple add operations for downstream optimisations
    patterns.add<DatapathCompressOpAddConversion>(patterns.getContext());
  if (lowerCompress)
    // Lower compressors to a complete gate-level implementation
    patterns.add<DatapathCompressOpConversion>(patterns.getContext(), analysis);

  if (failed(applyPatternsGreedilyWithTimingInfo(
          getOperation(), std::move(patterns), analysis)))
    return signalPassFailure();

  // Verify that all Datapath operations have been successfully converted.
  // Walk the operation and check for any remaining Datapath dialect
  // operations.
  auto result = getOperation()->walk([&](Operation *op) {
    if (llvm::isa<datapath::CompressOp>(op) && !lowerCompress &&
        !lowerCompressToAdd)
      return WalkResult::advance();
    if (llvm::isa_and_nonnull<datapath::DatapathDialect>(op->getDialect())) {
      op->emitError("Datapath operation not converted: ") << *op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
