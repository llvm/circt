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
    if (op.getNumResults() > 16 || forceBooth)
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

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);
    // AND Array Construction:
    // partialProducts[i] = ({b[i],..., b[i]} & a) << i
    assert(op.getNumResults() <= width &&
           "Cannot return more results than the operator width");

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto repl =
          rewriter.createOrFold<comb::ReplicateOp>(loc, bBits[i], width);
      auto ppRow = rewriter.createOrFold<comb::AndOp>(loc, repl, a);
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
    Location loc = op.getLoc();
    auto zeroFalse = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    auto zeroWidth = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));

    // Detect leading zeros in multiplicand due to zero-extension
    // and truncate to reduce partial product bits
    // {'0, a} * {'0, b}
    auto rowWidth = width;
    auto knownBitsA = comb::computeKnownBits(a);
    if (!knownBitsA.Zero.isZero()) {
      if (knownBitsA.Zero.countLeadingOnes() > 1) {
        // Retain one leading zero to represent 2*{1'b0, a} = {a, 1'b0}
        // {'0, a} -> {1'b0, a}
        rowWidth -= knownBitsA.Zero.countLeadingOnes() - 1;
        a = rewriter.createOrFold<comb::ExtractOp>(loc, a, 0, rowWidth);
      }
    }

    // TODO - replace with a concatenation to aid longest path analysis
    auto oneRowWidth =
        hw::ConstantOp::create(rewriter, loc, APInt(rowWidth, 1));
    // Booth encoding will select each row from {-2a, -1a, 0, 1a, 2a}
    Value twoA = rewriter.createOrFold<comb::ShlOp>(loc, a, oneRowWidth);

    // Encode based on the bits of b
    // TODO: sort a and b based on non-zero bits to encode the smaller input
    SmallVector<Value> bBits = extractBits(rewriter, b);

    // Identify zero bits of b to reduce height of partial product array
    auto knownBitsB = comb::computeKnownBits(b);
    if (!knownBitsB.Zero.isZero()) {
      for (unsigned i = 0; i < width; ++i)
        if (knownBitsB.Zero[i])
          bBits[i] = zeroFalse;
    }

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);

    // Booth encoding halves array height by grouping three bits at a time:
    // partialProducts[i] = a * (-2*b[2*i+1] + b[2*i] + b[2*i-1]) << 2*i
    // encNeg \approx (-2*b[2*i+1] + b[2*i] + b[2*i-1]) <= 0
    // encOne = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 1
    // encTwo = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 2
    Value encNegPrev;

    // For even width - additional row contains the final sign correction
    for (unsigned i = 0; i <= width; i += 2) {
      // Get Booth bits: b[i+1], b[i], b[i-1] (b[-1] = 0)
      Value bim1 = (i == 0) ? zeroFalse : bBits[i - 1];
      Value bi = (i < width) ? bBits[i] : zeroFalse;
      Value bip1 = (i + 1 < width) ? bBits[i + 1] : zeroFalse;

      // Is the encoding zero or negative (an approximation)
      Value encNeg = bip1;
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

      // Sign-extension Optimisation:
      // Section 7.2.2 of "Application Specific Arithmetic" by Dinechin &
      // Kumm Handle sign-extension and padding to full width s = encNeg
      // (sign-bit) {s, s, s, s, s, pp} = {1, 1, 1, 1, 1, pp}
      //                     + {0, 0, 0, 0,!s, '0}
      // Applying this to every row we create an upper-triangle of 1s that
      // can be optimised away since they will not affect the final sum.
      // {!s3,  0,!s2,  0,!s1,  0}
      // {  1,  1,  1,  1,  1, p1}
      // {  1,  1,  1,   p2      }
      // {  1,       p3          }
      if (rowWidth < width) {
        auto padding = width - rowWidth;
        auto encNegInv = bip1Inv;

        // Sign-extension trick not worth it for padding < 3
        if (padding < 3) {
          Value encNegPad =
              rewriter.createOrFold<comb::ReplicateOp>(loc, encNeg, padding);
          ppRow = rewriter.createOrFold<comb::ConcatOp>(
              loc, ValueRange{encNegPad, ppRow}); // Pad to full width
        } else if (i == 0) {
          // First row = {!encNeg, encNeg, encNeg, ppRow}
          ppRow = rewriter.createOrFold<comb::ConcatOp>(
              loc, ValueRange{encNegInv, encNeg, encNeg, ppRow});
        } else {
          // Remaining rows = {1, !encNeg, ppRow}
          ppRow = rewriter.createOrFold<comb::ConcatOp>(
              loc, ValueRange{constOne, encNegInv, ppRow});
        }

        // Zero pad to full width
        auto rowWidth = ppRow.getType().getIntOrFloatBitWidth();
        if (rowWidth < width) {
          auto zeroPad =
              hw::ConstantOp::create(rewriter, loc, APInt(width - rowWidth, 0));
          ppRow = rewriter.createOrFold<comb::ConcatOp>(
              loc, ValueRange{zeroPad, ppRow});
        }
      }

      // No sign-correction in the first row
      if (i == 0) {
        partialProducts.push_back(ppRow);
        encNegPrev = encNeg;
        continue;
      }

      // Insert a sign-correction from the previous row
      assert(i >= 2 && "Expected i to be at least 2 for sign correction");
      // {ppRow, 0, encNegPrev} << 2*(i-1)
      Value withSignCorrection = rewriter.createOrFold<comb::ConcatOp>(
          loc, ValueRange{ppRow, zeroFalse, encNegPrev});
      Value ppAlignPre = rewriter.createOrFold<comb::ExtractOp>(
          loc, withSignCorrection, 0, width);
      Value shiftBy =
          hw::ConstantOp::create(rewriter, loc, APInt(width, i - 2));
      Value ppAlign =
          rewriter.createOrFold<comb::ShlOp>(loc, ppAlignPre, shiftBy);
      partialProducts.push_back(ppAlign);
      encNegPrev = encNeg;

      if (partialProducts.size() == op.getNumResults())
        break;
    }

    // Zero-pad to match the required output width
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
  else
    // Lower compressors to a complete gate-level implementation
    patterns.add<DatapathCompressOpConversion>(patterns.getContext(), analysis);

  if (failed(applyPatternsGreedilyWithTimingInfo(
          getOperation(), std::move(patterns), analysis)))
    return signalPassFailure();

  // Verify that all Datapath operations have been successfully converted.
  // Walk the operation and check for any remaining Datapath dialect
  // operations.
  auto result = getOperation()->walk([&](Operation *op) {
    if (llvm::isa_and_nonnull<datapath::DatapathDialect>(op->getDialect())) {
      op->emitError("Datapath operation not converted: ") << *op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
