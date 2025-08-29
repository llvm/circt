//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

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
                               aig::IncrementalLongestPathAnalysis *analysis)
      : mlir::OpRewritePattern<CompressOp>(context), analysis(analysis) {}

  LogicalResult
  matchAndRewrite(CompressOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();

    SmallVector<SmallVector<Value>> addends;
    for (auto input : inputs) {
      addends.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    }

    // Wallace tree reduction
    // TODO: Implement a more efficient compression algorithm to compete with
    // yosys's `alumacc` lowering - a coarse grained timing model would help to
    // sort the inputs according to arrival time.
    auto targetAddends = op.getNumResults();
    if (analysis) {
      // Sort the addends row based on the delay of the input.
      for (size_t j = 0; j < addends[0].size(); ++j) {
        SmallVector<std::pair<int64_t, Value>> delays;
        for (auto &addend : addends) {
          auto delay = analysis->getOrComputeMaxDelay(addend[j], 0);
          if (failed(delay))
            return rewriter.notifyMatchFailure(op,
                                               "Failed to get delay for input");
          delays.push_back(std::make_pair(*delay, addend[j]));
        }
        std::stable_sort(delays.begin(), delays.end(),
                         [](const std::pair<int64_t, Value> &a,
                            const std::pair<int64_t, Value> &b) {
                           return a.first < b.first;
                         });
        for (size_t i = 0; i < addends.size(); ++i)
          addends[i][j] = delays[i].second;
      }
    }
    rewriter.replaceOp(op, comb::wallaceReduction(rewriter, loc, width,
                                                  targetAddends, addends));
    return success();
  }

private:
  aig::IncrementalLongestPathAnalysis *analysis = nullptr;
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

    // Use width as a heuristic to guide partial product implementation
    if (width > 16 || forceBooth)
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
      auto repl = comb::ReplicateOp::create(rewriter, loc, bBits[i], width);
      auto ppRow = comb::AndOp::create(rewriter, loc, repl, a);
      auto shiftBy = hw::ConstantOp::create(rewriter, loc, APInt(width, i));
      auto ppAlign = comb::ShlOp::create(rewriter, loc, ppRow, shiftBy);
      partialProducts.push_back(ppAlign);
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
    auto oneWidth = hw::ConstantOp::create(rewriter, loc, APInt(width, 1));
    Value twoA = comb::ShlOp::create(rewriter, loc, a, oneWidth);

    SmallVector<Value> bBits = extractBits(rewriter, b);

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
      Value encOne = comb::XorOp::create(rewriter, loc, bi, bim1, true);
      // Is the encoding two = (bip1 & ~bi & ~bim1) | (~bip1 & bi & bim1)
      Value constOne = hw::ConstantOp::create(rewriter, loc, APInt(1, 1));
      Value biInv = comb::XorOp::create(rewriter, loc, bi, constOne, true);
      Value bip1Inv = comb::XorOp::create(rewriter, loc, bip1, constOne, true);
      Value bim1Inv = comb::XorOp::create(rewriter, loc, bim1, constOne, true);

      Value andLeft = comb::AndOp::create(rewriter, loc,
                                          ValueRange{bip1Inv, bi, bim1}, true);
      Value andRight = comb::AndOp::create(
          rewriter, loc, ValueRange{bip1, biInv, bim1Inv}, true);
      Value encTwo = comb::OrOp::create(rewriter, loc, andLeft, andRight, true);

      Value encNegRepl =
          comb::ReplicateOp::create(rewriter, loc, encNeg, width);
      Value encOneRepl =
          comb::ReplicateOp::create(rewriter, loc, encOne, width);
      Value encTwoRepl =
          comb::ReplicateOp::create(rewriter, loc, encTwo, width);

      // Select between 2*a or 1*a or 0*a
      Value selTwoA = comb::AndOp::create(rewriter, loc, encTwoRepl, twoA);
      Value selOneA = comb::AndOp::create(rewriter, loc, encOneRepl, a);
      Value magA = comb::OrOp::create(rewriter, loc, selTwoA, selOneA, true);

      // Conditionally invert the row
      Value ppRow = comb::XorOp::create(rewriter, loc, magA, encNegRepl, true);

      // No sign-correction in the first row
      if (i == 0) {
        partialProducts.push_back(ppRow);
        encNegPrev = encNeg;
        continue;
      }

      // Insert a sign-correction from the previous row
      assert(i >= 2 && "Expected i to be at least 2 for sign correction");
      // {ppRow, 0, encNegPrev} << 2*(i-1)
      Value withSignCorrection = comb::ConcatOp::create(
          rewriter, loc, ValueRange{ppRow, zeroFalse, encNegPrev});
      Value ppAlignPre =
          comb::ExtractOp::create(rewriter, loc, withSignCorrection, 0, width);
      Value shiftBy =
          hw::ConstantOp::create(rewriter, loc, APInt(width, i - 2));
      Value ppAlign = comb::ShlOp::create(rewriter, loc, ppAlignPre, shiftBy);
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
    aig::IncrementalLongestPathAnalysis *analysis) {
  // TODO: Topologically sort the operations in the module to ensure that all
  // dependencies are processed before their users.
  mlir::GreedyRewriteConfig config;
  // Set the listener to update timing information
  // HACK: Setting max iterations to 2 to ensure that the patterns are one-shot,
  // making sure target operations are datapath operations are replaced.
  config.setMaxIterations(2).setListener(analysis).setUseTopDownTraversal(true);

  // Apply the patterns greedily
  if (failed(mlir::applyPatternsGreedily(op, std::move(patterns), config)))
    return failure();

  return success();
}

void ConvertDatapathToCombPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  patterns.add<DatapathPartialProductOpConversion>(patterns.getContext(),
                                                   forceBooth);
  aig::IncrementalLongestPathAnalysis *analysis = nullptr;
  if (timingAware)
    analysis = &getAnalysis<aig::IncrementalLongestPathAnalysis>();
  if (lowerCompressToAdd)
    // Lower compressors to simple add operations for downstream optimisations
    patterns.add<DatapathCompressOpAddConversion>(patterns.getContext());
  else
    // Lower compressors to a complete gate-level implementation
    patterns.add<DatapathCompressOpConversion>(patterns.getContext(), analysis);

  if (failed(applyPatternsGreedilyWithTimingInfo(
          getOperation(), std::move(patterns), analysis)))
    return signalPassFailure();
}
