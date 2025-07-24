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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
struct DatapathCompressOpAddConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();
    // Sum all the inputs - set that to result value 0
    auto addOp = rewriter.create<comb::AddOp>(loc, inputs, true);
    // Replace remaining results with zeros
    auto zeroOp = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    SmallVector<Value> results(op.getNumResults() - 1, zeroOp);
    results.push_back(addOp);
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Replace compressor by a wallace tree of full-adders
struct DatapathCompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();

    SmallVector<SmallVector<Value>> addends;
    for (auto input : inputs) {
      addends.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    }

    // Wallace tree reduction
    // TODO - implement a more efficient compression algorithm to compete with
    // yosys's `alumacc` lowering - a coarse grained timing model would help to
    // sort the inputs according to arrival time.
    auto targetAddends = op.getNumResults();
    rewriter.replaceOp(op, comb::wallaceReduction(rewriter, loc, width,
                                                  targetAddends, addends));
    return success();
  }
};

struct DatapathPartialProductOpConversion
    : OpConversionPattern<PartialProductOp> {
  using OpConversionPattern<PartialProductOp>::OpConversionPattern;

  DatapathPartialProductOpConversion(MLIRContext *context, bool forceBooth)
      : OpConversionPattern<PartialProductOp>(context),
        forceBooth(forceBooth) {};

  const bool forceBooth;

  LogicalResult
  matchAndRewrite(PartialProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

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
  static LogicalResult lowerAndArray(ConversionPatternRewriter &rewriter,
                                     Value a, Value b, PartialProductOp op,
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
      auto repl = rewriter.create<comb::ReplicateOp>(loc, bBits[i], width);
      auto ppRow = rewriter.create<comb::AndOp>(loc, repl, a);
      auto shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i));
      auto ppAlign = rewriter.create<comb::ShlOp>(loc, ppRow, shiftBy);
      partialProducts.push_back(ppAlign);
    }

    rewriter.replaceOp(op, partialProducts);
    return success();
  }

  static LogicalResult lowerBoothArray(ConversionPatternRewriter &rewriter,
                                       Value a, Value b, PartialProductOp op,
                                       unsigned width) {
    Location loc = op.getLoc();
    auto zeroFalse = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    auto zeroWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    auto oneWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 1));
    Value twoA = rewriter.create<comb::ShlOp>(loc, a, oneWidth);

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
      Value encOne = rewriter.create<comb::XorOp>(loc, bi, bim1, true);
      // Is the encoding two = (bip1 & ~bi & ~bim1) | (~bip1 & bi & bim1)
      Value constOne = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
      Value biInv = rewriter.create<comb::XorOp>(loc, bi, constOne, true);
      Value bip1Inv = rewriter.create<comb::XorOp>(loc, bip1, constOne, true);
      Value bim1Inv = rewriter.create<comb::XorOp>(loc, bim1, constOne, true);

      Value andLeft = rewriter.create<comb::AndOp>(
          loc, ValueRange{bip1Inv, bi, bim1}, true);
      Value andRight = rewriter.create<comb::AndOp>(
          loc, ValueRange{bip1, biInv, bim1Inv}, true);
      Value encTwo = rewriter.create<comb::OrOp>(loc, andLeft, andRight, true);

      Value encNegRepl = rewriter.create<comb::ReplicateOp>(loc, encNeg, width);
      Value encOneRepl = rewriter.create<comb::ReplicateOp>(loc, encOne, width);
      Value encTwoRepl = rewriter.create<comb::ReplicateOp>(loc, encTwo, width);

      // Select between 2*a or 1*a or 0*a
      Value selTwoA = rewriter.create<comb::AndOp>(loc, encTwoRepl, twoA);
      Value selOneA = rewriter.create<comb::AndOp>(loc, encOneRepl, a);
      Value magA = rewriter.create<comb::OrOp>(loc, selTwoA, selOneA, true);

      // Conditionally invert the row
      Value ppRow = rewriter.create<comb::XorOp>(loc, magA, encNegRepl, true);

      // No sign-correction in the first row
      if (i == 0) {
        partialProducts.push_back(ppRow);
        encNegPrev = encNeg;
        continue;
      }

      // Insert a sign-correction from the previous row
      assert(i >= 2 && "Expected i to be at least 2 for sign correction");
      // {ppRow, 0, encNegPrev} << 2*(i-1)
      Value withSignCorrection = rewriter.create<comb::ConcatOp>(
          loc, ValueRange{ppRow, zeroFalse, encNegPrev});
      Value ppAlignPre =
          rewriter.create<comb::ExtractOp>(loc, withSignCorrection, 0, width);
      Value shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i - 2));
      Value ppAlign = rewriter.create<comb::ShlOp>(loc, ppAlignPre, shiftBy);
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

void ConvertDatapathToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<DatapathDialect>();

  RewritePatternSet patterns(&getContext());

  patterns.add<DatapathPartialProductOpConversion>(patterns.getContext(),
                                                   forceBooth);

  if (lowerCompressToAdd)
    // Lower compressors to simple add operations for downstream optimisations
    patterns.add<DatapathCompressOpAddConversion>(patterns.getContext());
  else
    // Lower compressors to a complete gate-level implementation
    patterns.add<DatapathCompressOpConversion>(patterns.getContext());

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
